//! Variable-time BN254 arithmetic for proof verification.
//!
//! Use [`batch`] for untrusted bytes. It validates canonical encoding, curve
//! membership, and G2 subgroup membership. Typed curve values are not
//! validated because their fields are public.
//!
//! This crate uses input-dependent branches, memory access, caches, and early
//! exits. Use it only for public inputs. Do not use it with secrets.
//!
//! The default build uses `no_std` and `alloc`. The `std` feature enables
//! prepared verifier data and runtime IFMA dispatch. The `force-portable`
//! feature disables assembly and IFMA backends.

#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]

extern crate alloc;

#[cfg(any(
    all(narsil_mont4_x86_64_adx, not(feature = "force-portable")),
    all(
        target_arch = "aarch64",
        target_vendor = "apple",
        not(feature = "force-portable")
    )
))]
mod abi;
pub mod batch;
#[cfg(all(
    any(narsil_avx512_ifma, narsil_x86_runtime_ifma),
    not(feature = "force-portable")
))]
mod batch8;
/// Compute eight pairing products with AVX-512 IFMA.
#[cfg(all(narsil_avx512_ifma, not(feature = "force-portable")))]
pub use batch8::multi_pairing8;

mod const_tower;
pub mod consts;
#[cfg(all(
    feature = "std",
    any(narsil_avx512_ifma, narsil_x86_runtime_ifma),
    not(feature = "force-portable")
))]
mod final_exp_ifma;
pub mod fp;
pub mod fp12;
pub mod fp2;
mod fp2_fast;
pub mod fp6;
pub mod fr;
pub mod g1;
mod g1_fast;
pub mod g2;
mod g2_fast;
mod limb;
#[cfg(all(
    feature = "std",
    any(narsil_avx512_ifma, narsil_x86_runtime_ifma),
    not(feature = "force-portable")
))]
mod miller_coeff_ifma;
mod msm;
pub use msm::msm_variable_time_affine;
pub mod pairing;
#[cfg(feature = "std")]
mod prepared_verifier;
#[cfg(feature = "std")]
pub use prepared_verifier::{
    CubicResidueClass, FixedPair, LivePair, MAX_PREPARED_VERIFIER_TERMS, PairingResidueWitness,
    PreparedTerm, PreparedVerifier, PreparedVerifierError, ValidatedOnlineTerms,
};
#[cfg(test)]
mod sos_tests;
#[cfg(kani)]
mod verify_kani;
mod wnaf;
#[cfg(any(
    all(
        any(target_arch = "x86", target_arch = "x86_64"),
        any(all(narsil_x86_runtime_ifma, not(feature = "force-portable")), test)
    ),
    feature = "backend-report"
))]
mod x86_runtime;

/// Report the detected x86 capability and the real batch backend selected.
///
/// This hook exists only for repository benchmark/diagnostic harnesses. It is
/// absent from ordinary builds and does not change the production API or ABI.
#[cfg(feature = "backend-report")]
#[doc(hidden)]
pub fn backend_report_for_harness() -> &'static str {
    x86_runtime::backend_report()
}

#[cfg(test)]
mod arkworks_bn254_0_5_tests;

pub use batch::{
    AltBn128BatchError, FR_MAX_ELEMS, G1_BYTES, G1Bytes, G2_BYTES, G2Bytes, InputError,
    MSM_MAX_POINTS, PAIR_BYTES, PAIRING_MAX_PAIRS, PairBytes, PodG1G2Pair, PodG1Point, PodG2Point,
    PodPairingResult, PodScalar, SCALAR_BYTES, ScalarBytes, Version, alt_bn128_fr_batch_invert,
    alt_bn128_fr_lincomb, alt_bn128_g1_msm, alt_bn128_pairing_check, fr_batch_invert, fr_lincomb,
    g1_msm, pairing_product_is_one,
};
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
mod readme_doctests {}

pub use fp::Fp;
pub use fp2::Fp2;
pub use fp6::Fp6;
pub use fp12::Fp12;
pub use fr::Fr;
pub use g1::{G1Affine, G1Projective};
pub use g2::{G2Affine, G2Projective};
pub use pairing::{Gt, multi_pairing, pairing};

/// BN254 base field.
pub type Fq = Fp;
/// Quadratic extension of the BN254 base field.
pub type Fq2 = Fp2;
/// Sextic extension of the BN254 base field.
pub type Fq6 = Fp6;
/// Degree-twelve extension of the BN254 base field.
pub type Fq12 = Fp12;

/// BN254 pairing engine facade.
#[derive(Clone, Copy, Debug, Default)]
pub struct Bn254;

impl Bn254 {
    /// Compute one optimal Ate pairing.
    #[inline]
    pub fn pairing(p: &G1Affine, q: &G2Affine) -> Gt {
        pairing::pairing(p, q)
    }

    /// Compute a product of pairings with one final exponentiation.
    #[inline]
    pub fn multi_pairing(pairs: &[(&G1Affine, &G2Affine)]) -> Gt {
        pairing::multi_pairing(pairs)
    }

    /// Compute the Miller-loop value before final exponentiation.
    #[inline]
    pub fn miller_loop(p: &G1Affine, q: &G2Affine) -> Fp12 {
        pairing::miller_loop(p, q)
    }

    /// Map a Miller-loop value into the target group.
    #[inline]
    pub fn final_exponentiation(value: &Fp12) -> Gt {
        pairing::final_exponentiation(value)
    }
}

#[cfg(test)]
mod pairing_tests {
    use super::*;

    #[test]
    fn bilinearity_smoke() {
        let p = G1Affine::generator();
        let q = G2Affine::generator();
        let a = Fr::from_u64(7);
        let e1 = pairing(&p, &q).pow_u64(7);
        let pa = G1Projective::from(p).mul_scalar(a).to_affine();
        let e2 = pairing(&pa, &q);
        assert_eq!(e1, e2);

        let qa = G2Projective::from(q).mul_scalar(a).to_affine();
        let e3 = pairing(&p, &qa);
        assert_eq!(e1, e3);
    }

    #[test]
    fn identity_pairing() {
        let p = G1Affine::generator();
        let q = G2Affine::generator();
        assert_eq!(pairing(&G1Affine::identity(), &q), Fp12::ONE);
        assert_eq!(pairing(&p, &G2Affine::identity()), Fp12::ONE);
    }

    #[test]
    fn g1_order() {
        // [r] P = O
        let p = G1Projective::generator();
        // [r-1]P = -P
        let rm1 = Fr::from_raw([
            0x43e1f593f0000000, // r-1 low
            0x2833e84879b97091,
            0xb85045b68181585d,
            0x30644e72e131a029,
        ]);
        let q = p.mul_scalar(rm1).to_affine();
        assert_eq!(q.negate(), G1Affine::generator());
    }
}
