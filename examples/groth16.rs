//! Groth16 pairing product:
//! `e(alpha, beta) * e(L, gamma) * e(C, delta) * e(-A, B) = 1`.
//!
//! The points are a synthetic accepting instance, not a circuit proof.

use helius_narsil::{
    FixedPair, Fr, G1Affine, G1Projective, G2Affine, G2Projective, LivePair, PreparedTerm,
    PreparedVerifier, PreparedVerifierError,
};

fn g1(k: u64) -> G1Affine {
    G1Projective::generator()
        .mul_scalar(Fr::from_u64(k))
        .to_affine()
}

fn g2(k: u64) -> G2Affine {
    G2Projective::from(G2Affine::generator())
        .mul_scalar(Fr::from_u64(k))
        .to_affine()
}

fn main() -> Result<(), PreparedVerifierError> {
    let alpha = g1(4);
    let beta = g2(1);
    let gamma = g2(1);
    let delta = g2(2);

    let vk = PreparedVerifier::new(
        &[gamma, delta],
        &[FixedPair {
            g1: alpha,
            g2: beta,
        }],
    )?;

    let a = g1(1);
    let b = g2(7);
    let c = g1(1);
    let public = g1(1);

    assert!(vk.verify(
        &[
            PreparedTerm {
                g1: public,
                prepared_g2: 0,
            },
            PreparedTerm {
                g1: c,
                prepared_g2: 1,
            },
        ],
        &[LivePair {
            g1: a.negate(),
            g2: b,
        }],
    )?);

    Ok(())
}
