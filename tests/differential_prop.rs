//! Property-based differential tests against arkworks and algebraic
//! identities, driven only through the public API.
//!
//! Inputs are adversarial: uniform residues, values within 2^16 of the
//! modulus, sparse bit patterns, and values whose radix-52 limbs sit at the
//! 2^52 boundary (the carry edges of the IFMA backend). On hosts where the
//! build dispatches to a vector or assembly backend at run time (x86-64 with
//! AVX-512 IFMA, aarch64-apple scalar assembly), every equality below is a
//! differential test of that backend against arkworks and against the
//! crate's own scalar composition. On other hosts the same assertions pin
//! the portable path.
//!
//! The lazy radix-52 forms of the IFMA backend (`FpVec8L<B>`, `LAZY_CAP`) are
//! crate-private, so no test can call them. Their public drivers are the
//! eight-lane batch entries, which this file exercises under
//! `narsil_avx512_ifma` against the scalar composition. The bound algebra
//! behind them is proved separately in `src/verify_kani.rs`.
//!
//! Failure persistence: every block below uses [`config`], which pins the
//! seed file to `tests/differential_prop.proptest-regressions`. That file is
//! tracked, so a counterexample found once is replayed on every later run.

use ark_bn254::{
    Bn254, Fq as ArkFq, Fq2 as ArkFq2, Fq6 as ArkFq6, Fq12 as ArkFq12, Fr as ArkFr,
    G1Affine as ArkG1Affine, G1Projective as ArkG1Projective, G2Affine as ArkG2Affine,
    G2Projective as ArkG2Projective,
};
use ark_ec::{CurveGroup, PrimeGroup, pairing::Pairing};
use ark_ff::{AdditiveGroup, BigInt, Field, PrimeField};
use helius_narsil::{
    Fp, Fp2, Fp6, Fp12, Fr, G1Affine, G1Projective, G2Affine, G2Projective, consts, pairing,
};
use proptest::prelude::*;
use proptest::test_runner::FileFailurePersistence;

/// The default `SourceParallel` persistence resolves its path by walking up
/// for `lib.rs` or `main.rs`. An integration test has neither, so proptest
/// warns once and then discards every counterexample it finds. Pinning the
/// path is what makes a found failure reproducible.
fn config(cases: u32) -> ProptestConfig {
    ProptestConfig {
        cases,
        failure_persistence: Some(Box::new(FileFailurePersistence::Direct(
            "tests/differential_prop.proptest-regressions",
        ))),
        ..ProptestConfig::default()
    }
}

const MASK52: u64 = (1 << 52) - 1;

fn gte(a: &[u64; 4], b: &[u64; 4]) -> bool {
    for i in (0..4).rev() {
        if a[i] != b[i] {
            return a[i] > b[i];
        }
    }
    true
}

fn sub4(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
    let mut out = [0u64; 4];
    let mut borrow = 0u64;
    for i in 0..4 {
        let (d, b1) = a[i].overflowing_sub(b[i]);
        let (d, b2) = d.overflowing_sub(borrow);
        out[i] = d;
        borrow = u64::from(b1) | u64::from(b2);
    }
    out
}

fn reduce(mut v: [u64; 4], m: &[u64; 4]) -> [u64; 4] {
    while gte(&v, m) {
        v = sub4(&v, m);
    }
    v
}

/// Repack five radix-52 limbs into the 4x64 value mod 2^256. Mirrors the
/// IFMA domain layout, so boundary limbs land on its carry edges.
fn from_radix52(l: [u64; 5]) -> [u64; 4] {
    [
        l[0] | (l[1] << 52),
        (l[1] >> 12) | (l[2] << 40),
        (l[2] >> 24) | (l[3] << 28),
        (l[3] >> 36) | (l[4] << 16),
    ]
}

fn limb52() -> impl Strategy<Value = u64> {
    prop_oneof![
        3 => Just(MASK52),
        2 => Just(MASK52 - 1),
        2 => Just(0u64),
        1 => Just(1u64),
        4 => any::<u64>().prop_map(|v| v & MASK52),
    ]
}

/// Canonical residues below `m`, biased toward reduction and carry edges.
fn residue(m: &'static [u64; 4]) -> impl Strategy<Value = [u64; 4]> {
    prop_oneof![
        5 => any::<[u64; 4]>().prop_map(|v| reduce(v, m)),
        3 => (1u64..=65_536).prop_map(|k| sub4(m, &[k, 0, 0, 0])),
        2 => (any::<u64>(), 0usize..4).prop_map(|(v, i)| {
            let mut out = [0u64; 4];
            out[i] = v;
            reduce(out, m)
        }),
        2 => prop::collection::vec(0u32..254, 0..5).prop_map(|bits| {
            let mut out = [0u64; 4];
            for bit in bits {
                out[(bit / 64) as usize] |= 1u64 << (bit % 64);
            }
            reduce(out, m)
        }),
        4 => [limb52(), limb52(), limb52(), limb52(), limb52()]
            .prop_map(|l| reduce(from_radix52(l), m)),
        1 => Just([0u64; 4]),
        1 => Just([1u64, 0, 0, 0]),
    ]
}

fn fp() -> BoxedStrategy<[u64; 4]> {
    residue(&consts::P).boxed()
}

fn fr() -> BoxedStrategy<[u64; 4]> {
    residue(&consts::R).boxed()
}

fn ark_fq(limbs: [u64; 4]) -> ArkFq {
    ArkFq::from_bigint(BigInt::new(limbs)).expect("residue is canonical")
}

fn ark_fr(limbs: [u64; 4]) -> ArkFr {
    ArkFr::from_bigint(BigInt::new(limbs)).expect("residue is canonical")
}

fn fp_from_ark(v: ArkFq) -> Fp {
    Fp::from_raw(v.into_bigint().0)
}

fn fp2_from_ark(v: ArkFq2) -> Fp2 {
    Fp2::new(fp_from_ark(v.c0), fp_from_ark(v.c1))
}

fn fp6_from_ark(v: ArkFq6) -> Fp6 {
    Fp6::new(fp2_from_ark(v.c0), fp2_from_ark(v.c1), fp2_from_ark(v.c2))
}

fn fp12_from_ark(v: ArkFq12) -> Fp12 {
    Fp12::new(fp6_from_ark(v.c0), fp6_from_ark(v.c1))
}

fn ark_fq2(l: &[[u64; 4]]) -> ArkFq2 {
    ArkFq2::new(ark_fq(l[0]), ark_fq(l[1]))
}

fn ark_fq12(l: &[[u64; 4]; 12]) -> ArkFq12 {
    ArkFq12::new(
        ArkFq6::new(ark_fq2(&l[0..2]), ark_fq2(&l[2..4]), ark_fq2(&l[4..6])),
        ArkFq6::new(ark_fq2(&l[6..8]), ark_fq2(&l[8..10]), ark_fq2(&l[10..12])),
    )
}

fn g1_pair(scalar: [u64; 4]) -> (G1Affine, ArkG1Affine) {
    let ark = (ArkG1Projective::generator() * ark_fr(scalar)).into_affine();
    let ours = G1Projective::generator()
        .mul_scalar(Fr::from_raw(scalar))
        .to_affine();
    (ours, ark)
}

fn g2_pair(scalar: [u64; 4]) -> (G2Affine, ArkG2Affine) {
    let ark = (ArkG2Projective::generator() * ark_fr(scalar)).into_affine();
    let ours = G2Projective::from(G2Affine::generator())
        .mul_scalar(Fr::from_raw(scalar))
        .to_affine();
    (ours, ark)
}

proptest! {
    #![proptest_config(config(512))]

    #[test]
    fn fp_arithmetic_matches_arkworks(a in fp(), b in fp()) {
        let (ha, hb) = (Fp::from_raw(a), Fp::from_raw(b));
        let (aa, ab) = (ark_fq(a), ark_fq(b));

        prop_assert_eq!(ha + hb, fp_from_ark(aa + ab));
        prop_assert_eq!(ha - hb, fp_from_ark(aa - ab));
        prop_assert_eq!(ha * hb, fp_from_ark(aa * ab));
        prop_assert_eq!(ha.square(), fp_from_ark(aa.square()));
        prop_assert_eq!(ha.double(), fp_from_ark(aa.double()));
        prop_assert_eq!(ha.negate(), fp_from_ark(-aa));
        prop_assert_eq!(ha.invert(), aa.inverse().map(fp_from_ark));
        match ha.sqrt() {
            Some(r) => prop_assert_eq!(r.square(), ha),
            None => prop_assert!(aa.sqrt().is_none()),
        }
        prop_assert_eq!(Fp::from_bytes_be(&ha.to_bytes_be()), Some(ha));
        prop_assert_eq!(Fp::from_bytes_le(&ha.to_bytes_le()), Some(ha));
        prop_assert_eq!(ha.to_raw(), a);
    }

    /// Both operands within 2^16 below p, so every sum enters the reduction
    /// step just under 2p and every difference just above -p. Random residues
    /// hit that band with negligible probability.
    #[test]
    fn fp_sums_approaching_two_p_match_arkworks(i in 1u64..=65_536, j in 1u64..=65_536) {
        let a = sub4(&consts::P, &[i, 0, 0, 0]);
        let b = sub4(&consts::P, &[j, 0, 0, 0]);
        let (ha, hb) = (Fp::from_raw(a), Fp::from_raw(b));
        let (aa, ab) = (ark_fq(a), ark_fq(b));

        prop_assert_eq!(ha + hb, fp_from_ark(aa + ab));
        prop_assert_eq!(ha - hb, fp_from_ark(aa - ab));
        prop_assert_eq!(ha * hb, fp_from_ark(aa * ab));
        prop_assert_eq!(ha.double(), fp_from_ark(aa.double()));
        prop_assert_eq!(ha.square(), fp_from_ark(aa.square()));
    }

    #[test]
    fn fr_arithmetic_matches_arkworks(a in fr(), b in fr()) {
        let (ha, hb) = (Fr::from_raw(a), Fr::from_raw(b));
        let (aa, ab) = (ark_fr(a), ark_fr(b));
        let fr_from_ark = |v: ArkFr| Fr::from_raw(v.into_bigint().0);

        prop_assert_eq!(ha + hb, fr_from_ark(aa + ab));
        prop_assert_eq!(ha - hb, fr_from_ark(aa - ab));
        prop_assert_eq!(ha * hb, fr_from_ark(aa * ab));
        prop_assert_eq!(ha.square(), fr_from_ark(aa.square()));
        prop_assert_eq!(ha.negate(), fr_from_ark(-aa));
        prop_assert_eq!(ha.invert(), aa.inverse().map(fr_from_ark));
        prop_assert_eq!(ha.to_raw(), a);
    }
}

proptest! {
    #![proptest_config(config(256))]

    #[test]
    fn fp2_arithmetic_matches_arkworks(
        a in prop::array::uniform2(fp()),
        b in prop::array::uniform2(fp()),
    ) {
        let ha = Fp2::new(Fp::from_raw(a[0]), Fp::from_raw(a[1]));
        let hb = Fp2::new(Fp::from_raw(b[0]), Fp::from_raw(b[1]));
        let aa = ark_fq2(&a);
        let ab = ark_fq2(&b);
        let xi = ArkFq2::new(ArkFq::from(9u64), ArkFq::from(1u64));

        prop_assert_eq!(ha + hb, fp2_from_ark(aa + ab));
        prop_assert_eq!(ha - hb, fp2_from_ark(aa - ab));
        prop_assert_eq!(ha * hb, fp2_from_ark(aa * ab));
        prop_assert_eq!(ha.square(), fp2_from_ark(aa.square()));
        prop_assert_eq!(ha.mul_by_nonresidue(), fp2_from_ark(aa * xi));
        prop_assert_eq!(ha.conjugate(), fp2_from_ark(ArkFq2::new(aa.c0, -aa.c1)));
        prop_assert_eq!(ha.invert(), aa.inverse().map(fp2_from_ark));
        let mut fa = aa;
        fa.frobenius_map_in_place(1);
        prop_assert_eq!(ha.frobenius_map(), fp2_from_ark(fa));
    }
}

proptest! {
    #![proptest_config(config(64))]

    #[test]
    fn fp12_arithmetic_matches_arkworks(
        a in prop::array::uniform12(fp()),
        b in prop::array::uniform12(fp()),
    ) {
        let (aa, ab) = (ark_fq12(&a), ark_fq12(&b));
        let (ha, hb) = (fp12_from_ark(aa), fp12_from_ark(ab));

        prop_assert_eq!(ha * hb, fp12_from_ark(aa * ab));
        prop_assert_eq!(ha + hb, fp12_from_ark(aa + ab));
        prop_assert_eq!(ha - hb, fp12_from_ark(aa - ab));
        prop_assert_eq!(ha.square(), fp12_from_ark(aa.square()));
        prop_assert_eq!(ha.invert(), aa.inverse().map(fp12_from_ark));
        prop_assert_eq!(
            ha.conjugate(),
            fp12_from_ark(ArkFq12::new(aa.c0, -aa.c1))
        );
        for (k, frob) in [
            Fp12::frobenius_map,
            Fp12::frobenius_map_squared,
            Fp12::frobenius_map_cubed,
        ]
        .into_iter()
        .enumerate()
        {
            let mut fa = aa;
            fa.frobenius_map_in_place(k + 1);
            prop_assert_eq!(frob(ha), fp12_from_ark(fa));
        }
    }

    #[test]
    fn fp12_sparse_034_product_matches_dense(
        a in prop::array::uniform12(fp()),
        c0 in prop::array::uniform2(fp()),
        c3 in prop::array::uniform2(fp()),
        c4 in prop::array::uniform2(fp()),
    ) {
        let ha = fp12_from_ark(ark_fq12(&a));
        let s0 = Fp2::new(Fp::from_raw(c0[0]), Fp::from_raw(c0[1]));
        let s3 = Fp2::new(Fp::from_raw(c3[0]), Fp::from_raw(c3[1]));
        let s4 = Fp2::new(Fp::from_raw(c4[0]), Fp::from_raw(c4[1]));
        let sparse = Fp12::new(
            Fp6::new(s0, Fp2::ZERO, Fp2::ZERO),
            Fp6::new(s3, s4, Fp2::ZERO),
        );

        let dense = ha * sparse;
        prop_assert_eq!(ha.mul_by_034(s0, s3, s4), dense);
        let mut assigned = ha;
        assigned.mul_by_034_assign(s0, s3, s4);
        prop_assert_eq!(assigned, dense);
    }
}

proptest! {
    #![proptest_config(config(64))]

    #[test]
    fn pairing_matches_arkworks_and_is_bilinear(a in fr(), b in fr()) {
        let (p, ark_p) = g1_pair(a);
        let (q, ark_q) = g2_pair(b);

        let ours = pairing(&p, &q);
        prop_assert_eq!(ours, fp12_from_ark(Bn254::pairing(ark_p, ark_q).0));

        // e([a]G1, [b]G2) = e(G1, G2)^(a*b), checked as e([ab]G1, G2).
        let ab = Fr::from_raw(a) * Fr::from_raw(b);
        let pab = G1Projective::generator().mul_scalar(ab).to_affine();
        prop_assert_eq!(pairing(&pab, &G2Affine::generator()), ours);

        // Cancellation product must vanish on every dispatch route.
        let cancel = helius_narsil::multi_pairing(&[(&p, &q), (&p.negate(), &q)]);
        prop_assert_eq!(cancel, Fp12::ONE);
    }

    // On an AVX-512 IFMA host `multi_pairing` takes the vector Miller path
    // while the single-pair product below replays the scalar route, so this
    // equality is the scalar-versus-IFMA differential. Elsewhere it pins the
    // portable composition.
    #[test]
    fn multi_pairing_matches_single_pairing_product(
        scalars in prop::collection::vec((fr(), fr()), 1..9),
        identity_slots in prop::collection::vec(0usize..9, 0..3),
    ) {
        let mut typed: Vec<(G1Affine, G2Affine)> = scalars
            .iter()
            .map(|(a, b)| (g1_pair(*a).0, g2_pair(*b).0))
            .collect();
        for slot in identity_slots {
            if slot < typed.len() {
                if slot % 2 == 0 {
                    typed[slot].0 = G1Affine::identity();
                } else {
                    typed[slot].1 = G2Affine::identity();
                }
            }
        }
        let refs: Vec<_> = typed.iter().map(|(p, q)| (p, q)).collect();

        let product = refs
            .iter()
            .fold(Fp12::ONE, |acc, (p, q)| acc * pairing(p, q));
        prop_assert_eq!(helius_narsil::multi_pairing(&refs), product);
    }

    #[test]
    fn final_exponentiation_distributes_over_miller_products(
        scalars in prop::collection::vec((fr(), fr()), 2..6),
    ) {
        use helius_narsil::pairing::{final_exponentiation, multi_miller_loop};

        let typed: Vec<(G1Affine, G2Affine)> = scalars
            .iter()
            .map(|(a, b)| (g1_pair(*a).0, g2_pair(*b).0))
            .collect();
        let refs: Vec<_> = typed.iter().map(|(p, q)| (p, q)).collect();

        let fused = final_exponentiation(&multi_miller_loop(&refs));
        let product = refs
            .iter()
            .fold(Fp12::ONE, |acc, (p, q)| acc * pairing(p, q));
        prop_assert_eq!(fused, product);
    }

    #[test]
    fn gt_cyclotomic_ops_match_generic_ops(a in fr(), b in fr()) {
        let g = pairing(&g1_pair(a).0, &g2_pair(b).0);

        prop_assert_eq!(g.cyclotomic_square(), g.square());
        prop_assert_eq!(g.pow_x(), g.pow_u64(consts::BN_X));
        // Final exponentiation output lives in the cyclotomic subgroup,
        // where conjugation inverts.
        prop_assert_eq!(g.conjugate() * g, Fp12::ONE);
    }
}

#[cfg(feature = "std")]
mod prepared {
    use super::*;
    use helius_narsil::pairing::{miller_loop, miller_loop_prepared, prepare_g2};
    use helius_narsil::{FixedPair, LivePair, PreparedTerm, PreparedVerifier};

    proptest! {
        #![proptest_config(config(32))]

        #[test]
        fn prepared_miller_replays_live_miller(a in fr(), b in fr()) {
            let p = g1_pair(a).0;
            let q = g2_pair(b).0;
            let prepared = prepare_g2(&q).expect("subgroup point");
            prop_assert_eq!(miller_loop_prepared(&p, &prepared), miller_loop(&p, &q));
        }

        #[test]
        fn prepared_verifier_agrees_with_multi_pairing(a in fr(), b in fr(), s in fr()) {
            let p = g1_pair(a).0;
            let q = g2_pair(b).0;
            prop_assume!(!p.is_identity() && !q.is_identity());
            let s = Fr::from_raw(s);
            let sp = p.to_curve().mul_scalar(s).to_affine();
            let sq = q.to_curve().mul_scalar(s).to_affine();
            prop_assume!(!sq.is_identity());

            // e([s]P, Q) * e(-P, [s]Q) = 1 through every verifier entry.
            let verifier =
                PreparedVerifier::new(&[q, sq], &[]).expect("valid subgroup points");
            let terms = [
                PreparedTerm { g1: sp, prepared_g2: 0 },
                PreparedTerm { g1: p.negate(), prepared_g2: 1 },
            ];
            prop_assert_eq!(verifier.verify(&terms, &[]), Ok(true));

            let live = [
                LivePair { g1: sp, g2: q },
                LivePair { g1: p.negate(), g2: sq },
            ];
            prop_assert_eq!(verifier.verify(&[], &live), Ok(true));

            let fixed = PreparedVerifier::new(
                &[],
                &[
                    FixedPair { g1: sp, g2: q },
                    FixedPair { g1: p.negate(), g2: sq },
                ],
            )
            .expect("valid subgroup points");
            prop_assert_eq!(fixed.verify(&[], &[]), Ok(true));

            // A shifted online term must break the cancellation.
            let shifted = [
                PreparedTerm {
                    g1: sp.to_curve().add_mixed(p).to_affine(),
                    prepared_g2: 0,
                },
                PreparedTerm { g1: p.negate(), prepared_g2: 1 },
            ];
            let expected = helius_narsil::multi_pairing(&[
                (&shifted[0].g1, &q),
                (&p.negate(), &sq),
            ]);
            prop_assert_eq!(
                verifier.verify(&shifted, &[]),
                Ok(expected == Fp12::ONE)
            );
        }
    }
}

mod batch_bytes {
    use super::*;
    use helius_narsil::{G1Bytes, G2Bytes, PairBytes, pairing_product_is_one};

    proptest! {
        #![proptest_config(config(32))]

        // The byte facade routes through the batched (on IFMA hosts,
        // eight-lane) validation and pairing pipeline. Typed multi_pairing
        // is the scalar-composition oracle.
        #[test]
        fn byte_facade_matches_typed_multi_pairing(
            scalars in prop::collection::vec((fr(), fr()), 1..9),
        ) {
            let typed: Vec<(G1Affine, G2Affine)> = scalars
                .iter()
                .map(|(a, b)| (g1_pair(*a).0, g2_pair(*b).0))
                .collect();
            let refs: Vec<_> = typed.iter().map(|(p, q)| (p, q)).collect();
            let encoded: Vec<_> = typed
                .iter()
                .map(|(p, q)| PairBytes {
                    g1: G1Bytes::from_affine(p),
                    g2: G2Bytes::from_affine(q),
                })
                .collect();

            prop_assert_eq!(
                pairing_product_is_one(&encoded),
                Ok(helius_narsil::multi_pairing(&refs) == Fp12::ONE)
            );
        }
    }
}

/// The scalar batch entries. `g1_msm` walks the windowed bucket path and
/// `fr_batch_invert` the Montgomery prefix chain, neither of which the tower
/// lanes above reach.
mod batch_scalar {
    use super::*;
    use helius_narsil::{
        G1Bytes, ScalarBytes, fr_batch_invert, fr_lincomb, g1_msm, msm_variable_time_affine,
    };

    proptest! {
        #![proptest_config(config(48))]

        #[test]
        fn g1_msm_matches_arkworks_and_typed_msm(terms in prop::collection::vec((fr(), fr()), 1..17)) {
            let points: Vec<(G1Affine, ArkG1Affine)> =
                terms.iter().map(|(a, _)| g1_pair(*a)).collect();
            let scalars: Vec<[u64; 4]> = terms.iter().map(|(_, s)| *s).collect();

            let expected = scalars
                .iter()
                .zip(&points)
                .fold(ArkG1Projective::ZERO, |acc, (s, (_, ark))| {
                    acc + ArkG1Projective::from(*ark) * ark_fr(*s)
                })
                .into_affine();

            let ours: Vec<G1Affine> = points.iter().map(|(p, _)| *p).collect();
            let typed = msm_variable_time_affine(&ours, &scalars);
            prop_assert_eq!(typed.is_identity(), expected.infinity);
            if !expected.infinity {
                prop_assert_eq!(typed.x, fp_from_ark(expected.x));
                prop_assert_eq!(typed.y, fp_from_ark(expected.y));
            }

            let encoded: Vec<G1Bytes> = ours.iter().map(G1Bytes::from_affine).collect();
            let scalar_bytes: Vec<ScalarBytes> = scalars
                .iter()
                .map(|s| ScalarBytes::from_fr(Fr::from_raw(*s)))
                .collect();
            prop_assert_eq!(
                g1_msm(&encoded, &scalar_bytes),
                Ok(G1Bytes::from_affine(&typed))
            );
        }

        #[test]
        fn fr_lincomb_matches_the_scalar_dot_product(terms in prop::collection::vec((fr(), fr()), 1..17)) {
            let expected = terms
                .iter()
                .fold(ArkFr::ZERO, |acc, (a, b)| acc + ark_fr(*a) * ark_fr(*b));
            let a: Vec<ScalarBytes> = terms
                .iter()
                .map(|(a, _)| ScalarBytes::from_fr(Fr::from_raw(*a)))
                .collect();
            let b: Vec<ScalarBytes> = terms
                .iter()
                .map(|(_, b)| ScalarBytes::from_fr(Fr::from_raw(*b)))
                .collect();

            prop_assert_eq!(
                fr_lincomb(&a, &b),
                Ok(ScalarBytes::from_fr(Fr::from_raw(expected.into_bigint().0)))
            );
        }

        #[test]
        fn fr_batch_invert_matches_elementwise_inversion(values in prop::collection::vec(fr(), 1..17)) {
            let encoded: Vec<ScalarBytes> = values
                .iter()
                .map(|v| ScalarBytes::from_fr(Fr::from_raw(*v)))
                .collect();

            if values.iter().any(|v| v == &[0u64; 4]) {
                prop_assert!(fr_batch_invert(&encoded).is_err());
                return Ok(());
            }
            let expected: Vec<ScalarBytes> = values
                .iter()
                .map(|v| {
                    let inverse = ark_fr(*v).inverse().expect("nonzero scalar");
                    ScalarBytes::from_fr(Fr::from_raw(inverse.into_bigint().0))
                })
                .collect();
            prop_assert_eq!(fr_batch_invert(&encoded), Ok(expected));
        }
    }
}

/// The eight-lane IFMA batch entry. It is the only public caller of the lazy
/// radix-52 forms, so on an IFMA host these equalities are the differential
/// that covers `FpVec8L` bound tracking, the fold trajectories, and the
/// cross-chunk lane accumulation against the scalar route.
#[cfg(all(narsil_avx512_ifma, not(feature = "force-portable")))]
mod lazy_ifma {
    use super::*;
    use helius_narsil::multi_pairing8;

    proptest! {
        #![proptest_config(config(24))]

        // Pair counts span a partial chunk, one full chunk, and several
        // chunks, so the lane accumulator folds across chunk boundaries.
        #[test]
        fn multi_pairing8_matches_the_scalar_product(scalars in prop::collection::vec((fr(), fr()), 1..25)) {
            let typed: Vec<(G1Affine, G2Affine)> = scalars
                .iter()
                .map(|(a, b)| (g1_pair(*a).0, g2_pair(*b).0))
                .collect();
            let refs: Vec<_> = typed.iter().map(|(p, q)| (p, q)).collect();

            let product = refs
                .iter()
                .fold(Fp12::ONE, |acc, (p, q)| acc * pairing(p, q));
            prop_assert_eq!(multi_pairing8(&typed), product);
            prop_assert_eq!(multi_pairing8(&typed), helius_narsil::multi_pairing(&refs));
        }

        // Identity lanes are filtered before the vector path, so the residual
        // chunk shape differs from the input length.
        #[test]
        fn multi_pairing8_cancels_with_identity_lanes(a in fr(), b in fr(), pad in 0usize..8) {
            let p = g1_pair(a).0;
            let q = g2_pair(b).0;
            let mut pairs = vec![(p, q), (p.negate(), q)];
            pairs.extend(core::iter::repeat_n((G1Affine::identity(), q), pad));
            prop_assert_eq!(multi_pairing8(&pairs), Fp12::ONE);
        }
    }
}
