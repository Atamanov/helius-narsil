//! Optimal Ate pairing over BN254 (BN_SNARK1).

pub(crate) mod final_exp;
pub(crate) mod miller;
#[cfg(all(test, feature = "std"))]
mod redteam_tests;
pub use final_exp::final_exponentiation;
#[cfg(feature = "std")]
pub use miller::{G2Prepared, miller_loop_prepared, prepare_g2, prepare_g2_unchecked};
pub use miller::{miller_loop, multi_miller_loop};

use crate::fp12::Fp12;
use crate::g1::{G1Affine, G1Projective};
use crate::g2::G2Affine;

/// `e: G1 x G2 -> GT`.
pub fn pairing(p: &G1Affine, q: &G2Affine) -> Fp12 {
    if p.is_identity() || q.is_identity() {
        return Fp12::ONE;
    }
    let f = miller_loop(p, q);
    final_exponentiation(&f)
}

/// Product of pairings with a single final exponentiation.
pub fn multi_pairing(pairs: &[(&G1Affine, &G2Affine)]) -> Fp12 {
    // A common verification shape is `prod e(P_i, Q)` with one fixed G2
    // verifier-key point. Bilinearity turns it into `e(sum P_i, Q)`, removing
    // all redundant G2 line schedules and sparse Fp12 multiplications. Keep
    // the heterogeneous path untouched so distinct-Q inputs pay only this
    // short equality scan.
    let mut nonzero = pairs
        .iter()
        .copied()
        .filter(|(p, q)| !p.is_identity() && !q.is_identity());
    if let Some((first_p, common_q)) = nonzero.next() {
        let mut sum = G1Projective::from(*first_p);
        let mut all_q_equal = true;
        for (p, q) in nonzero {
            if q != common_q {
                all_q_equal = false;
                break;
            }
            sum = sum.add_mixed(*p);
        }
        if all_q_equal {
            if sum.is_identity() {
                return Fp12::ONE;
            }
            let sum = sum.to_affine();
            return pairing(&sum, common_q);
        }
    } else {
        return Fp12::ONE;
    }

    let f = multi_miller_loop(pairs);
    if f.is_one() {
        return Fp12::ONE;
    }
    final_exponentiation(&f)
}

/// Pairing target group. Elements live in the r-order cyclotomic subgroup
/// of Fp12 after final exponentiation.
pub type Gt = Fp12;

#[cfg(test)]
mod tests {
    use alloc::{vec, vec::Vec};

    use super::*;
    use crate::Fr;
    use crate::g2::G2Projective;

    fn assert_matches_individual_product(g1: &[G1Affine], g2: &[G2Affine]) {
        let refs: Vec<_> = g1.iter().zip(g2).collect();
        let expected = refs
            .iter()
            .fold(Fp12::ONE, |product, (p, q)| product * pairing(p, q));
        assert_eq!(multi_pairing(&refs), expected);
    }

    #[test]
    fn common_q_reduction_matches_individual_pairings() {
        let p = G1Affine::generator();
        let two_p = G1Projective::from(p).double().to_affine();
        let q = G2Affine::generator();

        let expected = pairing(&p, &q) * pairing(&two_p, &q);
        assert_eq!(multi_pairing(&[(&p, &q), (&two_p, &q)]), expected);

        let neg_p = p.negate();
        assert_eq!(multi_pairing(&[(&p, &q), (&neg_p, &q)]), Fp12::ONE);
    }

    #[test]
    fn heterogeneous_q_path_matches_individual_pairings() {
        let p = G1Affine::generator();
        let two_p = G1Projective::from(p).double().to_affine();
        let q = G2Affine::generator();
        let two_q = G2Projective::from(q).double().to_affine();

        let expected = pairing(&p, &q) * pairing(&two_p, &two_q);
        assert_eq!(multi_pairing(&[(&p, &q), (&two_p, &two_q)]), expected);
    }

    #[test]
    fn heterogeneous_fused_path_covers_late_mismatch_and_infinities() {
        let generator_p = G1Projective::generator();
        let generator_q = G2Projective::from(G2Affine::generator());

        for count in [3usize, 4, 8, 16] {
            let mut g1: Vec<_> = (0..count)
                .map(|index| {
                    generator_p
                        .mul_scalar(Fr::from_u64(index as u64 + 1))
                        .to_affine()
                })
                .collect();
            let mut g2 = vec![generator_q.to_affine(); count];
            // Force the equality scan to reach the last element before the
            // heterogeneous fused loop is selected.
            g2[count - 1] = generator_q
                .mul_scalar(Fr::from_u64(count as u64 + 1))
                .to_affine();
            assert_matches_individual_product(&g1, &g2);

            g1[1] = G1Affine::identity();
            g2[count / 2] = G2Affine::identity();
            assert_matches_individual_product(&g1, &g2);
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn public_prepared_g2_replays_live_miller_and_rejects_invalid_points() {
        let p = G1Affine::generator();
        let q = G2Affine::generator();
        let prepared = prepare_g2(&q).expect("generator is a valid subgroup point");
        assert_eq!(miller_loop_prepared(&p, &prepared), miller_loop(&p, &q));

        let identity = prepare_g2(&G2Affine::identity()).expect("identity is valid");
        assert_eq!(miller_loop_prepared(&p, &identity), Fp12::ONE);

        let invalid = G2Affine {
            x: crate::Fp2::ZERO,
            y: crate::Fp2::ZERO,
            infinity: false,
        };
        assert!(prepare_g2(&invalid).is_none());
    }
}
