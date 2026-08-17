//! One multi-Miller accumulator over a mixed set of live and prepared terms.

use alloc::vec::Vec;

use super::miller::{G2Miller, G2Prepared, multi_miller_loop_mixed as mixed_accumulator};
use crate::fp12::Fp12;
use crate::g1::G1Affine;
use crate::g2::G2Affine;

/// One term of a multi-Miller product.
///
/// A live term generates its G2 line schedule inside the loop. A prepared term
/// replays a schedule [`prepare_g2`](super::prepare_g2) built earlier. The
/// schedule fields stay private, so a term cannot name a schedule that no
/// validated curve point produced.
#[derive(Clone, Copy)]
pub struct MillerTerm<'a> {
    g1: &'a G1Affine,
    g2: G2Miller<'a>,
}

impl<'a> MillerTerm<'a> {
    /// Pair `g1` with a G2 point whose lines this loop generates.
    #[inline]
    pub const fn live(g1: &'a G1Affine, g2: &'a G2Affine) -> Self {
        Self {
            g1,
            g2: G2Miller::Live(g2),
        }
    }

    /// Pair `g1` with a schedule prepared earlier.
    #[inline]
    pub const fn prepared(g1: &'a G1Affine, schedule: &'a G2Prepared) -> Self {
        Self {
            g1,
            g2: G2Miller::Prepared(schedule),
        }
    }

    #[inline]
    pub(crate) const fn from_source(g1: &'a G1Affine, g2: G2Miller<'a>) -> Self {
        Self { g1, g2 }
    }

    #[inline]
    const fn source(self) -> (&'a G1Affine, G2Miller<'a>) {
        (self.g1, self.g2)
    }

    /// The pair the eight-lane engine can take. Identity terms contribute one
    /// and the engine's formulas reject them, so they stay on the accumulator.
    #[cfg(all(
        any(narsil_avx512_ifma, narsil_x86_runtime_ifma),
        not(feature = "force-portable")
    ))]
    #[inline]
    fn live_pair(self) -> Option<(G1Affine, G2Affine)> {
        match self.g2 {
            G2Miller::Live(q) if !self.g1.is_identity() && !q.is_identity() => Some((*self.g1, *q)),
            _ => None,
        }
    }
}

/// Miller product over live and prepared terms with one shared square chain.
///
/// Callers apply the final exponentiation themselves, and identity terms need
/// no filtering. Every ordering of `terms` returns the same field element, so
/// the entry is free to route a whole eight-lane group of live terms through
/// the vector engine and fold everything that group leaves behind, prepared
/// terms included, into a single accumulator.
///
/// Splitting a mixed set across several calls costs one extra Fp12 square
/// chain per extra call. Use this entry once instead.
pub fn multi_miller_loop_mixed(terms: &[MillerTerm<'_>]) -> Fp12 {
    #[cfg(all(
        any(narsil_avx512_ifma, narsil_x86_runtime_ifma),
        not(feature = "force-portable")
    ))]
    if crate::batch8::ifma_available() {
        let engine = terms.iter().filter_map(|term| term.live_pair()).count();
        let lanes = super::miller::vertical_prefix(engine);
        if lanes != 0 {
            let mut live = Vec::with_capacity(lanes);
            let mut rest = Vec::with_capacity(terms.len() - lanes);
            for term in terms {
                match term.live_pair() {
                    Some(pair) if live.len() < lanes => live.push(pair),
                    _ => rest.push(term.source()),
                }
            }
            let product = crate::batch8::multi_miller_product8(&live).unwrap_or(Fp12::ONE);
            return product * mixed_accumulator(&rest);
        }
    }

    let sources: Vec<_> = terms.iter().map(|term| term.source()).collect();
    mixed_accumulator(&sources)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pairing::{miller_loop, miller_loop_prepared, multi_miller_loop, prepare_g2};
    use crate::{Fr, G1Projective, G2Projective};
    use alloc::vec;

    fn p(value: u64) -> G1Affine {
        G1Projective::generator()
            .mul_scalar(Fr::from_u64(value))
            .to_affine()
    }

    fn q(value: u64) -> G2Affine {
        G2Projective::from(G2Affine::generator())
            .mul_scalar(Fr::from_u64(value))
            .to_affine()
    }

    /// The product of separate per-kind entries, the shape a caller is forced
    /// into without this module.
    fn separate(live: &[(G1Affine, G2Affine)], prepared: &[(G1Affine, G2Affine)]) -> Fp12 {
        let refs: Vec<_> = live.iter().map(|(g1, g2)| (g1, g2)).collect();
        let mut product = multi_miller_loop(&refs);
        for (g1, g2) in prepared {
            let schedule = prepare_g2(g2).expect("test G2 is a valid subgroup point");
            product *= miller_loop_prepared(g1, &schedule);
        }
        product
    }

    #[test]
    fn mixed_batch_matches_the_separate_entries_at_every_lane_boundary() {
        // Term counts around the eight-lane group boundary and around both
        // host crossovers, so no route split can move the product.
        for live_count in 0..=11_usize {
            for prepared_count in 0..=6_usize {
                let live: Vec<(G1Affine, G2Affine)> = (0..live_count)
                    .map(|index| (p(index as u64 + 1), q(index as u64 + 3)))
                    .collect();
                let fixed: Vec<(G1Affine, G2Affine)> = (0..prepared_count)
                    .map(|index| (p(index as u64 + 21), q(index as u64 + 31)))
                    .collect();
                let schedules: Vec<G2Prepared> = fixed
                    .iter()
                    .map(|(_, g2)| prepare_g2(g2).expect("valid subgroup point"))
                    .collect();
                let mut terms: Vec<MillerTerm<'_>> = fixed
                    .iter()
                    .zip(&schedules)
                    .map(|((g1, _), schedule)| MillerTerm::prepared(g1, schedule))
                    .collect();
                terms.extend(live.iter().map(|(g1, g2)| MillerTerm::live(g1, g2)));

                assert_eq!(
                    multi_miller_loop_mixed(&terms),
                    separate(&live, &fixed),
                    "live {live_count}, prepared {prepared_count}"
                );
            }
        }
    }

    #[test]
    fn term_order_does_not_move_the_product() {
        let live = [(p(2), q(5)), (p(3), q(7)), (p(11), q(13))];
        let fixed = [(p(4), q(9)), (p(6), q(17))];
        let schedules: Vec<G2Prepared> = fixed
            .iter()
            .map(|(_, g2)| prepare_g2(g2).expect("valid subgroup point"))
            .collect();
        let interleaved = vec![
            MillerTerm::live(&live[0].0, &live[0].1),
            MillerTerm::prepared(&fixed[0].0, &schedules[0]),
            MillerTerm::live(&live[1].0, &live[1].1),
            MillerTerm::prepared(&fixed[1].0, &schedules[1]),
            MillerTerm::live(&live[2].0, &live[2].1),
        ];
        let grouped = vec![
            interleaved[1],
            interleaved[3],
            interleaved[0],
            interleaved[2],
            interleaved[4],
        ];
        assert_eq!(
            multi_miller_loop_mixed(&interleaved),
            multi_miller_loop_mixed(&grouped)
        );
        assert_eq!(
            multi_miller_loop_mixed(&interleaved),
            separate(&live, &fixed)
        );
    }

    #[test]
    fn identity_terms_and_the_empty_set_contribute_one() {
        let identity_g1 = G1Affine::identity();
        let identity_g2 = G2Affine::identity();
        let schedule = prepare_g2(&identity_g2).expect("the identity prepares");
        let live_g1 = p(3);
        let live_g2 = q(5);

        assert_eq!(multi_miller_loop_mixed(&[]), Fp12::ONE);
        assert_eq!(
            multi_miller_loop_mixed(&[
                MillerTerm::live(&identity_g1, &live_g2),
                MillerTerm::live(&live_g1, &identity_g2),
                MillerTerm::prepared(&live_g1, &schedule),
            ]),
            Fp12::ONE
        );
        assert_eq!(
            multi_miller_loop_mixed(&[
                MillerTerm::prepared(&identity_g1, &schedule),
                MillerTerm::live(&live_g1, &live_g2),
            ]),
            miller_loop(&live_g1, &live_g2)
        );
    }
}
