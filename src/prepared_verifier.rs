//! Explicit prepared contexts for repeated small-batch verifiers.
//!
//! This API is intentionally separate from the alt_bn128 syscall byte facade. It accepts
//! typed points, validates them at construction or use, and precomputes only
//! public verifier material. It never caches a verification verdict.

use alloc::vec::Vec;

use crate::pairing::{
    MillerTerm, final_exponentiation,
    miller::{
        G2Miller, G2Prepared, miller_residue_relation_mixed, multi_miller_loop_mixed, prepare,
    },
    multi_miller_loop,
};
use crate::{Fp, Fp2, Fp6, Fp12, G1Affine, G2Affine, batch::InputError};

/// Maximum number of public schedules, fixed terms, or online terms accepted
/// by a small-batch verifier operation.
pub const MAX_PREPARED_VERIFIER_TERMS: usize = 5;

/// One fully fixed pairing term evaluated when a context is constructed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FixedPair {
    /// Validated G1 partner.
    pub g1: G1Affine,
    /// Validated G2 partner.
    pub g2: G2Affine,
}

/// One online G1 value paired with a context-owned prepared G2 schedule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PreparedTerm {
    /// Online G1 partner, validated before verification arithmetic.
    pub g1: G1Affine,
    /// Index into the `prepared_g2` slice supplied to
    /// [`PreparedVerifier::new`].
    pub prepared_g2: usize,
}

/// One fully online typed pairing term.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LivePair {
    /// Online G1 partner.
    pub g1: G1Affine,
    /// Online G2 partner.
    pub g2: G2Affine,
}

/// Multiplier that makes a valid Miller product a cubic residue.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CubicResidueClass {
    /// Do not scale the Miller product.
    #[default]
    Identity,
    /// Multiply by the fixed primitive 27th root.
    Root,
    /// Multiply by the square of the fixed primitive 27th root.
    RootSquared,
}

/// Auxiliary data for a pairing residue check.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PairingResidueWitness {
    root: Fp12,
    inverse: Fp12,
    class: CubicResidueClass,
}

impl PairingResidueWitness {
    /// Construct public witness data. Verification rejects a wrong inverse.
    pub const fn new(root: Fp12, inverse: Fp12, class: CubicResidueClass) -> Self {
        Self {
            root,
            inverse,
            class,
        }
    }
}

/// Construction or verification error for [`PreparedVerifier`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum PreparedVerifierError {
    /// A typed point failed the same curve/subgroup validation used by the byte
    /// facade. The contained error identifies the failed predicate.
    #[error("invalid prepared-verifier point: {0}")]
    InvalidPoint(#[from] InputError),
    /// A prepared G2 was the identity. Identity terms need no schedule and are
    /// rejected to prevent meaningless context state.
    #[error("prepared G2 must be nonidentity")]
    IdentityPreparedG2,
    /// An online prepared term referred outside the context's schedule table.
    #[error("prepared G2 index is outside the verifier context")]
    PreparedIndexOutOfBounds,
    /// A context or online verification exceeded the deliberately bounded
    /// one-through-five-pair workload supported by this API.
    #[error("prepared verifier supports at most five terms")]
    TooManyTerms,
}

struct PreparedG2 {
    point: G2Affine,
    schedule: G2Prepared,
}

/// Validated fixed verifier material for repeated pairing-product checks.
///
/// Each configured G2 owns its complete optimal-Ate line schedule. Fully fixed
/// pairs are collapsed into one raw Miller product and one GT product.
/// Prepared and live terms share one online Miller accumulator.
pub struct PreparedVerifier {
    prepared_g2: Vec<PreparedG2>,
    fixed_miller_product: Fp12,
    fixed_gt_product: Fp12,
}

/// Validated online terms for one prepared verifier context.
pub struct ValidatedOnlineTerms<'a> {
    sources: Vec<(&'a G1Affine, G2Miller<'a>)>,
    pairs: Vec<(G1Affine, G2Affine)>,
}

impl<'a> ValidatedOnlineTerms<'a> {
    /// Evaluate the online Miller product without final exponentiation.
    ///
    /// The result is not a target-group value. Apply final exponentiation
    /// before a pairing comparison.
    #[inline]
    pub fn miller_loop(&self) -> Fp12 {
        online_miller(&self.sources, &self.pairs)
    }

    /// Lend the validated terms to a caller-owned multi-Miller accumulator.
    ///
    /// [`Self::miller_loop`] closes one accumulator per context. A caller
    /// holding several contexts collects these terms and calls
    /// [`crate::pairing::multi_miller_loop_mixed`] once, which pays one Fp12
    /// square chain instead of one per context.
    #[inline]
    pub fn terms(&self) -> impl ExactSizeIterator<Item = MillerTerm<'a>> + '_ {
        self.sources
            .iter()
            .map(|&(g1, g2)| MillerTerm::from_source(g1, g2))
    }
}

/// Online Miller product for one verifier call.
///
/// Authorized IFMA hosts route a term count above the host crossover through
/// the masked eight-lane engine, whose dependent chain length does not grow
/// with the term count. Fewer terms keep the shared accumulator, which replays
/// prepared line schedules. Every route fully reduces, so the product is
/// identical bit for bit.
fn online_miller(sources: &[(&G1Affine, G2Miller<'_>)], pairs: &[(G1Affine, G2Affine)]) -> Fp12 {
    #[cfg(all(
        any(narsil_avx512_ifma, narsil_x86_runtime_ifma),
        not(feature = "force-portable")
    ))]
    if pairs.len() >= crate::x86_runtime::vertical_min_miller_terms()
        && let Some(product) = crate::batch8::multi_miller_product8(pairs)
    {
        return product;
    }
    let _ = pairs;
    multi_miller_loop_mixed(sources)
}

impl PreparedVerifier {
    /// Validate and prepare public verifier material.
    ///
    /// `prepared_g2` is validated in slice order. `fixed_terms` is then
    /// validated in pair order, G1 before G2. Construction performs the fixed
    /// pairing product once. Subsequent calls never recompute those terms.
    pub fn new(
        prepared_g2: &[G2Affine],
        fixed_terms: &[FixedPair],
    ) -> Result<Self, PreparedVerifierError> {
        if prepared_g2.len() > MAX_PREPARED_VERIFIER_TERMS
            || fixed_terms.len() > MAX_PREPARED_VERIFIER_TERMS
        {
            return Err(PreparedVerifierError::TooManyTerms);
        }
        let mut schedules = Vec::with_capacity(prepared_g2.len());
        for point in prepared_g2 {
            validate_g2(point)?;
            if point.is_identity() {
                return Err(PreparedVerifierError::IdentityPreparedG2);
            }
            schedules.push(PreparedG2 {
                point: *point,
                schedule: prepare(point),
            });
        }

        for term in fixed_terms {
            validate_g1(&term.g1)?;
            validate_g2(&term.g2)?;
        }
        let fixed_refs: Vec<_> = fixed_terms
            .iter()
            .map(|term| (&term.g1, &term.g2))
            .collect();
        let fixed_miller_product = multi_miller_loop(&fixed_refs);
        let fixed_gt_product = if fixed_miller_product.is_one() {
            Fp12::ONE
        } else {
            final_exponentiation(&fixed_miller_product)
        };

        Ok(Self {
            prepared_g2: schedules,
            fixed_miller_product,
            fixed_gt_product,
        })
    }

    /// Number of validated G2 schedules owned by this context.
    #[inline]
    pub fn prepared_g2_len(&self) -> usize {
        self.prepared_g2.len()
    }

    /// Evaluate online terms and test the complete fixed-plus-online product.
    ///
    /// Prepared terms are validated first in slice order (G1, then index).
    /// Live terms follow in slice order (G1 before G2). No Miller or final
    /// exponentiation work begins until the complete online validation pass
    /// succeeds.
    pub fn verify(
        &self,
        prepared_terms: &[PreparedTerm],
        live_terms: &[LivePair],
    ) -> Result<bool, PreparedVerifierError> {
        let online_terms = self.validate_online(prepared_terms, live_terms)?;
        let online = if online_terms.sources.is_empty() {
            Fp12::ONE
        } else {
            final_exponentiation(&online_terms.miller_loop())
        };
        Ok((self.fixed_gt_product * online).is_one())
    }

    /// Test the fixed-plus-online product for terms validated earlier.
    ///
    /// Splitting validation from arithmetic lets a caller amortize the
    /// validation cost over repeated evaluation of the same online terms.
    pub fn verify_validated(&self, online_terms: &ValidatedOnlineTerms<'_>) -> bool {
        let online = if online_terms.sources.is_empty() {
            Fp12::ONE
        } else {
            final_exponentiation(&online_terms.miller_loop())
        };
        (self.fixed_gt_product * online).is_one()
    }

    /// Verify terms the caller validated earlier and return the online
    /// target-group product with the verdict.
    ///
    /// Point validation is skipped, so the verdict is meaningful only when
    /// every point was validated earlier against this context. Index and
    /// term-count checks still apply. The returned Fp12 is the online product
    /// after final exponentiation, the value the fixed product is tested
    /// against.
    pub fn verify_prevalidated(
        &self,
        prepared_terms: &[PreparedTerm],
        live_terms: &[LivePair],
    ) -> Result<(bool, Fp12), PreparedVerifierError> {
        let online_terms = prepared_terms
            .len()
            .checked_add(live_terms.len())
            .ok_or(PreparedVerifierError::TooManyTerms)?;
        if online_terms > MAX_PREPARED_VERIFIER_TERMS {
            return Err(PreparedVerifierError::TooManyTerms);
        }
        for term in prepared_terms {
            if term.prepared_g2 >= self.prepared_g2.len() {
                return Err(PreparedVerifierError::PreparedIndexOutOfBounds);
            }
        }

        let terms = self.online_terms(prepared_terms, live_terms);
        let online = if terms.sources.is_empty() {
            Fp12::ONE
        } else {
            final_exponentiation(&terms.miller_loop())
        };
        Ok(((self.fixed_gt_product * online).is_one(), online))
    }

    /// Validate online terms without running pairing arithmetic.
    pub fn validate_online<'a>(
        &'a self,
        prepared_terms: &'a [PreparedTerm],
        live_terms: &'a [LivePair],
    ) -> Result<ValidatedOnlineTerms<'a>, PreparedVerifierError> {
        self.validated_sources(prepared_terms, live_terms)
    }

    /// Verify a pairing product with a prover-supplied residue witness.
    ///
    /// This implements the BN254 residue check from
    /// <https://eprint.iacr.org/2024/640>. Point validation runs before the
    /// relation check. An invalid witness returns `false`. Witness generation
    /// is separate and this operation is variable time.
    pub fn verify_with_witness(
        &self,
        prepared_terms: &[PreparedTerm],
        live_terms: &[LivePair],
        witness: &PairingResidueWitness,
    ) -> Result<bool, PreparedVerifierError> {
        let terms = self.validated_sources(prepared_terms, live_terms)?;
        if terms.sources.is_empty() {
            return Ok(self.fixed_gt_product.is_one());
        }
        if witness.root * witness.inverse != Fp12::ONE {
            return Ok(false);
        }
        let online = miller_residue_relation_mixed(&terms.sources, witness.root, witness.inverse);
        Ok(residue_relation_is_one(
            self.fixed_miller_product * online,
            witness.class,
        ))
    }

    fn validated_sources<'a>(
        &'a self,
        prepared_terms: &'a [PreparedTerm],
        live_terms: &'a [LivePair],
    ) -> Result<ValidatedOnlineTerms<'a>, PreparedVerifierError> {
        let online_terms = prepared_terms
            .len()
            .checked_add(live_terms.len())
            .ok_or(PreparedVerifierError::TooManyTerms)?;
        if online_terms > MAX_PREPARED_VERIFIER_TERMS {
            return Err(PreparedVerifierError::TooManyTerms);
        }
        for term in prepared_terms {
            validate_g1(&term.g1)?;
            if term.prepared_g2 >= self.prepared_g2.len() {
                return Err(PreparedVerifierError::PreparedIndexOutOfBounds);
            }
        }
        for term in live_terms {
            validate_g1(&term.g1)?;
            validate_g2(&term.g2)?;
        }

        Ok(self.online_terms(prepared_terms, live_terms))
    }

    /// Collect non-identity online terms. Bounds and validation are the
    /// caller's responsibility.
    fn online_terms<'a>(
        &'a self,
        prepared_terms: &'a [PreparedTerm],
        live_terms: &'a [LivePair],
    ) -> ValidatedOnlineTerms<'a> {
        let capacity = prepared_terms.len() + live_terms.len();
        let mut sources = Vec::with_capacity(capacity);
        let mut pairs = Vec::with_capacity(capacity);
        for term in prepared_terms {
            if term.g1.is_identity() {
                continue;
            }
            let fixed = &self.prepared_g2[term.prepared_g2];
            debug_assert!(!fixed.point.is_identity());
            sources.push((&term.g1, G2Miller::Prepared(&fixed.schedule)));
            pairs.push((term.g1, fixed.point));
        }
        for term in live_terms {
            if term.g1.is_identity() || term.g2.is_identity() {
                continue;
            }
            sources.push((&term.g1, G2Miller::Live(&term.g2)));
            pairs.push((term.g1, term.g2));
        }
        ValidatedOnlineTerms { sources, pairs }
    }
}

// This Fp3 element has order 27. Its powers define the witness class.
const RESIDUE_ROOT: Fp12 = Fp12::new(
    Fp6::new(
        Fp2::ZERO,
        Fp2::ZERO,
        Fp2::new(
            Fp::from_raw_unchecked([
                4_697_320_424_466_575_638,
                7_573_149_608_656_793_195,
                958_080_529_559_898_411,
                148_160_130_192_640_236,
            ]),
            Fp::from_raw_unchecked([
                16_815_426_526_155_404_214,
                8_598_384_272_826_145_338,
                3_568_933_804_943_838_745,
                5_739_724_454_296_374,
            ]),
        ),
    ),
    Fp6::ZERO,
);

fn residue_scale(class: CubicResidueClass) -> Fp12 {
    match class {
        CubicResidueClass::Identity => Fp12::ONE,
        CubicResidueClass::Root => RESIDUE_ROOT,
        CubicResidueClass::RootSquared => RESIDUE_ROOT.square(),
    }
}

fn residue_relation_is_one(relation: Fp12, class: CubicResidueClass) -> bool {
    (relation * residue_scale(class)).is_one()
}

fn validate_g1(point: &G1Affine) -> Result<(), PreparedVerifierError> {
    if !point.is_identity() && !point.is_on_curve() {
        return Err(InputError::NotOnCurve.into());
    }
    Ok(())
}

fn validate_g2(point: &G2Affine) -> Result<(), PreparedVerifierError> {
    if point.is_identity() {
        return Ok(());
    }
    if !point.is_on_curve() {
        return Err(InputError::NotOnCurve.into());
    }
    if !point.is_in_correct_subgroup_assuming_on_curve() {
        return Err(InputError::NotInSubgroup.into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pairing::multi_pairing;
    use crate::{Fr, G1Projective, G2Projective};

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

    fn fp12(values: [[u64; 4]; 12]) -> Fp12 {
        let values = values.map(Fp::from_raw);
        Fp12::new(
            Fp6::new(
                Fp2::new(values[0], values[1]),
                Fp2::new(values[2], values[3]),
                Fp2::new(values[4], values[5]),
            ),
            Fp6::new(
                Fp2::new(values[6], values[7]),
                Fp2::new(values[8], values[9]),
                Fp2::new(values[10], values[11]),
            ),
        )
    }

    #[test]
    fn kzg_like_two_pair_equation_matches_direct_pairing() {
        let context = PreparedVerifier::new(&[q(1)], &[]).unwrap();
        let prepared = [PreparedTerm {
            g1: p(2),
            prepared_g2: 0,
        }];
        let live = [LivePair {
            g1: p(1).negate(),
            g2: q(2),
        }];
        assert!(context.verify(&prepared, &live).unwrap());

        let expected = multi_pairing(&[(&prepared[0].g1, &q(1)), (&live[0].g1, &live[0].g2)]);
        assert!(expected.is_one());
    }

    #[test]
    fn groth16_like_fixed_gt_and_three_online_terms_match_direct_pairing() {
        let fixed = [FixedPair { g1: p(4), g2: q(1) }];
        let context = PreparedVerifier::new(&[q(1), q(2)], &fixed).unwrap();
        let prepared = [
            PreparedTerm {
                g1: p(1),
                prepared_g2: 0,
            },
            PreparedTerm {
                g1: p(1),
                prepared_g2: 1,
            },
        ];
        let live = [LivePair {
            g1: p(1).negate(),
            g2: q(7),
        }];
        assert!(context.verify(&prepared, &live).unwrap());

        let bad = [
            PreparedTerm {
                g1: p(2),
                prepared_g2: 0,
            },
            prepared[1],
        ];
        assert!(!context.verify(&bad, &live).unwrap());
    }

    /// Four and five online terms take the lane-engine route, so this pins
    /// that route against the engine-independent direct pairing product.
    #[test]
    fn four_and_five_online_terms_match_direct_pairing() {
        for live_count in [2_usize, 3] {
            let fixed = [FixedPair { g1: p(6), g2: q(3) }];
            let context = PreparedVerifier::new(&[q(1), q(2)], &fixed).unwrap();
            let prepared = [
                PreparedTerm {
                    g1: p(2),
                    prepared_g2: 0,
                },
                PreparedTerm {
                    g1: p(3),
                    prepared_g2: 1,
                },
            ];
            let live: Vec<LivePair> = (0..live_count)
                .map(|index| LivePair {
                    g1: p(4 + index as u64),
                    g2: q(5 + index as u64),
                })
                .collect();
            let mut product_refs: Vec<(&G1Affine, &G2Affine)> = Vec::new();
            let fixed_g2 = [q(1), q(2), q(3)];
            product_refs.push((&prepared[0].g1, &fixed_g2[0]));
            product_refs.push((&prepared[1].g1, &fixed_g2[1]));
            for pair in &live {
                product_refs.push((&pair.g1, &pair.g2));
            }
            product_refs.push((&fixed[0].g1, &fixed_g2[2]));
            let direct = multi_pairing(&product_refs);
            let verdict = context.verify(&prepared, &live).unwrap();
            assert_eq!(verdict, direct == Fp12::ONE);
            let (pre_verdict, online) = context.verify_prevalidated(&prepared, &live).unwrap();
            assert_eq!(pre_verdict, verdict);
            let mut online_refs: Vec<(&G1Affine, &G2Affine)> = Vec::new();
            online_refs.push((&prepared[0].g1, &fixed_g2[0]));
            online_refs.push((&prepared[1].g1, &fixed_g2[1]));
            for pair in &live {
                online_refs.push((&pair.g1, &pair.g2));
            }
            assert_eq!(online, multi_pairing(&online_refs), "live {live_count}");
        }
    }

    #[test]
    fn prevalidated_verify_matches_verify_and_returns_the_online_product() {
        let fixed = [FixedPair {
            g1: p(9).negate(),
            g2: q(1),
        }];
        let context = PreparedVerifier::new(&[q(1), q(2)], &fixed).unwrap();
        let prepared = [
            PreparedTerm {
                g1: p(2),
                prepared_g2: 0,
            },
            PreparedTerm {
                g1: p(1),
                prepared_g2: 1,
            },
        ];
        let live = [LivePair { g1: p(1), g2: q(5) }];
        let (verdict, online) = context.verify_prevalidated(&prepared, &live).unwrap();
        assert_eq!(verdict, context.verify(&prepared, &live).unwrap());
        assert_eq!(
            online,
            multi_pairing(&[
                (&prepared[0].g1, &q(1)),
                (&prepared[1].g1, &q(2)),
                (&live[0].g1, &live[0].g2),
            ])
        );
        assert!(verdict);

        let wrong = [
            PreparedTerm {
                g1: p(3),
                prepared_g2: 0,
            },
            prepared[1],
        ];
        let (verdict, online) = context.verify_prevalidated(&wrong, &live).unwrap();
        assert!(!verdict);
        assert_ne!(online, Fp12::ONE);

        assert_eq!(
            context.verify_prevalidated(
                &[PreparedTerm {
                    g1: p(1),
                    prepared_g2: usize::MAX,
                }],
                &[],
            ),
            Err(PreparedVerifierError::PreparedIndexOutOfBounds)
        );
    }

    #[test]
    fn validation_and_index_errors_precede_all_pairing_arithmetic() {
        let context = PreparedVerifier::new(&[q(1)], &[]).unwrap();
        let invalid_g1 = G1Affine {
            x: crate::Fp::ONE,
            y: crate::Fp::ONE,
            infinity: false,
        };
        assert_eq!(
            context.verify(
                &[PreparedTerm {
                    g1: invalid_g1,
                    prepared_g2: usize::MAX,
                }],
                &[]
            ),
            Err(PreparedVerifierError::InvalidPoint(InputError::NotOnCurve))
        );
        assert_eq!(
            context.verify(
                &[PreparedTerm {
                    g1: p(1),
                    prepared_g2: usize::MAX,
                }],
                &[]
            ),
            Err(PreparedVerifierError::PreparedIndexOutOfBounds)
        );

        let invalid_g2 = G2Affine {
            x: crate::Fp2::ONE,
            y: crate::Fp2::ONE,
            infinity: false,
        };
        assert!(matches!(
            PreparedVerifier::new(&[invalid_g2], &[]),
            Err(PreparedVerifierError::InvalidPoint(InputError::NotOnCurve))
        ));
        assert!(matches!(
            PreparedVerifier::new(&[G2Affine::identity()], &[]),
            Err(PreparedVerifierError::IdentityPreparedG2)
        ));

        let invalid_fixed_g1 = FixedPair {
            g1: invalid_g1,
            g2: invalid_g2,
        };
        assert!(matches!(
            PreparedVerifier::new(&[], &[invalid_fixed_g1]),
            Err(PreparedVerifierError::InvalidPoint(InputError::NotOnCurve))
        ));

        let context = PreparedVerifier::new(&[q(1)], &[]).unwrap();
        assert_eq!(
            context.verify(
                &[],
                &[LivePair {
                    g1: invalid_g1,
                    g2: invalid_g2,
                }]
            ),
            Err(PreparedVerifierError::InvalidPoint(InputError::NotOnCurve))
        );
    }

    #[test]
    fn context_and_online_terms_are_bounded_before_allocation_or_validation() {
        let six_g2 = [q(1), q(2), q(3), q(4), q(5), q(6)];
        assert!(matches!(
            PreparedVerifier::new(&six_g2, &[]),
            Err(PreparedVerifierError::TooManyTerms)
        ));

        let six_fixed = [FixedPair { g1: p(1), g2: q(1) }; 6];
        assert!(matches!(
            PreparedVerifier::new(&[], &six_fixed),
            Err(PreparedVerifierError::TooManyTerms)
        ));

        let context = PreparedVerifier::new(&[q(1)], &[]).unwrap();
        let six_prepared = [PreparedTerm {
            g1: p(1),
            prepared_g2: 0,
        }; 6];
        assert_eq!(
            context.verify(&six_prepared, &[]),
            Err(PreparedVerifierError::TooManyTerms)
        );
        let six_live = [LivePair { g1: p(1), g2: q(1) }; 6];
        assert_eq!(
            context.verify(&[], &six_live),
            Err(PreparedVerifierError::TooManyTerms)
        );
    }

    #[test]
    fn residue_scale_has_exact_order_and_subfield() {
        assert_eq!(RESIDUE_ROOT.pow_u64(27), Fp12::ONE);
        assert_ne!(RESIDUE_ROOT.pow_u64(9), Fp12::ONE);
        assert_eq!(RESIDUE_ROOT.frobenius_map_cubed(), RESIDUE_ROOT);
        assert_eq!(
            residue_scale(CubicResidueClass::RootSquared),
            RESIDUE_ROOT.square()
        );
    }

    #[test]
    fn witness_validation_rejects_wrong_inverse_after_point_validation() {
        let context = PreparedVerifier::new(&[q(1)], &[]).unwrap();
        let prepared = [PreparedTerm {
            g1: p(1),
            prepared_g2: 0,
        }];
        let witness =
            PairingResidueWitness::new(Fp12::ONE, Fp12::ZERO, CubicResidueClass::Identity);
        assert!(
            !context
                .verify_with_witness(&prepared, &[], &witness)
                .unwrap()
        );

        let invalid_g1 = G1Affine {
            x: crate::Fp::ONE,
            y: crate::Fp::ONE,
            infinity: false,
        };
        assert_eq!(
            context.verify_with_witness(
                &[PreparedTerm {
                    g1: invalid_g1,
                    prepared_g2: usize::MAX,
                }],
                &[],
                &witness,
            ),
            Err(PreparedVerifierError::InvalidPoint(InputError::NotOnCurve))
        );
    }

    #[test]
    fn empty_online_witness_check_uses_the_fixed_result() {
        let valid = PreparedVerifier::new(
            &[],
            &[
                FixedPair { g1: p(1), g2: q(1) },
                FixedPair {
                    g1: p(1).negate(),
                    g2: q(1),
                },
            ],
        )
        .unwrap();
        let invalid = PreparedVerifier::new(&[], &[FixedPair { g1: p(1), g2: q(1) }]).unwrap();
        let witness = PairingResidueWitness::new(Fp12::ZERO, Fp12::ZERO, CubicResidueClass::Root);
        assert!(valid.verify_with_witness(&[], &[], &witness).unwrap());
        assert!(!invalid.verify_with_witness(&[], &[], &witness).unwrap());
    }

    #[test]
    fn generated_witness_accepts_a_kzg_pairing_equation() {
        let root = fp12([
            [
                12274218754903275212,
                15791406168501439385,
                1740723370600005415,
                2682105288990535617,
            ],
            [
                1807459676185143227,
                6179248009439724992,
                846821426112826270,
                1500330907165842253,
            ],
            [
                7047638830849624412,
                18292692833437538653,
                5927968271358249273,
                383744126228639942,
            ],
            [
                1475167556951738433,
                17523273350862414270,
                6221657431126890399,
                2497234512255412984,
            ],
            [
                533585287900028464,
                458199493651665179,
                16023629795512442969,
                1013474267654159678,
            ],
            [
                2902366895750293879,
                13196472936373282866,
                7469897830310185042,
                1957929298519922520,
            ],
            [
                4941669018700063836,
                6068843742716532816,
                1864010403333396437,
                296464012004109990,
            ],
            [
                14268628611012427653,
                8774810988636916428,
                8318884562710262654,
                3362127789948292012,
            ],
            [
                7352730025691977050,
                8278353736991635168,
                5947418650549193195,
                1817732944663282150,
            ],
            [
                702146310757385825,
                14323044134397895369,
                14052669610000457033,
                1503792897305606873,
            ],
            [
                7004219929198430402,
                5345807093393552687,
                14856743116278816999,
                1267397077035180357,
            ],
            [
                5695392706028005540,
                3603861717721981902,
                18130162519841012278,
                1134037055399694587,
            ],
        ]);
        let inverse = fp12([
            [
                12070271757161191044,
                11622944671148031983,
                8022313166833257262,
                3447914505644857954,
            ],
            [
                2696063439946186455,
                10628208811600786405,
                16573728980519463853,
                3419300733151995131,
            ],
            [
                6059545768901814467,
                5375804560051955987,
                13007823751729624033,
                171727094480632356,
            ],
            [
                4522887996003278513,
                13422022781223540841,
                6227390526368198822,
                2252240955438852006,
            ],
            [
                13442194637805522283,
                4629553780526494953,
                1605784331563153248,
                2274783208938795784,
            ],
            [
                15420274940570360740,
                877702660778154908,
                4514107088998262957,
                777473868408688905,
            ],
            [
                3031477824980187915,
                16025599519804155072,
                15831616474008298165,
                3283711105442462846,
            ],
            [
                16290769798079619950,
                3111331154494046936,
                15904667260910514094,
                3479718513334113177,
            ],
            [
                12465249464494930685,
                4661675820172526212,
                11123855807686148241,
                2512681522336823872,
            ],
            [
                4563072504229389102,
                584485681133349187,
                8643475515989763181,
                1497921177256037252,
            ],
            [
                13829516827347295091,
                14380830993695725550,
                7664268547275728619,
                466245022872056957,
            ],
            [
                4323882412257501532,
                189027086721504725,
                13777399269647702489,
                1524148709724664556,
            ],
        ]);
        let context = PreparedVerifier::new(&[q(1)], &[]).unwrap();
        let prepared = [PreparedTerm {
            g1: p(2),
            prepared_g2: 0,
        }];
        let live = [LivePair {
            g1: p(1).negate(),
            g2: q(2),
        }];
        let witness = PairingResidueWitness::new(root, inverse, CubicResidueClass::RootSquared);
        assert!(
            context
                .verify_with_witness(&prepared, &live, &witness)
                .unwrap()
        );
        let wrong_class = PairingResidueWitness::new(root, inverse, CubicResidueClass::Root);
        assert!(
            !context
                .verify_with_witness(&prepared, &live, &wrong_class)
                .unwrap()
        );
        let wrong_equation = [PreparedTerm {
            g1: p(3),
            prepared_g2: 0,
        }];
        assert!(
            !context
                .verify_with_witness(&wrong_equation, &live, &witness)
                .unwrap()
        );
    }
}
