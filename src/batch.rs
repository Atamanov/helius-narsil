//! Agave-shaped, alt_bn128-compatible batch operations.
//!
//! This facade deliberately accepts canonical wire values. It measures and
//! optimizes the work a validator actually performs: decode, validate,
//! arithmetic, and encode. All algorithms in this module are variable-time and
//! are suitable only when their inputs are public.
//!
//! # Encoding contract
//!
//! All values are canonical big-endian, alt_bn128 style. A G1 point is
//! `x | y` (64 bytes). A G2 point is `x1 | x0 | y1 | y0` (128 bytes, Fp2
//! imaginary limb first). A scalar is a canonical 32-byte Fr element. The
//! all-zero point encoding is the point at infinity. Non-canonical field
//! elements (`>= p`, or scalars `>= r`) are rejected, never reduced.
//!
//! # Validation order
//!
//! Error precedence is consensus-visible and fixed: length checks first
//! (mismatch, then empty, then cap), then per-element decoding in input
//! order with G1 validated before G2 within a pair and all points validated
//! before any scalar. G2 points get a full r-subgroup check. G1 needs none
//! (its cofactor is 1).
//!
//! # Repeated fixed-G2 calls
//!
//! In `std` builds, except under `force-portable`, calls retain up to four
//! validated G2 points in a thread-local LRU. For repeated G2 values used by
//! multi-pair arithmetic the same entry lazily retains the complete Miller
//! line schedule. A warm exact-byte hit never weakens G1-before-G2 validation
//! order. Invalid inputs and identity terms are never inserted. `no_std` and
//! `force-portable` builds validate the same inputs without the cache.

use alloc::{vec, vec::Vec};
use core::fmt;

use crate::{
    Fp, Fp2, Fp12, Fr, G1Affine, G1Projective, G2Affine,
    consts::{FR_MONT_ONE, FR_MONT_R2, R},
    fr::{invert_raw as fr_invert_raw, mont_mul as fr_mont_mul},
    limb,
    msm::msm_variable_time_affine,
    pairing::multi_pairing,
};

#[cfg(all(feature = "std", not(feature = "force-portable")))]
use crate::pairing::{
    final_exponentiation,
    miller::{G2Miller, multi_miller_loop_mixed},
};

#[cfg(all(feature = "std", not(feature = "force-portable")))]
mod validated_g2_cache {
    use alloc::vec::Vec;
    use core::cell::RefCell;
    use std::sync::Arc;

    use super::{G2Bytes, InputError};
    use crate::pairing::miller::{G2Prepared, prepare};
    use crate::{G1Affine, G2Affine};

    /// LRU order is deterministic. Hits move to the back and misses evict the
    /// front entry.
    pub(super) const CAPACITY: usize = 4;

    struct Entry {
        key: G2Bytes,
        point: G2Affine,
        prepared: Option<Arc<G2Prepared>>,
    }

    #[derive(Default)]
    struct Cache {
        entries: Vec<Entry>,
        #[cfg(test)]
        hits: usize,
        #[cfg(test)]
        misses: usize,
        #[cfg(test)]
        evictions: usize,
    }

    std::thread_local! {
        static CACHE: RefCell<Cache> = RefCell::new(Cache::default());
    }

    pub(super) fn product_is_one(g1: &G1Affine, key: &G2Bytes) -> Result<bool, InputError> {
        CACHE.with(|cell| {
            let mut cache = cell.borrow_mut();
            if let Some(index) = cache.entries.iter().position(|entry| entry.key == *key) {
                #[cfg(test)]
                {
                    cache.hits += 1;
                }
                let entry = cache.entries.remove(index);
                debug_assert!(!entry.point.is_identity());
                cache.entries.push(entry);
                return Ok(g1.is_identity());
            }

            // Exact-byte lookup has failed. Only a fully canonical, on-curve,
            // subgroup-checked point can advance the cache state.
            let point = key.to_affine()?;
            if g1.is_identity() || point.is_identity() {
                return Ok(true);
            }
            #[cfg(test)]
            {
                cache.misses += 1;
            }
            if cache.entries.len() == CAPACITY {
                cache.entries.remove(0);
                #[cfg(test)]
                {
                    cache.evictions += 1;
                }
            }
            cache.entries.push(Entry {
                key: *key,
                point,
                prepared: None,
            });
            Ok(false)
        })
    }

    /// Validate one G2 in consensus order, consulting only exact byte keys.
    /// A miss is retained only when the complete pair is live. Identity terms
    /// never populate the cache, but an existing warm hit may still be reused.
    pub(super) fn validate(key: &G2Bytes, retain: bool) -> Result<G2Affine, InputError> {
        CACHE.with(|cell| {
            let mut cache = cell.borrow_mut();
            if let Some(index) = cache.entries.iter().position(|entry| entry.key == *key) {
                #[cfg(test)]
                {
                    cache.hits += 1;
                }
                let entry = cache.entries.remove(index);
                let point = entry.point;
                cache.entries.push(entry);
                return Ok(point);
            }

            let point = key.to_affine()?;
            if !retain || point.is_identity() {
                return Ok(point);
            }
            #[cfg(test)]
            {
                cache.misses += 1;
            }
            if cache.entries.len() == CAPACITY {
                cache.entries.remove(0);
                #[cfg(test)]
                {
                    cache.evictions += 1;
                }
            }
            cache.entries.push(Entry {
                key: *key,
                point,
                prepared: None,
            });
            Ok(point)
        })
    }

    /// Record one batch-validated point under `validate`'s exact retention
    /// and LRU-motion rules. The caller proves the point decoded from `key`
    /// and passed the curve and subgroup checks. Without this insert a
    /// batch-validated small product never warms the prepared-schedule LRU.
    #[cfg(helius_avx512_ifma)]
    pub(super) fn retain_batch_validated(key: &G2Bytes, point: &G2Affine, retain: bool) {
        CACHE.with(|cell| {
            let mut cache = cell.borrow_mut();
            if let Some(index) = cache.entries.iter().position(|entry| entry.key == *key) {
                #[cfg(test)]
                {
                    cache.hits += 1;
                }
                let entry = cache.entries.remove(index);
                cache.entries.push(entry);
                return;
            }
            if !retain || point.is_identity() {
                return;
            }
            #[cfg(test)]
            {
                cache.misses += 1;
            }
            if cache.entries.len() == CAPACITY {
                cache.entries.remove(0);
                #[cfg(test)]
                {
                    cache.evictions += 1;
                }
            }
            cache.entries.push(Entry {
                key: *key,
                point: *point,
                prepared: None,
            });
        });
    }

    /// Return a cached schedule or prepare the validated point exactly once.
    /// If the four-entry LRU evicted this key earlier in the same large call,
    /// the returned schedule is ephemeral and does not perturb the cache.
    pub(super) fn prepared(key: &G2Bytes, point: &G2Affine) -> Arc<G2Prepared> {
        CACHE.with(|cell| {
            let mut cache = cell.borrow_mut();
            if let Some(index) = cache.entries.iter().position(|entry| entry.key == *key) {
                let mut entry = cache.entries.remove(index);
                debug_assert_eq!(entry.point, *point);
                let schedule = entry
                    .prepared
                    .get_or_insert_with(|| Arc::new(prepare(point)))
                    .clone();
                cache.entries.push(entry);
                schedule
            } else {
                Arc::new(prepare(point))
            }
        })
    }

    #[cfg(test)]
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub(super) struct Snapshot {
        pub keys: Vec<G2Bytes>,
        pub prepared_keys: Vec<G2Bytes>,
        pub hits: usize,
        pub misses: usize,
        pub evictions: usize,
    }

    #[cfg(test)]
    pub(super) fn reset() {
        CACHE.with(|cell| *cell.borrow_mut() = Cache::default());
    }

    #[cfg(test)]
    pub(super) fn snapshot() -> Snapshot {
        CACHE.with(|cell| {
            let cache = cell.borrow();
            Snapshot {
                keys: cache.entries.iter().map(|entry| entry.key).collect(),
                prepared_keys: cache
                    .entries
                    .iter()
                    .filter_map(|entry| entry.prepared.as_ref().map(|_| entry.key))
                    .collect(),
                hits: cache.hits,
                misses: cache.misses,
                evictions: cache.evictions,
            }
        })
    }
}

/// Per-call cap on [`g1_msm`] points. Exceeding it is [`InputError::CapExceeded`].
pub const MSM_MAX_POINTS: usize = 2048;
/// Per-call cap on [`pairing_product_is_one`] pairs.
pub const PAIRING_MAX_PAIRS: usize = 256;
/// Per-call cap on [`fr_lincomb`] and [`fr_batch_invert`] elements.
pub const FR_MAX_ELEMS: usize = 2048;
/// Encoded G1 point size: `x | y`, 32 bytes each.
pub const G1_BYTES: usize = 64;
/// Encoded G2 point size: `x1 | x0 | y1 | y0`, 32 bytes each.
pub const G2_BYTES: usize = 128;
/// Encoded pairing input size: one G1 point followed by one G2 point.
pub const PAIR_BYTES: usize = G1_BYTES + G2_BYTES;
/// Encoded scalar size: one canonical big-endian Fr element.
pub const SCALAR_BYTES: usize = 32;

/// Version marker matching Agave's batch-syscall crate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Version {
    /// Initial syscall ABI: the caps, encodings, and error precedence above.
    V0,
}

/// G1 affine point encoded as alt_bn128 `x | y`, or all zeroes for infinity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(transparent)]
pub struct G1Bytes(pub [u8; 64]);

/// G2 affine point encoded as alt_bn128 `x1 | x0 | y1 | y0`, or all zeroes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(transparent)]
pub struct G2Bytes(pub [u8; 128]);

/// Canonical big-endian scalar-field element.
#[derive(Clone, Copy, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(transparent)]
pub struct ScalarBytes(pub [u8; 32]);

/// One contiguous alt_bn128 pairing input.
#[derive(Clone, Copy, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct PairBytes {
    /// G1 partner, validated first within the pair.
    pub g1: G1Bytes,
    /// G2 partner. Validation includes the r-subgroup check.
    pub g2: G2Bytes,
}

/// alt_bn128 pairing verdict word used at the syscall boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(transparent)]
pub struct PodPairingResult(pub [u8; 32]);

// Exact Agave-facing spellings. `pub use` preserves tuple-struct constructors,
// unlike type aliases, while the shorter names remain convenient internally.
pub use G1Bytes as PodG1Point;
pub use G2Bytes as PodG2Point;
pub use PairBytes as PodG1G2Pair;
pub use ScalarBytes as PodScalar;

/// Stable validation taxonomy shared with the Agave batch-syscall design.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputError {
    /// Input length is not a whole number of encoded elements.
    InvalidLength,
    /// Field element or scalar encoding is `>= p` (or `>= r`).
    NonCanonical,
    /// Decoded point does not satisfy the curve equation.
    NotOnCurve,
    /// G2 point is on the twist but outside the r-order subgroup.
    NotInSubgroup,
    /// Input is empty, or a batch-inversion element is zero.
    ZeroInput,
    /// Element count exceeds the operation's per-call cap.
    CapExceeded,
    /// Paired input slices disagree in element count.
    LengthMismatch,
}

pub use InputError as AltBn128BatchError;

impl fmt::Display for InputError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidLength => "input length is not a whole number of elements",
            Self::NonCanonical => "field element encoding is not canonical",
            Self::NotOnCurve => "point is not on the curve",
            Self::NotInSubgroup => "G2 point is not in the r-order subgroup",
            Self::ZeroInput => "input is empty or contains a non-invertible zero",
            Self::CapExceeded => "input exceeds the per-call cap",
            Self::LengthMismatch => "input arrays disagree in count",
        })
    }
}

// `core::error::Error` keeps the Agave-facing error usable by `thiserror`
// consumers without forcing the arithmetic crate to enable `std`.
impl core::error::Error for InputError {}

impl G1Bytes {
    /// Decode and fully validate: canonical coordinates, then on-curve.
    /// All-zero bytes decode to infinity. G1 needs no subgroup check.
    #[inline]
    pub fn to_affine(&self) -> Result<G1Affine, InputError> {
        decode_g1(self)
    }

    /// Encode to canonical `x | y` bytes. Infinity encodes as all zeroes.
    #[inline]
    pub fn from_affine(point: &G1Affine) -> Self {
        encode_g1(point)
    }
}

impl G2Bytes {
    /// Decode and fully validate: canonical coordinates, on-curve, then
    /// r-subgroup membership. All-zero bytes decode to infinity.
    #[inline]
    pub fn to_affine(&self) -> Result<G2Affine, InputError> {
        decode_g2(self)
    }

    /// Encode to canonical `x1 | x0 | y1 | y0` bytes. Infinity is all zeroes.
    #[inline]
    pub fn from_affine(point: &G2Affine) -> Self {
        encode_g2(point)
    }
}

impl ScalarBytes {
    /// Decode a canonical big-endian scalar. Values `>= r` are
    /// [`InputError::NonCanonical`], never reduced.
    #[inline]
    pub fn to_fr(&self) -> Result<Fr, InputError> {
        Fr::from_bytes_be(&self.0).ok_or(InputError::NonCanonical)
    }

    /// Encode to canonical big-endian bytes.
    #[inline]
    pub fn from_fr(scalar: Fr) -> Self {
        Self(scalar.to_bytes_be())
    }
}

impl PodPairingResult {
    /// Encode a pairing verdict: 32 bytes, big-endian 1 for true, 0 for false.
    #[inline]
    pub fn from_verdict(verdict: bool) -> Self {
        let mut output = [0u8; 32];
        output[31] = u8::from(verdict);
        Self(output)
    }

    /// True iff the word encodes a passing pairing check.
    #[inline]
    pub fn verdict(&self) -> bool {
        self.0[31] == 1
    }
}

/// Agave-compatible spelling of [`g1_msm`].
#[inline]
pub fn alt_bn128_g1_msm(
    _version: Version,
    points: &[PodG1Point],
    scalars: &[PodScalar],
) -> Result<PodG1Point, AltBn128BatchError> {
    g1_msm(points, scalars)
}

/// Agave-compatible spelling of [`pairing_product_is_one`].
#[inline]
pub fn alt_bn128_pairing_check(
    _version: Version,
    pairs: &[PodG1G2Pair],
) -> Result<bool, AltBn128BatchError> {
    pairing_product_is_one(pairs)
}

/// Agave-compatible spelling of [`fr_lincomb`].
#[inline]
pub fn alt_bn128_fr_lincomb(
    _version: Version,
    a: &[PodScalar],
    b: &[PodScalar],
) -> Result<PodScalar, AltBn128BatchError> {
    fr_lincomb(a, b)
}

/// Agave-compatible spelling of [`fr_batch_invert`].
#[inline]
pub fn alt_bn128_fr_batch_invert(
    _version: Version,
    values: &[PodScalar],
) -> Result<Vec<PodScalar>, AltBn128BatchError> {
    fr_batch_invert(values)
}

/// Variable-time G1 multi-scalar multiplication with Agave validation order.
///
/// Returns the canonical encoding of `sum(scalars[i] * points[i])`.
/// Precedence: [`InputError::LengthMismatch`], [`InputError::ZeroInput`]
/// (empty), [`InputError::CapExceeded`] (over [`MSM_MAX_POINTS`]), then every
/// point validated in order before any scalar is examined.
pub fn g1_msm(points: &[G1Bytes], scalars: &[ScalarBytes]) -> Result<G1Bytes, InputError> {
    if points.len() != scalars.len() {
        return Err(InputError::LengthMismatch);
    }
    if points.is_empty() {
        return Err(InputError::ZeroInput);
    }
    if points.len() > MSM_MAX_POINTS {
        return Err(InputError::CapExceeded);
    }

    // Consensus-visible precedence: every point is validated before any scalar.
    let mut bases = Vec::with_capacity(points.len());
    for point in points {
        bases.push(point.to_affine()?);
    }
    let mut exponents = Vec::with_capacity(scalars.len());
    for scalar in scalars {
        exponents.push(decode_scalar_raw(scalar)?);
    }

    Ok(encode_g1(&msm_variable_time_affine(&bases, &exponents)))
}

/// True iff the product of all pairings is the GT identity.
///
/// Precedence: [`InputError::ZeroInput`] (empty), [`InputError::CapExceeded`]
/// (over [`PAIRING_MAX_PAIRS`]), then pairs validated in order, G1 before G2
/// within each pair. Every pair is validated even when its partner is
/// infinity. A batch whose pairs all involve infinity yields `Ok(true)`.
pub fn pairing_product_is_one(pairs: &[PairBytes]) -> Result<bool, InputError> {
    if pairs.is_empty() {
        return Err(InputError::ZeroInput);
    }
    if pairs.len() > PAIRING_MAX_PAIRS {
        return Err(InputError::CapExceeded);
    }

    // EIP-197 defines both groups with the same prime order. For one validated
    // pair, the discrete-log product is zero exactly when either point is the
    // identity. G1 remains first in the consensus-visible validation order.
    if pairs.len() == 1 {
        let pair = &pairs[0];
        let g1 = pair.g1.to_affine()?;
        #[cfg(all(feature = "std", not(feature = "force-portable")))]
        return validated_g2_cache::product_is_one(&g1, &pair.g2);
        #[cfg(not(all(feature = "std", not(feature = "force-portable"))))]
        {
            let g2 = pair.g2.to_affine()?;
            return Ok(g1.is_identity() || g2.is_identity());
        }
    }

    // Validate both partners before deciding whether an infinity pair can be
    // removed from the arithmetic product.
    let mut decoded = Vec::with_capacity(pairs.len());
    let mut validated_g2 = Vec::<(G2Bytes, G2Affine)>::with_capacity(pairs.len());

    #[cfg(all(helius_avx512_ifma, not(feature = "force-portable")))]
    {
        if should_batch_validate_g2(pairs) {
            for chunk in pairs.chunks(8) {
                let g1 = core::array::from_fn::<_, 8, _>(|lane| {
                    chunk
                        .get(lane)
                        .map(|pair| pair.g1.to_affine())
                        .unwrap_or(Ok(G1Affine::identity()))
                });
                let g2 = core::array::from_fn::<_, 8, _>(|lane| {
                    chunk
                        .get(lane)
                        .map(|pair| decode_g2_coordinates(&pair.g2))
                        .unwrap_or(Ok(G2Affine::identity()))
                });
                let safe = core::array::from_fn(|lane| g2[lane].unwrap_or(G2Affine::identity()));
                let (on_curve, in_subgroup) = crate::batch8::g2_validation_masks8(&safe);
                for lane in 0..chunk.len() {
                    let p = g1[lane]?;
                    let pair = &chunk[lane];
                    let q = if let Some((_, point)) =
                        validated_g2.iter().find(|(key, _)| key == &pair.g2)
                    {
                        *point
                    } else {
                        let point = g2[lane]?;
                        if on_curve & (1 << lane) == 0 {
                            return Err(InputError::NotOnCurve);
                        }
                        if in_subgroup & (1 << lane) == 0 {
                            return Err(InputError::NotInSubgroup);
                        }
                        #[cfg(feature = "std")]
                        validated_g2_cache::retain_batch_validated(
                            &pair.g2,
                            &point,
                            !p.is_identity(),
                        );
                        validated_g2.push((pair.g2, point));
                        point
                    };
                    if !p.infinity && !q.infinity {
                        decoded.push((p, q));
                    }
                }
            }
        } else {
            for pair in pairs {
                // G1 validation is consensus-visible and must precede both G2
                // validation and cache lookup for this pair.
                let g1 = pair.g1.to_affine()?;
                let g2 = if let Some((_, point)) = validated_g2
                    .iter()
                    .find(|(encoding, _)| encoding == &pair.g2)
                {
                    *point
                } else {
                    let point = validate_g2_for_pair(&pair.g2, &g1)?;
                    validated_g2.push((pair.g2, point));
                    point
                };
                if !g1.infinity && !g2.infinity {
                    decoded.push((g1, g2));
                }
            }
        }
    }
    #[cfg(not(all(helius_avx512_ifma, not(feature = "force-portable"))))]
    for pair in pairs {
        let g1 = pair.g1.to_affine()?;
        let g2 = if let Some((_, point)) = validated_g2.iter().find(|(key, _)| key == &pair.g2) {
            *point
        } else {
            let point = validate_g2_for_pair(&pair.g2, &g1)?;
            validated_g2.push((pair.g2, point));
            point
        };
        if !g1.infinity && !g2.infinity {
            decoded.push((g1, g2));
        }
    }
    let group_count = distinct_g2_up_to_sign_count(&decoded);
    let grouped =
        should_group_pairs(decoded.len(), group_count).then(|| group_pairs_by_g2(&decoded));
    let routed = grouped.as_deref().unwrap_or(decoded.as_slice());
    if routed.is_empty() {
        return Ok(true);
    }

    // Every input has been validated, identity terms have been removed, and
    // cancelling grouped terms have been dropped. In the prime-order target
    // group, one remaining nonidentity pairing cannot equal one. Apply the same
    // EIP-197 theorem as the direct n=1 path after routing rather than paying a
    // complete Miller loop and final exponentiation.
    if routed.len() == 1 {
        return Ok(false);
    }

    // Three or more distinct terms run as independent lane chains. The shared
    // accumulator's dependent chain grows with the term count, the masked
    // eight-lane engine's does not. Two terms keep the prepared route below.
    #[cfg(all(
        any(helius_avx512_ifma, helius_x86_runtime_ifma),
        not(feature = "force-portable")
    ))]
    if routed.len() >= crate::pairing::miller::VERTICAL_MIN_MILLER_TERMS
        && crate::batch8::ifma_available()
    {
        return Ok(crate::batch8::multi_pairing8(routed) == Fp12::ONE);
    }

    // Small repeated-verifier workloads use fixed G2 values. Prepare only
    // after all validation, identity filtering, grouping, and theorem exits so
    // collapsed batches never pay line generation. Exact warm keys replay the
    // cached schedule. Cold keys are prepared once for this Miller loop and
    // retained when their four-entry LRU entry is still resident.
    #[cfg(all(feature = "std", not(feature = "force-portable")))]
    if routed.len() <= 5 {
        let schedules: Vec<_> = routed
            .iter()
            .map(|(_, g2)| {
                let (key, point) = validated_g2
                    .iter()
                    .find(|(_, point)| point == g2)
                    .expect("routed G2 was validated from an exact input key");
                validated_g2_cache::prepared(key, point)
            })
            .collect();
        let mixed: Vec<_> = routed
            .iter()
            .zip(&schedules)
            .map(|((g1, _), schedule)| (g1, G2Miller::Prepared(schedule.as_ref())))
            .collect();
        let miller = multi_miller_loop_mixed(&mixed);
        return Ok(final_exponentiation(&miller) == Fp12::ONE);
    }

    let refs: Vec<_> = routed.iter().map(|(g1, g2)| (g1, g2)).collect();
    Ok(multi_pairing(&refs) == Fp12::ONE)
}

#[inline]
fn validate_g2_for_pair(key: &G2Bytes, g1: &G1Affine) -> Result<G2Affine, InputError> {
    #[cfg(all(feature = "std", not(feature = "force-portable")))]
    {
        validated_g2_cache::validate(key, !g1.is_identity())
    }
    #[cfg(not(all(feature = "std", not(feature = "force-portable"))))]
    {
        let _ = g1;
        key.to_affine()
    }
}

const MIN_MULTI_KEY_ROUTING_PAIRS: usize = 16;

#[inline]
fn should_group_pairs(live_count: usize, group_count: usize) -> bool {
    group_count < live_count
        && (group_count == 1 || live_count >= MIN_MULTI_KEY_ROUTING_PAIRS)
        && group_count <= core::cmp::max(2, live_count / 4)
}

#[inline]
#[cfg(any(test, all(helius_avx512_ifma, not(feature = "force-portable"))))]
fn should_batch_validate_g2(pairs: &[PairBytes]) -> bool {
    let first = &pairs[0].g2;
    let mut second = None;
    for pair in &pairs[1..] {
        let key = &pair.g2;
        if key != first {
            if let Some(second) = second {
                if key != second {
                    return true;
                }
            } else {
                second = Some(key);
            }
        }
    }
    second.is_some() && pairs.len() < MIN_MULTI_KEY_ROUTING_PAIRS
}

/// Count first-seen G2 groups up to sign without performing any G1 arithmetic.
/// `e(P, -Q) = e(-P, Q)`, so both encodings belong to one algebraic group.
fn distinct_g2_up_to_sign_count(pairs: &[(G1Affine, G2Affine)]) -> usize {
    debug_assert!(pairs.len() <= PAIRING_MAX_PAIRS);
    let mut distinct = Vec::<G2Affine>::with_capacity(pairs.len());
    for (_, g2) in pairs {
        let negative = g2.negate();
        if !distinct.iter().any(|seen| seen == g2 || seen == &negative) {
            distinct.push(*g2);
        }
    }
    distinct.len()
}

/// Collapse a fully validated pairing batch by equal G2 value up to sign.
///
/// The output order is the order in which each distinct G2 first appeared.
/// A negative G2 contributes the negated G1 to its representative's sum because
/// `e(P, -Q) = e(-P, Q)`. G1 additions remain projective until the group is
/// complete. Zero sums and identity partners contribute no pairing and are
/// omitted.
fn group_pairs_by_g2(pairs: &[(G1Affine, G2Affine)]) -> Vec<(G1Affine, G2Affine)> {
    debug_assert!(pairs.len() <= PAIRING_MAX_PAIRS);
    let mut groups = Vec::<(G1Projective, G2Affine)>::with_capacity(pairs.len());
    for (g1, g2) in pairs {
        if g1.is_identity() || g2.is_identity() {
            continue;
        }
        if let Some((sum, _)) = groups.iter_mut().find(|(_, grouped_g2)| grouped_g2 == g2) {
            *sum = sum.add_mixed(*g1);
        } else if let Some((sum, _)) = groups
            .iter_mut()
            .find(|(_, grouped_g2)| grouped_g2 == &g2.negate())
        {
            *sum = sum.add_mixed(g1.negate());
        } else {
            groups.push((G1Projective::from(*g1), *g2));
        }
    }
    groups
        .into_iter()
        .filter(|(sum, _)| !sum.is_identity())
        .map(|(sum, g2)| (sum.to_affine(), g2))
        .collect()
}

/// `sum(a[i] * b[i])` in Fr, with one canonical output reduction.
///
/// Precedence: [`InputError::LengthMismatch`], [`InputError::ZeroInput`]
/// (empty), [`InputError::CapExceeded`] (over [`FR_MAX_ELEMS`]), then pairs
/// decoded in order. Any scalar `>= r` is [`InputError::NonCanonical`].
pub fn fr_lincomb(a: &[ScalarBytes], b: &[ScalarBytes]) -> Result<ScalarBytes, InputError> {
    if a.len() != b.len() {
        return Err(InputError::LengthMismatch);
    }
    if a.is_empty() {
        return Err(InputError::ZeroInput);
    }
    if a.len() > FR_MAX_ELEMS {
        return Err(InputError::CapExceeded);
    }

    // A raw/raw Montgomery multiply produces `a*b/R`. Accumulating those
    // products and multiplying by R^2 once at the end gives the canonical dot
    // product. This removes two Montgomery multiplications per input pair.
    let mut acc = [0u64; 4];
    for (x, y) in a.iter().zip(b) {
        let x = decode_scalar_raw(x)?;
        let y = decode_scalar_raw(y)?;
        let product = fr_mont_mul(&x, &y).0;
        acc = limb::add_mod(&acc, &product, &R);
    }
    let canonical = fr_mont_mul(&acc, &FR_MONT_R2).0;
    Ok(ScalarBytes(encode_scalar_raw(canonical)))
}

/// Montgomery's batch inversion trick: one inversion and `3n - 3` products.
///
/// Returns the canonical inverse of every element, in input order.
/// Precedence: [`InputError::ZeroInput`] (empty), [`InputError::CapExceeded`]
/// (over [`FR_MAX_ELEMS`]), then elements decoded in order. A zero element is
/// also [`InputError::ZeroInput`], surfaced at its position in that scan.
pub fn fr_batch_invert(values: &[ScalarBytes]) -> Result<Vec<ScalarBytes>, InputError> {
    if values.is_empty() {
        return Err(InputError::ZeroInput);
    }
    if values.len() > FR_MAX_ELEMS {
        return Err(InputError::CapExceeded);
    }

    // Keep all values in their canonical-limb scale. Starting the prefix at R
    // makes its scale descend by one per raw input. The raw inverse of the
    // final scaled product then has exactly the compensating scale, so every
    // backward output is canonical without per-element Montgomery conversion.
    // The first `M(R, a[0])` and final `M(a[0]^-1, R)` identities are elided,
    // as is the unused last accumulator update: exactly 3n-3 products remain.
    let mut field = Vec::with_capacity(values.len());
    let mut prefix = Vec::with_capacity(values.len());
    let mut product = FR_MONT_ONE;
    for (index, value) in values.iter().enumerate() {
        let value = decode_scalar_raw(value)?;
        if limb::is_zero(&value) {
            return Err(InputError::ZeroInput);
        }
        prefix.push(product);
        product = if index == 0 {
            value
        } else {
            fr_mont_mul(&product, &value).0
        };
        field.push(value);
    }

    let mut product_inverse =
        fr_invert_raw(product).expect("product of nonzero field elements is nonzero");
    let mut output = vec![ScalarBytes([0; 32]); field.len()];
    for i in (0..field.len()).rev() {
        if i == 0 {
            output[0] = ScalarBytes(encode_scalar_raw(product_inverse));
            break;
        }
        let inverse = fr_mont_mul(&product_inverse, &prefix[i]).0;
        output[i] = ScalarBytes(encode_scalar_raw(inverse));
        product_inverse = fr_mont_mul(&product_inverse, &field[i]).0;
    }
    Ok(output)
}

#[inline]
fn decode_g1(bytes: &G1Bytes) -> Result<G1Affine, InputError> {
    if bytes.0.iter().all(|byte| *byte == 0) {
        return Ok(G1Affine::identity());
    }
    let x = decode_fp(&bytes.0[0..32])?;
    let y = decode_fp(&bytes.0[32..64])?;
    let point = G1Affine {
        x,
        y,
        infinity: false,
    };
    if !point.is_on_curve() {
        return Err(InputError::NotOnCurve);
    }
    Ok(point)
}

#[inline]
fn decode_g2(bytes: &G2Bytes) -> Result<G2Affine, InputError> {
    let point = decode_g2_coordinates(bytes)?;
    if !point.is_on_curve() {
        return Err(InputError::NotOnCurve);
    }
    if !point.is_in_correct_subgroup_assuming_on_curve() {
        return Err(InputError::NotInSubgroup);
    }
    Ok(point)
}

#[inline]
fn decode_g2_coordinates(bytes: &G2Bytes) -> Result<G2Affine, InputError> {
    if bytes.0.iter().all(|byte| *byte == 0) {
        return Ok(G2Affine::identity());
    }
    let x1 = decode_fp(&bytes.0[0..32])?;
    let x0 = decode_fp(&bytes.0[32..64])?;
    let y1 = decode_fp(&bytes.0[64..96])?;
    let y0 = decode_fp(&bytes.0[96..128])?;
    let point = G2Affine {
        x: Fp2::new(x0, x1),
        y: Fp2::new(y0, y1),
        infinity: false,
    };
    Ok(point)
}

#[inline]
fn decode_fp(bytes: &[u8]) -> Result<Fp, InputError> {
    let bytes: &[u8; 32] = bytes.try_into().map_err(|_| InputError::InvalidLength)?;
    Fp::from_bytes_be(bytes).ok_or(InputError::NonCanonical)
}

#[inline]
fn decode_scalar_raw(bytes: &ScalarBytes) -> Result<[u64; 4], InputError> {
    let bytes = &bytes.0;
    let limbs = [
        u64::from_be_bytes(bytes[24..32].try_into().unwrap()),
        u64::from_be_bytes(bytes[16..24].try_into().unwrap()),
        u64::from_be_bytes(bytes[8..16].try_into().unwrap()),
        u64::from_be_bytes(bytes[0..8].try_into().unwrap()),
    ];
    if limb::gte(&limbs, &R) {
        return Err(InputError::NonCanonical);
    }
    Ok(limbs)
}

#[inline]
fn encode_scalar_raw(limbs: [u64; 4]) -> [u8; 32] {
    let mut out = [0u8; 32];
    for (i, limb) in limbs.iter().enumerate() {
        let start = 24 - 8 * i;
        out[start..start + 8].copy_from_slice(&limb.to_be_bytes());
    }
    out
}

#[inline]
fn encode_g1(point: &G1Affine) -> G1Bytes {
    if point.infinity {
        return G1Bytes([0; 64]);
    }
    let mut out = [0u8; 64];
    out[0..32].copy_from_slice(&point.x.to_bytes_be());
    out[32..64].copy_from_slice(&point.y.to_bytes_be());
    G1Bytes(out)
}

#[inline]
fn encode_g2(point: &G2Affine) -> G2Bytes {
    if point.infinity {
        return G2Bytes([0; 128]);
    }
    let mut out = [0u8; 128];
    out[0..32].copy_from_slice(&point.x.c1.to_bytes_be());
    out[32..64].copy_from_slice(&point.x.c0.to_bytes_be());
    out[64..96].copy_from_slice(&point.y.c1.to_bytes_be());
    out[96..128].copy_from_slice(&point.y.c0.to_bytes_be());
    G2Bytes(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn be_field_round_trips() {
        for value in [0, 1, 2, u64::MAX] {
            let fp = Fp::from_u64(value);
            assert_eq!(Fp::from_bytes_be(&fp.to_bytes_be()), Some(fp));
            let fr = Fr::from_u64(value);
            assert_eq!(Fr::from_bytes_be(&fr.to_bytes_be()), Some(fr));
        }
    }

    #[test]
    fn g2_generator_passes_subgroup_check() {
        let generator = G2Affine::generator();
        assert!(generator.is_on_curve());
        assert!(generator.is_in_correct_subgroup_assuming_on_curve());
    }

    #[test]
    fn single_pair_identity_theorem_matches_full_pairing() {
        let p = G1Projective::generator();
        let q = crate::g2::G2Projective::from(G2Affine::generator());
        let g1 = [
            G1Affine::identity(),
            p.to_affine(),
            p.mul_scalar(Fr::from_u64(17)).to_affine(),
        ];
        let g2 = [
            G2Affine::identity(),
            q.to_affine(),
            q.mul_scalar(Fr::from_u64(19)).to_affine(),
        ];
        for p in g1 {
            for q in g2 {
                assert_eq!(
                    pairing_product_is_one(&[test_pair_unconditional(p, q)]),
                    Ok(crate::pairing::pairing(&p, &q).is_one()),
                );
            }
        }
    }

    #[test]
    fn group_router_thresholds_are_pinned() {
        for (live, groups, expected) in [
            (1usize, 1usize, false),
            (2, 1, true),
            (2, 2, false),
            (3, 2, false),
            (4, 2, false),
            (5, 2, false),
            (8usize, 1usize, true),
            (8, 2, false),
            (8, 3, false),
            (8, 4, false),
            (8, 8, false),
            (16, 1, true),
            (16, 2, true),
            (16, 3, true),
            (16, 4, true),
            (16, 16, false),
            (7, 7, false),
            (15, 7, false),
            (28, 7, true),
        ] {
            assert_eq!(
                should_group_pairs(live, groups),
                expected,
                "live={live}, groups={groups}"
            );
        }
    }

    #[test]
    fn g2_validation_route_thresholds_are_pinned() {
        let g1 = G1Affine::generator();
        for count in [8_usize, 16] {
            let points: Vec<_> = (1..=count as u64)
                .map(test_g2_multiple_unconditional)
                .collect();
            let expected = if count == 8 {
                [false, true, true, true, true]
            } else {
                [false, false, true, true, true]
            };
            for (distinct, expected) in [1_usize, 2, 3, 4, count].into_iter().zip(expected) {
                let pairs: Vec<_> = (0..count)
                    .map(|index| test_pair_unconditional(g1, points[index % distinct]))
                    .collect();
                assert_eq!(
                    should_batch_validate_g2(&pairs),
                    expected,
                    "count={count}, distinct={distinct}"
                );
            }
        }
    }

    #[test]
    fn equal_g2_grouping_is_ordered_and_drops_cancellation() {
        use crate::g2::G2Projective;

        let p = G1Projective::generator();
        let p1 = p.to_affine();
        let p2 = p.double().to_affine();
        let p3 = p.mul_scalar(Fr::from_u64(3)).to_affine();
        let q = G2Projective::from(G2Affine::generator());
        let q1 = q.to_affine();
        let q2 = q.double().to_affine();
        let q3 = q.mul_scalar(Fr::from_u64(3)).to_affine();
        let grouped = group_pairs_by_g2(&[
            (p1, q2),
            (p1, q1),
            (p2, q2),
            (p1, q3),
            (p1.negate(), q3),
            (G1Affine::identity(), q1),
        ]);
        assert_eq!(
            distinct_g2_up_to_sign_count(&[
                (p1, q2),
                (p1, q1),
                (p2, q2),
                (p1, q3),
                (p1.negate(), q3),
                (G1Affine::identity(), q1),
            ]),
            3
        );
        assert_eq!(grouped, vec![(p3, q2), (p1, q1)]);
    }

    #[test]
    fn signed_g2_grouping_moves_the_sign_to_g1() {
        use crate::g2::G2Projective;

        let p = G1Projective::generator();
        let p1 = p.to_affine();
        let p2 = p.double().to_affine();
        let q1 = G2Projective::from(G2Affine::generator()).to_affine();
        let negative_q1 = q1.negate();

        let grouped = group_pairs_by_g2(&[(p2, q1), (p1, negative_q1)]);
        assert_eq!(
            distinct_g2_up_to_sign_count(&[(p2, q1), (p1, negative_q1)]),
            1
        );
        assert_eq!(grouped, vec![(p1, q1)]);

        let cancelled = group_pairs_by_g2(&[(p1, q1), (p1, negative_q1)]);
        assert!(cancelled.is_empty());

        // First-seen order fixes the representative even when the negative
        // encoding arrives first. The later positive-Q term moves its sign to
        // G1, rather than silently changing the group's representative.
        let negative_first = group_pairs_by_g2(&[(p2, negative_q1), (p1, q1)]);
        assert_eq!(negative_first, vec![(p1, negative_q1)]);
    }

    #[test]
    fn routed_single_term_theorem_covers_small_common_q_batches() {
        use crate::g2::G2Projective;

        let p = G1Projective::generator();
        let q = G2Projective::from(G2Affine::generator()).to_affine();
        for count in 2..=5 {
            let pairs: Vec<_> = (1..=count)
                .map(|multiple| {
                    test_pair_unconditional(
                        p.mul_scalar(Fr::from_u64(multiple as u64)).to_affine(),
                        q,
                    )
                })
                .collect();
            assert_eq!(pairing_product_is_one(&pairs), Ok(false), "count={count}");
        }

        let signed_cancel = [
            test_pair_unconditional(p.to_affine(), q),
            test_pair_unconditional(p.to_affine(), q.negate()),
        ];
        assert_eq!(pairing_product_is_one(&signed_cancel), Ok(true));
    }

    #[test]
    fn distinct_small_batches_match_the_full_typed_pairing_product() {
        use crate::g2::G2Projective;

        let g1 = G1Projective::generator();
        let g2 = G2Projective::from(G2Affine::generator());
        for count in 2..=5 {
            let decoded: Vec<_> = (1..=count)
                .map(|multiple| {
                    (
                        g1.mul_scalar(Fr::from_u64(multiple as u64 + 1)).to_affine(),
                        g2.mul_scalar(Fr::from_u64(3 * multiple as u64 + 1))
                            .to_affine(),
                    )
                })
                .collect();
            let pairs: Vec<_> = decoded
                .iter()
                .map(|(p, q)| test_pair_unconditional(*p, *q))
                .collect();
            let expected = decoded.iter().fold(Fp12::ONE, |product, (p, q)| {
                product * crate::pairing::pairing(p, q)
            });
            assert_eq!(
                pairing_product_is_one(&pairs),
                Ok(expected.is_one()),
                "count={count}"
            );
        }
    }

    #[test]
    fn small_batch_routing_matches_full_pairing_across_signed_and_identity_shapes() {
        use crate::g2::G2Projective;

        let g1 = G1Projective::generator();
        let g2 = G2Projective::from(G2Affine::generator());
        for seed in 1..=12_u64 {
            for count in 2..=5 {
                let decoded: Vec<_> = (0..count)
                    .map(|index| {
                        let mut p = g1
                            .mul_scalar(Fr::from_u64(seed + index as u64 + 1))
                            .to_affine();
                        let mut q = g2
                            .mul_scalar(Fr::from_u64(seed * 3 + (index % 3) as u64 + 1))
                            .to_affine();
                        if (seed + index as u64).is_multiple_of(4) {
                            q = q.negate();
                        }
                        if (seed + index as u64).is_multiple_of(11) {
                            p = G1Affine::identity();
                        }
                        (p, q)
                    })
                    .collect();
                let encoded: Vec<_> = decoded
                    .iter()
                    .map(|(p, q)| test_pair_unconditional(*p, *q))
                    .collect();
                let refs: Vec<_> = decoded.iter().map(|(p, q)| (p, q)).collect();
                assert_eq!(
                    pairing_product_is_one(&encoded),
                    Ok(crate::pairing::multi_pairing(&refs).is_one()),
                    "seed={seed}, count={count}"
                );
            }
        }
    }

    #[cfg(all(feature = "std", not(feature = "force-portable")))]
    fn test_pair(g1: G1Affine, g2: G2Affine) -> PairBytes {
        test_pair_unconditional(g1, g2)
    }

    fn test_pair_unconditional(g1: G1Affine, g2: G2Affine) -> PairBytes {
        PairBytes {
            g1: G1Bytes::from_affine(&g1),
            g2: G2Bytes::from_affine(&g2),
        }
    }

    fn test_g2_multiple_unconditional(value: u64) -> G2Affine {
        use crate::g2::G2Projective;
        G2Projective::from(G2Affine::generator())
            .mul_scalar(Fr::from_u64(value))
            .to_affine()
    }

    #[cfg(all(feature = "std", not(feature = "force-portable")))]
    fn test_g2_multiple(value: u64) -> G2Affine {
        test_g2_multiple_unconditional(value)
    }

    #[cfg(all(feature = "std", not(feature = "force-portable")))]
    #[test]
    fn validated_g2_cache_records_exact_key_hits_and_distinct_misses() {
        use crate::pairing::pairing;

        validated_g2_cache::reset();
        let p = G1Affine::generator();
        let q1 = test_g2_multiple(1);
        let q2 = test_g2_multiple(2);
        for q in [q1, q1, q2] {
            let pair = test_pair(p, q);
            assert_eq!(
                pairing_product_is_one(&[pair]).unwrap(),
                pairing(&p, &q).is_one()
            );
        }
        let state = validated_g2_cache::snapshot();
        assert_eq!((state.hits, state.misses, state.evictions), (1, 2, 0));
        assert_eq!(
            state.keys,
            vec![G2Bytes::from_affine(&q1), G2Bytes::from_affine(&q2)]
        );
    }

    #[cfg(all(feature = "std", not(feature = "force-portable")))]
    #[test]
    fn prepared_g2_cache_is_lazy_for_theorem_exits_and_warm_for_small_batches() {
        validated_g2_cache::reset();
        let p = G1Projective::generator();
        let q1 = test_g2_multiple(1);
        let q2 = test_g2_multiple(2);

        let collapsing = [
            test_pair(p.to_affine(), q1),
            test_pair(p.double().to_affine(), q1),
        ];
        assert_eq!(pairing_product_is_one(&collapsing), Ok(false));
        assert!(validated_g2_cache::snapshot().prepared_keys.is_empty());

        let distinct = [
            test_pair(p.to_affine(), q1),
            test_pair(p.double().to_affine(), q2),
        ];
        let first = pairing_product_is_one(&distinct).unwrap();
        let cold = validated_g2_cache::snapshot();
        assert_eq!(cold.prepared_keys.len(), 2);
        let second = pairing_product_is_one(&distinct).unwrap();
        let warm = validated_g2_cache::snapshot();
        assert_eq!(second, first);
        assert_eq!(warm.prepared_keys.len(), 2);
        assert!(warm.hits >= cold.hits + 2);
    }

    #[cfg(all(feature = "std", not(feature = "force-portable")))]
    #[test]
    fn large_common_g2_batches_validate_one_exact_key() {
        let p = G1Projective::generator();
        let q = test_g2_multiple(1);

        for count in [8_usize, 16] {
            validated_g2_cache::reset();
            let pairs: Vec<_> = (1..=count)
                .map(|value| test_pair(p.mul_scalar(Fr::from_u64(value as u64)).to_affine(), q))
                .collect();
            assert_eq!(pairing_product_is_one(&pairs), Ok(false));

            let state = validated_g2_cache::snapshot();
            assert_eq!(state.keys, vec![G2Bytes::from_affine(&q)]);
            assert_eq!((state.hits, state.misses, state.evictions), (0, 1, 0));
            assert!(state.prepared_keys.is_empty());
        }
    }

    #[cfg(all(feature = "std", not(feature = "force-portable")))]
    #[test]
    fn validated_g2_cache_has_deterministic_lru_eviction() {
        validated_g2_cache::reset();
        let p = G1Affine::generator();
        let points: Vec<_> = (1..=validated_g2_cache::CAPACITY as u64 + 1)
            .map(test_g2_multiple)
            .collect();
        for q in &points[..validated_g2_cache::CAPACITY] {
            pairing_product_is_one(&[test_pair(p, *q)]).unwrap();
        }
        // Refresh the oldest entry, then insert one more point. The second
        // original key must be the deterministic eviction victim.
        pairing_product_is_one(&[test_pair(p, points[0])]).unwrap();
        pairing_product_is_one(&[test_pair(p, points[validated_g2_cache::CAPACITY])]).unwrap();
        let state = validated_g2_cache::snapshot();
        assert_eq!((state.hits, state.misses, state.evictions), (1, 5, 1));
        assert_eq!(
            state.keys,
            vec![
                G2Bytes::from_affine(&points[2]),
                G2Bytes::from_affine(&points[3]),
                G2Bytes::from_affine(&points[0]),
                G2Bytes::from_affine(&points[4]),
            ]
        );
    }

    #[cfg(all(feature = "std", not(feature = "force-portable")))]
    #[test]
    fn invalid_and_identity_inputs_cannot_populate_or_bypass_validation() {
        validated_g2_cache::reset();
        let p = G1Affine::generator();
        let q = test_g2_multiple(7);
        pairing_product_is_one(&[test_pair(p, q)]).unwrap();
        let warm = validated_g2_cache::snapshot();

        let mut bad_g1 = test_pair(p, q);
        bad_g1.g1.0[..32].fill(0xff);
        assert_eq!(
            pairing_product_is_one(&[bad_g1]),
            Err(InputError::NonCanonical)
        );
        assert_eq!(validated_g2_cache::snapshot(), warm);

        let mut bad_g2 = test_pair(p, q);
        bad_g2.g2.0[..32].fill(0xff);
        assert_eq!(
            pairing_product_is_one(&[bad_g2]),
            Err(InputError::NonCanonical)
        );
        assert_eq!(validated_g2_cache::snapshot(), warm);

        validated_g2_cache::reset();
        assert!(pairing_product_is_one(&[test_pair(G1Affine::identity(), q)]).unwrap());
        assert!(pairing_product_is_one(&[test_pair(p, G2Affine::identity())]).unwrap());
        let identities = test_pair(G1Affine::identity(), G2Affine::identity());
        assert!(pairing_product_is_one(&[identities]).unwrap());
        assert!(validated_g2_cache::snapshot().keys.is_empty());

        let mut identity_bad_g2 = test_pair(G1Affine::identity(), q);
        identity_bad_g2.g2.0[..32].fill(0xff);
        assert_eq!(
            pairing_product_is_one(&[identity_bad_g2]),
            Err(InputError::NonCanonical)
        );

        let mut bad_both = test_pair(p, q);
        bad_both.g1.0[63] ^= 1;
        bad_both.g2.0[..32].fill(0xff);
        assert_eq!(
            pairing_product_is_one(&[bad_both]),
            Err(InputError::NotOnCurve)
        );

        validated_g2_cache::reset();
        assert!(!pairing_product_is_one(&[test_pair(p, q)]).unwrap());
        assert!(pairing_product_is_one(&[test_pair(G1Affine::identity(), q)]).unwrap());
        let state = validated_g2_cache::snapshot();
        assert_eq!((state.hits, state.misses, state.evictions), (1, 1, 0));
        assert_eq!(state.keys, vec![G2Bytes::from_affine(&q)]);
    }

    #[cfg(all(feature = "std", not(feature = "force-portable")))]
    #[test]
    fn multi_pair_errors_preserve_g1_before_g2_and_never_cache_invalid_g2() {
        validated_g2_cache::reset();
        let p = G1Affine::generator();
        let q = test_g2_multiple(23);

        let identity = test_pair(G1Affine::identity(), G2Affine::identity());
        let mut identity_bad_g2 = test_pair(G1Affine::identity(), q);
        identity_bad_g2.g2.0[..32].fill(0xff);
        assert_eq!(
            pairing_product_is_one(&[identity, identity_bad_g2]),
            Err(InputError::NonCanonical)
        );
        assert!(validated_g2_cache::snapshot().keys.is_empty());

        let mut bad_both = test_pair(p, q);
        bad_both.g1.0[63] ^= 1;
        bad_both.g2.0[..32].fill(0xff);
        assert_eq!(
            pairing_product_is_one(&[identity, bad_both]),
            Err(InputError::NotOnCurve)
        );
        assert!(validated_g2_cache::snapshot().keys.is_empty());
    }

    #[cfg(all(feature = "std", not(feature = "force-portable")))]
    #[test]
    fn five_distinct_keys_have_the_exact_backend_cache_shape() {
        validated_g2_cache::reset();
        let p = G1Projective::generator();
        let pairs: Vec<_> = (1..=5_u64)
            .map(|value| {
                test_pair(
                    p.mul_scalar(Fr::from_u64(value + 1)).to_affine(),
                    test_g2_multiple(value),
                )
            })
            .collect();
        let first = pairing_product_is_one(&pairs).unwrap();
        let cold = validated_g2_cache::snapshot();
        #[cfg(any(helius_avx512_ifma, helius_x86_runtime_ifma))]
        let expected = if crate::batch8::ifma_available() {
            (validated_g2_cache::CAPACITY, 0)
        } else {
            (validated_g2_cache::CAPACITY, validated_g2_cache::CAPACITY)
        };
        #[cfg(not(any(helius_avx512_ifma, helius_x86_runtime_ifma)))]
        let expected = (validated_g2_cache::CAPACITY, validated_g2_cache::CAPACITY);
        assert_eq!(cold.keys.len(), expected.0);
        // Every tier retains the four keys that survive the bounded LRU, so a
        // later small product can warm-replay a fixed verifier key. IFMA n=5
        // consumes the validated points in the lane engine and must not build
        // dead scalar line schedules. Scalar tiers prepare the surviving keys.
        assert_eq!(cold.prepared_keys.len(), expected.1);
        let second = pairing_product_is_one(&pairs).unwrap();
        let warm = validated_g2_cache::snapshot();
        assert_eq!(second, first);
        assert_eq!(warm.keys.len(), expected.0);
        assert_eq!(warm.prepared_keys.len(), expected.1);
    }

    #[cfg(all(feature = "std", not(feature = "force-portable")))]
    #[test]
    fn validated_g2_cache_is_thread_local() {
        validated_g2_cache::reset();
        let p = G1Affine::generator();
        let q1 = test_g2_multiple(11);
        let q2 = test_g2_multiple(13);
        pairing_product_is_one(&[test_pair(p, q1)]).unwrap();
        let outer = validated_g2_cache::snapshot();

        let inner = std::thread::spawn(move || {
            assert!(validated_g2_cache::snapshot().keys.is_empty());
            pairing_product_is_one(&[test_pair(p, q2)]).unwrap();
            validated_g2_cache::snapshot()
        })
        .join()
        .unwrap();
        assert_eq!(inner.keys, vec![G2Bytes::from_affine(&q2)]);
        assert_eq!((inner.hits, inner.misses), (0, 1));
        assert_eq!(validated_g2_cache::snapshot(), outer);
    }
}
