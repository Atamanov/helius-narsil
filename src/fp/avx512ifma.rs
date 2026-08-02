//! AVX-512 IFMA 8-way batched Montgomery arithmetic (radix-52).
//!
//! Eight independent `Fp` values ride the eight 64-bit lanes of ZMM
//! registers in structure-of-arrays form: five limbs of 52 bits each, in the
//! radix-52 Montgomery domain `v * 2^260 mod p`. `vpmadd52{l,h}uq` performs
//! the 52x52-bit partial products. Per-element conversion between the 4x64
//! `2^256` domain and the 5x52 `2^260` domain costs one batched Montgomery
//! multiplication by a fixed constant each way, so the tier only pays off
//! when several multiplications amortize one conversion pair, such as the MSM
//! bucket-accumulation phase.
//!
//! On generic x86-64 this module is reachable only below the
//! `#[target_feature]` whole-batch entry. CPUID+XCR0 runtime dispatch guards
//! that entry. An unsupported host stays on the scalar backend. When IFMA is
//! already a compile-time target feature, the same implementation also backs
//! typed and MSM operations.

use core::arch::x86_64::*;

use crate::consts::{MONT_ONE, P, P_INV};
use crate::fp::Fp;

const MASK52: u64 = (1 << 52) - 1;
/// `-p^{-1} mod 2^52`: the low 52 bits of the radix-64 constant, because the
/// inverse of `p` modulo `2^52` is the inverse modulo `2^64` truncated.
const P_INV_52: u64 = P_INV & MASK52;
const P52: [u64; 5] = to_radix52(P);
const MONT52_BITS: u32 = 5 * 52;
const MODULUS_BITS: u32 = 256 - P[3].leading_zeros();
const SOS_MAX_ARITY: usize = 6;
const LOOSE3_MAX_LIMB: u64 = 3 * MASK52;
const LOOSE3_MAX_CARRY: u64 = 2;
const LOOSE3_MAX_TOP_LIMB: u64 = 3 * P52[4] + LOOSE3_MAX_CARRY;

const _: () = assert!(SOS_MAX_ARITY < 1 << (MONT52_BITS - MODULUS_BITS));
const _: () = assert!(LOOSE3_MAX_LIMB < 1 << 54);
const _: () = assert!(LOOSE3_MAX_TOP_LIMB <= MASK52);

/// Largest admissible lazy bound multiple. `LAZY_CAP * p < 2^260` keeps
/// every bound-tracked value inside five normalized limbs, and its
/// 254-bit-shifted quotient stays below `2^6` for the fold step.
const LAZY_CAP: u64 = 84;
const _: () = assert!(mul_p52(LAZY_CAP).1 == 0);
const _: () = {
    let (limbs, _) = mul_p52(LAZY_CAP);
    assert!(limbs[4] >> 46 < 1 << 6);
};

/// `2^254 - p` in normalized radix-52 limbs, the additive fold constant:
/// `v = q * 2^254 + r` becomes `r + q * EPS52`, congruent mod `p`.
const EPS52: [u64; 5] = to_radix52(const_sub(&[0, 0, 0, 1 << 62], &P));
// The fold quotient is below 2^6, so eps * q never carries past limb 4.
const _: () = assert!(EPS52[4] << 6 <= MASK52);

/// `k * p` in normalized radix-52 limbs plus the carry past limb 4.
const fn mul_p52(k: u64) -> ([u64; 5], u64) {
    let mut out = [0u64; 5];
    let mut carry = 0u128;
    let mut j = 0;
    while j < 5 {
        let acc = k as u128 * P52[j] as u128 + carry;
        out[j] = (acc as u64) & MASK52;
        carry = acc >> 52;
        j += 1;
    }
    (out, carry as u64)
}

/// Whether `s * p <= r * 2^260`, the pre-reduction bound that makes an
/// `r`-round subtraction tail canonicalize a sum of bounded products.
const fn sos_sum_ok(s: u64, r: u64) -> bool {
    mul_p52(s).1 < r
}

/// Limb representation of `k * p` whose limbs cover an `m`-term per-limb
/// subtraction of normalized limbs without borrowing. Lending `m * 2^52`
/// down each limb telescopes, so the value stays exactly `k * p`.
const fn bias52(k: u64, m: u64) -> [u64; 5] {
    let (kp, carry) = mul_p52(k);
    assert!(carry == 0 && kp[4] >= m);
    [
        kp[0] + (m << 52),
        kp[1] + (m << 52) - m,
        kp[2] + (m << 52) - m,
        kp[3] + (m << 52) - m,
        kp[4] - m,
    ]
}

/// The lent lower limbs cover any `m` normalized limbs. The top limb has no
/// lend, so it must dominate the subtrahend top limbs, which value bounds
/// summing to `csum * p` cap at `csum * (P52[4] + 1)`.
const fn bias_ok(k: u64, m: u64, csum: u64) -> bool {
    let (kp, carry) = mul_p52(k);
    carry == 0 && kp[4] >= m + csum * (P52[4] + 1)
}

/// Smallest borrow-free bias multiple for an `m`-term subtraction of values
/// bounded by `csum * p` in total.
const fn min_bias(m: u64, csum: u64) -> u64 {
    let mut k = 1;
    while !bias_ok(k, m, csum) {
        k += 1;
        assert!(k <= LAZY_CAP);
    }
    k
}

const fn lt52(a: &[u64; 5], b: &[u64; 5]) -> bool {
    let mut j = 4;
    loop {
        if a[j] < b[j] {
            return true;
        }
        if a[j] > b[j] {
            return false;
        }
        if j == 0 {
            return false;
        }
        j -= 1;
    }
}

/// Exact post-fold bound: folding a value below `b * p` leaves a value
/// below `fold_bound(b) * p`.
const fn fold_bound(b: u64) -> u64 {
    let (bp, carry) = mul_p52(b);
    assert!(b >= 1 && carry == 0);
    // Largest input is b*p - 1; its 254-bit quotient caps q.
    let mut vmax = bp;
    let mut j = 0;
    while j < 5 {
        if vmax[j] > 0 {
            vmax[j] -= 1;
            break;
        }
        vmax[j] = MASK52;
        j += 1;
    }
    let q = vmax[4] >> 46;
    // out_max = (2^254 - 1) + q * eps.
    let mut out = [MASK52, MASK52, MASK52, MASK52, (1u64 << 46) - 1];
    let mut c = 0u128;
    let mut j = 0;
    while j < 5 {
        let acc = out[j] as u128 + q as u128 * EPS52[j] as u128 + c;
        out[j] = (acc as u64) & MASK52;
        c = acc >> 52;
        j += 1;
    }
    assert!(c == 0);
    let mut bound = 1;
    loop {
        let (cp, cc) = mul_p52(bound);
        if cc == 0 && lt52(&out, &cp) {
            return bound;
        }
        bound += 1;
        assert!(bound <= LAZY_CAP);
    }
}

/// Folds that bring a bound of `b` down to at most 3, where plain
/// subtraction rounds finish cheaper than another fold.
const fn folds_needed(mut b: u64) -> usize {
    let mut n = 0;
    while b > 3 {
        b = fold_bound(b);
        n += 1;
        assert!(n <= 4);
    }
    n
}

const fn folded_bound(mut b: u64) -> u64 {
    while b > 3 {
        b = fold_bound(b);
    }
    b
}

const fn max_u64(a: u64, b: u64) -> u64 {
    if a >= b { a } else { b }
}

/// Repack canonical 4x64 little-endian limbs into 5x52. Pure bit slicing.
/// value-preserving for any 256-bit input.
const fn to_radix52(x: [u64; 4]) -> [u64; 5] {
    [
        x[0] & MASK52,
        ((x[0] >> 52) | (x[1] << 12)) & MASK52,
        ((x[1] >> 40) | (x[2] << 24)) & MASK52,
        ((x[2] >> 28) | (x[3] << 36)) & MASK52,
        x[3] >> 16,
    ]
}

/// Inverse of [`to_radix52`]. Limbs must be normalized (below `2^52`).
#[inline(always)]
fn from_radix52(l: &[u64; 5]) -> [u64; 4] {
    debug_assert!(l.iter().all(|&limb| limb <= MASK52));
    [
        l[0] | (l[1] << 52),
        (l[1] >> 12) | (l[2] << 40),
        (l[2] >> 24) | (l[3] << 28),
        (l[3] >> 36) | (l[4] << 16),
    ]
}

const fn const_gte(a: &[u64; 4], b: &[u64; 4]) -> bool {
    let mut i = 3usize;
    loop {
        if a[i] > b[i] {
            return true;
        }
        if a[i] < b[i] {
            return false;
        }
        if i == 0 {
            return true;
        }
        i -= 1;
    }
}

const fn const_sub(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
    let mut out = [0u64; 4];
    let mut borrow = 0u64;
    let mut i = 0usize;
    while i < 4 {
        let (d, b1) = a[i].overflowing_sub(b[i]);
        let (d, b2) = d.overflowing_sub(borrow);
        out[i] = d;
        borrow = (b1 as u64) | (b2 as u64);
        i += 1;
    }
    out
}

/// `2a mod p` for `a < p` (`2a < 2^255` never overflows four limbs).
const fn double_mod_p(a: [u64; 4]) -> [u64; 4] {
    let mut r = [
        a[0] << 1,
        (a[1] << 1) | (a[0] >> 63),
        (a[2] << 1) | (a[1] >> 63),
        (a[3] << 1) | (a[2] >> 63),
    ];
    if const_gte(&r, &P) {
        r = const_sub(&r, &P);
    }
    r
}

const fn pow2_shift_mod_p(mut v: [u64; 4], count: usize) -> [u64; 4] {
    let mut i = 0usize;
    while i < count {
        v = double_mod_p(v);
        i += 1;
    }
    v
}

/// Domain-entry constant `2^264 mod p`:
/// `mont52(x * 2^256, 2^264) = x * 2^260`, moving the 4x64 Montgomery domain
/// into the radix-52 one.
const C_IN_52: [u64; 5] = to_radix52(pow2_shift_mod_p(MONT_ONE, 8));
/// Domain-exit constant `2^256 mod p`:
/// `mont52(x * 2^260, 2^256) = x * 2^256`.
const C_OUT_52: [u64; 5] = to_radix52(MONT_ONE);

/// Eight `Fp` values in radix-52 Montgomery form (`v * 2^260 mod p`),
/// structure-of-arrays: `l[j]` holds limb `j` of all eight lanes. Invariant:
/// every lane is canonical (normalized 52-bit limbs, value below `p`).
/// Unsafe constructors establish the ISA precondition. All other constructors
/// derive a value from an existing `FpVec8`.
#[derive(Clone, Copy)]
pub(crate) struct FpVec8 {
    l: [__m512i; 5],
}

// Loose values cannot enter a Montgomery product before reduction.
#[derive(Clone, Copy)]
struct FpVec8Loose2 {
    l: [__m512i; 5],
}

#[derive(Clone, Copy)]
struct FpVec8Loose3 {
    l: [__m512i; 5],
}

// Loads use unaligned intrinsics. Unsafe constructors require the IFMA ISA.

#[inline(always)]
fn splat(v: u64) -> __m512i {
    unsafe { _mm512_set1_epi64(v as i64) }
}

#[inline(always)]
fn splat5(v: &[u64; 5]) -> [__m512i; 5] {
    [
        splat(v[0]),
        splat(v[1]),
        splat(v[2]),
        splat(v[3]),
        splat(v[4]),
    ]
}

/// Repack eight contiguous 4x64 values into five radix-52 limb vectors with
/// in-register permutes and shifts: the vector form of [`to_radix52`].
/// Value-preserving for any input bits.
#[inline(always)]
unsafe fn repack52_8(values: &[Fp; 8]) -> [__m512i; 5] {
    unsafe {
        let base = values.as_ptr() as *const u8;
        let m0 = _mm512_loadu_si512(base as *const _);
        let m1 = _mm512_loadu_si512(base.add(64) as *const _);
        let m2 = _mm512_loadu_si512(base.add(128) as *const _);
        let m3 = _mm512_loadu_si512(base.add(192) as *const _);
        // Words j of lanes 0..=3 land in slots 0..=3, then merge with the
        // high four lanes.
        let merge = _mm512_loadu_si512([0u64, 1, 2, 3, 8, 9, 10, 11].as_ptr().cast());
        let mut w = [_mm512_setzero_si512(); 4];
        for (j, word) in w.iter_mut().enumerate() {
            let j = j as u64;
            let idx = _mm512_loadu_si512([j, j + 4, j + 8, j + 12, 0, 0, 0, 0].as_ptr().cast());
            let low = _mm512_permutex2var_epi64(m0, idx, m1);
            let high = _mm512_permutex2var_epi64(m2, idx, m3);
            *word = _mm512_permutex2var_epi64(low, merge, high);
        }
        let mask = splat(MASK52);
        [
            _mm512_and_si512(w[0], mask),
            _mm512_and_si512(
                _mm512_or_si512(_mm512_srli_epi64::<52>(w[0]), _mm512_slli_epi64::<12>(w[1])),
                mask,
            ),
            _mm512_and_si512(
                _mm512_or_si512(_mm512_srli_epi64::<40>(w[1]), _mm512_slli_epi64::<24>(w[2])),
                mask,
            ),
            _mm512_and_si512(
                _mm512_or_si512(_mm512_srli_epi64::<28>(w[2]), _mm512_slli_epi64::<36>(w[3])),
                mask,
            ),
            _mm512_srli_epi64::<16>(w[3]),
        ]
    }
}

/// 8-way radix-52 CIOS Montgomery multiplication.
///
/// Contract: lanes of `a` and `b` are canonical radix-52 residues below `p`.
/// Returns canonical residues of `a * b * 2^-260 mod p` per lane.
///
/// Accumulator bound: each of the six 64-bit accumulators absorbs at most
/// four sub-`2^52` products per round plus one propagated carry, so after
/// five rounds every accumulator stays below `5 * 4 * 2^52 < 2^57` -- far
/// from wrapping. The pre-reduction result is below `2p` (since
/// `p^2/2^260` is negligible next to `p`), so one masked subtraction
/// canonicalizes.
#[inline(always)]
fn mont_mul_8(a: &[__m512i; 5], b: &[__m512i; 5]) -> [__m512i; 5] {
    unsafe {
        let zero = _mm512_setzero_si512();
        let pinv = splat(P_INV_52);
        let p = splat5(&P52);
        let mut t = [zero; 6];
        for &bi in b {
            for j in 0..5 {
                t[j] = _mm512_madd52lo_epu64(t[j], a[j], bi);
                t[j + 1] = _mm512_madd52hi_epu64(t[j + 1], a[j], bi);
            }
            // q only depends on the low 52 bits of t[0]. Vpmadd52luq ignores
            // the accumulated high garbage in both multiplicands.
            let q = _mm512_madd52lo_epu64(zero, t[0], pinv);
            for j in 0..5 {
                t[j] = _mm512_madd52lo_epu64(t[j], p[j], q);
                t[j + 1] = _mm512_madd52hi_epu64(t[j + 1], p[j], q);
            }
            // t[0] is now 0 mod 2^52. Shift the window down one limb.
            let carry = _mm512_srli_epi64::<52>(t[0]);
            t[1] = _mm512_add_epi64(t[1], carry);
            for j in 0..5 {
                t[j] = t[j + 1];
            }
            t[5] = zero;
        }
        let mut out = [t[0], t[1], t[2], t[3], t[4]];
        normalize_and_reduce(&mut out);
        out
    }
}

#[inline(always)]
fn mont_mul_8_three(a: [&[__m512i; 5]; 3], b: [&[__m512i; 5]; 3]) -> [[__m512i; 5]; 3] {
    unsafe {
        let zero = _mm512_setzero_si512();
        let pinv = splat(P_INV_52);
        let p = splat5(&P52);
        let mut t = [[zero; 6]; 3];
        let b_rounds: [[__m512i; 3]; 5] = core::array::from_fn(|k| [b[0][k], b[1][k], b[2][k]]);
        for bk in b_rounds {
            for j in 0..5 {
                for stream in 0..3 {
                    t[stream][j] = _mm512_madd52lo_epu64(t[stream][j], a[stream][j], bk[stream]);
                    t[stream][j + 1] =
                        _mm512_madd52hi_epu64(t[stream][j + 1], a[stream][j], bk[stream]);
                }
            }
            let q: [__m512i; 3] =
                core::array::from_fn(|stream| _mm512_madd52lo_epu64(zero, t[stream][0], pinv));
            for j in 0..5 {
                for stream in 0..3 {
                    t[stream][j] = _mm512_madd52lo_epu64(t[stream][j], p[j], q[stream]);
                    t[stream][j + 1] = _mm512_madd52hi_epu64(t[stream][j + 1], p[j], q[stream]);
                }
            }
            for ts in &mut t {
                let carry = _mm512_srli_epi64::<52>(ts[0]);
                ts[1] = _mm512_add_epi64(ts[1], carry);
                for j in 0..5 {
                    ts[j] = ts[j + 1];
                }
                ts[5] = zero;
            }
        }
        core::array::from_fn(|stream| {
            let mut out = [
                t[stream][0],
                t[stream][1],
                t[stream][2],
                t[stream][3],
                t[stream][4],
            ];
            normalize_and_reduce(&mut out);
            out
        })
    }
}

/// Carry-normalize five limbs to 52 bits. Requires the value below `2^260`,
/// so the top limb never carries out.
#[inline(always)]
unsafe fn carry_normalize(t: &mut [__m512i; 5]) {
    unsafe {
        let mask = splat(MASK52);
        let mut carry = _mm512_setzero_si512();
        for limb in &mut t[..4] {
            let v = _mm512_add_epi64(*limb, carry);
            *limb = _mm512_and_si512(v, mask);
            carry = _mm512_srli_epi64::<52>(v);
        }
        t[4] = _mm512_add_epi64(t[4], carry);
    }
}

/// One conditional subtraction of `p` on normalized limbs: lanes at or above
/// `p` drop by `p`, so a value below `(k+1)p` needs `k` rounds to become
/// canonical.
#[inline(always)]
unsafe fn cond_sub_p(t: &mut [__m512i; 5]) {
    unsafe {
        let zero = _mm512_setzero_si512();
        let mask = splat(MASK52);
        let p = splat5(&P52);
        // d = t - p with a borrow chain over 52-bit limbs.
        let mut d = [zero; 5];
        let mut borrow = zero;
        for j in 0..5 {
            let v = _mm512_sub_epi64(_mm512_sub_epi64(t[j], p[j]), borrow);
            d[j] = _mm512_and_si512(v, mask);
            borrow = _mm512_srli_epi64::<63>(v);
        }
        // Lanes that borrow (t < p) keep t. 0 - borrow builds the keep mask
        // arithmetically, avoiding the k-register compare round trip.
        let keep = _mm512_sub_epi64(zero, borrow);
        for j in 0..5 {
            t[j] = _mm512_ternarylogic_epi64::<0xE4>(t[j], d[j], keep);
        }
    }
}

/// Carry-normalize, then subtract `p` in lanes where the value is at least
/// `p`. Requires the pre-normalization value to be below `2p`.
#[inline(always)]
unsafe fn normalize_and_reduce(t: &mut [__m512i; 5]) {
    unsafe {
        carry_normalize(t);
        cond_sub_p(t);
    }
}

/// Access to the five radix-52 limb vectors of a batched value. Sealed to
/// this module: only forms whose limbs are normalized below `2^52` implement
/// it, which the `vpmadd52` truncation semantics require.
trait Radix52Limbs: Copy {
    fn limbs(&self) -> &[__m512i; 5];
}

impl Radix52Limbs for FpVec8 {
    #[inline(always)]
    fn limbs(&self) -> &[__m512i; 5] {
        &self.l
    }
}

#[cfg(feature = "std")]
impl<const B: u64> Radix52Limbs for FpVec8L<B> {
    #[inline(always)]
    fn limbs(&self) -> &[__m512i; 5] {
        &self.l
    }
}

/// Eight lane radix-52 CIOS Montgomery sum of products.
///
/// Each accumulator stays below `2^59` for the admitted arities. The result
/// is below `(R+1)p` when the operand value bounds satisfy
/// `sum(A_i * B_i) * p <= R * 2^260`, which every public entry asserts. `R`
/// conditional subtractions then canonicalize.
#[inline(always)]
fn mont_sos_mac_8<L: Radix52Limbs, Rv: Radix52Limbs, const N: usize, const R: usize>(
    a: &[L; N],
    b: &[Rv; N],
) -> [__m512i; 5] {
    const {
        assert!(N == 2 || N == 6);
        assert!(2_u64 * 5 * (N as u64 + 1) * (1 << 52) < 1 << 59);
        assert!(R >= 1 && R <= 3);
    }
    unsafe {
        let zero = _mm512_setzero_si512();
        let pinv = splat(P_INV_52);
        let p = splat5(&P52);
        let mut t = [zero; 6];
        for k in 0..5 {
            for term in 0..N {
                let bk = b[term].limbs()[k];
                for j in 0..5 {
                    t[j] = _mm512_madd52lo_epu64(t[j], a[term].limbs()[j], bk);
                    t[j + 1] = _mm512_madd52hi_epu64(t[j + 1], a[term].limbs()[j], bk);
                }
            }
            // One Montgomery reduction step. Q cancels t[0] mod 2^52.
            // vpmadd52luq reads only the low 52 bits of the accumulated t[0],
            // so its high garbage is irrelevant.
            let q = _mm512_madd52lo_epu64(zero, t[0], pinv);
            for j in 0..5 {
                t[j] = _mm512_madd52lo_epu64(t[j], p[j], q);
                t[j + 1] = _mm512_madd52hi_epu64(t[j + 1], p[j], q);
            }
            let carry = _mm512_srli_epi64::<52>(t[0]);
            t[1] = _mm512_add_epi64(t[1], carry);
            let next0 = t[1];
            let next1 = t[2];
            let next2 = t[3];
            let next3 = t[4];
            let next4 = t[5];
            t[0] = next0;
            t[1] = next1;
            t[2] = next2;
            t[3] = next3;
            t[4] = next4;
            t[5] = zero;
        }
        let mut out = [t[0], t[1], t[2], t[3], t[4]];
        carry_normalize(&mut out);
        for _ in 0..R {
            cond_sub_p(&mut out);
        }
        out
    }
}

/// Two interleaved SoS streams of unequal arity and per-stream subtraction
/// rounds. The five-round Montgomery recurrence and the value bound argument
/// are those of [`mont_sos_mac_8`], applied to each stream with its own term
/// count.
#[inline(always)]
fn mont_sos_mac_8_pair_split<
    L: Radix52Limbs,
    Rv: Radix52Limbs,
    const N0: usize,
    const N1: usize,
    const R0: usize,
    const R1: usize,
>(
    a0: &[L; N0],
    b0: &[Rv; N0],
    a1: &[L; N1],
    b1: &[Rv; N1],
) -> [[__m512i; 5]; 2] {
    const {
        assert!(N0 >= N1 && N1 >= 1 && N0 <= 12);
        // No accumulator wraps 2^64: five rounds add at most `N + 1` low and
        // `N + 1` high sub-2^52 products to any accumulator position.
        assert!(2 * 5 * (N0 as u64 + 1) * (1 << 52) < 1 << 62);
        assert!(R0 >= 1 && R0 <= 3 && R1 >= 1 && R1 <= 3);
    }
    unsafe {
        let zero = _mm512_setzero_si512();
        let pinv = splat(P_INV_52);
        let p = splat5(&P52);
        let mut t = [[zero; 6]; 2];
        for k in 0..5 {
            for term in 0..N0 {
                for j in 0..5 {
                    let bk = b0[term].limbs()[k];
                    t[0][j] = _mm512_madd52lo_epu64(t[0][j], a0[term].limbs()[j], bk);
                    t[0][j + 1] = _mm512_madd52hi_epu64(t[0][j + 1], a0[term].limbs()[j], bk);
                    if term < N1 {
                        let bk = b1[term].limbs()[k];
                        t[1][j] = _mm512_madd52lo_epu64(t[1][j], a1[term].limbs()[j], bk);
                        t[1][j + 1] = _mm512_madd52hi_epu64(t[1][j + 1], a1[term].limbs()[j], bk);
                    }
                }
            }
            let q: [__m512i; 2] =
                core::array::from_fn(|stream| _mm512_madd52lo_epu64(zero, t[stream][0], pinv));
            for j in 0..5 {
                for stream in 0..2 {
                    t[stream][j] = _mm512_madd52lo_epu64(t[stream][j], p[j], q[stream]);
                    t[stream][j + 1] = _mm512_madd52hi_epu64(t[stream][j + 1], p[j], q[stream]);
                }
            }
            for ts in &mut t {
                let carry = _mm512_srli_epi64::<52>(ts[0]);
                ts[1] = _mm512_add_epi64(ts[1], carry);
                for j in 0..5 {
                    ts[j] = ts[j + 1];
                }
                ts[5] = zero;
            }
        }
        let mut out0 = [t[0][0], t[0][1], t[0][2], t[0][3], t[0][4]];
        carry_normalize(&mut out0);
        for _ in 0..R0 {
            cond_sub_p(&mut out0);
        }
        let mut out1 = [t[1][0], t[1][1], t[1][2], t[1][3], t[1][4]];
        carry_normalize(&mut out1);
        for _ in 0..R1 {
            cond_sub_p(&mut out1);
        }
        [out0, out1]
    }
}

impl FpVec8 {
    #[cfg(feature = "std")]
    #[inline(always)]
    pub(crate) fn copy_from(&mut self, value: &Self) {
        self.l[0] = value.l[0];
        self.l[1] = value.l[1];
        self.l[2] = value.l[2];
        self.l[3] = value.l[3];
        self.l[4] = value.l[4];
    }

    /// Odd canonical lanes add p before the cross-limb shift. Since x + p is
    /// below 2p and 2p is below 2^255, limb four cannot carry out.
    #[inline(always)]
    pub(crate) fn half(&self) -> Self {
        unsafe {
            let zero = _mm512_setzero_si512();
            let one = splat(1);
            let radix_mask = splat(MASK52);
            let p = splat5(&P52);
            let odd = _mm512_cmpneq_epi64_mask(_mm512_and_si512(self.l[0], one), zero);
            let mut limbs = [zero; 5];
            let mut carry = zero;

            for i in 0..5 {
                let addend = _mm512_mask_blend_epi64(odd, zero, p[i]);
                let sum = _mm512_add_epi64(_mm512_add_epi64(self.l[i], addend), carry);
                limbs[i] = _mm512_and_si512(sum, radix_mask);
                carry = _mm512_srli_epi64::<52>(sum);
            }

            let mut out = [zero; 5];
            for i in 0..4 {
                let high_bit = _mm512_slli_epi64::<51>(_mm512_and_si512(limbs[i + 1], one));
                out[i] = _mm512_or_si512(_mm512_srli_epi64::<1>(limbs[i]), high_bit);
            }
            out[4] = _mm512_srli_epi64::<1>(limbs[4]);
            Self { l: out }
        }
    }

    /// Convert eight canonical `Fp` (4x64 Montgomery) into the batched
    /// radix-52 domain: repack, then one batched multiplication by
    /// `2^264 mod p`.
    ///
    /// # Safety
    ///
    /// The current CPU and OS must support AVX-512F and AVX-512IFMA.
    #[inline(always)]
    pub(crate) unsafe fn load(values: &[Fp; 8]) -> Self {
        let mut out = core::mem::MaybeUninit::uninit();
        unsafe { Self::load_into(out.as_mut_ptr(), values) };
        unsafe { out.assume_init() }
    }

    #[inline(always)]
    pub(crate) unsafe fn load_into(out: *mut Self, values: &[Fp; 8]) {
        unsafe { Self::load_into_ifma(out, values) }
    }

    #[target_feature(enable = "avx512f,avx512ifma")]
    #[inline]
    unsafe fn load_into_ifma(out: *mut Self, values: &[Fp; 8]) {
        unsafe {
            let raw = repack52_8(values);
            out.write(Self {
                l: mont_mul_8(&raw, &splat5(&C_IN_52)),
            });
        }
    }

    /// Build a per-pairing scale vector for [`Self::load_scaled_into`]. Lane
    /// pairs 0/1 and 2/3 hold `c0 * 2^264 mod p` and `c3 * 2^264 mod p`, the
    /// high lanes hold `2^264 mod p`, so the domain-entry multiplication also
    /// applies the scaling exactly. Outlined: one call per registered anchor.
    ///
    /// # Safety
    ///
    /// The current CPU and OS must support AVX-512F and AVX-512IFMA.
    #[cfg(feature = "std")]
    #[target_feature(enable = "avx512f,avx512ifma")]
    #[inline(never)]
    pub(crate) unsafe fn line_scale_pair(c0: Fp, c3: Fp) -> Self {
        let s0 = to_radix52(pow2_shift_mod_p(c0.0, 8));
        let s3 = to_radix52(pow2_shift_mod_p(c3.0, 8));
        let mut lanes = [[0u64; 8]; 5];
        for j in 0..5 {
            lanes[j] = [
                s0[j], s0[j], s3[j], s3[j], C_IN_52[j], C_IN_52[j], C_IN_52[j], C_IN_52[j],
            ];
        }
        // The shifted values stay below `p`, so the canonical limb invariant
        // holds without a reduction.
        unsafe {
            Self {
                l: [
                    _mm512_loadu_si512(lanes[0].as_ptr() as *const _),
                    _mm512_loadu_si512(lanes[1].as_ptr() as *const _),
                    _mm512_loadu_si512(lanes[2].as_ptr() as *const _),
                    _mm512_loadu_si512(lanes[3].as_ptr() as *const _),
                    _mm512_loadu_si512(lanes[4].as_ptr() as *const _),
                ],
            }
        }
    }

    /// Convert eight canonical `Fp` into the batched radix-52 domain while
    /// multiplying lane `i` by the scale embedded in `scale` (built by
    /// [`Self::line_scale8`]): lane `i` becomes `v_i * s_i * 2^260`.
    ///
    /// # Safety
    ///
    /// The current CPU and OS must support AVX-512F and AVX-512IFMA.
    #[cfg(feature = "std")]
    #[inline(always)]
    pub(crate) unsafe fn load_scaled_into(out: *mut Self, values: &[Fp; 8], scale: &Self) {
        unsafe { Self::load_scaled_into_ifma(out, values, scale) }
    }

    #[cfg(feature = "std")]
    #[target_feature(enable = "avx512f,avx512ifma")]
    #[inline]
    unsafe fn load_scaled_into_ifma(out: *mut Self, values: &[Fp; 8], scale: &Self) {
        unsafe {
            let raw = repack52_8(values);
            out.write(Self {
                l: mont_mul_8(&raw, &scale.l),
            });
        }
    }

    /// Convert back to eight canonical `Fp`: one batched multiplication by
    /// `2^256 mod p`, then repack.
    #[inline(always)]
    pub(crate) fn store(&self) -> [Fp; 8] {
        // SAFETY: `self` can exist only after an unsafe IFMA constructor.
        unsafe { self.store_ifma() }
    }

    #[target_feature(enable = "avx512f,avx512ifma")]
    #[inline]
    unsafe fn store_ifma(&self) -> [Fp; 8] {
        let out = mont_mul_8(&self.l, &splat5(&C_OUT_52));
        let mut lanes = [[0u64; 8]; 5];
        unsafe {
            for j in 0..5 {
                _mm512_storeu_si512(lanes[j].as_mut_ptr() as *mut _, out[j]);
            }
        }
        core::array::from_fn(|lane| {
            Fp(from_radix52(&[
                lanes[0][lane],
                lanes[1][lane],
                lanes[2][lane],
                lanes[3][lane],
                lanes[4][lane],
            ]))
        })
    }

    /// Select `self` in lanes whose bit is set and `fallback` in every other
    /// lane. Both inputs remain in the radix-52 Montgomery domain.
    #[inline(always)]
    pub(crate) fn select_lanes(&self, active_mask: u8, fallback: &Self) -> Self {
        unsafe {
            Self {
                l: core::array::from_fn(|i| {
                    _mm512_mask_blend_epi64(active_mask, fallback.l[i], self.l[i])
                }),
            }
        }
    }

    #[inline(always)]
    pub(crate) fn mul(&self, rhs: &Self) -> Self {
        Self {
            l: mont_mul_8(&self.l, &rhs.l),
        }
    }

    /// Dedicated squaring entry point. Currently the general product (a
    /// 15-vs-25 partial-product schedule is a follow-up if the tier lands).
    #[cfg(helius_avx512_ifma)]
    #[inline(always)]
    pub(crate) fn square(&self) -> Self {
        self.mul(self)
    }

    #[inline]
    pub(crate) fn sos_mac_2(a: &[FpVec8; 2], b: &[FpVec8; 2]) -> Self {
        // Each input is created only by an unsafe IFMA constructor.
        unsafe { Self::sos_mac_2_ifma(a, b) }
    }

    #[target_feature(enable = "avx512f,avx512ifma")]
    #[inline(never)]
    unsafe fn sos_mac_2_ifma(a: &[FpVec8; 2], b: &[FpVec8; 2]) -> Self {
        Self {
            l: mont_sos_mac_8::<_, _, 2, 1>(a, b),
        }
    }

    #[inline]
    pub(crate) fn sos_mac_6(a: &[FpVec8; 6], b: &[FpVec8; 6]) -> Self {
        // SAFETY: each input can exist only after an unsafe IFMA constructor.
        unsafe { Self::sos_mac_6_ifma(a, b) }
    }

    #[target_feature(enable = "avx512f,avx512ifma")]
    #[inline(never)]
    unsafe fn sos_mac_6_ifma(a: &[FpVec8; 6], b: &[FpVec8; 6]) -> Self {
        Self {
            l: mont_sos_mac_8::<_, _, 6, 1>(a, b),
        }
    }

    /// Fused unequal-arity pair over bound-tracked lazy rows: six-term rows
    /// on stream 0, three-term half rows on stream 1. Each stream is
    /// canonical because its own `sum * p <= R * 2^260` budget holds and `R`
    /// subtraction rounds follow.
    ///
    /// # Safety
    ///
    /// The caller must already run with AVX-512F and AVX-512IFMA enabled.
    #[cfg(feature = "std")]
    #[inline(always)]
    pub(crate) unsafe fn sos_mac_6_3_pair_fused_lazy<
        const LA: u64,
        const RB: u64,
        const R0: usize,
        const R1: usize,
    >(
        a0: &[FpVec8L<LA>; 6],
        b0: &[FpVec8L<RB>; 6],
        a1: &[FpVec8L<LA>; 3],
        b1: &[FpVec8L<RB>; 3],
    ) -> [Self; 2] {
        const { assert!(sos_sum_ok(6 * LA * RB, R0 as u64)) };
        const { assert!(sos_sum_ok(3 * LA * RB, R1 as u64)) };
        mont_sos_mac_8_pair_split::<_, _, 6, 3, R0, R1>(a0, b0, a1, b1).map(|l| Self { l })
    }

    /// Fused unequal-arity pair for one dense Fp12 product: twelve-term full
    /// rows on stream 0, six-term half rows on stream 1. Bound contract as
    /// [`Self::sos_mac_6_3_pair_fused_lazy`].
    ///
    /// # Safety
    ///
    /// The caller must already run with AVX-512F and AVX-512IFMA enabled.
    #[cfg(feature = "std")]
    #[inline(always)]
    pub(crate) unsafe fn sos_mac_12_6_pair_fused_lazy<
        const LA: u64,
        const RB: u64,
        const R0: usize,
        const R1: usize,
    >(
        a0: &[FpVec8L<LA>; 12],
        b0: &[FpVec8L<RB>; 12],
        a1: &[FpVec8L<LA>; 6],
        b1: &[FpVec8L<RB>; 6],
    ) -> [Self; 2] {
        const { assert!(sos_sum_ok(12 * LA * RB, R0 as u64)) };
        const { assert!(sos_sum_ok(6 * LA * RB, R1 as u64)) };
        mont_sos_mac_8_pair_split::<_, _, 12, 6, R0, R1>(a0, b0, a1, b1).map(|l| Self { l })
    }

    /// Two-term lazy sum of products. `2 * LA * RB * p <= 2^260` keeps the
    /// pre-reduction value below `2p`, so the result is canonical.
    #[cfg(feature = "std")]
    #[inline]
    pub(crate) fn sos_mac_2_lazy<const LA: u64, const RB: u64>(
        a: &[FpVec8L<LA>; 2],
        b: &[FpVec8L<RB>; 2],
    ) -> Self {
        // SAFETY: each input can exist only after an unsafe IFMA constructor.
        unsafe { Self::sos_mac_2_lazy_ifma(a, b) }
    }

    #[cfg(feature = "std")]
    #[target_feature(enable = "avx512f,avx512ifma")]
    #[inline(never)]
    unsafe fn sos_mac_2_lazy_ifma<const LA: u64, const RB: u64>(
        a: &[FpVec8L<LA>; 2],
        b: &[FpVec8L<RB>; 2],
    ) -> Self {
        const { assert!(sos_sum_ok(2 * LA * RB, 1)) };
        Self {
            l: mont_sos_mac_8::<_, _, 2, 1>(a, b),
        }
    }

    /// Three interleaved lazy products. `AB * BB * p <= 2^260` keeps every
    /// stream's pre-reduction value below `2p`, so the results are canonical.
    #[cfg(feature = "std")]
    #[inline]
    pub(crate) fn mul_three_lazy<const AB: u64, const BB: u64>(
        a0: &FpVec8L<AB>,
        b0: &FpVec8L<BB>,
        a1: &FpVec8L<AB>,
        b1: &FpVec8L<BB>,
        a2: &FpVec8L<AB>,
        b2: &FpVec8L<BB>,
    ) -> [Self; 3] {
        // SAFETY: each input can exist only after an unsafe IFMA constructor.
        unsafe { Self::mul_three_lazy_ifma(a0, b0, a1, b1, a2, b2) }
    }

    #[cfg(feature = "std")]
    #[target_feature(enable = "avx512f,avx512ifma")]
    #[inline(never)]
    unsafe fn mul_three_lazy_ifma<const AB: u64, const BB: u64>(
        a0: &FpVec8L<AB>,
        b0: &FpVec8L<BB>,
        a1: &FpVec8L<AB>,
        b1: &FpVec8L<BB>,
        a2: &FpVec8L<AB>,
        b2: &FpVec8L<BB>,
    ) -> [Self; 3] {
        const { assert!(sos_sum_ok(AB * BB, 1)) };
        mont_mul_8_three([&a0.l, &a1.l, &a2.l], [&b0.l, &b1.l, &b2.l]).map(|l| Self { l })
    }

    // The mixed-addition formula is all-subtraction. Add is kept as part of
    // the differential-gated op set for future batched formulas.
    #[inline(always)]
    pub(crate) fn add(&self, rhs: &Self) -> Self {
        unsafe {
            let mut t = [
                _mm512_add_epi64(self.l[0], rhs.l[0]),
                _mm512_add_epi64(self.l[1], rhs.l[1]),
                _mm512_add_epi64(self.l[2], rhs.l[2]),
                _mm512_add_epi64(self.l[3], rhs.l[3]),
                _mm512_add_epi64(self.l[4], rhs.l[4]),
            ];
            // Sum < 2p. The shared normalize/reduce path canonicalizes.
            normalize_and_reduce(&mut t);
            Self { l: t }
        }
    }

    #[inline(always)]
    pub(crate) fn sub(&self, rhs: &Self) -> Self {
        unsafe {
            let zero = _mm512_setzero_si512();
            let mask = splat(MASK52);
            let p = splat5(&P52);
            // d = a - b over 52-bit limbs. Lanes that borrow add p back.
            let mut d = [zero; 5];
            let mut borrow = zero;
            for (j, limb) in d.iter_mut().enumerate() {
                let v = _mm512_sub_epi64(_mm512_sub_epi64(self.l[j], rhs.l[j]), borrow);
                *limb = _mm512_and_si512(v, mask);
                borrow = _mm512_srli_epi64::<63>(v);
            }
            // 0 - borrow is all-ones exactly in lanes that owe an add-back,
            // masking p without a k-register round trip.
            let negative = _mm512_sub_epi64(zero, borrow);
            let mut carry = zero;
            let mut out = [zero; 5];
            for j in 0..5 {
                let addend = _mm512_and_si512(p[j], negative);
                let v = _mm512_add_epi64(_mm512_add_epi64(d[j], addend), carry);
                out[j] = _mm512_and_si512(v, mask);
                carry = _mm512_srli_epi64::<52>(v);
            }
            // For a < b, radix subtraction represents 2^260 + a - b. Adding
            // p intentionally drops the carry past limb 4, leaving a-b+p.
            Self { l: out }
        }
    }

    /// The all-zero vector (value 0 in every lane. Canonical radix-52).
    ///
    /// # Safety
    ///
    /// The current CPU and OS must support AVX-512F and AVX-512IFMA.
    #[inline(always)]
    pub(crate) unsafe fn zero() -> Self {
        Self {
            l: [unsafe { _mm512_setzero_si512() }; 5],
        }
    }

    /// Enter the bound-tracked lazy domain. Canonical values are below `p`.
    #[cfg(feature = "std")]
    #[inline(always)]
    pub(crate) fn lazy(&self) -> FpVec8L<1> {
        FpVec8L { l: self.l }
    }

    /// Per-lane additive inverse `p - self` (`0` maps to `0`).
    #[inline(always)]
    pub(crate) fn negate(&self) -> Self {
        unsafe { Self::zero() }.sub(self)
    }

    /// Per-lane `2 * self`.
    #[inline(always)]
    pub(crate) fn double(&self) -> Self {
        self.add(self)
    }

    #[inline(always)]
    pub(crate) fn triple(&self) -> Self {
        self.add_loose(self).add(self).reduce()
    }

    /// Per-lane zero test (canonical representation makes this exact).
    #[inline(always)]
    #[cfg(helius_avx512_ifma)]
    pub(crate) fn is_zero_mask(&self) -> u8 {
        unsafe {
            let all = _mm512_or_si512(
                _mm512_or_si512(self.l[0], self.l[1]),
                _mm512_or_si512(_mm512_or_si512(self.l[2], self.l[3]), self.l[4]),
            );
            _mm512_cmpeq_epi64_mask(all, _mm512_setzero_si512())
        }
    }

    /// Per-lane selection without leaving the resident radix-52 domain.
    #[inline(always)]
    #[cfg(any(feature = "std", helius_avx512_ifma))]
    pub(crate) fn select(mask: u8, if_true: &Self, if_false: &Self) -> Self {
        unsafe {
            Self {
                l: core::array::from_fn(|limb| {
                    _mm512_mask_blend_epi64(mask, if_false.l[limb], if_true.l[limb])
                }),
            }
        }
    }

    /// Select each lane from either `self` (indices 0..=7) or `other`
    /// (indices 8..=15). The same coefficient map is applied to every
    /// radix-52 limb, so values never leave the resident Montgomery domain.
    #[inline(always)]
    #[cfg(feature = "std")]
    pub(crate) fn permute2(&self, other: &Self, indices: [u64; 8]) -> Self {
        unsafe {
            let index = _mm512_loadu_si512(indices.as_ptr().cast());
            Self {
                l: core::array::from_fn(|limb| {
                    _mm512_permutex2var_epi64(self.l[limb], index, other.l[limb])
                }),
            }
        }
    }

    /// Negate only the selected lanes, retaining the other lanes unchanged.
    #[inline(always)]
    #[cfg(feature = "std")]
    pub(crate) fn negate_lanes(&self, mask: u8) -> Self {
        Self::select(mask, &self.negate(), self)
    }

    /// Zero every lane whose mask bit is clear (zero is canonical).
    #[inline(always)]
    #[cfg(feature = "std")]
    pub(crate) fn keep_lanes(self, mask: u8) -> Self {
        unsafe {
            Self {
                l: self.l.map(|limb| _mm512_maskz_mov_epi64(mask, limb)),
            }
        }
    }

    #[inline(always)]
    fn add_loose(&self, rhs: &Self) -> FpVec8Loose2 {
        unsafe {
            FpVec8Loose2 {
                l: core::array::from_fn(|limb| _mm512_add_epi64(self.l[limb], rhs.l[limb])),
            }
        }
    }
}

impl FpVec8Loose2 {
    #[inline(always)]
    fn add(&self, rhs: &FpVec8) -> FpVec8Loose3 {
        unsafe {
            FpVec8Loose3 {
                l: core::array::from_fn(|limb| _mm512_add_epi64(self.l[limb], rhs.l[limb])),
            }
        }
    }
}

impl FpVec8Loose3 {
    #[inline(always)]
    fn reduce(&self) -> FpVec8 {
        unsafe {
            let mut t = self.l;
            // Value < 3p, so two subtraction rounds canonicalize.
            carry_normalize(&mut t);
            cond_sub_p(&mut t);
            cond_sub_p(&mut t);
            FpVec8 { l: t }
        }
    }
}

/// Eight lanes in the radix-52 Montgomery domain with normalized limbs and
/// a type-tracked value bound: every lane is below `B * p`. A wrong bound is
/// silent arithmetic corruption, so every producer proves its output bound
/// with a const assert and [`Self::retag`] is the only unproven narrowing.
#[cfg(feature = "std")]
#[derive(Clone, Copy)]
pub(crate) struct FpVec8L<const B: u64> {
    l: [__m512i; 5],
}

/// One additive fold: `v = q * 2^254 + r` becomes `r + q * eps` with
/// `eps = 2^254 - p`, congruent mod `p`. Requires normalized limbs and a
/// value below `LAZY_CAP * p`, so `q < 2^6` and no product carries past the
/// top limb.
#[cfg(feature = "std")]
#[inline(always)]
unsafe fn fold_254(t: &mut [__m512i; 5]) {
    unsafe {
        let q = _mm512_srli_epi64::<46>(t[4]);
        t[4] = _mm512_and_si512(t[4], splat((1 << 46) - 1));
        let eps = splat5(&EPS52);
        for j in 0..5 {
            t[j] = _mm512_madd52lo_epu64(t[j], eps[j], q);
        }
        for j in 0..4 {
            t[j + 1] = _mm512_madd52hi_epu64(t[j + 1], eps[j], q);
        }
        carry_normalize(t);
    }
}

#[cfg(feature = "std")]
impl<const B: u64> FpVec8L<B> {
    /// Lift to a wider bound without any work.
    #[inline(always)]
    pub(crate) fn relax<const C: u64>(self) -> FpVec8L<C> {
        const { assert!(C >= B && C <= LAZY_CAP) };
        FpVec8L { l: self.l }
    }

    /// Zero every lane whose mask bit is clear.
    #[inline(always)]
    pub(crate) fn keep_lanes(self, mask: u8) -> Self {
        unsafe {
            Self {
                l: self.l.map(|limb| _mm512_maskz_mov_epi64(mask, limb)),
            }
        }
    }

    /// Narrow the tracked bound on the caller's per-lane proof.
    ///
    /// # Safety
    ///
    /// Every lane's value must be below `C * p`. An overstated narrowing is
    /// silent arithmetic corruption downstream.
    #[inline(always)]
    pub(crate) unsafe fn retag<const C: u64>(self) -> FpVec8L<C> {
        const { assert!(C >= 1 && C <= LAZY_CAP) };
        #[cfg(debug_assertions)]
        assert!(self.lanes_below(C));
        FpVec8L { l: self.l }
    }

    #[cfg(debug_assertions)]
    fn lanes_below(&self, bound: u64) -> bool {
        let mut lanes = [[0u64; 8]; 5];
        unsafe {
            for (row, limb) in lanes.iter_mut().zip(self.l) {
                _mm512_storeu_si512(row.as_mut_ptr() as *mut _, limb);
            }
        }
        let (bp, carry) = mul_p52(bound);
        debug_assert!(carry == 0);
        (0..8).all(|lane| {
            let v = [
                lanes[0][lane],
                lanes[1][lane],
                lanes[2][lane],
                lanes[3][lane],
                lanes[4][lane],
            ];
            v.iter().all(|&limb| limb <= MASK52) && lt52(&v, &bp)
        })
    }

    #[inline(always)]
    pub(crate) fn add_lazy<const C: u64, const OUT: u64>(&self, rhs: &FpVec8L<C>) -> FpVec8L<OUT> {
        const { assert!(OUT >= B + C && OUT <= LAZY_CAP) };
        unsafe {
            let mut t = core::array::from_fn(|j| _mm512_add_epi64(self.l[j], rhs.l[j]));
            carry_normalize(&mut t);
            FpVec8L { l: t }
        }
    }

    #[inline(always)]
    pub(crate) fn double_lazy<const OUT: u64>(&self) -> FpVec8L<OUT> {
        const { assert!(OUT >= 2 * B && OUT <= LAZY_CAP) };
        unsafe {
            let mut t = self.l.map(|limb| _mm512_slli_epi64::<1>(limb));
            carry_normalize(&mut t);
            FpVec8L { l: t }
        }
    }

    /// `self - rhs + K*p` for the smallest borrow-free `K`.
    #[inline(always)]
    pub(crate) fn sub_lazy<const C: u64, const OUT: u64>(&self, rhs: &FpVec8L<C>) -> FpVec8L<OUT> {
        const { assert!(OUT >= B + min_bias(1, C) && OUT <= LAZY_CAP) };
        let bias = const { bias52(min_bias(1, C), 1) };
        unsafe {
            let bias = splat5(&bias);
            let mut t = core::array::from_fn(|j| {
                _mm512_sub_epi64(_mm512_add_epi64(self.l[j], bias[j]), rhs.l[j])
            });
            carry_normalize(&mut t);
            FpVec8L { l: t }
        }
    }

    /// `self - y - z + K*p` for the smallest borrow-free `K`.
    #[inline(always)]
    pub(crate) fn sub2_lazy<const CY: u64, const CZ: u64, const OUT: u64>(
        &self,
        y: &FpVec8L<CY>,
        z: &FpVec8L<CZ>,
    ) -> FpVec8L<OUT> {
        const { assert!(OUT >= B + min_bias(2, CY + CZ) && OUT <= LAZY_CAP) };
        let bias = const { bias52(min_bias(2, CY + CZ), 2) };
        unsafe {
            let bias = splat5(&bias);
            let mut t = core::array::from_fn(|j| {
                let v = _mm512_sub_epi64(_mm512_add_epi64(self.l[j], bias[j]), y.l[j]);
                _mm512_sub_epi64(v, z.l[j])
            });
            carry_normalize(&mut t);
            FpVec8L { l: t }
        }
    }

    /// Multiply adjacent (real, imag) lane pairs by `xi = 9 + u`:
    /// even lanes become `9r - i + K*p`, odd lanes `9i + r`.
    #[inline(always)]
    pub(crate) fn xi_lazy<const OUT: u64>(&self) -> FpVec8L<OUT> {
        const { assert!(OUT >= 9 * B + min_bias(1, B) && OUT <= LAZY_CAP) };
        let bias = const { bias52(min_bias(1, B), 1) };
        unsafe {
            let index = _mm512_loadu_si512([1u64, 0, 3, 2, 5, 4, 7, 6].as_ptr().cast());
            let bias = splat5(&bias);
            let mut t = core::array::from_fn(|j| {
                let swapped = _mm512_permutexvar_epi64(index, self.l[j]);
                let nine = _mm512_add_epi64(_mm512_slli_epi64::<3>(self.l[j]), self.l[j]);
                let real = _mm512_sub_epi64(_mm512_add_epi64(nine, bias[j]), swapped);
                let imag = _mm512_add_epi64(nine, swapped);
                _mm512_mask_blend_epi64(0x55, imag, real)
            });
            carry_normalize(&mut t);
            FpVec8L { l: t }
        }
    }

    /// `K*p - self` on masked lanes, `self` elsewhere.
    #[inline(always)]
    pub(crate) fn negate_lanes_lazy<const OUT: u64>(&self, mask: u8) -> FpVec8L<OUT> {
        const { assert!(OUT > min_bias(1, B) && OUT <= LAZY_CAP) };
        let bias = const { bias52(min_bias(1, B), 1) };
        unsafe {
            let bias = splat5(&bias);
            let mut t = core::array::from_fn(|j| {
                let negated = _mm512_sub_epi64(bias[j], self.l[j]);
                _mm512_mask_blend_epi64(mask, self.l[j], negated)
            });
            carry_normalize(&mut t);
            FpVec8L { l: t }
        }
    }

    /// `3*self + 2*rhs` on lanes in `add_mask`, `3*self - 2*rhs + K*p`
    /// elsewhere.
    #[inline(always)]
    pub(crate) fn triple_pm_double<const C: u64, const OUT: u64>(
        &self,
        rhs: &FpVec8L<C>,
        add_mask: u8,
    ) -> FpVec8L<OUT> {
        const { assert!(OUT >= 3 * B + max_u64(2 * C, min_bias(2, 2 * C)) && OUT <= LAZY_CAP) };
        let bias = const { bias52(min_bias(2, 2 * C), 2) };
        unsafe {
            let bias = splat5(&bias);
            let mut t = core::array::from_fn(|j| {
                let three = _mm512_add_epi64(_mm512_slli_epi64::<1>(self.l[j]), self.l[j]);
                let two = _mm512_slli_epi64::<1>(rhs.l[j]);
                let added = _mm512_add_epi64(three, two);
                let subtracted = _mm512_sub_epi64(_mm512_add_epi64(three, bias[j]), two);
                _mm512_mask_blend_epi64(add_mask, subtracted, added)
            });
            carry_normalize(&mut t);
            FpVec8L { l: t }
        }
    }

    /// Same coefficient map on every limb, lanes 0..=7 from `self` and
    /// 8..=15 from `other`.
    #[inline(always)]
    pub(crate) fn permute2<const C: u64, const OUT: u64>(
        &self,
        other: &FpVec8L<C>,
        indices: [u64; 8],
    ) -> FpVec8L<OUT> {
        const { assert!(OUT >= B && OUT >= C && OUT <= LAZY_CAP) };
        unsafe {
            let index = _mm512_loadu_si512(indices.as_ptr().cast());
            FpVec8L {
                l: core::array::from_fn(|limb| {
                    _mm512_permutex2var_epi64(self.l[limb], index, other.l[limb])
                }),
            }
        }
    }

    #[inline(always)]
    pub(crate) fn select(mask: u8, if_true: &Self, if_false: &Self) -> Self {
        unsafe {
            Self {
                l: core::array::from_fn(|limb| {
                    _mm512_mask_blend_epi64(mask, if_false.l[limb], if_true.l[limb])
                }),
            }
        }
    }

    /// Canonicalize: fold the bound down to at most 3, then finish with
    /// plain subtraction rounds.
    #[inline(always)]
    pub(crate) fn reduce_full(self) -> FpVec8 {
        const { assert!(B >= 1 && B <= LAZY_CAP) };
        let mut t = self.l;
        unsafe {
            for _ in 0..const { folds_needed(B) } {
                fold_254(&mut t);
            }
            for _ in 0..const { folded_bound(B) - 1 } {
                cond_sub_p(&mut t);
            }
        }
        FpVec8 { l: t }
    }
}

/// Eight independent Jacobian mixed additions (general case of
/// madd-2007-bl, a = 0), batched over the IFMA lanes.
///
/// Inputs are eight buckets `(x1, y1, z1)` and eight affine points
/// `(x2, y2)` in ordinary 4x64 Montgomery form. The kernel is total over
/// canonical residues: identity buckets (`z1 = 0`) and `h == 0` lanes
/// (shared x-coordinate: doubling or cancellation) produce garbage-but-
/// canonical outputs that the caller MUST mask and redo scalar-side. See
/// `flush_bucket_batch`, which passes such lanes deliberately.
/// Returns the eight sums plus the `h == 0` lane mask. The caller must ignore
/// masked lane outputs and recompute them with scalar code.
///
/// 11 batched muls (3 of them squarings) + 7 batched subs + 8 domain
/// conversions, versus 8 x 11 scalar muls on the scalar path.
#[cfg(helius_avx512_ifma)]
pub(crate) fn g1_madd_batch8(
    x1: &[Fp; 8],
    y1: &[Fp; 8],
    z1: &[Fp; 8],
    x2: &[Fp; 8],
    y2: &[Fp; 8],
) -> ([Fp; 8], [Fp; 8], [Fp; 8], u8) {
    let x1v = unsafe { FpVec8::load(x1) };
    let y1v = unsafe { FpVec8::load(y1) };
    let z1v = unsafe { FpVec8::load(z1) };
    let x2v = unsafe { FpVec8::load(x2) };
    let y2v = unsafe { FpVec8::load(y2) };

    let z1z1 = z1v.square();
    let h = x2v.mul(&z1z1).sub(&x1v);
    let r = z1z1.mul(&z1v).mul(&y2v).sub(&y1v);
    let h_zero = h.is_zero_mask();
    let z3 = z1v.mul(&h);
    let h2 = h.square();
    let mut ry = r.square();
    let u1h = x1v.mul(&h2);
    let h3 = h2.mul(&h);
    ry = ry.sub(&u1h).sub(&u1h);
    let x3 = ry.sub(&h3);
    let y3 = u1h.sub(&x3).mul(&r).sub(&h3.mul(&y1v));

    (x3.store(), y3.store(), z3.store(), h_zero)
}

// Direct intrinsic tests intentionally require an IFMA target baseline. A
// generic test binary must exercise this module only through runtime dispatch,
// so running the suite on an older CPU can never enter an unguarded helper.
#[cfg(all(test, helius_avx512_ifma))]
mod tests {
    use super::*;

    fn load(values: &[Fp; 8]) -> FpVec8 {
        unsafe { FpVec8::load(values) }
    }
    use crate::consts::{MONT_ONE, MONT_R2, P};
    use crate::limb;

    fn next_residue(state: &mut u64) -> [u64; 4] {
        let mut value = [0u64; 4];
        for limb in &mut value {
            *state ^= *state << 13;
            *state ^= *state >> 7;
            *state ^= *state << 17;
            *limb = *state;
        }
        while limb::gte(&value, &P) {
            value = limb::sub_noborrow(&value, &P);
        }
        value
    }

    fn reduced(mut value: [u64; 4]) -> [u64; 4] {
        while limb::gte(&value, &P) {
            value = limb::sub_noborrow(&value, &P);
        }
        value
    }

    /// Carry-edge corpus: canonical values whose radix-52 limbs sit at the
    /// 2^52-1 boundary, plus the usual field edges.
    fn edge_corpus() -> Vec<[u64; 4]> {
        let p_minus_one = limb::sub_noborrow(&P, &[1, 0, 0, 0]);
        let mut cases = vec![
            [0; 4],
            [1, 0, 0, 0],
            [MASK52, 0, 0, 0],
            [u64::MAX, MASK52, 0, 0],
            MONT_ONE,
            MONT_R2,
            p_minus_one,
            reduced([u64::MAX; 4]),
            reduced([u64::MAX, u64::MAX, u64::MAX, (1 << 62) - 1]),
            reduced([u64::MAX, 0, u64::MAX, 0]),
            reduced([0, u64::MAX, 0, 0x1000_0000_0000_0000]),
            // All radix-52 limbs saturated: (2^260 - 1) mod 2^256 pattern.
            reduced([u64::MAX, u64::MAX, u64::MAX, 0x000f_ffff_ffff_ffff]),
        ];
        let mut state = 0x243f_6a88_85a3_08d3u64;
        for _ in 0..256 {
            cases.push(next_residue(&mut state));
        }
        cases
    }

    fn fp8(values: &[[u64; 4]]) -> [Fp; 8] {
        core::array::from_fn(|i| Fp(values[i % values.len()]))
    }

    #[test]
    fn radix52_repack_roundtrips() {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        for _ in 0..10_000 {
            let v = next_residue(&mut state);
            assert_eq!(from_radix52(&to_radix52(v)), v);
        }
        assert_eq!(from_radix52(&P52), P);
    }

    #[test]
    fn constants_are_consistent() {
        // p * (-p^{-1}) = -1 mod 2^52.
        let p0 = P[0] & MASK52;
        assert_eq!(p0.wrapping_mul(P_INV_52) & MASK52, MASK52);
        // C_IN = 2^264 mod p: check via Fp arithmetic.
        let two_pow_8 = Fp::from_u64(256);
        let expected = Fp::from_raw(MONT_ONE) * two_pow_8; // (2^256 mod p)*2^8
        assert_eq!(to_radix52(expected.to_raw()), C_IN_52);
    }

    #[test]
    fn load_store_roundtrips_on_edges() {
        for chunk in edge_corpus().chunks(8) {
            let input = fp8(chunk);
            assert_eq!(load(&input).store(), input);
        }
    }

    #[test]
    fn batched_ops_match_scalar_on_edges_and_random() {
        let cases = edge_corpus();
        let mut state = 0x6a09_e667_f3bc_c909u64;
        let rounds = if cfg!(debug_assertions) {
            2_000
        } else {
            65_536
        };
        let inv2 = Fp::from_u64(2).invert().unwrap();
        for round in 0..rounds {
            let a: [Fp; 8] = if round < cases.len() {
                fp8(&cases[round..(round + 8).min(cases.len())])
            } else {
                core::array::from_fn(|_| Fp(next_residue(&mut state)))
            };
            let b: [Fp; 8] = core::array::from_fn(|_| Fp(next_residue(&mut state)));
            let av = load(&a);
            let bv = load(&b);
            let mul = av.mul(&bv).store();
            let sqr = av.square().store();
            let half = av.half().store();
            let add = av.add(&bv).store();
            let sub = av.sub(&bv).store();
            for lane in 0..8 {
                assert_eq!(
                    mul[lane],
                    a[lane] * b[lane],
                    "mul round {round} lane {lane}"
                );
                assert_eq!(sqr[lane], a[lane].square(), "sqr round {round} lane {lane}");
                assert_eq!(half[lane], a[lane] * inv2, "half round {round} lane {lane}");
                assert_eq!(
                    add[lane],
                    a[lane] + b[lane],
                    "add round {round} lane {lane}"
                );
                assert_eq!(
                    sub[lane],
                    a[lane] - b[lane],
                    "sub round {round} lane {lane}"
                );
            }
            let zero_mask = av.is_zero_mask();
            for (lane, value) in a.iter().enumerate() {
                assert_eq!(zero_mask >> lane & 1 == 1, value.is_zero());
            }
        }
    }

    #[test]
    #[ignore = "million-case release stress gate; run explicitly before changing the IFMA kernel"]
    fn million_products_match_portable() {
        let mut state = 0xd1b5_4a32_d192_ed03u64;
        for case in 0..125_000u64 {
            let a: [Fp; 8] = core::array::from_fn(|_| Fp(next_residue(&mut state)));
            let b: [Fp; 8] = core::array::from_fn(|_| Fp(next_residue(&mut state)));
            let mul = load(&a).mul(&load(&b)).store();
            for lane in 0..8 {
                let expected = crate::fp::portable::mont_mul(&a[lane].0, &b[lane].0);
                assert_eq!(mul[lane], expected, "case {case} lane {lane}");
            }
        }
    }

    #[test]
    fn batched_madd_matches_scalar() {
        use crate::g1::G1Projective;
        use crate::{Fr, G1Affine};

        let mut state = 0x0123_4567_89ab_cdefu64;
        let random_point = |state: &mut u64| -> G1Affine {
            let scalar = next_residue(state);
            G1Projective::generator()
                .mul_scalar(Fr::from_raw(scalar))
                .to_affine()
        };

        for case in 0..64 {
            let points: [G1Affine; 8] = core::array::from_fn(|_| random_point(&mut state));
            let mut buckets: [G1Projective; 8] = core::array::from_fn(|_| {
                random_point(&mut state)
                    .to_curve()
                    .add_projective(random_point(&mut state).to_curve())
            });
            // Lane 3: h == 0 doubling case (bucket is the point itself, z=1).
            // Lane 5: h == 0 cancellation case (bucket is the negation).
            buckets[3] = points[3].to_curve();
            buckets[5] = points[5].negate().to_curve();

            let x1 = core::array::from_fn(|i| buckets[i].x);
            let y1 = core::array::from_fn(|i| buckets[i].y);
            let z1 = core::array::from_fn(|i| buckets[i].z);
            let x2 = core::array::from_fn(|i| points[i].x);
            let y2 = core::array::from_fn(|i| points[i].y);
            let (x3, y3, z3, h_zero) = g1_madd_batch8(&x1, &y1, &z1, &x2, &y2);

            assert_eq!(h_zero, 0b0010_1000, "case {case}");
            for lane in 0..8 {
                if h_zero >> lane & 1 == 1 {
                    continue;
                }
                let expected = buckets[lane].add_mixed(points[lane]);
                let actual = G1Projective {
                    x: x3[lane],
                    y: y3[lane],
                    z: z3[lane],
                };
                assert_eq!(
                    actual.to_affine(),
                    expected.to_affine(),
                    "case {case} lane {lane}"
                );
            }
        }
    }

    #[test]
    fn sos_mac_matches_scalar_sum_of_products() {
        fn check<const N: usize>(
            state: &mut u64,
            rounds: usize,
            sos_mac: fn(&[FpVec8; N], &[FpVec8; N]) -> FpVec8,
        ) {
            for round in 0..rounds {
                let a_terms: [[Fp; 8]; N] =
                    core::array::from_fn(|_| core::array::from_fn(|_| Fp(next_residue(state))));
                let b_terms: [[Fp; 8]; N] =
                    core::array::from_fn(|_| core::array::from_fn(|_| Fp(next_residue(state))));
                let expected: [Fp; 8] = core::array::from_fn(|lane| {
                    (0..N).fold(Fp::ZERO, |acc, term| {
                        acc + a_terms[term][lane] * b_terms[term][lane]
                    })
                });
                let av: [FpVec8; N] = core::array::from_fn(|term| load(&a_terms[term]));
                let bv: [FpVec8; N] = core::array::from_fn(|term| load(&b_terms[term]));
                let got = sos_mac(&av, &bv).store();
                assert_eq!(got, expected, "arity={N} round={round}");
            }
        }

        let mut state = 0x1234_5678_9abc_def0u64;
        let rounds = if cfg!(debug_assertions) { 300 } else { 6_144 };
        check(&mut state, rounds, FpVec8::sos_mac_2);
        check(&mut state, rounds, FpVec8::sos_mac_6);

        let pm1 = Fp(limb::sub_noborrow(&P, &[1, 0, 0, 0]));
        let lanes = [pm1; 8];
        let v = load(&lanes);
        assert_eq!(
            FpVec8::sos_mac_2(&[v; 2], &[v; 2]).store(),
            [pm1 * pm1 + pm1 * pm1; 8]
        );
        assert_eq!(
            FpVec8::sos_mac_6(&[v; SOS_MAX_ARITY], &[v; SOS_MAX_ARITY]).store(),
            [(0..SOS_MAX_ARITY).fold(Fp::ZERO, |acc, _| acc + pm1 * pm1); 8]
        );
    }

    #[test]
    #[cfg(feature = "std")]
    fn interleaved_products_match_serial_products() {
        let mut state = 0x3c6e_f372_fe94_f82bu64;
        let rounds = if cfg!(debug_assertions) { 300 } else { 6_144 };
        for round in 0..rounds {
            let values: [FpVec8; 6] = core::array::from_fn(|_| {
                load(&core::array::from_fn(|_| Fp(next_residue(&mut state))))
            });
            let lazy: [FpVec8L<1>; 6] = core::array::from_fn(|index| values[index].lazy());
            let [p0, p1, p2] = FpVec8::mul_three_lazy::<1, 1>(
                &lazy[0], &lazy[1], &lazy[2], &lazy[3], &lazy[4], &lazy[5],
            );
            assert_eq!(
                p0.store(),
                values[0].mul(&values[1]).store(),
                "round={round}"
            );
            assert_eq!(
                p1.store(),
                values[2].mul(&values[3]).store(),
                "round={round}"
            );
            assert_eq!(
                p2.store(),
                values[4].mul(&values[5]).store(),
                "round={round}"
            );

            let a0: [FpVec8; 6] = core::array::from_fn(|index| values[index]);
            let b0: [FpVec8; 6] = core::array::from_fn(|index| values[5 - index]);
            let a1: [FpVec8; 3] = core::array::from_fn(|index| values[(index + 2) % 6]);
            let b1: [FpVec8; 3] = core::array::from_fn(|index| values[(index + 4) % 6]);
            let la0 = a0.map(|value| value.lazy());
            let lb0 = b0.map(|value| value.lazy());
            let la1 = a1.map(|value| value.lazy());
            let lb1 = b1.map(|value| value.lazy());
            let [s0, s1] = unsafe {
                FpVec8::sos_mac_6_3_pair_fused_lazy::<1, 1, 1, 1>(&la0, &lb0, &la1, &lb1)
            };
            assert_eq!(
                s0.store(),
                FpVec8::sos_mac_6(&a0, &b0).store(),
                "round={round}"
            );
            let short_expected: [Fp; 8] = {
                let (a1s, b1s) = (a1.map(|v| v.store()), b1.map(|v| v.store()));
                core::array::from_fn(|lane| {
                    (0..3).fold(Fp::ZERO, |acc, t| acc + a1s[t][lane] * b1s[t][lane])
                })
            };
            assert_eq!(s1.store(), short_expected, "round={round}");
        }
    }

    #[test]
    fn loose_triple_matches_canonical_addition() {
        let mut state = 0x0ddc_0ffe_e15e_beefu64;
        let rounds = if cfg!(debug_assertions) { 300 } else { 6_144 };
        for round in 0..rounds {
            let values = core::array::from_fn(|_| Fp(next_residue(&mut state)));
            let expected = values.map(|value| value + value + value);
            assert_eq!(load(&values).triple().store(), expected, "round={round}");
        }

        let pm1 = Fp(limb::sub_noborrow(&P, &[1, 0, 0, 0]));
        let pm3 = Fp(limb::sub_noborrow(&P, &[3, 0, 0, 0]));
        assert_eq!(load(&[pm1; 8]).triple().store(), [pm3; 8]);

        let half_pm1 = Fp(crate::consts::derive::shr4(pm1.0, 1));
        let cases = [
            Fp::ZERO,
            Fp::ONE,
            half_pm1,
            half_pm1,
            pm1,
            pm1,
            Fp::ZERO,
            Fp::ONE,
        ];
        assert_eq!(
            load(&cases).triple().store(),
            cases.map(|value| value + value + value)
        );
    }

    #[cfg(feature = "std")]
    #[test]
    fn lazy_bound_constants_are_pinned() {
        // bias52 keeps the exact value k*p after carry normalization.
        for (k, m) in [(1, 1), (2, 1), (3, 2), (12, 1), (13, 2), (22, 1)] {
            let bias = bias52(k, m);
            let mut normalized = [0u64; 5];
            let mut carry = 0u64;
            for j in 0..5 {
                let acc = bias[j] + carry;
                normalized[j] = acc & MASK52;
                carry = acc >> 52;
            }
            let (kp, kp_carry) = mul_p52(k);
            assert_eq!(kp_carry, 0);
            assert_eq!((normalized, carry), (kp, 0), "bias {k} {m}");
        }
        // Minimal biases the kernels rely on.
        assert_eq!(min_bias(1, 1), 2);
        assert_eq!(min_bias(1, 2), 3);
        assert_eq!(min_bias(1, 11), 12);
        assert_eq!(min_bias(1, 21), 22);
        assert_eq!(min_bias(2, 2), 3);
        assert_eq!(min_bias(2, 12), 13);
        // Fold trajectory used by reduce_full.
        assert_eq!(fold_bound(39), 11);
        assert_eq!(fold_bound(11), 4);
        assert_eq!(fold_bound(4), 3);
        assert_eq!(folds_needed(39), 3);
        assert_eq!(folded_bound(39), 3);
        assert_eq!(folds_needed(14), 2);
        assert_eq!(folds_needed(6), 1);
        assert_eq!(folds_needed(2), 0);
        // Sum budgets of the live sos entries.
        assert!(sos_sum_ok(6 * 13, 1));
        assert!(sos_sum_ok(6 * 23, 2));
        assert!(sos_sum_ok(3 * 13, 1));
        assert!(sos_sum_ok(3 * 23, 1));
        assert!(sos_sum_ok(12 * 13, 2));
        assert!(sos_sum_ok(5 * 4, 1));
        assert!(sos_sum_ok(2 * 3, 1));
        assert_eq!(folds_needed(15), 2);
    }

    #[cfg(feature = "std")]
    #[test]
    fn lazy_ops_match_scalar() {
        let nine = Fp::from_u64(9);
        let mut state = 0x5851_f42d_4c95_7f2du64;
        let rounds = if cfg!(debug_assertions) { 200 } else { 4_096 };
        for round in 0..rounds {
            let a: [Fp; 8] = core::array::from_fn(|_| Fp(next_residue(&mut state)));
            let b: [Fp; 8] = core::array::from_fn(|_| Fp(next_residue(&mut state)));
            let c: [Fp; 8] = core::array::from_fn(|_| Fp(next_residue(&mut state)));
            let av = load(&a).lazy();
            let bv = load(&b).lazy();
            let cv = load(&c).lazy();

            let sum = av.add_lazy::<1, 2>(&bv);
            assert_eq!(
                sum.reduce_full().store(),
                core::array::from_fn(|i| a[i] + b[i]),
                "add round {round}"
            );
            assert_eq!(
                av.sub_lazy::<1, 3>(&bv).reduce_full().store(),
                core::array::from_fn(|i| a[i] - b[i]),
                "sub round {round}"
            );
            assert_eq!(
                av.sub2_lazy::<1, 1, 4>(&bv, &cv).reduce_full().store(),
                core::array::from_fn(|i| a[i] - b[i] - c[i]),
                "sub2 round {round}"
            );
            assert_eq!(
                av.double_lazy::<2>().reduce_full().store(),
                core::array::from_fn(|i| a[i] + a[i]),
                "double round {round}"
            );
            // xi on pairs: even lane 9r - i, odd lane 9i + r.
            let xi = av.xi_lazy::<11>().reduce_full().store();
            for pair in 0..4 {
                let (r, i) = (a[2 * pair], a[2 * pair + 1]);
                assert_eq!(xi[2 * pair], r * nine - i, "xi real round {round}");
                assert_eq!(xi[2 * pair + 1], i * nine + r, "xi imag round {round}");
            }
            let negated = av.negate_lanes_lazy::<3>(0x55).reduce_full().store();
            for lane in 0..8usize {
                let want = if lane.is_multiple_of(2) {
                    -a[lane]
                } else {
                    a[lane]
                };
                assert_eq!(negated[lane], want, "negate round {round}");
            }
            // Deep chain: sum at bound 2 through xi and a fold-heavy reduce.
            let chained = sum.xi_lazy::<21>().reduce_full().store();
            for pair in 0..4 {
                let (r, i) = (a[2 * pair] + b[2 * pair], a[2 * pair + 1] + b[2 * pair + 1]);
                assert_eq!(chained[2 * pair], r * nine - i, "chain real round {round}");
                assert_eq!(
                    chained[2 * pair + 1],
                    i * nine + r,
                    "chain imag round {round}"
                );
            }
            let mixed = sum
                .relax::<12>()
                .triple_pm_double::<1, 39>(&bv, 0xc0)
                .reduce_full()
                .store();
            for lane in 0..8 {
                let t = a[lane] + b[lane];
                let three_t = t + t + t;
                let two_b = b[lane] + b[lane];
                let want = if lane >= 6 {
                    three_t + two_b
                } else {
                    three_t - two_b
                };
                assert_eq!(mixed[lane], want, "triple_pm round {round}");
            }
            // keep_lanes zeroes the masked-out lanes, retag narrows.
            let kept = unsafe { sum.keep_lanes(0x3c).retag::<2>() }
                .reduce_full()
                .store();
            for lane in 0..8 {
                let want = if (0x3c >> lane) & 1 == 1 {
                    a[lane] + b[lane]
                } else {
                    Fp::ZERO
                };
                assert_eq!(kept[lane], want, "keep round {round}");
            }
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn lazy_sos_entries_match_scalar() {
        let mut state = 0x94d0_49bb_1331_11ebu64;
        let rounds = if cfg!(debug_assertions) { 100 } else { 2_048 };
        for round in 0..rounds {
            let a_terms: [[Fp; 8]; 6] =
                core::array::from_fn(|_| core::array::from_fn(|_| Fp(next_residue(&mut state))));
            let b_terms: [[Fp; 8]; 6] =
                core::array::from_fn(|_| core::array::from_fn(|_| Fp(next_residue(&mut state))));
            let c_terms: [[Fp; 8]; 6] =
                core::array::from_fn(|_| core::array::from_fn(|_| Fp(next_residue(&mut state))));
            let av: [FpVec8L<1>; 6] = core::array::from_fn(|t| load(&a_terms[t]).lazy());
            // Rows at bound 13 built by lazy negation, matching the FE plans.
            let bv: [FpVec8L<13>; 6] = core::array::from_fn(|t| {
                load(&b_terms[t])
                    .lazy()
                    .relax::<11>()
                    .negate_lanes_lazy::<13>(0x55)
            });
            let cv: [FpVec8L<13>; 6] = core::array::from_fn(|t| {
                load(&c_terms[t])
                    .lazy()
                    .relax::<11>()
                    .negate_lanes_lazy::<13>(0x55)
            });
            let signed = |x: Fp, lane: usize| if lane.is_multiple_of(2) { -x } else { x };
            let expected0: [Fp; 8] = core::array::from_fn(|lane| {
                (0..6).fold(Fp::ZERO, |acc, t| {
                    acc + a_terms[t][lane] * signed(b_terms[t][lane], lane)
                })
            });
            let expected1_short: [Fp; 8] = core::array::from_fn(|lane| {
                (0..3).fold(Fp::ZERO, |acc, t| {
                    acc + a_terms[t][lane] * signed(c_terms[t][lane], lane)
                })
            });
            let short_av: [FpVec8L<1>; 3] = core::array::from_fn(|t| av[t]);
            let short_cv: [FpVec8L<13>; 3] = core::array::from_fn(|t| cv[t]);
            // The sparse-034 bound pair: full rows at 13 with one round,
            // half rows at 13 with one round.
            let [got0, got1] = unsafe {
                FpVec8::sos_mac_6_3_pair_fused_lazy::<1, 13, 1, 1>(&av, &bv, &short_av, &short_cv)
            };
            assert_eq!(got0.store(), expected0, "pair stream 0 round {round}");
            assert_eq!(got1.store(), expected1_short, "pair stream 1 round {round}");

            // Two-round tail at the doubling-stage bound.
            let bw: [FpVec8L<23>; 6] = core::array::from_fn(|t| {
                load(&b_terms[t])
                    .lazy()
                    .relax::<21>()
                    .negate_lanes_lazy::<23>(0x55)
            });
            let cw: [FpVec8L<23>; 3] = core::array::from_fn(|t| {
                load(&c_terms[t])
                    .lazy()
                    .relax::<21>()
                    .negate_lanes_lazy::<23>(0x55)
            });
            let [got0, got1] = unsafe {
                FpVec8::sos_mac_6_3_pair_fused_lazy::<1, 23, 2, 1>(&av, &bw, &short_av, &cw)
            };
            assert_eq!(got0.store(), expected0, "wide stream 0 round {round}");
            assert_eq!(got1.store(), expected1_short, "wide stream 1 round {round}");

            // The dense final-exponentiation pair: twelve-term full rows with
            // two rounds against six-term half rows with one round.
            let dense_av: [FpVec8L<1>; 12] = core::array::from_fn(|t| av[t % 6]);
            let dense_bv: [FpVec8L<13>; 12] = core::array::from_fn(|t| bv[t % 6]);
            let expected_dense: [Fp; 8] = core::array::from_fn(|lane| {
                let one = expected0[lane];
                one + one
            });
            let [got0, got1] = unsafe {
                FpVec8::sos_mac_12_6_pair_fused_lazy::<1, 13, 2, 1>(&dense_av, &dense_bv, &av, &cv)
            };
            assert_eq!(got0.store(), expected_dense, "dense stream 0 round {round}");
            let expected1: [Fp; 8] = core::array::from_fn(|lane| {
                (0..6).fold(Fp::ZERO, |acc, t| {
                    acc + a_terms[t][lane] * signed(c_terms[t][lane], lane)
                })
            });
            assert_eq!(got1.store(), expected1, "dense stream 1 round {round}");

            // Interleaved products with lazily bounded operands. The doubled
            // and negated forms exercise non-canonical values.
            let doubled: [FpVec8L<4>; 3] =
                core::array::from_fn(|t| load(&b_terms[t]).lazy().double_lazy::<2>().relax::<4>());
            let lefts: [FpVec8L<5>; 3] = core::array::from_fn(|t| {
                load(&a_terms[t])
                    .lazy()
                    .relax::<3>()
                    .negate_lanes_lazy::<5>(0x55)
            });
            let [m0, m1, m2] = FpVec8::mul_three_lazy::<5, 4>(
                &lefts[0],
                &doubled[0],
                &lefts[1],
                &doubled[1],
                &lefts[2],
                &doubled[2],
            );
            for (t, got) in [m0, m1, m2].iter().enumerate() {
                let expected: [Fp; 8] = core::array::from_fn(|lane| {
                    signed(a_terms[t][lane], lane) * (b_terms[t][lane] + b_terms[t][lane])
                });
                assert_eq!(got.store(), expected, "mul_three stream {t} round {round}");
            }
        }
    }

    #[test]
    fn loose_range_limits_are_pinned() {
        assert_eq!(MONT52_BITS, 260);
        assert_eq!(MODULUS_BITS, 254);
        assert_eq!(SOS_MAX_ARITY, 6);
        const { assert!(SOS_MAX_ARITY < 1 << (MONT52_BITS - MODULUS_BITS)) };
        assert_eq!(LOOSE3_MAX_LIMB, 3 * ((1_u64 << 52) - 1));
        const { assert!(LOOSE3_MAX_LIMB < 1 << 54) };
        assert_eq!(LOOSE3_MAX_CARRY, 2);
        assert_eq!(LOOSE3_MAX_TOP_LIMB, 3 * P52[4] + LOOSE3_MAX_CARRY);
        const { assert!(LOOSE3_MAX_TOP_LIMB <= MASK52) };
    }
}
