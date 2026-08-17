//! Monomorphized Fp2 ops (raw 8x u64) for G2 and miller hot paths.

use crate::consts::P;
use crate::fp::Fp;
// Tower/miller: shared out-of-line asm (keeps miller I-cache tight).
#[cfg(all(
    target_arch = "aarch64",
    target_vendor = "apple",
    not(feature = "force-portable"),
))]
use crate::fp::aarch64::mont_mul;
#[cfg(not(any(
    all(
        target_arch = "aarch64",
        target_vendor = "apple",
        not(feature = "force-portable")
    ),
    all(narsil_mont4_x86_64_adx, not(feature = "force-portable"))
)))]
use crate::fp::portable::mont_mul;
#[cfg(all(narsil_mont4_x86_64_adx, not(feature = "force-portable")))]
use crate::fp::x86_64::mont_mul;
use crate::fp2::Fp2;
use crate::limb::{add_mod, sub_mod};

pub type F2 = ([u64; 4], [u64; 4]); // (c0, c1)

#[inline(always)]
fn fm(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
    mont_mul(a, b).0
}
#[inline(always)]
fn fa(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
    add_mod(a, b, &P)
}
#[inline(always)]
fn fsub(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
    sub_mod(a, b, &P)
}
#[inline(always)]
fn fd(a: &[u64; 4]) -> [u64; 4] {
    fa(a, a)
}
#[inline(always)]
fn fneg(a: &[u64; 4]) -> [u64; 4] {
    // Plain p-a is wrong at zero (p-0 = p needs a reduce). Sub_mod(0, a)
    // gives -a mod p and maps zero to zero.
    fsub(&[0; 4], a)
}

#[inline(always)]
pub fn f2_from(f: Fp2) -> F2 {
    (f.c0.0, f.c1.0)
}
#[inline(always)]
pub fn f2_to(f: F2) -> Fp2 {
    Fp2 {
        c0: Fp(f.0),
        c1: Fp(f.1),
    }
}

// X86 may outline these operations to bound the Miller loop's instruction
// footprint. Other targets inline them. NARSIL_MILLER_INLINE selects the x86
// inline variant.
macro_rules! miller_helper {
    ($(#[$doc:meta])* pub fn $name:ident $signature:tt -> F2 $body:block) => {
        $(#[$doc])*
        #[cfg_attr(
            all(target_arch = "x86_64", not(narsil_miller_inline)),
            inline(never)
        )]
        #[cfg_attr(
            any(not(target_arch = "x86_64"), narsil_miller_inline),
            inline(always)
        )]
        pub fn $name $signature -> F2 $body
    };
}

miller_helper! {
pub fn f2_add(a: F2, b: F2) -> F2 {
    (fa(&a.0, &b.0), fa(&a.1, &b.1))
}
}
miller_helper! {
pub fn f2_sub(a: F2, b: F2) -> F2 {
    (fsub(&a.0, &b.0), fsub(&a.1, &b.1))
}
}
miller_helper! {
pub fn f2_dbl(a: F2) -> F2 {
    (fd(&a.0), fd(&a.1))
}
}
miller_helper! {
pub fn f2_neg(a: F2) -> F2 {
    (fneg(&a.0), fneg(&a.1))
}
}

/// Schoolbook sums-of-products (4M, 2 interleaved reductions):
/// `c0 = a0*b0 - a1*b1`, `c1 = a0*b1 + a1*b0`. Both lanes in one dual kernel.
#[cfg(not(all(narsil_f2mul_asm, not(feature = "force-portable"))))]
#[inline(always)]
pub fn f2_mul(a: F2, b: F2) -> F2 {
    crate::fp::sos::sosd2(&a.0, &a.1, &b.0, &b.1)
}

/// Lazy double-width Karatsuba (3M, 2 reductions) through the generated
/// leaf, the same value as the sums-of-products route. The three raw
/// products stay unreduced, so the Karatsuba identity buys a whole 4x4
/// product; `4p < 2^256` is what admits the uncorrected operand sums.
#[cfg(all(narsil_f2mul_asm, not(feature = "force-portable")))]
#[inline(always)]
pub fn f2_mul(a: F2, b: F2) -> F2 {
    crate::fp::x86_64::f2mul(&[a.0, a.1], &[b.0, b.1])
}

/// SoS square: `c0 = a0^2 - a1^2`, `c1 = a0*a1 + a1*a0` (4M, 2 reductions,
/// no modular add/sub). Both lanes in one dual kernel.
#[inline(always)]
pub fn f2_sqr(a: F2) -> F2 {
    crate::fp::sos::sosd2(&a.0, &a.1, &a.0, &a.1)
}

/// The tier's Fp2 square. The ADX tiers take the complex identity, which
/// pays two Montgomery products where the fused SoS kernel pays four.
#[cfg(not(all(narsil_mont4_x86_64_adx, not(feature = "force-portable"))))]
pub use f2_sqr as f2_square;
#[cfg(all(narsil_f2sqr_asm, not(feature = "force-portable")))]
pub use f2_sqr_fused as f2_square;
#[cfg(all(
    narsil_mont4_x86_64_adx,
    not(narsil_f2sqr_asm),
    not(feature = "force-portable")
))]
pub use f2_sqr_lazy as f2_square;

/// Karatsuba: 3 Montgomery products + 3 reductions vs `f2_mul`'s fused
/// 4 + 2, for five extra modular add/subs. Canonical in/out. The
/// differential-test reference. No production tier routes here now, the
/// three-call form loses to the one-call fused dual kernel on every
/// measured x86 host.
#[cfg(test)]
#[cfg_attr(target_arch = "x86_64", inline(never))]
#[cfg_attr(not(target_arch = "x86_64"), inline(always))]
pub fn f2_mul_karatsuba(a: F2, b: F2) -> F2 {
    let t0 = fm(&a.0, &b.0);
    let t1 = fm(&a.1, &b.1);
    let t2 = fm(&fa(&a.0, &a.1), &fa(&b.0, &b.1));
    (fsub(&t0, &t1), fsub(&fsub(&t2, &t0), &t1))
}

/// (a+bu)^2 = (a+b)(a-b) + 2ab u: 2 Montgomery products + 2 reductions vs
/// `f2_sqr`'s fused 4 + 2, for three extra modular add/subs. Canonical
/// in/out. The ADX tier routes the Miller G2-step squares here (mont_mul
/// dominates there). Other tiers keep the fused dual kernel.
#[cfg(any(test, all(narsil_mont4_x86_64_adx, not(feature = "force-portable"))))]
#[cfg_attr(target_arch = "x86_64", inline(never))]
#[cfg_attr(not(target_arch = "x86_64"), inline(always))]
pub fn f2_sqr_lazy(a: F2) -> F2 {
    let ab = fm(&a.0, &a.1);
    (fm(&fa(&a.0, &a.1), &fsub(&a.0, &a.1)), fd(&ab))
}

/// The same complex identity as [`f2_sqr_lazy`], but both Montgomery chains
/// live in one leaf instead of two sequential `mont_mul` calls, so the second
/// chain fills the first one's latency. Same product count, same value.
#[cfg(all(narsil_f2sqr_asm, not(feature = "force-portable")))]
#[inline(always)]
pub fn f2_sqr_fused(a: F2) -> F2 {
    crate::fp::x86_64::f2sqr(&a.0, &a.1)
}

/// `g^2 - 3*e^2` over Fp2 through the lazy double-width leaf. `f` must be
/// `3*e`: the tripling folds into the second operand of each raw product, so
/// the leaf pays 4 products and 2 Montgomery reductions where the composed
/// route pays 4 products and 4 reductions plus three Fp2 helper round trips.
#[cfg(all(narsil_g2_ysqr_asm, not(feature = "force-portable")))]
#[inline(always)]
pub fn f2_ysqr(g: F2, e: F2, f: F2) -> F2 {
    crate::fp::x86_64::g2_ysqr(&[g.0, g.1], &[e.0, e.1], &[f.0, f.1])
}

#[inline(always)]
pub fn f2_mul_fp(a: F2, f: &[u64; 4]) -> F2 {
    (fm(&a.0, f), fm(&a.1, f))
}

#[inline(always)]
pub fn f2_is_zero(a: F2) -> bool {
    (a.0[0] | a.0[1] | a.0[2] | a.0[3] | a.1[0] | a.1[1] | a.1[2] | a.1[3]) == 0
}

#[inline(always)]
pub fn f2_is_one(a: F2) -> bool {
    use crate::consts::MONT_ONE;
    a.0 == MONT_ONE && (a.1[0] | a.1[1] | a.1[2] | a.1[3]) == 0
}

#[inline(always)]
pub fn f2_one() -> F2 {
    use crate::consts::MONT_ONE;
    (MONT_ONE, [0; 4])
}
