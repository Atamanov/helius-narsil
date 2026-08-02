//! Kani proof harnesses for the radix-52 lazy-reduction bound arithmetic.
//!
//! The IFMA backend (`fp/avx512ifma.rs`) tracks per-value bounds in the type
//! system (`FpVec8L<B>`, `LAZY_CAP = 84`) and discharges each producer bound
//! with const asserts over pure quotient functions (`fold_bound`,
//! `folds_needed`, `min_bias`). This module re-derives those functions and
//! the per-lane semantics of the fold/reduce steps as exact u128 integer
//! models, then proves the bound claims for every admissible bound and every
//! value, not only the corpus a test can reach. The vector kernels
//! themselves are out of scope here; they are covered differentially.
//!
//! The module compiles only under `cfg(kani)`. Run every harness with
//! `cargo kani -p helius-narsil`, or one with `--harness <name>`; a prefix
//! selects a group, for example `--harness fold`. Add
//! `CC=/usr/bin/cc CXX=/usr/bin/c++ AR=/usr/bin/ar` on hosts whose default
//! clang is broken.
//!
//! Kani ships its own nightly rustc. When that nightly is older than the
//! workspace `rust-version`, cargo refuses the build before Kani runs, and
//! the floor has to drop for the length of the run.
//!
//! Limb equality goes through [`eq52`], never `==`. Array comparison lowers
//! to `memcmp`, which the model checker has no definition for, and every
//! check in the harness then reports as undetermined.

use crate::consts::P;

const MASK52: u64 = (1 << 52) - 1;
/// Mirror of `fp::avx512ifma::LAZY_CAP`. `LAZY_CAP * p < 2^260`.
const LAZY_CAP: u64 = 84;

const fn to_radix52(x: [u64; 4]) -> [u64; 5] {
    [
        x[0] & MASK52,
        ((x[0] >> 52) | (x[1] << 12)) & MASK52,
        ((x[1] >> 40) | (x[2] << 24)) & MASK52,
        ((x[2] >> 28) | (x[3] << 36)) & MASK52,
        x[3] >> 16,
    ]
}

const fn sub4(a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
    let mut out = [0u64; 4];
    let mut borrow = 0u64;
    let mut i = 0;
    while i < 4 {
        let (d, b1) = a[i].overflowing_sub(b[i]);
        let (d, b2) = d.overflowing_sub(borrow);
        out[i] = d;
        borrow = b1 as u64 | b2 as u64;
        i += 1;
    }
    out
}

const P52: [u64; 5] = to_radix52(P);
/// `2^254 - p`, the additive fold constant.
const EPS52: [u64; 5] = to_radix52(sub4([0, 0, 0, 1 << 62], P));

/// `k * p` as normalized limbs plus the carry past limb 4.
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

const fn lt52(a: &[u64; 5], b: &[u64; 5]) -> bool {
    let mut j = 4;
    loop {
        if a[j] != b[j] {
            return a[j] < b[j];
        }
        if j == 0 {
            return false;
        }
        j -= 1;
    }
}

/// Mirror of the source `fold_bound`: exact post-fold bound multiple.
const fn fold_bound(b: u64) -> u64 {
    let (bp, carry) = mul_p52(b);
    assert!(b >= 1 && carry == 0);
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

/// Smallest borrow-free bias multiple for an `m`-term subtraction of values
/// bounded by `csum * p` in total. Mirror of the source `min_bias`.
const fn min_bias(m: u64, csum: u64) -> u64 {
    let mut k = 1;
    loop {
        let (kp, carry) = mul_p52(k);
        if carry == 0 && kp[4] >= m + csum * (P52[4] + 1) {
            return k;
        }
        k += 1;
        assert!(k <= LAZY_CAP);
    }
}

const TABLE: usize = (LAZY_CAP + 1) as usize;

/// Compile-time tables so a harness indexes with a symbolic bound instead of
/// unwinding the search loops symbolically. Const evaluation of the mirrors
/// also re-runs every internal assert.
const FOLD_BOUND: [u64; TABLE] = {
    let mut t = [0u64; TABLE];
    let mut b = 1;
    while b <= LAZY_CAP {
        t[b as usize] = fold_bound(b);
        b += 1;
    }
    t
};

const FOLDS_NEEDED: [u64; TABLE] = {
    let mut t = [0u64; TABLE];
    let mut b = 1;
    while b <= LAZY_CAP {
        let mut cur = b;
        let mut n = 0;
        while cur > 3 {
            cur = FOLD_BOUND[cur as usize];
            n += 1;
            assert!(n <= 4);
        }
        t[b as usize] = n;
        b += 1;
    }
    t
};

const FOLDED_BOUND: [u64; TABLE] = {
    let mut t = [0u64; TABLE];
    let mut b = 1;
    while b <= LAZY_CAP {
        let mut cur = b;
        while cur > 3 {
            cur = FOLD_BOUND[cur as usize];
        }
        t[b as usize] = cur;
        b += 1;
    }
    t
};

/// `min_bias(1, c)` exceeds `LAZY_CAP` once `c` reaches it, so no bias exists
/// for a total of `LAZY_CAP * p`. Entries at and above the cap stay zero and
/// no harness may read them.
const MIN_BIAS1: [u64; TABLE] = {
    let mut t = [0u64; TABLE];
    let mut c = 1;
    while c < LAZY_CAP {
        t[c as usize] = min_bias(1, c);
        c += 1;
    }
    t
};

fn normalized(l: &[u64; 5]) -> bool {
    l.iter().all(|&limb| limb <= MASK52)
}

/// Limb-wise equality. Array `==` lowers to `memcmp`, which the model checker
/// has no definition for and reports as an unreachable missing function.
fn eq52(a: &[u64; 5], b: &[u64; 5]) -> bool {
    let mut j = 0;
    while j < 5 {
        if a[j] != b[j] {
            return false;
        }
        j += 1;
    }
    true
}

/// Exact sum of two normalized limb vectors. The carry out of limb 4 is
/// returned, never dropped.
fn add_exact(a: &[u64; 5], b: &[u64; 5]) -> ([u64; 5], u64) {
    let mut out = [0u64; 5];
    let mut carry = 0u64;
    for j in 0..5 {
        let v = a[j] as u128 + b[j] as u128 + carry as u128;
        out[j] = (v as u64) & MASK52;
        carry = (v >> 52) as u64;
    }
    (out, carry)
}

/// Per-lane model of the SIMD carry normalization. Asserts that no 64-bit
/// lane wraps and that limb 4 takes no carry out, the two silent-corruption
/// modes of the vector code.
fn carry_normalize_checked(t: [u64; 5]) -> [u64; 5] {
    let mut out = [0u64; 5];
    let mut carry = 0u64;
    for j in 0..4 {
        let v = t[j] as u128 + carry as u128;
        assert!(v < 1u128 << 64, "64-bit lane wraps in carry normalization");
        out[j] = (v as u64) & MASK52;
        carry = (v >> 52) as u64;
    }
    let v = t[4] as u128 + carry as u128;
    assert!(v <= MASK52 as u128, "value escapes five normalized limbs");
    out[4] = v as u64;
    out
}

/// Per-lane model of `fold_254`: `v = q * 2^254 + r` becomes `r + q * eps`.
/// Returns the folded limbs and the removed multiple `q` of `p`.
fn fold_model(v: [u64; 5]) -> ([u64; 5], u64) {
    let q = v[4] >> 46;
    let mut t = v;
    t[4] &= (1 << 46) - 1;
    for j in 0..5 {
        let lo = (q as u128 * EPS52[j] as u128) as u64 & MASK52;
        let sum = t[j] as u128 + lo as u128;
        assert!(sum < 1u128 << 64, "fold low product wraps a 64-bit lane");
        t[j] = sum as u64;
    }
    for j in 0..4 {
        let hi = (q as u128 * EPS52[j] as u128 >> 52) as u64;
        let sum = t[j + 1] as u128 + hi as u128;
        assert!(sum < 1u128 << 64, "fold high product wraps a 64-bit lane");
        t[j + 1] = sum as u64;
    }
    // The vector code has no high-half sink for limb 4. The product must
    // fit 52 bits or the fold silently drops value.
    assert!(q as u128 * EPS52[4] as u128 <= MASK52 as u128);
    (carry_normalize_checked(t), q)
}

/// One conditional subtraction of `p` on normalized limbs.
fn cond_sub_model(t: [u64; 5]) -> ([u64; 5], u64) {
    if lt52(&t, &P52) {
        return (t, 0);
    }
    let mut out = [0u64; 5];
    let mut borrow = 0u64;
    for j in 0..5 {
        let (d, b1) = t[j].overflowing_sub(P52[j]);
        let (d, b2) = d.overflowing_sub(borrow);
        out[j] = d & MASK52;
        borrow = u64::from(b1 | b2 | (d > MASK52));
    }
    assert!(borrow == 0, "subtraction of p borrows although t >= p");
    (out, 1)
}

/// `prev == cur + removed * p` over exact limb arithmetic.
fn assert_removed_multiple(prev: &[u64; 5], cur: &[u64; 5], removed: u64) {
    let (rp, carry) = mul_p52(removed);
    assert!(carry == 0, "removed multiple of p escapes five limbs");
    let (sum, carry) = add_exact(cur, &rp);
    assert!(carry == 0);
    assert!(
        eq52(&sum, prev),
        "fold/reduce step does not preserve the value"
    );
}

fn any_bound(max: u64) -> u64 {
    let b: u64 = kani::any();
    kani::assume(b >= 1 && b <= max);
    b
}

/// Normalized limbs of a value below `bound * p`.
fn any_value_below(bound: u64) -> [u64; 5] {
    let v: [u64; 5] = kani::any();
    kani::assume(normalized(&v));
    let (bp, carry) = mul_p52(bound);
    kani::assume(carry == 0 && lt52(&v, &bp));
    v
}

/// `LAZY_CAP` admissibility: every tracked multiple of `p` fits five
/// normalized limbs, and the largest fold quotient fits six bits.
#[kani::proof]
fn lazy_cap_multiples_fit_five_limbs() {
    let b = any_bound(LAZY_CAP);
    let (limbs, carry) = mul_p52(b);
    assert!(carry == 0);
    assert!(normalized(&limbs));
    assert!(limbs[4] >> 46 < 1 << 6);
    assert!(EPS52[4] << 6 <= MASK52);
}

/// Soundness of the `fold_bound` table: one fold of any value below `b * p`
/// keeps limbs normalized, removes an exact multiple of `p`, and lands below
/// `fold_bound(b) * p`.
#[kani::proof]
#[kani::unwind(8)]
fn fold_bound_is_sound_for_every_bound() {
    let b = any_bound(LAZY_CAP);
    let v = any_value_below(b);
    let (out, q) = fold_model(v);
    assert!(normalized(&out));
    assert!(q < 1 << 6);
    assert_removed_multiple(&v, &out, q);
    let (fbp, carry) = mul_p52(FOLD_BOUND[b as usize]);
    assert!(carry == 0);
    assert!(lt52(&out, &fbp));
}

/// The fold trajectory of every admissible bound reaches at most 3 within
/// four folds, and the table values agree with the recomputation.
#[kani::proof]
#[kani::unwind(8)]
fn fold_trajectory_reaches_three_within_four_folds() {
    let b = any_bound(LAZY_CAP);
    let mut cur = b;
    let mut n = 0u64;
    while cur > 3 {
        cur = FOLD_BOUND[cur as usize];
        n += 1;
        assert!(n <= 4);
        assert!((1..=LAZY_CAP).contains(&cur));
    }
    assert!(n == FOLDS_NEEDED[b as usize]);
    assert!(cur == FOLDED_BOUND[b as usize]);
    assert!(cur <= 3);
}

/// End-to-end `reduce_full` model: `folds_needed(b)` folds then
/// `folded_bound(b) - 1` conditional subtractions canonicalize any value
/// below `b * p` while removing only exact multiples of `p`.
#[kani::proof]
#[kani::unwind(8)]
fn reduce_full_canonicalizes_every_lazy_value() {
    let b = any_bound(LAZY_CAP);
    let mut cur = any_value_below(b);
    let mut bound = b;

    let folds = FOLDS_NEEDED[b as usize];
    let mut i = 0u64;
    while i < folds {
        let prev = cur;
        let (next, q) = fold_model(prev);
        assert_removed_multiple(&prev, &next, q);
        bound = FOLD_BOUND[bound as usize];
        let (bp, carry) = mul_p52(bound);
        assert!(carry == 0 && lt52(&next, &bp));
        cur = next;
        i += 1;
    }
    assert!(bound == FOLDED_BOUND[b as usize] && bound <= 3);

    let mut j = 1u64;
    while j < bound {
        let prev = cur;
        let (next, removed) = cond_sub_model(prev);
        assert_removed_multiple(&prev, &next, removed);
        cur = next;
        j += 1;
    }
    assert!(lt52(&cur, &P52), "reduce_full result is not canonical");
}

/// `add_lazy` bound propagation: the sum of values below `B * p` and
/// `C * p` normalizes without lane wrap and stays below `(B + C) * p`.
#[kani::proof]
fn add_lazy_bound_propagates() {
    let bb = any_bound(LAZY_CAP - 1);
    let cc = any_bound(LAZY_CAP - bb);
    let a = any_value_below(bb);
    let c = any_value_below(cc);

    let mut t = [0u64; 5];
    for j in 0..5 {
        t[j] = a[j] + c[j];
    }
    let out = carry_normalize_checked(t);
    let (sum, carry) = add_exact(&a, &c);
    assert!(carry == 0 && eq52(&sum, &out));
    let (bp, carry) = mul_p52(bb + cc);
    assert!(carry == 0);
    assert!(lt52(&out, &bp));
}

/// `sub_lazy` bias: `a + min_bias(1, C) * p - r` is borrow-free per limb,
/// exact, and below `(B + min_bias(1, C)) * p`.
#[kani::proof]
#[kani::unwind(8)]
fn sub_lazy_bias_is_borrow_free_and_bounded() {
    let cc = any_bound(LAZY_CAP - 1);
    let k = MIN_BIAS1[cc as usize];
    kani::assume(k < LAZY_CAP);
    let bb = any_bound(LAZY_CAP - k);
    let a = any_value_below(bb);
    let r = any_value_below(cc);

    // bias52(k, 1): k * p with 2^52 lent down each limb.
    let (kp, carry) = mul_p52(k);
    assert!(carry == 0 && kp[4] >= 1);
    let bias = [
        kp[0] + (1 << 52),
        kp[1] + (1 << 52) - 1,
        kp[2] + (1 << 52) - 1,
        kp[3] + (1 << 52) - 1,
        kp[4] - 1,
    ];

    let mut t = [0u64; 5];
    for j in 0..5 {
        let v = a[j] as i128 + bias[j] as i128 - r[j] as i128;
        assert!(v >= 0, "biased subtraction borrows in a 64-bit lane");
        assert!(v < 1i128 << 64);
        t[j] = v as u64;
    }
    let out = carry_normalize_checked(t);

    // out + r == a + k * p, so the step removed no value.
    let (lhs, lc) = add_exact(&out, &r);
    let (rhs, rc) = add_exact(&a, &kp);
    assert!(lc == rc && eq52(&lhs, &rhs));
    let (bp, carry) = mul_p52(bb + k);
    assert!(carry == 0);
    assert!(lt52(&out, &bp));
}
