//! Dual-lane Fp2 sums-of-products schedule for AArch64.
//!
//! `narsil_sosd2` returns both Fp halves of `(x0 + x1*u)*(y0 + y1*u)` in one
//! leaf: `lane0 = x0*y0 + x1*(p - y1)` and `lane1 = x0*y1 + x1*y0`, each
//! divided by R and reduced. The two five-word accumulators never leave
//! registers, so the only serial work is the two carry recurrences, and they
//! are independent of each other.
//!
//! Bounds, operands at most p (n = 4 limbs, w = 2^64, R = 2^256,
//! p < 0.1891*2^256, T = 2 products per lane):
//! * between rounds the accumulator is below `(T+1)p = 3p < 2^256`, so its
//!   fifth word is zero;
//! * the in-round peak before the shift is below `3p*w < 2^320`, so five
//!   words hold it and no accumulate row carries out;
//! * the result is below `p*(1 + T*p/R) < 2p`, so one conditional
//!   subtraction per lane is canonical.

use super::machine::Reg::{
    Sp, X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14, X15, X16, X17, X19, X20,
    X21, X22, X23, X24, X25, X26, X27, X28,
};
use super::machine::{MachineA64, Reg};
use super::schedule::column_products;

/// Register roles for `narsil_sosd2`.
pub const SOSD2_REGISTER_MAP: &[(Reg, &str)] = &[
    (X0, "z: eight result limbs, lane0 then lane1"),
    (X1, "x0 pointer"),
    (X2, "x1 pointer"),
    (X3, "y0 pointer"),
    (X4, "y1 pointer"),
    (X5, "consts pointer"),
    (X6, "-p^-1 mod 2^64"),
    (
        X7,
        "x0 limb of this round; then the lane0 Montgomery factor",
    ),
    (
        X8,
        "lane0 accumulator word 0 (shifts down one word per round)",
    ),
    (X9, "lane0 accumulator word 1"),
    (X10, "lane0 accumulator word 2"),
    (X11, "lane0 accumulator word 3"),
    (X12, "lane0 accumulator word 4"),
    (X13, "lane1 accumulator word 0"),
    (X14, "lane1 accumulator word 1"),
    (X15, "lane1 accumulator word 2"),
    (X16, "lane1 accumulator word 3"),
    (X17, "lane1 accumulator word 4"),
    (X19, "column word 0 of the current row; then result scratch"),
    (X20, "column word 1; then result scratch"),
    (X21, "column word 2; then result scratch"),
    (X22, "column word 3; then result scratch"),
    (X23, "column word 4"),
    (X24, "limb 0 of the row multiplicand (y0, p - y1, y1, or p)"),
    (X25, "limb 1 of the row multiplicand"),
    (X26, "limb 2 of the row multiplicand"),
    (X27, "limb 3 of the row multiplicand"),
    (
        X28,
        "x1 limb of this round; then the lane1 Montgomery factor",
    ),
];

/// Lane0 accumulator, `x0*y0 + x1*(p - y1)`.
const T: [Reg; 5] = [X8, X9, X10, X11, X12];
/// Lane1 accumulator, `x0*y1 + x1*y0`.
const U: [Reg; 5] = [X13, X14, X15, X16, X17];
/// Column words of the current row.
const S: [Reg; 5] = [X19, X20, X21, X22, X23];
/// The four limbs of the current row multiplicand.
const W: [Reg; 4] = [X24, X25, X26, X27];
const Z: Reg = X0;
const X0P: Reg = X1;
const X1P: Reg = X2;
const Y0P: Reg = X3;
const Y1P: Reg = X4;
const CONSTS: Reg = X5;
const PINV: Reg = X6;
/// Per-round scratch: the loaded `x0` limb, then the lane0 factor `m`.
const VA: Reg = X7;
/// Per-round scratch: the loaded `x1` limb, then the lane1 factor `m`.
const VB: Reg = X28;

/// Ten callee-saved slots plus the `p - y1` image, 16-byte aligned.
const FRAME: i32 = 112;
/// Frame offset of the `p - y1` image.
const NY1: i32 = 80;

/// `narsil_sosd2`: four dual-lane CIOS rounds, fully unrolled.
pub fn sosd2<M: MachineA64>(m: &mut M) {
    m.stp_pre(X19, X20, -FRAME);
    m.stp(X21, X22, Sp, 16, "");
    m.stp(X23, X24, Sp, 32, "");
    m.stp(X25, X26, Sp, 48, "");
    m.stp(X27, X28, Sp, 64, "");

    m.comment("");
    m.comment("p - y1 turns the subtracted lane0 term into a positive one,");
    m.comment("so every partial product below is a plain addition");
    m.ldr(PINV, CONSTS, 32, "-p^-1 mod 2^64");
    load_row(m, CONSTS, 0, "p0, p1");
    m.ldp(S[0], S[1], Y1P, 0, "y1 limbs 0, 1");
    m.ldp(S[2], S[3], Y1P, 16, "y1 limbs 2, 3");
    m.subs(W[0], W[0], S[0], "word 0 of p - y1");
    m.sbcs(W[1], W[1], S[1], "");
    m.sbcs(W[2], W[2], S[2], "");
    m.sbcs(W[3], W[3], S[3], "y1 <= p, so this cannot borrow out");
    m.stp(W[0], W[1], Sp, NY1, "");
    m.stp(W[2], W[3], Sp, NY1 + 16, "");

    // Unrolled, and the two lanes alternate row by row, so both carry
    // recurrences stay in flight through the whole kernel.
    for round in 0..4i32 {
        let limb = 8 * round;
        m.comment("");
        m.comment(&format!("round {round}: source limb {round} of x0 and x1"));
        m.ldr(VA, X0P, limb, "x0 limb");
        m.ldr(VB, X1P, limb, "x1 limb");
        load_row(m, Y0P, 0, "y0 limbs 0, 1");
        if round == 0 {
            // The first row of each lane writes the whole accumulator, so
            // neither lane needs zeroing and neither pays an accumulate row.
            column_products(m, T, VA, W, "y0_", "x0j");
            column_products(m, U, VB, W, "y0_", "x1j");
        } else {
            product_row(m, T, VA, "y0_", "x0j");
            product_row(m, U, VB, "y0_", "x1j");
        }
        load_row(m, Sp, NY1, "p - y1, limbs 0, 1");
        product_row(m, T, VB, "ny1_", "x1j");
        load_row(m, Y1P, 0, "y1 limbs 0, 1");
        product_row(m, U, VA, "y1_", "x0j");
        load_row(m, CONSTS, 0, "p0, p1");
        reduction_row(m, T, VA, round == 3);
        reduction_row(m, U, VB, round == 3);
    }

    m.comment("");
    m.comment("the row multiplicand still holds p, so the two conditional");
    m.comment("subtractions need no reload");
    final_reduce(m, T, 0);
    final_reduce(m, U, 32);

    m.comment("");
    m.ldp(X21, X22, Sp, 16, "");
    m.ldp(X23, X24, Sp, 32, "");
    m.ldp(X25, X26, Sp, 48, "");
    m.ldp(X27, X28, Sp, 64, "");
    m.ldp_post(X19, X20, Sp, FRAME);
    m.ret();
}

/// Load the four limbs of the next row multiplicand.
fn load_row<M: MachineA64>(m: &mut M, base: Reg, offset: i32, what: &str) {
    m.ldp(W[0], W[1], base, offset, what);
    m.ldp(W[2], W[3], base, offset + 16, "");
}

/// `acc += v * W`, five words, one column chain and one accumulate chain.
fn product_row<M: MachineA64>(m: &mut M, acc: [Reg; 5], v: Reg, limb: &str, factor: &str) {
    column_products(m, S, v, W, limb, factor);
    m.adds(acc[0], acc[0], S[0], "opens the accumulate chain");
    m.adcs(acc[1], acc[1], S[1], "");
    m.adcs(acc[2], acc[2], S[2], "");
    m.adcs(acc[3], acc[3], S[3], "");
    m.adc(acc[4], acc[4], S[4], "in-round peak < 3p*2^64 < 2^320");
}

/// `acc = (acc + m*p) / 2^64` with `m = acc0 * -p^-1`. W holds p. `last`
/// drops the fifth word the next round would have consumed.
fn reduction_row<M: MachineA64>(m: &mut M, acc: [Reg; 5], factor: Reg, last: bool) {
    m.mul(factor, PINV, acc[0], "m cancels word 0 of the accumulator");
    column_products(m, S, factor, W, "p", "m");
    m.cmn(acc[0], S[0], "word 0 is 0 mod 2^64; keep only its carry");
    m.adcs(
        acc[0],
        acc[1],
        S[1],
        "shift down one word: the division by R",
    );
    m.adcs(acc[1], acc[2], S[2], "");
    m.adcs(acc[2], acc[3], S[3], "");
    if last {
        m.adc(acc[3], acc[4], S[4], "after four rounds u < 2p < 2^256");
    } else {
        m.adcs(acc[3], acc[4], S[4], "");
        m.cset_hs(acc[4], "");
        m.claim_zero(acc[4], "between rounds u < 3p < 2^256");
    }
}

/// One branch-free conditional subtraction, then the four output limbs.
fn final_reduce<M: MachineA64>(m: &mut M, acc: [Reg; 5], offset: i32) {
    m.subs(S[0], acc[0], W[0], "word 0 of u - p");
    m.sbcs(S[1], acc[1], W[1], "");
    m.sbcs(S[2], acc[2], W[2], "");
    m.sbcs(S[3], acc[3], W[3], "C = (u >= p)");
    m.csel_hs(acc[0], S[0], acc[0], "no borrow: keep u - p");
    m.csel_hs(acc[1], S[1], acc[1], "");
    m.csel_hs(acc[2], S[2], acc[2], "");
    m.csel_hs(acc[3], S[3], acc[3], "");
    m.stp(acc[0], acc[1], Z, offset, "");
    m.stp(acc[2], acc[3], Z, offset + 16, "");
}
