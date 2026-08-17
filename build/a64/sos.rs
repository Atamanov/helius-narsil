//! Dual-lane Fp2 sums-of-products schedules for AArch64.
//!
//! `narsil_sosd2` returns both Fp halves of `(x0 + x1*u)*(y0 + y1*u)` and
//! `narsil_sosd6` both halves of a sum of three such products, each divided by
//! R and reduced. Every accumulator stays in registers, so the only serial
//! work is the two carry recurrences, and they are independent of each other.
//!
//! Bounds, operands at most p (n = 4 limbs, w = 2^64, R = 2^256,
//! p < 0.1891*2^256, T products per lane):
//! * between rounds the accumulator is below `(T+1)p`, which is `3p < 2^256`
//!   for T = 2 and `7p < 2^257` for T = 6;
//! * the in-round peak before the shift is below `(T+1)p*w`, which is
//!   `< 2^320` for T = 2 and `< 2^321` for T = 6, so the six-word accumulator
//!   of the larger kernel is what keeps the accumulate rows from carrying out;
//! * the result is below `p*(1 + T*p/R)`, so T = 2 needs one conditional
//!   subtraction per lane and T = 6 needs two.

use super::machine::Reg::{
    Sp, X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14, X15, X16, X17, X19, X20,
    X21, X22, X23, X24, X25, X26, X27, X28, Xzr,
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

/// Register roles for `narsil_sosd6`.
pub const SOSD6_REGISTER_MAP: &[(Reg, &str)] = &[
    (X0, "z: eight result limbs, lane0 then lane1"),
    (X1, "table: twelve operand pointers"),
    (X2, "consts pointer"),
    (X3, "-p^-1 mod 2^64"),
    (X4, "operand pointer read out of the table"),
    (
        X5,
        "x_i0 limb of this round; then the lane0 Montgomery factor",
    ),
    (
        X6,
        "x_i1 limb of this round; then the lane1 Montgomery factor",
    ),
    (
        X7,
        "lane0 accumulator word 0 (shifts down one word per round)",
    ),
    (X8, "lane0 accumulator word 1"),
    (X9, "lane0 accumulator word 2"),
    (X10, "lane0 accumulator word 3"),
    (X11, "lane0 accumulator word 4"),
    (X12, "lane0 accumulator word 5"),
    (X13, "lane1 accumulator word 0"),
    (X14, "lane1 accumulator word 1"),
    (X15, "lane1 accumulator word 2"),
    (X16, "lane1 accumulator word 3"),
    (X17, "lane1 accumulator word 4"),
    (X19, "lane1 accumulator word 5"),
    (X20, "column word 0 of the current row; then result scratch"),
    (X21, "column word 1; then result scratch"),
    (X22, "column word 2; then result scratch"),
    (X23, "column word 3; then result scratch"),
    (X24, "column word 4"),
    (
        X25,
        "limb 0 of the row multiplicand (y_i0, p - y_i1, y_i1, or p)",
    ),
    (X26, "limb 1 of the row multiplicand"),
    (X27, "limb 2 of the row multiplicand"),
    (X28, "limb 3 of the row multiplicand"),
];

/// The banks every dual-lane row works through, plus the reduction constant.
/// Both kernels drive the same row primitives through this one description.
struct Banks {
    /// Column words of the current row.
    s: [Reg; 5],
    /// The four limbs of the current row multiplicand.
    w: [Reg; 4],
    /// `-p^-1 mod 2^64`, on the critical path from word 0 to the factor.
    pinv: Reg,
}

impl Banks {
    /// Load the four limbs of the next row multiplicand.
    fn load_row<M: MachineA64>(&self, m: &mut M, base: Reg, offset: i32, what: &str) {
        m.ldp(self.w[0], self.w[1], base, offset, what);
        m.ldp(self.w[2], self.w[3], base, offset + 16, "");
    }

    /// `acc = v * W`: the first row of a lane writes the accumulator instead
    /// of adding to it, so no lane pays a zeroing pass or an accumulate chain.
    fn open<M: MachineA64>(&self, m: &mut M, acc: &[Reg], v: Reg, limb: &str, factor: &str) {
        let columns = [acc[0], acc[1], acc[2], acc[3], acc[4]];
        column_products(m, columns, v, self.w, limb, factor);
    }

    /// `acc += v * W`, one column chain then one accumulate chain. `carries`
    /// opens the top word from the chain carry: only the row that can push
    /// the accumulator past 2^320 pays for it, and the ADC on every other row
    /// is what the interpreter checks that claim against.
    fn product<M: MachineA64>(
        &self,
        m: &mut M,
        acc: &[Reg],
        v: Reg,
        limb: &str,
        factor: &str,
        carries: bool,
    ) {
        column_products(m, self.s, v, self.w, limb, factor);
        m.adds(acc[0], acc[0], self.s[0], "opens the accumulate chain");
        m.adcs(acc[1], acc[1], self.s[1], "");
        m.adcs(acc[2], acc[2], self.s[2], "");
        m.adcs(acc[3], acc[3], self.s[3], "");
        if carries {
            m.adcs(acc[4], acc[4], self.s[4], "");
            m.cset_hs(acc[5], "six rows of p*2^64 exceed 2^320, five do not");
        } else {
            m.adc(acc[4], acc[4], self.s[4], "row peak < 2^320");
        }
    }

    /// `acc = (acc + m*p) / 2^64` with `m = acc0 * -p^-1`. W holds p. `last`
    /// drops the carry word the next round would have consumed.
    fn reduce<M: MachineA64>(&self, m: &mut M, acc: &[Reg], factor: Reg, last: bool) {
        m.mul(
            factor,
            self.pinv,
            acc[0],
            "m cancels word 0 of the accumulator",
        );
        column_products(m, self.s, factor, self.w, "p", "m");
        m.cmn(
            acc[0],
            self.s[0],
            "word 0 is 0 mod 2^64; keep only its carry",
        );
        m.adcs(
            acc[0],
            acc[1],
            self.s[1],
            "shift down one word: the division by R",
        );
        m.adcs(acc[1], acc[2], self.s[2], "");
        m.adcs(acc[2], acc[3], self.s[3], "");
        match (acc.get(5), last) {
            (None, true) => m.adc(
                acc[3],
                acc[4],
                self.s[4],
                "after four rounds u < 2p < 2^256",
            ),
            (None, false) => {
                m.adcs(acc[3], acc[4], self.s[4], "");
                m.cset_hs(acc[4], "");
                m.claim_zero(acc[4], "between rounds u < 3p < 2^256");
            }
            (Some(&top), true) => {
                m.adcs(acc[3], acc[4], self.s[4], "");
                m.adc(acc[4], top, Xzr, "in-round peak < 7p*2^64 < 2^321");
                m.claim_zero(acc[4], "after four rounds u < 2.14p < 2^256");
            }
            (Some(&top), false) => {
                m.adcs(acc[3], acc[4], self.s[4], "");
                m.adc(acc[4], top, Xzr, "in-round peak < 7p*2^64 < 2^321");
            }
        }
    }

    /// Branch-free conditional subtractions. W holds p. One reaches [0, p)
    /// from below 2p, two from below 3p.
    fn canonicalize<M: MachineA64>(&self, m: &mut M, acc: &[Reg], subtractions: usize) {
        for _ in 0..subtractions {
            m.subs(self.s[0], acc[0], self.w[0], "word 0 of u - p");
            m.sbcs(self.s[1], acc[1], self.w[1], "");
            m.sbcs(self.s[2], acc[2], self.w[2], "");
            m.sbcs(self.s[3], acc[3], self.w[3], "C = (u >= p)");
            m.csel_hs(acc[0], self.s[0], acc[0], "no borrow: keep u - p");
            m.csel_hs(acc[1], self.s[1], acc[1], "");
            m.csel_hs(acc[2], self.s[2], acc[2], "");
            m.csel_hs(acc[3], self.s[3], acc[3], "");
        }
    }

    /// The four canonical limbs of one lane.
    fn store_lane<M: MachineA64>(&self, m: &mut M, acc: &[Reg], z: Reg, offset: i32) {
        m.stp(acc[0], acc[1], z, offset, "");
        m.stp(acc[2], acc[3], z, offset + 16, "");
    }
}

/// Spill the ten callee-saved registers both kernels use.
fn spill_callee_saved<M: MachineA64>(m: &mut M, frame: i32) {
    m.stp_pre(X19, X20, -frame);
    m.stp(X21, X22, Sp, 16, "");
    m.stp(X23, X24, Sp, 32, "");
    m.stp(X25, X26, Sp, 48, "");
    m.stp(X27, X28, Sp, 64, "");
}

fn restore_callee_saved<M: MachineA64>(m: &mut M, frame: i32) {
    m.ldp(X21, X22, Sp, 16, "");
    m.ldp(X23, X24, Sp, 32, "");
    m.ldp(X25, X26, Sp, 48, "");
    m.ldp(X27, X28, Sp, 64, "");
    m.ldp_post(X19, X20, Sp, frame);
    m.ret();
}

/// Lane0 accumulator of `narsil_sosd2`, `x0*y0 + x1*(p - y1)`.
const T2: [Reg; 5] = [X8, X9, X10, X11, X12];
/// Lane1 accumulator of `narsil_sosd2`, `x0*y1 + x1*y0`.
const U2: [Reg; 5] = [X13, X14, X15, X16, X17];
const BANKS2: Banks = Banks {
    s: [X19, X20, X21, X22, X23],
    w: [X24, X25, X26, X27],
    pinv: X6,
};
const Z2: Reg = X0;
const X0P: Reg = X1;
const X1P: Reg = X2;
const Y0P: Reg = X3;
const Y1P: Reg = X4;
const CONSTS2: Reg = X5;
/// Per-round scratch: the loaded `x0` limb, then the lane0 factor `m`.
const VA2: Reg = X7;
/// Per-round scratch: the loaded `x1` limb, then the lane1 factor `m`.
const VB2: Reg = X28;

/// Ten callee-saved slots plus the `p - y1` image, 16-byte aligned.
const FRAME2: i32 = 112;
/// Frame offset of the `p - y1` image.
const NY1: i32 = 80;

/// `narsil_sosd2`: four dual-lane CIOS rounds, fully unrolled.
pub fn sosd2<M: MachineA64>(m: &mut M) {
    spill_callee_saved(m, FRAME2);

    m.comment("");
    m.comment("p - y1 turns the subtracted lane0 term into a positive one,");
    m.comment("so every partial product below is a plain addition");
    m.ldr(BANKS2.pinv, CONSTS2, 32, "-p^-1 mod 2^64");
    BANKS2.load_row(m, CONSTS2, 0, "p0, p1");
    m.ldp(BANKS2.s[0], BANKS2.s[1], Y1P, 0, "y1 limbs 0, 1");
    m.ldp(BANKS2.s[2], BANKS2.s[3], Y1P, 16, "y1 limbs 2, 3");
    negp_image(m, &BANKS2, NY1, "y1");

    // Unrolled, and the two lanes alternate row by row, so both carry
    // recurrences stay in flight through the whole kernel.
    for round in 0..4i32 {
        let limb = 8 * round;
        m.comment("");
        m.comment(&format!("round {round}: source limb {round} of x0 and x1"));
        m.ldr(VA2, X0P, limb, "x0 limb");
        m.ldr(VB2, X1P, limb, "x1 limb");
        BANKS2.load_row(m, Y0P, 0, "y0 limbs 0, 1");
        if round == 0 {
            BANKS2.open(m, &T2, VA2, "y0_", "x0j");
            BANKS2.open(m, &U2, VB2, "y0_", "x1j");
        } else {
            BANKS2.product(m, &T2, VA2, "y0_", "x0j", false);
            BANKS2.product(m, &U2, VB2, "y0_", "x1j", false);
        }
        BANKS2.load_row(m, Sp, NY1, "p - y1, limbs 0, 1");
        BANKS2.product(m, &T2, VB2, "ny1_", "x1j", false);
        BANKS2.load_row(m, Y1P, 0, "y1 limbs 0, 1");
        BANKS2.product(m, &U2, VA2, "y1_", "x0j", false);
        BANKS2.load_row(m, CONSTS2, 0, "p0, p1");
        BANKS2.reduce(m, &T2, VA2, round == 3);
        BANKS2.reduce(m, &U2, VB2, round == 3);
    }

    m.comment("");
    m.comment("the row multiplicand still holds p, so the two conditional");
    m.comment("subtractions need no reload");
    BANKS2.canonicalize(m, &T2, 1);
    BANKS2.store_lane(m, &T2, Z2, 0);
    BANKS2.canonicalize(m, &U2, 1);
    BANKS2.store_lane(m, &U2, Z2, 32);

    m.comment("");
    restore_callee_saved(m, FRAME2);
}

/// Lane0 accumulator of `narsil_sosd6`, `sum_i x_i0*y_i0 + x_i1*(p - y_i1)`.
const T6: [Reg; 6] = [X7, X8, X9, X10, X11, X12];
/// Lane1 accumulator of `narsil_sosd6`, `sum_i x_i0*y_i1 + x_i1*y_i0`.
const U6: [Reg; 6] = [X13, X14, X15, X16, X17, X19];
const BANKS6: Banks = Banks {
    s: [X20, X21, X22, X23, X24],
    w: [X25, X26, X27, X28],
    pinv: X3,
};
const Z6: Reg = X0;
const TABLE: Reg = X1;
const CONSTS6: Reg = X2;
/// Scratch for one operand pointer read out of the table.
const PTR: Reg = X4;
/// Per-round scratch: the loaded `x_i0` limb, then the lane0 factor `m`.
const VA6: Reg = X5;
/// Per-round scratch: the loaded `x_i1` limb, then the lane1 factor `m`.
const VB6: Reg = X6;

/// Ten callee-saved slots plus the three `p - y_i1` images, 16-byte aligned.
const FRAME6: i32 = 176;
/// Frame offset of the `p - y_i1` image of product `i`.
const fn image(product: i32) -> i32 {
    80 + 32 * product
}

/// Table slot of operand `part` (0 = x_i0, 1 = x_i1, 2 = y_i0, 3 = y_i1) of
/// product `i`, in bytes.
const fn slot(product: i32, part: i32) -> i32 {
    8 * (4 * product + part)
}

/// `narsil_sosd6`: four dual-lane CIOS rounds over three Fp2 products, fully
/// unrolled. Six accumulator words per lane, two conditional subtractions.
pub fn sosd6<M: MachineA64>(m: &mut M) {
    spill_callee_saved(m, FRAME6);

    m.comment("");
    m.comment("p - y_i1 turns each subtracted lane0 term into a positive one,");
    m.comment("so every partial product below is a plain addition");
    m.ldr(BANKS6.pinv, CONSTS6, 32, "-p^-1 mod 2^64");
    for product in 0..3i32 {
        BANKS6.load_row(m, CONSTS6, 0, "p0, p1");
        m.ldr(PTR, TABLE, slot(product, 3), "y_i1 pointer");
        m.ldp(BANKS6.s[0], BANKS6.s[1], PTR, 0, "y_i1 limbs 0, 1");
        m.ldp(BANKS6.s[2], BANKS6.s[3], PTR, 16, "y_i1 limbs 2, 3");
        negp_image(m, &BANKS6, image(product), "y_i1");
    }

    // The two lanes alternate row by row, so both carry recurrences stay in
    // flight through the whole kernel.
    for round in 0..4i32 {
        let limb = 8 * round;
        for product in 0..3i32 {
            // The last product contributes each lane's sixth and final row.
            let sixth = product == 2;
            m.comment("");
            m.comment(&format!("round {round}, product {product}"));
            m.ldr(PTR, TABLE, slot(product, 0), "x_i0 pointer");
            m.ldr(VA6, PTR, limb, "x_i0 limb");
            m.ldr(PTR, TABLE, slot(product, 1), "x_i1 pointer");
            m.ldr(VB6, PTR, limb, "x_i1 limb");
            m.ldr(PTR, TABLE, slot(product, 2), "y_i0 pointer");
            BANKS6.load_row(m, PTR, 0, "y_i0 limbs 0, 1");
            if round == 0 && product == 0 {
                BANKS6.open(m, &T6, VA6, "y_i0_", "x_i0j");
                BANKS6.open(m, &U6, VB6, "y_i0_", "x_i1j");
            } else {
                BANKS6.product(m, &T6, VA6, "y_i0_", "x_i0j", false);
                BANKS6.product(m, &U6, VB6, "y_i0_", "x_i1j", false);
            }
            BANKS6.load_row(m, Sp, image(product), "p - y_i1, limbs 0, 1");
            BANKS6.product(m, &T6, VB6, "ny_i1_", "x_i1j", sixth);
            m.ldr(PTR, TABLE, slot(product, 3), "y_i1 pointer");
            BANKS6.load_row(m, PTR, 0, "y_i1 limbs 0, 1");
            BANKS6.product(m, &U6, VA6, "y_i1_", "x_i0j", sixth);
        }
        m.comment("");
        BANKS6.load_row(m, CONSTS6, 0, "p0, p1");
        BANKS6.reduce(m, &T6, VA6, round == 3);
        BANKS6.reduce(m, &U6, VB6, round == 3);
    }

    m.comment("");
    m.comment("the row multiplicand still holds p, so the four conditional");
    m.comment("subtractions need no reload");
    BANKS6.canonicalize(m, &T6, 2);
    BANKS6.store_lane(m, &T6, Z6, 0);
    BANKS6.canonicalize(m, &U6, 2);
    BANKS6.store_lane(m, &U6, Z6, 32);

    m.comment("");
    restore_callee_saved(m, FRAME6);
}

/// `W = p - S`, stored to the frame. W holds p and S holds the subtrahend.
fn negp_image<M: MachineA64>(m: &mut M, banks: &Banks, offset: i32, what: &str) {
    m.subs(
        banks.w[0],
        banks.w[0],
        banks.s[0],
        &format!("word 0 of p - {what}"),
    );
    m.sbcs(banks.w[1], banks.w[1], banks.s[1], "");
    m.sbcs(banks.w[2], banks.w[2], banks.s[2], "");
    m.sbcs(
        banks.w[3],
        banks.w[3],
        banks.s[3],
        &format!("{what} <= p, so this cannot borrow out"),
    );
    m.stp(banks.w[0], banks.w[1], Sp, offset, "");
    m.stp(banks.w[2], banks.w[3], Sp, offset + 16, "");
}
