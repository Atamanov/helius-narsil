//! Fused cyclotomic-square leaves for AArch64.
//!
//! `narsil_cyc_fp4` squares one Fp4 element `(r0 + r1*y)^2` with
//! `y^2 = xi = 9 + u`, and `narsil_cyc_fold` turns the six Fp4 halves into
//! the six Granger-Scott outputs. Three calls of the first and one of the
//! second replace the composed route's six dual-lane dispatches plus its
//! thirty-six single-width modular folds.
//!
//! Operand prep is lazy. A CIOS row reads its limb source one word at a
//! time, so that side only has to be below `2^256`, while the row
//! multiplicand enters the accumulator bound and must stay at most `p`.
//! Putting every scaling on the source side is what removes the folds.
//!
//! Bounds (BN254: `p < 0.1891*2^256 = 0.1891*R`, four limbs, T rows per
//! lane, all multiplicands at most p):
//! * the accumulator between rounds is below `(T+1)p` and the in-round peak
//!   below `(T+1)p*2^64 < 2^320`, both independent of the limb sources,
//!   which are single words by construction;
//! * after four rounds the lane holds `(sum of src*mult)/R + m*p/R`, so a
//!   lane whose sources are bounded by `S_i` ends below
//!   `p + p*sum(S_i)/R`. The conditional-subtraction count follows.
//!
//! The Fp4 leaf draws the first product from the complex identity
//! `(a + b*u)^2 = (a + b)(a - b) + (2a)b*u`, so the r0 square and the
//! xi-scaled r1 square cost three rows per lane, not four.

use super::machine::Reg::{
    Sp, X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14, X15, X16, X17, X19, X20,
    X21, X22, X23, X24, X25, X26, X27, X28, Xzr,
};
use super::machine::{MachineA64, Reg};
use super::sos::{Banks, restore_callee_saved, spill_callee_saved};

/// Register roles for `narsil_cyc_fp4`.
pub const CYC_FP4_REGISTER_MAP: &[(Reg, &str)] = &[
    (X0, "z: four result limb sets, t_a lane0 first"),
    (X1, "r0 pointer (two Fp halves)"),
    (X2, "r1 pointer"),
    (X3, "consts pointer, reread once per round for p"),
    (X4, "-p^-1 mod 2^64; a prep bank word before that"),
    (
        X5,
        "lane0 limb of this round; then the lane0 Montgomery factor",
    ),
    (
        X6,
        "lane1 limb of this round; then the lane1 Montgomery factor",
    ),
    (
        X7,
        "lane0 accumulator word 0 (shifts down one word per round)",
    ),
    (X8, "lane0 accumulator word 1"),
    (X9, "lane0 accumulator word 2"),
    (X10, "lane0 accumulator word 3"),
    (X11, "lane0 accumulator word 4"),
    (X12, "lane1 accumulator word 0"),
    (X13, "lane1 accumulator word 1"),
    (X14, "lane1 accumulator word 2"),
    (X15, "lane1 accumulator word 3"),
    (X16, "lane1 accumulator word 4"),
    (X17, "column word 0 of the current row; then result scratch"),
    (X19, "column word 1; then result scratch"),
    (X20, "column word 2; then result scratch"),
    (X21, "column word 3; then result scratch"),
    (X22, "column word 4"),
    (
        X23,
        "limb 0 of the row multiplicand; p during the reduction rows",
    ),
    (X24, "limb 1 of the row multiplicand"),
    (X25, "limb 2 of the row multiplicand"),
    (X26, "limb 3 of the row multiplicand"),
    (X27, "prep bank word; unused once the rows start"),
    (X28, "prep bank word; unused once the rows start"),
];

/// Register roles for `narsil_cyc_fold`.
pub const CYC_FOLD_REGISTER_MAP: &[(Reg, &str)] = &[
    (X0, "z: twelve result limb sets in repr(C) Fp12 order"),
    (X1, "t pointer: the six Fp4 halves"),
    (X2, "f pointer: the repr(C) Fp12 input"),
    (X3, "consts pointer"),
    (X4, "p limb 0"),
    (X5, "p limb 1"),
    (X6, "p limb 2"),
    (X7, "p limb 3"),
    (X8, "lane0 t half, then the lane0 combine sum"),
    (X9, "lane0 word 1"),
    (X10, "lane0 word 2"),
    (X11, "lane0 word 3"),
    (X12, "lane0 f half, then the lane0 inner fold"),
    (X13, "lane0 word 1"),
    (X14, "lane0 word 2"),
    (X15, "lane0 word 3"),
    (X16, "shared conditional-term scratch word 0"),
    (X17, "scratch word 1"),
    (X19, "scratch word 2"),
    (X20, "scratch word 3"),
    (X21, "lane1 t half, then the lane1 combine sum"),
    (X22, "lane1 word 1"),
    (X23, "lane1 word 2"),
    (X24, "lane1 word 3"),
    (X25, "lane1 f half, then the lane1 inner fold"),
    (X26, "lane1 word 1"),
    (X27, "lane1 word 2"),
    (X28, "lane1 word 3"),
];

/// A four-limb value held in registers.
type Bank = [Reg; 4];

const Z: Reg = X0;
const R0P: Reg = X1;
const R1P: Reg = X2;
const CONSTS: Reg = X3;
const PINV: Reg = X4;
/// Per-round scratch: the lane0 limb source word, then the lane0 factor.
const VA: Reg = X5;
/// Per-round scratch: the lane1 limb source word, then the lane1 factor.
const VB: Reg = X6;
const LANE0: [Reg; 5] = [X7, X8, X9, X10, X11];
const LANE1: [Reg; 5] = [X12, X13, X14, X15, X16];
const COLUMN: [Reg; 5] = [X17, X19, X20, X21, X22];
const ROW: Bank = [X23, X24, X25, X26];

/// Prep banks. Every register the rows later claim is free here, so the
/// whole modular prologue runs without touching the stack for scratch.
const PB: Bank = [X23, X24, X25, X26];
const PV: Bank = [X7, X8, X9, X10];
const PW: Bank = [X12, X13, X14, X15];
const PS: Bank = [X17, X19, X20, X21];
const PT: Bank = [X4, X5, X6, X11];
const PU: Bank = [X16, X22, X27, X28];

/// Frame: the five spilled callee-saved pairs, then twelve staged operands.
const SAVED_PAIRS: usize = 5;
const SAVES: i32 = 16 * SAVED_PAIRS as i32;
const SLOT: i32 = 32;
const SLOTS: i32 = 12;
const FP4_FRAME: i32 = SAVES + SLOT * SLOTS;

/// Frame offset of staged operand `index`.
const fn slot(index: i32) -> i32 {
    SAVES + SLOT * index
}

/// `p0 + p1`, below 2p: the lane0 source of the r0 square.
const S_SUM0: i32 = 0;
/// `(p0 - p1) mod p`: its multiplicand.
const M_DIF0: i32 = 1;
/// `2*p0`, below 2p: source of the lane1 cross term and of both t_b rows.
const S_DBL0: i32 = 2;
/// `2*p1`, below 2p: the second t_b source.
const S_DBL1: i32 = 3;
/// `q0 + q1`, below 2p: the lane1 source of the r1 square.
const S_SUM1: i32 = 4;
/// `(q0 + q1) mod p`: the lane0 multiplicand of the nine-scaled r1 square.
const M_SUM1: i32 = 5;
/// `(q0 - q1) mod p`: the lane1 multiplicand of the r1 square.
const M_DIF1: i32 = 6;
/// `9*(q0 - q1)` below 5p: the lane0 source carrying the xi real part.
const S_NINE0: i32 = 7;
/// `2*q0`, below 2p: the lane0 source of the crossed r1 term.
const S_DBLQ: i32 = 8;
/// `(2*q0) mod p`: the lane1 multiplicand of the nine-scaled cross term.
const M_DBLQ: i32 = 9;
/// `9*q1` below 5p: the lane1 source carrying the xi imaginary part.
const S_NINE1: i32 = 10;
/// `p - q1`: the multiplicand that turns the subtracted terms positive.
const M_NEGQ: i32 = 11;

/// Three rows per lane: sources below 5p, 2p and 2p keep the lane under
/// `p + 9p^2/R < 2.71p`, which two conditional subtractions canonicalize.
const BANKS_A: Banks = Banks {
    s: COLUMN,
    w: ROW,
    pinv: PINV,
    terms: 3,
    result: "after four rounds u < 2.71p < 3p < 2^256",
};

/// Two rows per lane, both sources below 2p: the lane ends under
/// `p + 4p^2/R < 1.76p`, one conditional subtraction.
const BANKS_B: Banks = Banks {
    s: COLUMN,
    w: ROW,
    pinv: PINV,
    terms: 2,
    result: "after four rounds u < 1.76p < 2p < 2^256",
};

/// `narsil_cyc_fp4`: one Fp4 square, prep and both CIOS blocks in one leaf.
pub fn cyc_fp4<M: MachineA64>(m: &mut M) {
    spill_callee_saved(m, FP4_FRAME, SAVED_PAIRS);
    prep(m);
    m.comment("");
    m.ldr(PINV, CONSTS, 32, "-p^-1 mod 2^64");
    block(m, &BANKS_A, &ROWS_A, "t_a", 2, 0);
    block(m, &BANKS_B, &ROWS_B, "t_b", 1, 64);
    m.comment("");
    restore_callee_saved(m, FP4_FRAME, SAVED_PAIRS);
}

/// Stage every row operand. Scalings ride the limb-source side, which only
/// has to stay below 2^256, so nine of the twelve need no conditional fold.
fn prep<M: MachineA64>(m: &mut M) {
    m.comment("");
    m.comment("prep: every row source and multiplicand, staged once");
    load4(m, PB, CONSTS, 0, "p");
    load4(m, PV, R0P, 0, "p0");
    load4(m, PW, R0P, 32, "p1");

    double4(m, PS, PV, "2*p0 < 2p");
    store4(m, PS, S_DBL0);
    double4(m, PS, PW, "2*p1 < 2p");
    store4(m, PS, S_DBL1);
    add4(m, PS, PV, PW, "p0 + p1 < 2p");
    store4(m, PS, S_SUM0);
    sub_mod4(m, PS, PV, PW, PB, PT, "p0 - p1");
    store4(m, PS, M_DIF0);

    load4(m, PV, R1P, 0, "q0");
    load4(m, PW, R1P, 32, "q1");

    double4(m, PS, PV, "2*q0 < 2p");
    store4(m, PS, S_DBLQ);
    cond_sub(m, PS, PB, PT, "2*q0 mod p");
    store4(m, PS, M_DBLQ);
    add4(m, PS, PV, PW, "q0 + q1 < 2p");
    store4(m, PS, S_SUM1);
    cond_sub(m, PS, PB, PT, "(q0 + q1) mod p");
    store4(m, PS, M_SUM1);
    negp4(m, PS, PB, PW, "q1");
    store4(m, PS, M_NEGQ);
    sub_mod4(m, PS, PV, PW, PB, PT, "q0 - q1");
    store4(m, PS, M_DIF1);

    nine4(m, PT, PU, PS, "q0 - q1");
    store4(m, PU, S_NINE0);
    nine4(m, PT, PU, PW, "q1");
    store4(m, PU, S_NINE1);
}

/// One CIOS row: which lane it accumulates into, the register that carries
/// its limb this round and its Montgomery factor next, the staged limb
/// source, and where the four-limb multiplicand sits.
struct Row {
    lane: [Reg; 5],
    scratch: Reg,
    source: i32,
    base: Reg,
    offset: i32,
    /// The two operand names, for the emitted column comments.
    what: (&'static str, &'static str),
}

impl Row {
    /// The first row of a lane writes the accumulator instead of adding to
    /// it, so no lane pays a zeroing pass. `source_held` and `row_held` skip
    /// a reload of a word an earlier row of the same round already placed.
    fn emit<M: MachineA64>(
        &self,
        m: &mut M,
        banks: &Banks,
        limb: i32,
        open: bool,
        source_held: bool,
        row_held: bool,
    ) {
        if !source_held {
            m.ldr(self.scratch, Sp, slot(self.source) + limb, "source limb");
        }
        if !row_held {
            banks.load_row(m, self.base, self.offset, "multiplicand limbs 0, 1");
        }
        let (multiplicand, source) = self.what;
        if open {
            banks.open(m, &self.lane, self.scratch, multiplicand, source);
        } else {
            banks.product(m, &self.lane, self.scratch, multiplicand, source, false);
        }
    }
}

/// `t_a = r0^2 + xi*r1^2`.
///
/// Lane0 is `(p0+p1)(p0-p1) + 9(q0-q1)(q0+q1) + (2q0)(p-q1)`, which is
/// `p0^2 - p1^2 + 9c - d` for `c = q0^2 - q1^2` and `d = 2*q0*q1`. Lane1 is
/// `(2p0)p1 + (q0+q1)(q0-q1) + (9q1)(2q0)`, which is `2*p0*p1 + c + 9d`.
const ROWS_A: [Row; 6] = [
    Row {
        lane: LANE0,
        scratch: VA,
        source: S_SUM0,
        base: Sp,
        offset: slot(M_DIF0),
        what: ("dif0_", "sum0j"),
    },
    Row {
        lane: LANE1,
        scratch: VB,
        source: S_DBL0,
        base: R0P,
        offset: 32,
        what: ("p1_", "dbl0j"),
    },
    Row {
        lane: LANE0,
        scratch: VA,
        source: S_NINE0,
        base: Sp,
        offset: slot(M_SUM1),
        what: ("sum1_", "nine0j"),
    },
    Row {
        lane: LANE1,
        scratch: VB,
        source: S_SUM1,
        base: Sp,
        offset: slot(M_DIF1),
        what: ("dif1_", "sum1j"),
    },
    Row {
        lane: LANE0,
        scratch: VA,
        source: S_DBLQ,
        base: Sp,
        offset: slot(M_NEGQ),
        what: ("negq_", "dblqj"),
    },
    Row {
        lane: LANE1,
        scratch: VB,
        source: S_NINE1,
        base: Sp,
        offset: slot(M_DBLQ),
        what: ("dblq_", "nine1j"),
    },
];

/// `t_b = 2*r0*r1`: the dual-lane Fp2 product with both sources doubled.
/// `p - q1` carries the subtracted lane0 term as a positive one.
const ROWS_B: [Row; 4] = [
    Row {
        lane: LANE0,
        scratch: VA,
        source: S_DBL0,
        base: R1P,
        offset: 0,
        what: ("q0_", "dbl0j"),
    },
    Row {
        lane: LANE1,
        scratch: VB,
        source: S_DBL1,
        base: R1P,
        offset: 0,
        what: ("q0_", "dbl1j"),
    },
    Row {
        lane: LANE0,
        scratch: VB,
        source: S_DBL1,
        base: Sp,
        offset: slot(M_NEGQ),
        what: ("negq_", "dbl1j"),
    },
    Row {
        lane: LANE1,
        scratch: VA,
        source: S_DBL0,
        base: R1P,
        offset: 32,
        what: ("q1_", "dbl0j"),
    },
];

/// Four dual-lane CIOS rounds over one row plan, then the canonical store.
/// The two lanes alternate row by row, so both carry recurrences stay in
/// flight through the whole block.
fn block<M: MachineA64>(
    m: &mut M,
    banks: &Banks,
    rows: &[Row],
    name: &str,
    subtractions: usize,
    output: i32,
) {
    for round in 0..4i32 {
        m.comment("");
        m.comment(&format!("{name} round {round}"));
        for (index, row) in rows.iter().enumerate() {
            // Each row writes only its own scratch, so the previous row on
            // the same scratch decides whether the source word is still live.
            let source_held = rows[..index]
                .iter()
                .rev()
                .find(|earlier| earlier.scratch == row.scratch)
                .is_some_and(|earlier| earlier.source == row.source);
            let row_held = index > 0
                && rows[index - 1].base == row.base
                && rows[index - 1].offset == row.offset;
            row.emit(
                m,
                banks,
                8 * round,
                round < 1 && index < 2,
                source_held,
                row_held,
            );
        }
        m.comment("");
        banks.load_row(m, CONSTS, 0, "p0, p1");
        banks.reduce(m, &LANE0, VA, round == 3);
        banks.reduce(m, &LANE1, VB, round == 3);
    }
    m.comment("");
    m.comment("the row multiplicand still holds p, so the conditional");
    m.comment("subtractions need no reload");
    banks.canonicalize(m, &LANE0, subtractions);
    banks.store_lane(m, &LANE0, Z, output);
    banks.canonicalize(m, &LANE1, subtractions);
    banks.store_lane(m, &LANE1, Z, output + 32);
}

// ---------------------------------------------------------------- fold leaf

const ZF: Reg = X0;
const TF: Reg = X1;
const FF: Reg = X2;
const CF: Reg = X3;
const FP: Bank = [X4, X5, X6, X7];
/// Lane0: the t half on entry, then the running `2u + t` sum.
const FA0: Bank = [X8, X9, X10, X11];
/// Lane0: the f half on entry, then the inner fold `u`.
const FB0: Bank = [X12, X13, X14, X15];
/// Shared, and only ever live inside one four-word group.
const FC: Bank = [X16, X17, X19, X20];
const FA1: Bank = [X21, X22, X23, X24];
const FB1: Bank = [X25, X26, X27, X28];

const FOLD_SAVED_PAIRS: usize = 5;
const FOLD_SAVES: i32 = 16 * FOLD_SAVED_PAIRS as i32;
/// The two canonical halves of `xi*t5`, the only value the fold stages.
const FOLD_XI: i32 = FOLD_SAVES;
const FOLD_FRAME: i32 = FOLD_SAVES + 64;

/// Byte offset of Fp2 `index` inside a six-element repr(C) block.
const fn fp2(index: i32) -> i32 {
    64 * index
}

/// `narsil_cyc_fold`: the six Granger-Scott combines on the reduced halves.
///
/// With the arkworks mapping `r0 = c0.c0, r4 = c0.c1, r3 = c0.c2,
/// r2 = c1.c0, r1 = c1.c1, r5 = c1.c2` and `(t0, t1)`, `(t2, t3)`,
/// `(t4, t5)` the three Fp4 squares, the outputs are `z0 = 3*t0 - 2*r0`,
/// `z1 = 3*t1 + 2*r1`, `z2 = 3*xi*t5 + 2*r2`, `z3 = 3*t4 - 2*r3`,
/// `z4 = 3*t2 - 2*r4` and `z5 = 3*t3 + 2*r5`, stored in repr(C) order.
pub fn cyc_fold<M: MachineA64>(m: &mut M) {
    spill_callee_saved(m, FOLD_FRAME, FOLD_SAVED_PAIRS);
    m.comment("");
    load4(m, FP, CF, 0, "p");

    m.comment("");
    m.comment("xi*t5 is the only combine operand that is not already a");
    m.comment("reduced Fp4 half, so it folds to canonical first");
    xi_scale(m);

    for (label, dst, t_off, f_off, minus) in [
        ("z0 = 3*t0 - 2*r0", fp2(0), fp2(0), fp2(0), true),
        ("z4 = 3*t2 - 2*r4", fp2(1), fp2(2), fp2(1), true),
        ("z3 = 3*t4 - 2*r3", fp2(2), fp2(4), fp2(2), true),
        ("z1 = 3*t1 + 2*r1", fp2(4), fp2(1), fp2(4), false),
        ("z5 = 3*t3 + 2*r5", fp2(5), fp2(3), fp2(5), false),
    ] {
        m.comment("");
        m.comment(label);
        load4(m, FA0, TF, t_off, "t lane0");
        load4(m, FA1, TF, t_off + 32, "t lane1");
        load4(m, FB0, FF, f_off, "r lane0");
        load4(m, FB1, FF, f_off + 32, "r lane1");
        combine(m, dst, minus);
    }

    m.comment("");
    m.comment("z2 = 3*xi*t5 + 2*r2");
    load4(m, FA0, Sp, FOLD_XI, "xi*t5 lane0");
    load4(m, FA1, Sp, FOLD_XI + 32, "xi*t5 lane1");
    load4(m, FB0, FF, fp2(3), "r2 lane0");
    load4(m, FB1, FF, fp2(3) + 32, "r2 lane1");
    combine(m, fp2(3), false);

    m.comment("");
    restore_callee_saved(m, FOLD_FRAME, FOLD_SAVED_PAIRS);
}

/// `z = 3*t -+ 2*r` for both lanes, from `t` in FA and `r` in FB.
///
/// `u = t -+ r mod p` is below p, so `2u + t` is below 3p and two
/// conditional subtractions reach the canonical residue. The value is the
/// composed route's `2*(t -+ r) + t`.
fn combine<M: MachineA64>(m: &mut M, dst: i32, minus: bool) {
    for (a, b) in [(FA0, FB0), (FA1, FB1)] {
        if minus {
            sub_mod4(m, b, a, b, FP, FC, "t - r");
        } else {
            add4(m, b, a, b, "t + r < 2p");
            cond_sub(m, b, FP, FC, "(t + r) mod p");
        }
    }
    for (a, b) in [(FA0, FB0), (FA1, FB1)] {
        double4(m, FC, b, "2u < 2p");
        add4(m, a, FC, a, "2u + t < 3p");
    }
    for (index, a) in [FA0, FA1].into_iter().enumerate() {
        cond_sub(m, a, FP, FC, "first of two: 3p to 2p");
        cond_sub(m, a, FP, FC, "second: 2p to the canonical residue");
        store_at(m, a, ZF, dst + 32 * index as i32);
    }
}

/// `xi*t5 = (9*a - b) + (a + 9*b)*u` for `t5 = a + b*u`, both halves folded
/// to canonical and staged. Nine-scaling runs as `2*(4x mod p) + x`, so the
/// intermediate stays below 3p and never leaves four words.
fn xi_scale<M: MachineA64>(m: &mut M) {
    load4(m, FA0, TF, fp2(5), "t5 lane0");
    load4(m, FA1, TF, fp2(5) + 32, "t5 lane1");

    nine_fold(m, FB0, FC, FA0, "a");
    sub_mod4(m, FB0, FB0, FA1, FP, FC, "9*a - b, below 3p");
    cond_sub(m, FB0, FP, FC, "3p to 2p");
    cond_sub(m, FB0, FP, FC, "2p to the canonical residue");
    store_at(m, FB0, Sp, FOLD_XI);

    nine_fold(m, FB1, FC, FA1, "b");
    add4(m, FB1, FB1, FA0, "9*b + a < 4p");
    cond_sub(m, FB1, FP, FC, "4p to 3p");
    cond_sub(m, FB1, FP, FC, "3p to 2p");
    cond_sub(m, FB1, FP, FC, "2p to the canonical residue");
    store_at(m, FB1, Sp, FOLD_XI + 32);
}

/// `d = 9*x` below 3p, for canonical `x`. Two conditional folds keep the
/// doubled value under p, so the final `8x + x` cannot leave four words.
fn nine_fold<M: MachineA64>(m: &mut M, d: Bank, scratch: Bank, x: Bank, what: &str) {
    double4(m, d, x, &format!("2*{what} < 2p"));
    cond_sub(m, d, FP, scratch, "");
    double4(m, scratch, d, "");
    cond_sub(m, scratch, FP, d, "4x mod p");
    double4(m, d, scratch, "8x below 2p");
    add4(m, d, d, x, &format!("9*{what} < 3p"));
}

// ------------------------------------------------------------ word helpers

fn load4<M: MachineA64>(m: &mut M, into: Bank, base: Reg, offset: i32, what: &str) {
    m.ldp(into[0], into[1], base, offset, what);
    m.ldp(into[2], into[3], base, offset + 16, "");
}

fn store_at<M: MachineA64>(m: &mut M, from: Bank, base: Reg, offset: i32) {
    m.stp(from[0], from[1], base, offset, "");
    m.stp(from[2], from[3], base, offset + 16, "");
}

/// Stage a prep value at its frame slot.
fn store4<M: MachineA64>(m: &mut M, from: Bank, index: i32) {
    store_at(m, from, Sp, slot(index));
}

/// `d = a + b`, which the caller's bound keeps below 2^256: the closing ADC
/// drops a carry-out the interpreter asserts is zero.
fn add4<M: MachineA64>(m: &mut M, d: Bank, a: Bank, b: Bank, what: &str) {
    m.adds(d[0], a[0], b[0], what);
    m.adcs(d[1], a[1], b[1], "");
    m.adcs(d[2], a[2], b[2], "");
    m.adc(d[3], a[3], b[3], "");
}

fn double4<M: MachineA64>(m: &mut M, d: Bank, a: Bank, what: &str) {
    add4(m, d, a, a, what);
}

/// `d = a - b` with `C = (a >= b)`.
fn sub4<M: MachineA64>(m: &mut M, d: Bank, a: Bank, b: Bank, what: &str) {
    m.subs(d[0], a[0], b[0], what);
    m.sbcs(d[1], a[1], b[1], "");
    m.sbcs(d[2], a[2], b[2], "");
    m.sbcs(d[3], a[3], b[3], "");
}

/// `d = p - x`, for `x` at most p, so the chain cannot borrow out.
fn negp4<M: MachineA64>(m: &mut M, d: Bank, modulus: Bank, x: Bank, what: &str) {
    sub4(m, d, modulus, x, &format!("p - {what} stays nonnegative"));
}

/// `v -= p` where that stays nonnegative: one conditional subtraction.
fn cond_sub<M: MachineA64>(m: &mut M, v: Bank, modulus: Bank, scratch: Bank, what: &str) {
    sub4(m, scratch, v, modulus, what);
    for k in 0..4 {
        m.csel_hs(v[k], scratch[k], v[k], "");
    }
}

/// `d = a - b mod p`, adding p back on a borrow. The closing ADCS drops that
/// borrow, exactly as the modular subtract leaf does. Correct whenever
/// `a - b` is above `-p`, which is what makes the result nonnegative.
fn sub_mod4<M: MachineA64>(
    m: &mut M,
    d: Bank,
    a: Bank,
    b: Bank,
    modulus: Bank,
    scratch: Bank,
    what: &str,
) {
    sub4(m, d, a, b, what);
    for k in 0..4 {
        m.csel_hs(scratch[k], Xzr, modulus[k], "no borrow: add nothing back");
    }
    m.adds(d[0], d[0], scratch[0], "");
    m.adcs(d[1], d[1], scratch[1], "");
    m.adcs(d[2], d[2], scratch[2], "");
    m.adcs(
        d[3],
        d[3],
        scratch[3],
        "the add-back carry out is the borrow",
    );
}

/// `d = 9*x` below 5p, for canonical `x`. One conditional fold holds the
/// doubling chain, so `8x` stays under 4p and `8x + x` under 5p < 2^256.
fn nine4<M: MachineA64>(m: &mut M, work: Bank, d: Bank, x: Bank, what: &str) {
    double4(m, work, x, &format!("2*{what} < 2p"));
    cond_sub(m, work, PB, d, "");
    double4(m, d, work, "4x below 2p");
    double4(m, d, d, "8x below 4p");
    add4(m, d, d, x, &format!("9*{what} < 5p < 2^256"));
}
