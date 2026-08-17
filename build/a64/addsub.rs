//! Modular add and subtract schedules for AArch64.
//!
//! Both take the modulus as four limbs, so one schedule serves every
//! four-limb modulus the crate reduces against, and both are bit-exact with
//! the portable `src/limb.rs` forms on all of `[0, 2^256)`. Canonicality
//! rests on the caller's bound: `a, b < p` puts `a + b` below `2p < 2^257`
//! and `a - b` in `(-p, p)`, so one conditional fold reaches `[0, p)`.
//!
//! One body serves two renderings, selected by [`MemoryShape`]. Under
//! [`MemoryShape::Pointers`] it is a leaf symbol that loads its operands and
//! stores its result. Under [`MemoryShape::Registers`] the loads, the stores
//! and the `ret` are gone and the operands arrive in registers, which is the
//! `asm!` form the tower inlines. The arithmetic in between is the same text,
//! so the interpreter verifies whichever shape it is given.
//!
//! Register use stops at x17. A leaf that touches no callee-saved register
//! needs no prologue, so neither schedule moves sp or reads the stack.

use super::inline::{InlineOperand, InlineRole};
use super::machine::Reg::{
    X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14, X15, X16, X17, Xzr,
};
use super::machine::{MachineA64, Reg};

/// Where the operands live at the schedule boundary.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MemoryShape {
    /// Pointer arguments, AAPCS64 leaf.
    Pointers,
    /// Operands and result in registers, no memory access at all.
    Registers,
}

/// Register roles for `narsil_add_mod`.
pub const ADD_MOD_REGISTER_MAP: &[(Reg, &str)] = &[
    (X0, "z: four result limbs"),
    (X1, "a pointer"),
    (X2, "b pointer"),
    (X3, "modulus pointer"),
    (X4, "a limb 0 on entry; then sum word 0"),
    (X5, "a limb 1; then sum word 1"),
    (X6, "a limb 2; then sum word 2"),
    (X7, "a limb 3; then sum word 3"),
    (X8, "b limb 0; then word 0 of sum - p"),
    (X9, "b limb 1; then word 1"),
    (X10, "b limb 2; then word 2"),
    (X11, "b limb 3; then word 3"),
    (X12, "p limb 0"),
    (X13, "p limb 1"),
    (X14, "p limb 2"),
    (X15, "p limb 3"),
    (X16, "sum word 4"),
    (X17, "scratch for the fifth borrow word"),
];

/// Register roles for `narsil_sub_mod`.
pub const SUB_MOD_REGISTER_MAP: &[(Reg, &str)] = &[
    (X0, "z: four result limbs"),
    (X1, "a pointer"),
    (X2, "b pointer"),
    (X3, "modulus pointer"),
    (X4, "a limb 0 on entry; then difference word 0"),
    (X5, "a limb 1; then difference word 1"),
    (X6, "a limb 2; then difference word 2"),
    (X7, "a limb 3; then difference word 3"),
    (X8, "b limb 0; then word 0 of the add-back term"),
    (X9, "b limb 1; then word 1"),
    (X10, "b limb 2; then word 2"),
    (X11, "b limb 3; then word 3"),
    (X12, "p limb 0"),
    (X13, "p limb 1"),
    (X14, "p limb 2"),
    (X15, "p limb 3"),
];

/// Left operand on entry, then the running result.
const T: [Reg; 4] = [X4, X5, X6, X7];
/// Right operand on entry, then the conditional term.
const U: [Reg; 4] = [X8, X9, X10, X11];
/// Modulus limbs.
const P: [Reg; 4] = [X12, X13, X14, X15];
const Z: Reg = X0;
const A: Reg = X1;
const B: Reg = X2;
const MODULUS: Reg = X3;
/// Word 4 of the sum: at most 1, because `a + b < 2^257`.
const CARRY: Reg = X16;
/// Consumes word 4 in the comparison chain. Its value is never read.
const SCRATCH: Reg = X17;

/// Inline operands of `add_mod`, one per register the schedule touches.
pub const ADD_MOD_INLINE_OPERANDS: &[InlineOperand] = &[
    InlineOperand {
        reg: X4,
        name: "z0",
        role: InlineRole::Result,
    },
    InlineOperand {
        reg: X5,
        name: "z1",
        role: InlineRole::Result,
    },
    InlineOperand {
        reg: X6,
        name: "z2",
        role: InlineRole::Result,
    },
    InlineOperand {
        reg: X7,
        name: "z3",
        role: InlineRole::Result,
    },
    InlineOperand {
        reg: X8,
        name: "b0",
        role: InlineRole::Operand,
    },
    InlineOperand {
        reg: X9,
        name: "b1",
        role: InlineRole::Operand,
    },
    InlineOperand {
        reg: X10,
        name: "b2",
        role: InlineRole::Operand,
    },
    InlineOperand {
        reg: X11,
        name: "b3",
        role: InlineRole::Operand,
    },
    InlineOperand {
        reg: X12,
        name: "p0",
        role: InlineRole::Modulus,
    },
    InlineOperand {
        reg: X13,
        name: "p1",
        role: InlineRole::Modulus,
    },
    InlineOperand {
        reg: X14,
        name: "p2",
        role: InlineRole::Modulus,
    },
    InlineOperand {
        reg: X15,
        name: "p3",
        role: InlineRole::Modulus,
    },
    InlineOperand {
        reg: X16,
        name: "carry",
        role: InlineRole::Scratch,
    },
    InlineOperand {
        reg: X17,
        name: "borrow",
        role: InlineRole::Scratch,
    },
];

/// Inline operands of `sub_mod`. It needs no scratch beyond the three groups.
pub const SUB_MOD_INLINE_OPERANDS: &[InlineOperand] = &[
    InlineOperand {
        reg: X4,
        name: "z0",
        role: InlineRole::Result,
    },
    InlineOperand {
        reg: X5,
        name: "z1",
        role: InlineRole::Result,
    },
    InlineOperand {
        reg: X6,
        name: "z2",
        role: InlineRole::Result,
    },
    InlineOperand {
        reg: X7,
        name: "z3",
        role: InlineRole::Result,
    },
    InlineOperand {
        reg: X8,
        name: "b0",
        role: InlineRole::Operand,
    },
    InlineOperand {
        reg: X9,
        name: "b1",
        role: InlineRole::Operand,
    },
    InlineOperand {
        reg: X10,
        name: "b2",
        role: InlineRole::Operand,
    },
    InlineOperand {
        reg: X11,
        name: "b3",
        role: InlineRole::Operand,
    },
    InlineOperand {
        reg: X12,
        name: "p0",
        role: InlineRole::Modulus,
    },
    InlineOperand {
        reg: X13,
        name: "p1",
        role: InlineRole::Modulus,
    },
    InlineOperand {
        reg: X14,
        name: "p2",
        role: InlineRole::Modulus,
    },
    InlineOperand {
        reg: X15,
        name: "p3",
        role: InlineRole::Modulus,
    },
];

/// `z = a + b mod p`.
pub fn add_mod<M: MachineA64>(m: &mut M, shape: MemoryShape) {
    if shape == MemoryShape::Pointers {
        load4(m, T, A, "a");
        load4(m, U, B, "b");
        load4(m, P, MODULUS, "p");
        m.comment("");
    }

    m.adds(T[0], T[0], U[0], "word 0 of a + b (opens the chain)");
    m.adcs(T[1], T[1], U[1], "");
    m.adcs(T[2], T[2], U[2], "");
    m.adcs(T[3], T[3], U[3], "");
    m.cset_hs(CARRY, "a + b < 2p < 2^257, so word 4 is 0 or 1");

    m.comment("");
    m.subs(U[0], T[0], P[0], "word 0 of the five-word sum - p");
    m.sbcs(U[1], T[1], P[1], "");
    m.sbcs(U[2], T[2], P[2], "");
    m.sbcs(U[3], T[3], P[3], "");
    m.sbcs(SCRATCH, CARRY, Xzr, "consume word 4; C = (a + b >= p)");
    m.csel_hs(T[0], U[0], T[0], "no borrow: keep sum - p");
    m.csel_hs(T[1], U[1], T[1], "");
    m.csel_hs(T[2], U[2], T[2], "");
    m.csel_hs(T[3], U[3], T[3], "");

    if shape == MemoryShape::Pointers {
        store4(m, T, Z);
    }
}

/// `z = a - b mod p`.
pub fn sub_mod<M: MachineA64>(m: &mut M, shape: MemoryShape) {
    if shape == MemoryShape::Pointers {
        load4(m, T, A, "a");
        load4(m, U, B, "b");
        load4(m, P, MODULUS, "p");
        m.comment("");
    }

    m.subs(T[0], T[0], U[0], "word 0 of a - b (opens the chain)");
    m.sbcs(T[1], T[1], U[1], "");
    m.sbcs(T[2], T[2], U[2], "");
    m.sbcs(T[3], T[3], U[3], "C = (a >= b)");

    m.comment("");
    m.comment("a - b > -p, so adding p back once on a borrow lands in [0, p)");
    m.csel_hs(U[0], Xzr, P[0], "no borrow: add nothing");
    m.csel_hs(U[1], Xzr, P[1], "");
    m.csel_hs(U[2], Xzr, P[2], "");
    m.csel_hs(U[3], Xzr, P[3], "");
    m.adds(T[0], T[0], U[0], "opens a second chain");
    m.adcs(T[1], T[1], U[1], "");
    m.adcs(T[2], T[2], U[2], "");
    m.adcs(
        T[3],
        T[3],
        U[3],
        "the add-back carry out is the borrow, dropped",
    );

    if shape == MemoryShape::Pointers {
        store4(m, T, Z);
    }
}

fn load4<M: MachineA64>(m: &mut M, into: [Reg; 4], base: Reg, what: &str) {
    m.ldp(into[0], into[1], base, 0, &format!("{what} limbs 0, 1"));
    m.ldp(into[2], into[3], base, 16, "");
}

fn store4<M: MachineA64>(m: &mut M, from: [Reg; 4], base: Reg) {
    m.comment("");
    m.stp(from[0], from[1], base, 0, "z0, z1");
    m.stp(from[2], from[3], base, 16, "z2, z3");
    m.ret();
}
