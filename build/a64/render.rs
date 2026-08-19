//! Render the complete AArch64 `.s` text: provenance header, one register
//! map and one body per kernel. The build script writes this text to OUT_DIR
//! and assembles it for aarch64-apple targets. No copy is checked in.

use super::addsub::{
    ADD_MOD_INLINE_OPERANDS, ADD_MOD_REGISTER_MAP, MemoryShape, SUB_MOD_INLINE_OPERANDS,
    SUB_MOD_REGISTER_MAP, add_mod, sub_mod,
};
use super::cyc::{CYC_FOLD_REGISTER_MAP, CYC_FP4_REGISTER_MAP, cyc_fold, cyc_fp4};
use super::emit::EmitterA64;
use super::inline::{EmitterInlineA64, InlineOperand};
use super::machine::Reg;
use super::schedule::{MONT4_REGISTER_MAP, mont4};
use super::sos::{SOSD2_REGISTER_MAP, SOSD4_REGISTER_MAP, SOSD6_REGISTER_MAP, sosd2, sosd4, sosd6};

/// Apple Mach-O adds the underscore. Rust names the symbol `narsil_mont4`.
pub const MONT4_SYMBOL: &str = "_narsil_mont4";
pub const SOSD2_SYMBOL: &str = "_narsil_sosd2";
pub const SOSD4_SYMBOL: &str = "_narsil_sosd4";
pub const SOSD6_SYMBOL: &str = "_narsil_sosd6";
pub const ADD_MOD_SYMBOL: &str = "_narsil_add_mod";
pub const SUB_MOD_SYMBOL: &str = "_narsil_sub_mod";
pub const CYC_FP4_SYMBOL: &str = "_narsil_cyc_fp4";
pub const CYC_FOLD_SYMBOL: &str = "_narsil_cyc_fold";

/// One emitted symbol: its schedule, its register roles, and the argument
/// list the wrapper in `src/fp/aarch64.rs` must match.
pub struct Kernel {
    pub symbol: &'static str,
    pub schedule_name: &'static str,
    schedule: fn(&mut EmitterA64),
    register_map: &'static [(Reg, &'static str)],
    signature: &'static [&'static str],
}

pub const A64_KERNELS: &[Kernel] = &[
    Kernel {
        symbol: MONT4_SYMBOL,
        schedule_name: "bn254-mont4-cios-unrolled",
        schedule: mont4::<EmitterA64>,
        register_map: MONT4_REGISTER_MAP,
        signature: &[
            "Arguments: (z: *mut u64x4, x: *const u64x4, y: *const u64x4,",
            "            consts: *const { p: [u64; 4], neg_p_inv: u64 })",
        ],
    },
    Kernel {
        symbol: SOSD2_SYMBOL,
        schedule_name: "bn254-sosd2-dual-lane-cios",
        schedule: sosd2::<EmitterA64>,
        register_map: SOSD2_REGISTER_MAP,
        signature: &[
            "Arguments: (z: *mut u64x8, x0: *const u64x4, x1: *const u64x4,",
            "            y0: *const u64x4, y1: *const u64x4,",
            "            consts: *const { p: [u64; 4], neg_p_inv: u64 })",
        ],
    },
    Kernel {
        symbol: SOSD4_SYMBOL,
        schedule_name: "bn254-sosd4-dual-lane-cios",
        schedule: sosd4::<EmitterA64>,
        register_map: SOSD4_REGISTER_MAP,
        signature: &[
            "Arguments: (z: *mut u64x8, table: *const [*const u64x4; 8],",
            "            consts: *const { p: [u64; 4], neg_p_inv: u64 })",
            "Table order: x00 x01 y00 y01 x10 x11 y10 y11",
        ],
    },
    Kernel {
        symbol: SOSD6_SYMBOL,
        schedule_name: "bn254-sosd6-dual-lane-cios",
        schedule: sosd6::<EmitterA64>,
        register_map: SOSD6_REGISTER_MAP,
        signature: &[
            "Arguments: (z: *mut u64x8, table: *const [*const u64x4; 12],",
            "            consts: *const { p: [u64; 4], neg_p_inv: u64 })",
            "Table order: x00 x01 y00 y01 x10 x11 y10 y11 x20 x21 y20 y21",
        ],
    },
    Kernel {
        symbol: CYC_FP4_SYMBOL,
        schedule_name: "bn254-cyc-fp4-square-fused",
        schedule: cyc_fp4::<EmitterA64>,
        register_map: CYC_FP4_REGISTER_MAP,
        signature: &[
            "Arguments: (z: *mut u64x16, r0: *const u64x8, r1: *const u64x8,",
            "            consts: *const { p: [u64; 4], neg_p_inv: u64 })",
            "z holds t_a then t_b, each two canonical Fp halves.",
        ],
    },
    Kernel {
        symbol: CYC_FOLD_SYMBOL,
        schedule_name: "bn254-cyc-granger-scott-combines",
        schedule: cyc_fold::<EmitterA64>,
        register_map: CYC_FOLD_REGISTER_MAP,
        signature: &[
            "Arguments: (z: *mut u64x48, t: *const u64x48, f: *const u64x48,",
            "            consts: *const { p: [u64; 4], neg_p_inv: u64 })",
            "t holds t0..t5 in Fp4-pair order, z and f are repr(C) Fp12.",
        ],
    },
    Kernel {
        symbol: ADD_MOD_SYMBOL,
        schedule_name: "add-mod-conditional-subtract",
        schedule: add_mod_leaf,
        register_map: ADD_MOD_REGISTER_MAP,
        signature: &[
            "Arguments: (z: *mut u64x4, a: *const u64x4, b: *const u64x4,",
            "            p: *const u64x4)",
        ],
    },
    Kernel {
        symbol: SUB_MOD_SYMBOL,
        schedule_name: "sub-mod-conditional-add",
        schedule: sub_mod_leaf,
        register_map: SUB_MOD_REGISTER_MAP,
        signature: &[
            "Arguments: (z: *mut u64x4, a: *const u64x4, b: *const u64x4,",
            "            p: *const u64x4)",
        ],
    },
];

fn add_mod_leaf(m: &mut EmitterA64) {
    add_mod(m, MemoryShape::Pointers);
}

fn sub_mod_leaf(m: &mut EmitterA64) {
    sub_mod(m, MemoryShape::Pointers);
}

/// One inline fold: the register-shape schedule and the Rust name that wraps
/// it in the generated fragment.
struct InlineFold {
    function: &'static str,
    schedule: fn(&mut EmitterInlineA64),
    operands: &'static [InlineOperand],
    /// The bound the conditional fold rests on.
    range: &'static str,
}

const INLINE_FOLDS: &[InlineFold] = &[
    InlineFold {
        function: "add_mod_inline",
        schedule: |m| add_mod(m, MemoryShape::Registers),
        operands: ADD_MOD_INLINE_OPERANDS,
        range: "a, b < p keeps a + b below 2p < 2^257, so one conditional subtraction reaches [0, p)",
    },
    InlineFold {
        function: "sub_mod_inline",
        schedule: |m| sub_mod(m, MemoryShape::Registers),
        operands: SUB_MOD_INLINE_OPERANDS,
        range: "a, b < p keeps a - b in (-p, p), so one conditional add-back reaches [0, p)",
    },
];

/// Render the Rust fragment the crate includes from OUT_DIR: one
/// `#[inline(always)]` function per fold, each wrapping the register-shape
/// schedule as an `asm!` template. Deterministic, like the assembly text.
pub fn render_a64_inline() -> String {
    let mut out = String::new();
    let mut line = |text: &str| {
        out.push_str(text);
        out.push('\n');
    };

    line("// @generated at build time by the helius-narsil build script.");
    line("// Source of truth: crates/helius-narsil/build/a64 (ADR 0001).");
    line("// The same schedule renders the leaf symbols in kernels_aarch64.s.");
    line("// Verified by: tests/kernelgen_verify.rs (interpreter + rendering).");

    for fold in INLINE_FOLDS {
        let mut emitter = EmitterInlineA64::new(fold.operands);
        (fold.schedule)(&mut emitter);
        let declarations = emitter.declarations();
        line("");
        line(&format!("/// {}", fold.range));
        line("#[inline(always)]");
        line(&format!(
            "pub(crate) fn {}(a: &[u64; 4], b: &[u64; 4], p: &[u64; 4]) -> [u64; 4] {{",
            fold.function,
        ));
        line("    let [mut z0, mut z1, mut z2, mut z3] = *a;");
        line("    // SAFETY: the block is register-only arithmetic. It reads and");
        line("    // writes no memory, needs no stack, and every register it names");
        line("    // is a declared operand, so pure, nomem and nostack all hold.");
        line("    unsafe {");
        line("        core::arch::asm!(");
        for instruction in emitter.into_lines() {
            line(&format!("            \"{instruction}\","));
        }
        for declaration in declarations {
            line(&format!("            {declaration},"));
        }
        line("            options(pure, nomem, nostack),");
        line("        );");
        line("    }");
        line("    [z0, z1, z2, z3]");
        line("}");
    }
    out
}

/// Render the full AArch64 kernel file. Byte-for-byte deterministic (no
/// timestamps, paths, or host data). Tests assert generate-twice identity.
pub fn render_aarch64() -> String {
    let mut out = String::new();
    let mut line = |text: &str| {
        out.push_str(text);
        out.push('\n');
    };

    line("/* @generated at build time by the helius-narsil build script.");
    line("   Source of truth: crates/helius-narsil/build/a64 (ADR 0001).");
    line("   Inspect a copy: NARSIL_DUMP_ASM=<absolute dir> cargo build.");
    line("   Verified by: tests/kernelgen_verify.rs (interpreter + determinism).");
    line("");
    for (kernel, (instructions, bytes)) in A64_KERNELS.iter().zip(kernel_sizes()) {
        line(&format!(
            "   {}: schedule {}, {instructions} instructions, {bytes} bytes",
            kernel.symbol, kernel.schedule_name,
        ));
    }
    line("");
    line("   AAPCS64 (Apple), leaf, straight line, no calls and no branches.");
    line("   Contract and safety boundary live beside the only callers in");
    line("   src/fp/aarch64.rs; carry bounds live in build/a64. */");

    for kernel in A64_KERNELS {
        let mut emitter = EmitterA64::new();
        (kernel.schedule)(&mut emitter);
        line("");
        line(&format!("/* {} register map:", kernel.symbol));
        for text in kernel.signature {
            line(&format!("   {text}"));
        }
        for (reg, role) in kernel.register_map {
            line(&format!("   {:<4} {}", reg.name(), role));
        }
        line("*/");
        line("    .text");
        line("    .p2align 5");
        line(&format!("    .globl {}", kernel.symbol));
        line(&format!("{}:", kernel.symbol));
        for body_line in emitter.into_lines() {
            line(&body_line);
        }
    }
    out
}

/// (instructions, bytes) for every kernel, in `A64_KERNELS` order.
pub fn kernel_sizes() -> Vec<(usize, usize)> {
    A64_KERNELS
        .iter()
        .map(|kernel| {
            let mut emitter = EmitterA64::new();
            (kernel.schedule)(&mut emitter);
            (emitter.instructions(), emitter.bytes())
        })
        .collect()
}
