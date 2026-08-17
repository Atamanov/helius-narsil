//! Render the complete AArch64 `.s` text: provenance header, one register
//! map and one body per kernel. The build script writes this text to OUT_DIR
//! and assembles it for aarch64-apple targets. No copy is checked in.

use super::emit::EmitterA64;
use super::machine::Reg;
use super::schedule::{MONT4_REGISTER_MAP, mont4};
use super::sos::{SOSD2_REGISTER_MAP, SOSD4_REGISTER_MAP, SOSD6_REGISTER_MAP, sosd2, sosd4, sosd6};

/// Apple Mach-O adds the underscore. Rust names the symbol `narsil_mont4`.
pub const MONT4_SYMBOL: &str = "_narsil_mont4";
pub const SOSD2_SYMBOL: &str = "_narsil_sosd2";
pub const SOSD4_SYMBOL: &str = "_narsil_sosd4";
pub const SOSD6_SYMBOL: &str = "_narsil_sosd6";

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
];

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
