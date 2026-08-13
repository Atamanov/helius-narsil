//! Render the complete x86-64 `.s` text: provenance header, per-kernel
//! register maps, and the emitted schedules. The build script writes this
//! text to OUT_DIR and assembles it. No copy is checked in.

use super::emit::Emitter;
use super::machine::Reg;
use super::schedule::{
    CYC_SQR_REGISTER_MAP, F2SQR_REGISTER_MAP, FP2_XI_COMPACT_REGISTER_MAP, FP4_SQR_REGISTER_MAP,
    FP6_REGISTER_MAP, FP12_034_REGISTER_MAP, FP12_034K_REGISTER_MAP, FP12_034L_REGISTER_MAP,
    FP12_MUL_REGISTER_MAP, FP12_SQR_MCL_REGISTER_MAP, FP12_SQR_REGISTER_MAP, G2_YSQR_REGISTER_MAP,
    MUL_REGISTER_MAP, MULPRE_REGISTER_MAP, REDC_REGISTER_MAP, SOS_REGISTER_MAP, SOSD2_REGISTER_MAP,
    SOSD2_SMALL_REGISTER_MAP, SOSD6_REGISTER_MAP, SQR_REGISTER_MAP, cyc_sqr_x86, f2sqr_small_x86,
    f2sqr_x86, fp2_xi_compact_x86, fp4_sqr_x86, fp6_mul_x86, fp12_034_x86, fp12_034k_x86,
    fp12_034l_x86, fp12_mul_x86, fp12_sqr_mcl_x86, fp12_sqr_x86, g2_ysqr_x86, mont4_mul,
    mont4_mulpre_x86, mont4_redc_x86, mont4_sqr, sos_rolled, sosd2_small_x86, sosd2_x86, sosd6_x86,
};

pub const MUL_SYMBOL: &str = "narsil_mont4_mul_x86";
pub const SQR_SYMBOL: &str = "narsil_mont4_sqr_x86";
pub const SOS_SYMBOL: &str = "narsil_sos_x86";
pub const SOSD2_SYMBOL: &str = "narsil_sosd2_x86";
pub const SOSD2_SMALL_SYMBOL: &str = "narsil_sosd2_small_x86";
pub const FP6_SYMBOL: &str = "narsil_fp6_mul_x86";
pub const FP12_034_SYMBOL: &str = "narsil_fp12_034_x86";
pub const FP4_SQR_SYMBOL: &str = "narsil_fp4_sqr_x86";
pub const FP12_SQR_SYMBOL: &str = "narsil_fp12_sqr_x86";
pub const FP12_MUL_SYMBOL: &str = "narsil_fp12_mul_x86";
pub const CYC_SQR_SYMBOL: &str = "narsil_cyc_sqr_x86";
pub const FP12_034L_SYMBOL: &str = "narsil_fp12_034l_x86";
pub const FP12_034K_SYMBOL: &str = "narsil_fp12_034k_x86";
pub const SOSD6_SYMBOL: &str = "narsil_sosd6_x86";
pub const MULPRE_SYMBOL: &str = "narsil_mont4_mulpre_x86";
pub const REDC_SYMBOL: &str = "narsil_mont4_redc_x86";
pub const FP2_XI_COMPACT_SYMBOL: &str = "narsil_fp2_xi_compact_x86";
pub const FP12_SQR_MCL_SYMBOL: &str = "narsil_fp12_sqr_mcl_x86";
pub const F2SQR_SYMBOL: &str = "narsil_f2sqr_x86";
pub const F2SQR_SMALL_SYMBOL: &str = "narsil_f2sqr_small_x86";
pub const G2_YSQR_SYMBOL: &str = "narsil_g2_ysqr_x86";

struct Kernel {
    symbol: &'static str,
    schedule_name: &'static str,
    register_map: &'static [(Reg, &'static str)],
    body: Vec<String>,
    instructions: usize,
    bytes: usize,
    rodata: Vec<(String, Vec<u64>)>,
    rodata_bytes: usize,
}

fn emit_kernel(
    symbol: &'static str,
    schedule_name: &'static str,
    register_map: &'static [(Reg, &'static str)],
    schedule: fn(&mut Emitter),
) -> Kernel {
    let mut emitter = Emitter::new();
    schedule(&mut emitter);
    let instructions = emitter.instructions();
    let bytes = emitter.bytes();
    let rodata_bytes = emitter.rodata_bytes();
    let (body, rodata) = emitter.into_parts();
    Kernel {
        symbol,
        schedule_name,
        register_map,
        instructions,
        bytes,
        body,
        rodata,
        rodata_bytes,
    }
}

fn kernels() -> [Kernel; 21] {
    [
        emit_kernel(
            FP2_XI_COMPACT_SYMBOL,
            "bn254-fp2-xi-compact-mu-reduce",
            FP2_XI_COMPACT_REGISTER_MAP,
            fp2_xi_compact_x86::<Emitter>,
        ),
        emit_kernel(
            FP12_SQR_MCL_SYMBOL,
            "bn254-fp12-sqr-mcl-compact-fp6-wrapper",
            FP12_SQR_MCL_REGISTER_MAP,
            fp12_sqr_mcl_x86::<Emitter>,
        ),
        emit_kernel(
            MUL_SYMBOL,
            "bn254-mont4-mul-cios-dual-chain",
            MUL_REGISTER_MAP,
            mont4_mul::<Emitter>,
        ),
        emit_kernel(
            SQR_SYMBOL,
            "bn254-mont4-sqr-cross-double",
            SQR_REGISTER_MAP,
            mont4_sqr::<Emitter>,
        ),
        emit_kernel(
            SOS_SYMBOL,
            "bn254-sos-rolled-dual-chain",
            SOS_REGISTER_MAP,
            sos_rolled::<Emitter>,
        ),
        emit_kernel(
            SOSD2_SYMBOL,
            "bn254-sosd2-dual-lane",
            SOSD2_REGISTER_MAP,
            sosd2_x86::<Emitter>,
        ),
        emit_kernel(
            SOSD2_SMALL_SYMBOL,
            "bn254-sosd2-dual-lane-rolled",
            SOSD2_SMALL_REGISTER_MAP,
            sosd2_small_x86::<Emitter>,
        ),
        emit_kernel(
            FP6_SYMBOL,
            "bn254-fp6-mul-rolled-dual-lane-t6",
            FP6_REGISTER_MAP,
            fp6_mul_x86::<Emitter>,
        ),
        emit_kernel(
            FP12_034_SYMBOL,
            "bn254-fp12-034-rolled-dual-lane-t6",
            FP12_034_REGISTER_MAP,
            fp12_034_x86::<Emitter>,
        ),
        emit_kernel(
            FP4_SQR_SYMBOL,
            "bn254-fp4-sqr-rolled-dual-lane",
            FP4_SQR_REGISTER_MAP,
            fp4_sqr_x86::<Emitter>,
        ),
        emit_kernel(
            FP12_SQR_SYMBOL,
            "bn254-fp12-sqr-lazy-karatsuba-dblwidth",
            FP12_SQR_REGISTER_MAP,
            fp12_sqr_x86::<Emitter>,
        ),
        emit_kernel(
            FP12_MUL_SYMBOL,
            "bn254-fp12-mul-lazy-karatsuba-dblwidth",
            FP12_MUL_REGISTER_MAP,
            fp12_mul_x86::<Emitter>,
        ),
        emit_kernel(
            CYC_SQR_SYMBOL,
            "bn254-cyc-sqr-lazy-fp4-dblwidth",
            CYC_SQR_REGISTER_MAP,
            cyc_sqr_x86::<Emitter>,
        ),
        emit_kernel(
            FP12_034L_SYMBOL,
            "bn254-fp12-034-lazy-sparse-dblwidth",
            FP12_034L_REGISTER_MAP,
            fp12_034l_x86::<Emitter>,
        ),
        emit_kernel(
            FP12_034K_SYMBOL,
            "bn254-fp12-034-karatsuba-w-walk",
            FP12_034K_REGISTER_MAP,
            fp12_034k_x86::<Emitter>,
        ),
        emit_kernel(
            SOSD6_SYMBOL,
            "bn254-sosd6-dual-lane-rolled-t6",
            SOSD6_REGISTER_MAP,
            sosd6_x86::<Emitter>,
        ),
        emit_kernel(
            MULPRE_SYMBOL,
            "bn254-mont4-mulpre-direct-rolled",
            MULPRE_REGISTER_MAP,
            mont4_mulpre_x86::<Emitter>,
        ),
        emit_kernel(
            REDC_SYMBOL,
            "bn254-mont4-redc-direct",
            REDC_REGISTER_MAP,
            mont4_redc_x86::<Emitter>,
        ),
        emit_kernel(
            F2SQR_SYMBOL,
            "bn254-f2sqr-dual-lane-complex",
            F2SQR_REGISTER_MAP,
            f2sqr_x86::<Emitter>,
        ),
        emit_kernel(
            F2SQR_SMALL_SYMBOL,
            "bn254-f2sqr-dual-lane-complex-rolled",
            F2SQR_REGISTER_MAP,
            f2sqr_small_x86::<Emitter>,
        ),
        emit_kernel(
            G2_YSQR_SYMBOL,
            "bn254-g2-ysqr-lazy-dblwidth",
            G2_YSQR_REGISTER_MAP,
            g2_ysqr_x86::<Emitter>,
        ),
    ]
}

/// Render the full x86-64 kernel file. Byte-for-byte deterministic (no
/// timestamps, paths, or host data). Tests assert generate-twice identity.
pub fn render_mont4_x86_64() -> String {
    let kernels = kernels();
    let mut out = String::new();
    let mut line = |text: &str| {
        out.push_str(text);
        out.push('\n');
    };

    line("/* @generated at build time by the helius-narsil build script.");
    line("   Source of truth: crates/helius-narsil/build/schedule.rs (ADR 0001).");
    line("   Inspect a copy: NARSIL_DUMP_ASM=<absolute dir> cargo build.");
    line("   Verified by: tests/kernelgen_verify.rs (interpreter + determinism).");
    line("");
    for kernel in &kernels {
        if kernel.rodata_bytes == 0 {
            line(&format!(
                "   {}: schedule {}, {} instructions, {} bytes",
                kernel.symbol, kernel.schedule_name, kernel.instructions, kernel.bytes,
            ));
        } else {
            line(&format!(
                "   {}: schedule {}, {} instructions, {} bytes, {} rodata bytes",
                kernel.symbol,
                kernel.schedule_name,
                kernel.instructions,
                kernel.bytes,
                kernel.rodata_bytes,
            ));
        }
    }
    line("");
    line("   System V AMD64; requires BMI2 (mulx) and ADX (adox/adcx).");
    line("   mont4 arguments: (z: *mut u64x4, x: *const u64x4, y: *const u64x4,");
    line("                     consts: *const { p: [u64; 4], neg_p_inv: u64 })");
    line("   mont4_mulpre arguments: (z: *mut u64x8, x, y: *const u64x4);");
    line("   z = x*y exactly; x,y may be below 2p (BN254 p < 2^254).");
    line("   mont4_redc arguments: (z: *mut u64x4, t: *const u64x8, consts);");
    line("   requires t < p*2^256 and returns canonical t/R mod p.");
    line("   sos arguments:   (z: *mut u64x4, pairs: *const *const u64,");
    line("                     t: u64 even pair count in 2..=10, consts as above);");
    line("   pairs holds 2t pointers a_0, b_0, ..., operands below or at p.");
    line("   sosd2 arguments: (z: *mut u64x8 lane0 then lane1, x0, x1, y0, y1:");
    line("                     *const u64x4 at most p, consts as above);");
    line("   lane0 = (x0*y0 + x1*(p - y1))/R, lane1 = (x0*y1 + x1*y0)/R mod p.");
    line("   sosd2_small: identical contract, rolled rounds (op-cache-compact);");
    line("   the Rust wrapper picks one at build time (NARSIL_SOSD2_SMALL).");
    line("   fp6_mul arguments: (z: *mut u64x24, a, b: *const u64x24 in repr(C)");
    line("                       Fp6 order c0.re, c0.im, .., c2.im, all Fp < p;");
    line("                       consts: *const { p: [u64; 4], neg_p_inv: u64,");
    line("                       mu: u64 = floor(2^310/p) });");
    line("   z = a*b in Fp6 = Fp2[v]/(v^3 - (9+u)), all outputs canonical.");
    line("   fp12_034 arguments: (z: *mut u64x48, f: *const u64x48 in repr(C)");
    line("                        Fp12 order (c0 then c1, each Fp6 as above),");
    line("                        z == f allowed; c: *const u64x24 = the sparse");
    line("                        coefficients c0, c3, c4 as contiguous Fp2s;");
    line("                        consts as fp6_mul);");
    line("   z = f * (c0 + c3*w + c4*v*w) in Fp12 = Fp6[w]/(w^2 - v), all Fp");
    line("   inputs canonical, outputs canonical (arkworks mul_by_034).");
    line("   fp4_sqr arguments: (z: *mut u64x16 = t0 then t1 as repr(C) Fp2");
    line("                       pairs, z aliasing neither input; r0, r1:");
    line("                       *const u64x8 repr(C) Fp2, all Fp < p;");
    line("                       consts as fp6_mul);");
    line("   t0 = r0^2 + xi*r1^2, t1 = 2*r0*r1 with xi = 9 + u (the Fp4");
    line("   square of the cyclotomic square), all outputs canonical.");
    line("   fp12_sqr arguments: (z: *mut u64x48, f: *const u64x48 in repr(C)");
    line("                        Fp12 order, z == f allowed; consts as");
    line("                        fp6_mul);");
    line("   z = f^2 via mcl's lazy double-width shape: 36 raw 4x4 products");
    line("   and 12 Montgomery reductions, cross terms held as 512-bit values");
    line("   mod p*2^256 (needs p < 2^254; BN254: yes), outputs canonical.");
    line("   fp12_mul arguments: (z: *mut u64x48, a, b: *const u64x48 in");
    line("                        repr(C) Fp12 order, z may alias a, b, or both.");
    line("                        consts as fp6_mul);");
    line("   z = a*b via the same lazy shape: 54 raw 4x4 products and 12");
    line("   Montgomery reductions (3 Fp6Dbl mulPre + double-width mulVadd");
    line("   and Karatsuba assembly mod p*2^256), outputs canonical.");
    line("   cyc_sqr arguments: (z: *mut u64x48, f: *const u64x48 in repr(C)");
    line("                       Fp12 order, z == f allowed; consts as");
    line("                       fp6_mul);");
    line("   z = the Granger-Scott cyclotomic square of f via three lazy Fp4");
    line("   squares (Fp2Dbl sqrPre complex method): 18 raw 4x4 products and");
    line("   12 Montgomery reductions, single-width z-combines; equals f^2");
    line("   exactly on the cyclotomic subgroup, and the composed formula");
    line("   bit for bit on any canonical input.");
    line("   fp12_034l arguments: identical contract to fp12_034 (same ABI");
    line("   and value); mcl mul_403's lazy double-width shape: 39 raw 4x4");
    line("   products and 12 Montgomery reductions where fp12_034 pays 72");
    line("   interleaved-reduction products, outputs canonical.");
    line("   fp12_034k arguments: identical contract to fp12_034 (same ABI");
    line("   and value); fp12_034's uniform W-basis walk with Karatsuba");
    line("   inside every Fp2 product: 54 raw 4x4 products and 12 Montgomery");
    line("   reductions, outputs canonical.");
    line("   sosd6 arguments: (z: *mut u64x8 lane0 then lane1, stage: *mut");
    line("                     u64x64 wrapper-built scratch, consts as mont4);");
    line("   stage: +0 the 24 x limbs transposed (x_i[j] at 8*(6j+i), operand");
    line("   order x00 x01 x10 x11 x20 x21), +192 five 64-byte y pair blocks");
    line("   [y00,y01] [y01,y00] [y10,y11] [y11,y10] [y20,y21]; the kernel");
    line("   overwrites the low vectors of blocks 1 and 3 with p - y01 and");
    line("   p - y11 in place. lane0 = (sum x_i0*y_i0 + x_i1*(p - y_i1))/R,");
    line("   lane1 = (sum x_i0*y_i1 + x_i1*y_i0)/R mod p, operands at most p,");
    line("   both lanes canonical.");
    line("   fp2_xi_compact arguments: (z: *mut u64x8, x: *const u64x8,");
    line("                             consts as fp6_mul), z and x distinct;");
    line("   z = x*(9+u), canonical, using one quotient reduction per Fp half.");
    line("   fp12_sqr_mcl arguments: (f: *mut u64x48 in repr(C) Fp12 order,");
    line("                            consts as fp6_mul), in-place;");
    line("   f = f^2 via MCL's high-level t0/t1/ab identity and two calls to");
    line("   the existing 36-product fp6_mul leaf (72 raw products total).");
    line("   This is a compact-wrapper screen, not MCL's 18-product Fp6Dbl");
    line("   lower primitive; all intermediate and output Fp values canonical.");
    line("   f2sqr arguments: (z: *mut u64x8 lane0 then lane1, x0, x1:");
    line("                     *const u64x4 below p, consts as mont4);");
    line("   lane0 = ((x0+x1)*(x0-x1))/R, lane1 = (x0*(2*x1))/R mod p, the");
    line("   complex square of x0 + x1*u with both Montgomery chains");
    line("   interleaved; both lanes canonical. f2sqr_small is the rolled");
    line("   twin with the identical contract (NARSIL_F2SQR_SMALL picks one).");
    line("   g2_ysqr arguments: (z: *mut u64x8, g, e, f: *const u64x8 in");
    line("                       repr(C) Fp2 order with f = 3e, all Fp < p;");
    line("                       consts as mont4), z distinct from g, e, f;");
    line("   z = g^2 - 3*e^2 in Fp2 via mcl's lazy double-width shape: 4 raw");
    line("   4x4 products and 2 Montgomery reductions where the composed");
    line("   route pays 4 products and 4 reductions. Needs 4p < 2^256");
    line("   (BN254: yes), outputs canonical.");
    line("   The quotient schedule and xi=9+u constants are BN254-specific;");
    line("   carry-bound proofs live in build/schedule.rs. */");
    line("");
    line("    .text");
    line("    .intel_syntax noprefix");

    for kernel in &kernels {
        line("");
        line(&format!("/* {} register map:", kernel.symbol));
        for (reg, role) in kernel.register_map {
            line(&format!("   {:<4} {}", reg.name(), role));
        }
        line("*/");
        line("    .p2align 4");
        line(&format!("    .globl {}", kernel.symbol));
        line(&format!("    .type {}, @function", kernel.symbol));
        line(&format!("{}:", kernel.symbol));
        for body_line in &kernel.body {
            line(body_line);
        }
        line(&format!("    .size {0}, . - {0}", kernel.symbol));
    }

    // Walk tables live in .rodata: data-cache traffic, not op-cache mass.
    if kernels.iter().any(|kernel| !kernel.rodata.is_empty()) {
        line("");
        line("    .section .rodata");
        line("    .p2align 3");
        for kernel in &kernels {
            for (label, values) in &kernel.rodata {
                line(&format!("{label}:"));
                for value in values {
                    line(&format!("    .quad {value:#x}"));
                }
            }
        }
    }

    line("");
    line("    .section .note.GNU-stack, \"\", @progbits");
    out
}

/// (instructions, bytes) for the main kernels in `[fp2_xi_compact,
/// fp12_sqr_mcl, mul, sqr, sos, sosd2, sosd2_small, fp6_mul, fp12_034,
/// fp4_sqr, fp12_sqr, fp12_mul, cyc_sqr, fp12_034l, fp12_034k, sosd6]`
/// order.
pub fn kernel_sizes() -> [(usize, usize); 16] {
    let kernels = kernels();
    core::array::from_fn(|i| (kernels[i].instructions, kernels[i].bytes))
}

/// `(instructions, bytes)` for the direct `[mulpre, redc]` primitives.
pub fn mul403_primitive_sizes() -> [(usize, usize); 2] {
    let kernels = kernels();
    core::array::from_fn(|i| (kernels[16 + i].instructions, kernels[16 + i].bytes))
}

/// `(instructions, bytes)` for the `[f2sqr, f2sqr_small]` Fp2 square leaves.
pub fn f2sqr_sizes() -> [(usize, usize); 2] {
    let kernels = kernels();
    core::array::from_fn(|i| (kernels[18 + i].instructions, kernels[18 + i].bytes))
}

/// `(instructions, bytes)` for the lazy double-width G2 `y` leaf.
pub fn g2_ysqr_size() -> (usize, usize) {
    let kernels = kernels();
    (kernels[20].instructions, kernels[20].bytes)
}
