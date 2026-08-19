//! Throughput probe for the row shapes the A64 CIOS leaves issue: one price
//! per row kind, so any row schedule can be costed before it is written.
//!
//! Prices are per lane. A schedule's cost is its row counts times these.

#[cfg(not(target_arch = "aarch64"))]
fn main() {}

#[cfg(target_arch = "aarch64")]
use core::arch::asm;
#[cfg(target_arch = "aarch64")]
use std::time::Instant;

#[cfg(target_arch = "aarch64")]
use helius_narsil::{Fp2, Fp6, G1Affine, G2Affine, pairing};

#[cfg(target_arch = "aarch64")]
macro_rules! rep8 {
    ($s:literal) => {
        concat!($s, $s, $s, $s, $s, $s, $s, $s)
    };
}

/// One dual-lane column chain feeding a five word accumulator: the row the
/// CIOS leaves issue. 16 multiplies, 16 flag ops per repetition.
#[cfg(target_arch = "aarch64")]
#[inline(never)]
fn cios_row(n: u64) {
    unsafe {
        asm!(
            "1:",
            rep8!(
                // lane A column
                "mul   x1, x6, x7\n umulh x2, x6, x7\n mul x3, x6, x7\n adds x2, x2, x3\n \
                 umulh x3, x6, x7\n mul x4, x6, x7\n adcs x3, x3, x4\n \
                 umulh x4, x6, x7\n mul x5, x6, x7\n adcs x4, x4, x5\n \
                 umulh x5, x6, x7\n cinc x5, x5, hs\n \
                 adds x24, x24, x1\n adcs x25, x25, x2\n adcs x26, x26, x3\n adcs x27, x27, x4\n adc x28, x28, x5\n \
                 mul   x1, x6, x7\n umulh x2, x6, x7\n mul x3, x6, x7\n adds x2, x2, x3\n \
                 umulh x3, x6, x7\n mul x4, x6, x7\n adcs x3, x3, x4\n \
                 umulh x4, x6, x7\n mul x5, x6, x7\n adcs x4, x4, x5\n \
                 umulh x5, x6, x7\n cinc x5, x5, hs\n \
                 adds x9, x9, x1\n adcs x10, x10, x2\n adcs x11, x11, x3\n adcs x12, x12, x4\n adc x13, x13, x5\n"
            ),
            "subs {n}, {n}, #1",
            "b.ne 1b",
            n = inout(reg) n => _,
            out("x1") _, out("x2") _, out("x3") _, out("x4") _, out("x5") _,
            inout("x6") 3u64 => _, inout("x7") 5u64 => _,
            inout("x9") 0u64 => _, inout("x10") 0u64 => _, inout("x11") 0u64 => _,
            inout("x12") 0u64 => _, inout("x13") 0u64 => _,
            inout("x24") 0u64 => _, inout("x25") 0u64 => _, inout("x26") 0u64 => _,
            inout("x27") 0u64 => _, inout("x28") 0u64 => _,
            options(nostack),
        );
    }
}

/// The same column chain accumulated into an eight word double-width bank,
/// carry rippling to the top. 16 multiplies, 18 flag ops per repetition.
#[cfg(target_arch = "aarch64")]
#[inline(never)]
fn wide_row(n: u64) {
    unsafe {
        asm!(
            "1:",
            rep8!(
                "mul   x1, x6, x7\n umulh x2, x6, x7\n mul x3, x6, x7\n adds x2, x2, x3\n \
                 umulh x3, x6, x7\n mul x4, x6, x7\n adcs x3, x3, x4\n \
                 umulh x4, x6, x7\n mul x5, x6, x7\n adcs x4, x4, x5\n \
                 umulh x5, x6, x7\n cinc x5, x5, hs\n \
                 adds x21, x21, x1\n adcs x22, x22, x2\n adcs x23, x23, x3\n adcs x24, x24, x4\n \
                 adcs x25, x25, x5\n adcs x26, x26, xzr\n \
                 mul   x1, x6, x7\n umulh x2, x6, x7\n mul x3, x6, x7\n adds x2, x2, x3\n \
                 umulh x3, x6, x7\n mul x4, x6, x7\n adcs x3, x3, x4\n \
                 umulh x4, x6, x7\n mul x5, x6, x7\n adcs x4, x4, x5\n \
                 umulh x5, x6, x7\n cinc x5, x5, hs\n \
                 adds x11, x11, x1\n adcs x12, x12, x2\n adcs x13, x13, x3\n adcs x14, x14, x4\n \
                 adcs x15, x15, x5\n adcs x16, x16, xzr\n"
            ),
            "subs {n}, {n}, #1",
            "b.ne 1b",
            n = inout(reg) n => _,
            out("x1") _, out("x2") _, out("x3") _, out("x4") _, out("x5") _,
            inout("x6") 3u64 => _, inout("x7") 5u64 => _,
            inout("x11") 0u64 => _, inout("x12") 0u64 => _, inout("x13") 0u64 => _,
            inout("x14") 0u64 => _, inout("x15") 0u64 => _, inout("x16") 0u64 => _,
            inout("x21") 0u64 => _, inout("x22") 0u64 => _, inout("x23") 0u64 => _,
            inout("x24") 0u64 => _, inout("x25") 0u64 => _, inout("x26") 0u64 => _,
            options(nostack),
        );
    }
}

/// One REDC round over a double-width bank: the factor multiply, the p column
/// chain, the cancel and the shift. 18 multiplies, 20 flag ops per repetition.
#[cfg(target_arch = "aarch64")]
#[inline(never)]
fn redc_round(n: u64) {
    unsafe {
        asm!(
            "1:",
            rep8!(
                "mul   x8, x6, x21\n \
                 mul   x1, x8, x7\n umulh x2, x8, x7\n mul x3, x8, x7\n adds x2, x2, x3\n \
                 umulh x3, x8, x7\n mul x4, x8, x7\n adcs x3, x3, x4\n \
                 umulh x4, x8, x7\n mul x5, x8, x7\n adcs x4, x4, x5\n \
                 umulh x5, x8, x7\n cinc x5, x5, hs\n \
                 cmn   x21, x1\n adcs x21, x22, x2\n adcs x22, x23, x3\n adcs x23, x24, x4\n \
                 adcs x24, x25, x5\n adcs x25, x26, xzr\n \
                 mul   x8, x6, x11\n \
                 mul   x1, x8, x7\n umulh x2, x8, x7\n mul x3, x8, x7\n adds x2, x2, x3\n \
                 umulh x3, x8, x7\n mul x4, x8, x7\n adcs x3, x3, x4\n \
                 umulh x4, x8, x7\n mul x5, x8, x7\n adcs x4, x4, x5\n \
                 umulh x5, x8, x7\n cinc x5, x5, hs\n \
                 cmn   x11, x1\n adcs x11, x12, x2\n adcs x12, x13, x3\n adcs x13, x14, x4\n \
                 adcs x14, x15, x5\n adcs x15, x16, xzr\n"
            ),
            "subs {n}, {n}, #1",
            "b.ne 1b",
            n = inout(reg) n => _,
            out("x1") _, out("x2") _, out("x3") _, out("x4") _, out("x5") _, out("x8") _,
            inout("x6") 3u64 => _, inout("x7") 5u64 => _,
            inout("x11") 1u64 => _, inout("x12") 0u64 => _, inout("x13") 0u64 => _,
            inout("x14") 0u64 => _, inout("x15") 0u64 => _, inout("x16") 0u64 => _,
            inout("x21") 1u64 => _, inout("x22") 0u64 => _, inout("x23") 0u64 => _,
            inout("x24") 0u64 => _, inout("x25") 0u64 => _, inout("x26") 0u64 => _,
            options(nostack),
        );
    }
}

/// Eight word accumulate of a staged double-width value: the build rows of the
/// lazy layer. 16 flag ops and 8 loads per repetition, no multiplies.
#[cfg(target_arch = "aarch64")]
#[inline(never)]
fn wide_add(n: u64, buf: *const u64) {
    unsafe {
        asm!(
            "1:",
            rep8!(
                "ldp x27, x28, [x0]\n adds x21, x21, x27\n adcs x22, x22, x28\n \
                 ldp x27, x28, [x0, #16]\n adcs x23, x23, x27\n adcs x24, x24, x28\n \
                 ldp x27, x28, [x0, #32]\n adcs x25, x25, x27\n adcs x26, x26, x28\n \
                 ldp x27, x28, [x0, #48]\n adcs x3, x3, x27\n adcs x4, x4, x28\n \
                 ldp x17, x8, [x0, #64]\n adds x11, x11, x17\n adcs x12, x12, x8\n \
                 ldp x17, x8, [x0, #80]\n adcs x13, x13, x17\n adcs x14, x14, x8\n \
                 ldp x17, x8, [x0, #96]\n adcs x15, x15, x17\n adcs x16, x16, x8\n \
                 ldp x17, x8, [x0, #112]\n adcs x9, x9, x17\n adcs x10, x10, x8\n"
            ),
            "subs {n}, {n}, #1",
            "b.ne 1b",
            n = inout(reg) n => _,
            inout("x0") buf => _,
            out("x8") _, out("x17") _, out("x27") _, out("x28") _,
            inout("x9") 0u64 => _, inout("x10") 0u64 => _, inout("x11") 0u64 => _,
            inout("x12") 0u64 => _, inout("x13") 0u64 => _, inout("x14") 0u64 => _,
            inout("x15") 0u64 => _, inout("x16") 0u64 => _,
            inout("x3") 0u64 => _, inout("x4") 0u64 => _, inout("x21") 0u64 => _,
            inout("x22") 0u64 => _, inout("x23") 0u64 => _, inout("x24") 0u64 => _,
            inout("x25") 0u64 => _, inout("x26") 0u64 => _,
            options(readonly, nostack),
        );
    }
}

/// One modular add with its conditional subtraction, both lanes.
/// 20 flag ops, 8 selects per repetition.
#[cfg(target_arch = "aarch64")]
#[inline(never)]
fn mod_add(n: u64) {
    unsafe {
        asm!(
            "1:",
            rep8!(
                "adds x20, x20, x1\n adcs x21, x21, x2\n adcs x22, x22, x3\n adcs x23, x23, x4\n \
                 cset x17, hs\n \
                 subs x24, x20, x5\n sbcs x25, x21, x6\n sbcs x26, x22, x7\n sbcs x27, x23, x8\n \
                 sbcs x17, x17, xzr\n \
                 csel x20, x24, x20, hs\n csel x21, x25, x21, hs\n csel x22, x26, x22, hs\n csel x23, x27, x23, hs\n \
                 adds x9, x9, x1\n adcs x10, x10, x2\n adcs x11, x11, x3\n adcs x12, x12, x4\n \
                 cset x28, hs\n \
                 subs x13, x9, x5\n sbcs x14, x10, x6\n sbcs x15, x11, x7\n sbcs x16, x12, x8\n \
                 sbcs x28, x28, xzr\n \
                 csel x9, x13, x9, hs\n csel x10, x14, x10, hs\n csel x11, x15, x11, hs\n csel x12, x16, x12, hs\n"
            ),
            "subs {n}, {n}, #1",
            "b.ne 1b",
            n = inout(reg) n => _,
            inout("x1") 1u64 => _, inout("x2") 2u64 => _, inout("x3") 3u64 => _, inout("x4") 4u64 => _,
            inout("x5") 5u64 => _, inout("x6") 6u64 => _, inout("x7") 7u64 => _, inout("x8") 8u64 => _,
            out("x13") _, out("x14") _, out("x15") _, out("x16") _, out("x17") _,
            out("x24") _, out("x25") _, out("x26") _, out("x27") _, out("x28") _,
            inout("x9") 0u64 => _, inout("x10") 0u64 => _, inout("x11") 0u64 => _, inout("x12") 0u64 => _,
            inout("x20") 0u64 => _, inout("x21") 0u64 => _, inout("x22") 0u64 => _, inout("x23") 0u64 => _,
            options(nostack),
        );
    }
}

/// Minimum wall time per unit over `rounds` runs of `n * 8` repetitions.
#[cfg(target_arch = "aarch64")]
fn probe(name: &str, rounds: u32, n: u64, f: &dyn Fn(u64)) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..rounds {
        let start = Instant::now();
        f(n);
        let elapsed = start.elapsed().as_secs_f64();
        best = best.min(elapsed);
    }
    let per = best / (n as f64 * 8.0) * 1e9;
    println!("{name:<12} {per:8.3} ns per repetition");
    per
}

#[cfg(target_arch = "aarch64")]
fn time<T>(name: &str, rounds: u32, n: u64, f: &dyn Fn() -> T) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..rounds {
        let start = Instant::now();
        for _ in 0..n {
            std::hint::black_box(f());
        }
        let elapsed = start.elapsed().as_secs_f64();
        best = best.min(elapsed);
    }
    let per = best / n as f64 * 1e9;
    println!("{name:<28} {per:10.2} ns");
    per
}

#[cfg(target_arch = "aarch64")]
fn main() {
    let buf = vec![0u64; 32];
    let rounds = 400;

    println!("row shapes, per repetition (two lanes each)");
    let cios = probe("cios_row", rounds, 2000, &cios_row);
    let wide = probe("wide_row", rounds, 2000, &wide_row);
    let redc = probe("redc_round", rounds, 2000, &redc_round);
    let add8 = probe("wide_add", rounds, 2000, &|n| wide_add(n, buf.as_ptr()));
    let madd = probe("mod_add", rounds, 2000, &mod_add);

    println!();
    println!(
        "per lane: cios {:.3} wide {:.3} redc {:.3} wide_add {:.3} mod_add {:.3}",
        cios / 2.0,
        wide / 2.0,
        redc / 2.0,
        add8 / 2.0,
        madd / 2.0
    );

    // Today's Fp4: sosd4 is 32 product rows + 8 reduce rows, sosd2 is
    // 16 + 8, and the Rust prep is 12 modular folds.
    let today = 48.0 * (cios / 2.0) + 16.0 * (cios / 2.0) + 12.0 * (madd / 2.0);
    // Lazy Fp4: 24 wide product rows, 16 REDC rounds, 14 eight-word
    // accumulates and 10 modular folds of prep.
    let lazy =
        24.0 * (wide / 2.0) + 16.0 * (redc / 2.0) + 14.0 * (add8 / 2.0) + 10.0 * (madd / 2.0);
    println!();
    println!("predicted Fp4 square: today {today:.1} ns, lazy {lazy:.1} ns");

    println!();
    let a2 = Fp2::ONE + Fp2::ONE;
    let b2 = a2.square();
    let _ = Fp6::new(a2, b2, a2);
    let m = pairing::miller_loop(&G1Affine::generator(), &G2Affine::generator());
    let f = pairing::final_exponentiation(&m);
    time("fp2_mul", rounds, 20000, &|| {
        std::hint::black_box(a2) * std::hint::black_box(b2)
    });
    time("fp12_cyclotomic_square", rounds, 20000, &|| {
        std::hint::black_box(f).cyclotomic_square()
    });
    time("fp12_mul", rounds, 20000, &|| {
        std::hint::black_box(m) * std::hint::black_box(f)
    });
    time("full_pairing", 20, 40, &|| {
        pairing::pairing(&G1Affine::generator(), &G2Affine::generator())
    });
}
