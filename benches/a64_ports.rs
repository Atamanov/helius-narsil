//! Issue-width probe for the AArch64 integer pipes.
//!
//! Each case issues a block of 64 instructions and reports time per block,
//! so call overhead is under two percent of every measurement. Dividing the
//! instruction count by the measured cycles gives the sustained rate of a
//! pipe class, which decides whether a hand kernel is bound by multiply
//! issue, by flag issue, or by chain latency. The dependent multiply chain
//! calibrates the clock, since its per instruction cost is the published
//! multiply latency.
#![cfg(target_arch = "aarch64")]

use core::arch::asm;
use criterion::{Criterion, black_box, criterion_group, criterion_main};

macro_rules! rep64 {
    ($s:literal) => {
        concat!(
            $s, $s, $s, $s, $s, $s, $s, $s, $s, $s, $s, $s, $s, $s, $s, $s, $s, $s, $s, $s, $s, $s,
            $s, $s, $s, $s, $s, $s, $s, $s, $s, $s, $s, $s, $s, $s, $s, $s, $s, $s, $s, $s, $s, $s,
            $s, $s, $s, $s, $s, $s, $s, $s, $s, $s, $s, $s, $s, $s, $s, $s, $s, $s, $s, $s,
        )
    };
}

/// 64 independent `mul` into eight rotating destinations.
#[inline(never)]
fn mul_64(a: u64, b: u64) -> u64 {
    let (r0, r1, r2, r3, r4, r5, r6, r7): (u64, u64, u64, u64, u64, u64, u64, u64);
    unsafe {
        asm!(
            rep64!(
                "mul {r0}, {a}, {b}\n mul {r1}, {a}, {b}\n mul {r2}, {a}, {b}\n mul {r3}, {a}, {b}\n \
                 mul {r4}, {a}, {b}\n mul {r5}, {a}, {b}\n mul {r6}, {a}, {b}\n mul {r7}, {a}, {b}\n"
            ),
            a = in(reg) a, b = in(reg) b,
            r0 = out(reg) r0, r1 = out(reg) r1, r2 = out(reg) r2, r3 = out(reg) r3,
            r4 = out(reg) r4, r5 = out(reg) r5, r6 = out(reg) r6, r7 = out(reg) r7,
            options(nomem, nostack),
        );
    }
    r0 ^ r1 ^ r2 ^ r3 ^ r4 ^ r5 ^ r6 ^ r7
}

/// 64 independent `umulh`.
#[inline(never)]
fn umulh_64(a: u64, b: u64) -> u64 {
    let (r0, r1, r2, r3, r4, r5, r6, r7): (u64, u64, u64, u64, u64, u64, u64, u64);
    unsafe {
        asm!(
            rep64!(
                "umulh {r0}, {a}, {b}\n umulh {r1}, {a}, {b}\n umulh {r2}, {a}, {b}\n umulh {r3}, {a}, {b}\n \
                 umulh {r4}, {a}, {b}\n umulh {r5}, {a}, {b}\n umulh {r6}, {a}, {b}\n umulh {r7}, {a}, {b}\n"
            ),
            a = in(reg) a, b = in(reg) b,
            r0 = out(reg) r0, r1 = out(reg) r1, r2 = out(reg) r2, r3 = out(reg) r3,
            r4 = out(reg) r4, r5 = out(reg) r5, r6 = out(reg) r6, r7 = out(reg) r7,
            options(nomem, nostack),
        );
    }
    r0 ^ r1 ^ r2 ^ r3 ^ r4 ^ r5 ^ r6 ^ r7
}

/// 32 `mul` + `umulh` pairs, the full 128-bit product shape a CIOS row
/// issues.
#[inline(never)]
fn mul_umulh_64(a: u64, b: u64) -> u64 {
    let (r0, r1, r2, r3, r4, r5, r6, r7): (u64, u64, u64, u64, u64, u64, u64, u64);
    unsafe {
        asm!(
            rep64!(
                "mul {r0}, {a}, {b}\n umulh {r1}, {a}, {b}\n mul {r2}, {a}, {b}\n umulh {r3}, {a}, {b}\n \
                 mul {r4}, {a}, {b}\n umulh {r5}, {a}, {b}\n mul {r6}, {a}, {b}\n umulh {r7}, {a}, {b}\n"
            ),
            a = in(reg) a, b = in(reg) b,
            r0 = out(reg) r0, r1 = out(reg) r1, r2 = out(reg) r2, r3 = out(reg) r3,
            r4 = out(reg) r4, r5 = out(reg) r5, r6 = out(reg) r6, r7 = out(reg) r7,
            options(nomem, nostack),
        );
    }
    r0 ^ r1 ^ r2 ^ r3 ^ r4 ^ r5 ^ r6 ^ r7
}

/// 64 dependent `mul`. Cost per instruction is multiply latency, which
/// calibrates the clock.
#[inline(never)]
fn mul_chain_64(a: u64, b: u64) -> u64 {
    let r: u64;
    unsafe {
        asm!(
            "mov {r}, {a}",
            rep64!("mul {r}, {r}, {b}\n"),
            a = in(reg) a, b = in(reg) b, r = out(reg) r,
            options(nomem, nostack),
        );
    }
    r
}

/// 64 independent `adds`, each opening its own flag chain.
#[inline(never)]
fn adds_64(a: u64, b: u64) -> u64 {
    let (r0, r1, r2, r3, r4, r5, r6, r7): (u64, u64, u64, u64, u64, u64, u64, u64);
    unsafe {
        asm!(
            rep64!(
                "adds {r0}, {a}, {b}\n adds {r1}, {a}, {b}\n adds {r2}, {a}, {b}\n adds {r3}, {a}, {b}\n \
                 adds {r4}, {a}, {b}\n adds {r5}, {a}, {b}\n adds {r6}, {a}, {b}\n adds {r7}, {a}, {b}\n"
            ),
            a = in(reg) a, b = in(reg) b,
            r0 = out(reg) r0, r1 = out(reg) r1, r2 = out(reg) r2, r3 = out(reg) r3,
            r4 = out(reg) r4, r5 = out(reg) r5, r6 = out(reg) r6, r7 = out(reg) r7,
            options(nomem, nostack),
        );
    }
    r0 ^ r1 ^ r2 ^ r3 ^ r4 ^ r5 ^ r6 ^ r7
}

/// One serial 64-long carry chain.
#[inline(never)]
fn chain_x1(a: u64, b: u64) -> u64 {
    let (r0, r1, r2, r3, r4, r5, r6, r7): (u64, u64, u64, u64, u64, u64, u64, u64);
    unsafe {
        asm!(
            "adds {r0}, {a}, {b}",
            rep64!(
                "adcs {r1}, {a}, {b}\n adcs {r2}, {a}, {b}\n adcs {r3}, {a}, {b}\n adcs {r4}, {a}, {b}\n \
                 adcs {r5}, {a}, {b}\n adcs {r6}, {a}, {b}\n adcs {r7}, {a}, {b}\n adcs {r0}, {a}, {b}\n"
            ),
            a = in(reg) a, b = in(reg) b,
            r0 = out(reg) r0, r1 = out(reg) r1, r2 = out(reg) r2, r3 = out(reg) r3,
            r4 = out(reg) r4, r5 = out(reg) r5, r6 = out(reg) r6, r7 = out(reg) r7,
            options(nomem, nostack),
        );
    }
    r0 ^ r1 ^ r2 ^ r3 ^ r4 ^ r5 ^ r6 ^ r7
}

/// Independent 4-long chains back to back, 64 flag ops in all. If NZCV
/// renaming lets independent chains overlap, this beats one long chain.
#[inline(never)]
fn chain_len4(a: u64, b: u64) -> u64 {
    let (r0, r1, r2, r3, r4, r5, r6, r7): (u64, u64, u64, u64, u64, u64, u64, u64);
    unsafe {
        asm!(
            rep64!(
                "adds {r0}, {a}, {b}\n adcs {r1}, {a}, {b}\n adcs {r2}, {a}, {b}\n adcs {r3}, {a}, {b}\n \
                 adds {r4}, {a}, {b}\n adcs {r5}, {a}, {b}\n adcs {r6}, {a}, {b}\n adcs {r7}, {a}, {b}\n"
            ),
            a = in(reg) a, b = in(reg) b,
            r0 = out(reg) r0, r1 = out(reg) r1, r2 = out(reg) r2, r3 = out(reg) r3,
            r4 = out(reg) r4, r5 = out(reg) r5, r6 = out(reg) r6, r7 = out(reg) r7,
            options(nomem, nostack),
        );
    }
    r0 ^ r1 ^ r2 ^ r3 ^ r4 ^ r5 ^ r6 ^ r7
}

/// A CIOS-shaped mix, one full product row per eight instructions. Four
/// products feeding a four-long carry chain, the ratio a real kernel issues.
#[inline(never)]
fn row_mix_64(a: u64, b: u64) -> u64 {
    let (r0, r1, r2, r3, r4, r5, r6, r7): (u64, u64, u64, u64, u64, u64, u64, u64);
    unsafe {
        asm!(
            rep64!(
                "mul {r0}, {a}, {b}\n umulh {r1}, {a}, {b}\n mul {r2}, {a}, {b}\n umulh {r3}, {a}, {b}\n \
                 adds {r4}, {r0}, {r1}\n adcs {r5}, {r2}, {r3}\n adcs {r6}, {r4}, {b}\n adcs {r7}, {r5}, {b}\n"
            ),
            a = in(reg) a, b = in(reg) b,
            r0 = out(reg) r0, r1 = out(reg) r1, r2 = out(reg) r2, r3 = out(reg) r3,
            r4 = out(reg) r4, r5 = out(reg) r5, r6 = out(reg) r6, r7 = out(reg) r7,
            options(nomem, nostack),
        );
    }
    r0 ^ r1 ^ r2 ^ r3 ^ r4 ^ r5 ^ r6 ^ r7
}

fn bench(c: &mut Criterion) {
    let a = black_box(0x9e37_79b9_7f4a_7c15u64);
    let b = black_box(0xbf58_476d_1ce4_e5b9u64);
    let mut g = c.benchmark_group("a64_ports");
    g.bench_function("mul_64", |t| t.iter(|| mul_64(black_box(a), black_box(b))));
    g.bench_function("umulh_64", |t| {
        t.iter(|| umulh_64(black_box(a), black_box(b)))
    });
    g.bench_function("mul_umulh_64", |t| {
        t.iter(|| mul_umulh_64(black_box(a), black_box(b)))
    });
    g.bench_function("mul_chain_64", |t| {
        t.iter(|| mul_chain_64(black_box(a), black_box(b)))
    });
    g.bench_function("adds_64", |t| {
        t.iter(|| adds_64(black_box(a), black_box(b)))
    });
    g.bench_function("chain_x1_len64", |t| {
        t.iter(|| chain_x1(black_box(a), black_box(b)))
    });
    g.bench_function("chain_x16_len4", |t| {
        t.iter(|| chain_len4(black_box(a), black_box(b)))
    });
    g.bench_function("row_mix_64", |t| {
        t.iter(|| row_mix_64(black_box(a), black_box(b)))
    });
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
