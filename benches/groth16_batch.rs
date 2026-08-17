//! Batch Groth16 verification, and the primitives the batch route spends on.
//!
//! The sequential row verifies the same statements one at a time, so the
//! crossover is read directly off the table.

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use helius_narsil::{
    Fp12, G1Affine, G2Affine,
    pairing::{final_exponentiation, miller_loop, multi_miller_loop},
};

#[path = "../tests/support/groth16_batch.rs"]
mod fixture;

use fixture::{Batch, Rng, g1, g2};

const SIZES: [usize; 8] = [2, 3, 4, 5, 6, 7, 8, 16];
const MULTI_VK_KEYS: usize = 2;

fn batch_rows(c: &mut Criterion) {
    let mut group = c.benchmark_group("groth16_batch");
    group.measurement_time(Duration::from_secs(6));
    for size in SIZES {
        let same = Batch::new(0x5eed_0001 + size as u64, size, 1);
        assert!(same.verify());
        assert!(same.verify_sequential());
        group.bench_with_input(BenchmarkId::new("same_vk", size), &same, |b, batch| {
            b.iter(|| black_box(batch.verify()))
        });
        group.bench_with_input(BenchmarkId::new("sequential", size), &same, |b, batch| {
            b.iter(|| black_box(batch.verify_sequential()))
        });
        if size % MULTI_VK_KEYS == 0 {
            let multi = Batch::new(0x5eed_0100 + size as u64, size, MULTI_VK_KEYS);
            assert!(multi.verify());
            group.bench_with_input(BenchmarkId::new("multi_vk", size), &multi, |b, batch| {
                b.iter(|| black_box(batch.verify()))
            });
        }
    }
    group.finish();
}

/// Per-term cost of the two Miller routes. The eight-lane engine runs a whole
/// pass whether or not its lanes are masked, so this table fixes where a group
/// belongs.
fn miller_routes(c: &mut Criterion) {
    let mut rng = Rng::new(0x1234_5678_9abc_def0);
    let points: Vec<(G1Affine, G2Affine)> = (0..16)
        .map(|_| (g1(rng.scalar()), g2(rng.scalar())))
        .collect();
    let mut group = c.benchmark_group("miller_terms");
    group.measurement_time(Duration::from_secs(6));
    for terms in [1usize, 2, 3, 4, 5, 6, 8, 11, 12, 14, 16] {
        let slice = &points[..terms];
        let refs: Vec<(&G1Affine, &G2Affine)> = slice.iter().map(|(p, q)| (p, q)).collect();
        group.bench_with_input(BenchmarkId::new("fused", terms), &refs, |b, refs| {
            b.iter(|| black_box(multi_miller_loop(refs)))
        });
        group.bench_with_input(
            BenchmarkId::new("independent", terms),
            &slice,
            |b, slice| {
                b.iter(|| {
                    let mut product = Fp12::ONE;
                    for (p, q) in slice.iter() {
                        product *= miller_loop(p, q);
                    }
                    black_box(product)
                })
            },
        );
    }
    group.finish();
}

fn primitives(c: &mut Criterion) {
    let mut rng = Rng::new(0xfeed_face_dead_beef);
    let point = g1(rng.scalar());
    let scalar = rng.scalar();
    let (p, q) = (g1(rng.scalar()), g2(rng.scalar()));
    let raw = miller_loop(&p, &q);
    let mut group = c.benchmark_group("primitive");
    group.bench_function("g1_mul_scalar", |b| {
        b.iter(|| black_box(black_box(point).to_curve().mul_scalar(black_box(scalar))))
    });
    group.bench_function("g1_to_affine", |b| {
        let projective = point.to_curve().mul_scalar(scalar);
        b.iter(|| black_box(black_box(projective).to_affine()))
    });
    group.bench_function("final_exponentiation", |b| {
        b.iter(|| black_box(final_exponentiation(black_box(&raw))))
    });
    group.finish();
}

criterion_group!(benches, batch_rows, miller_routes, primitives);
criterion_main!(benches);
