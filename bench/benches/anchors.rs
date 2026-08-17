//! Criterion anchors for gate A.
//!
//! Gate A cross-checks the campaign runner against a second statistical
//! harness. Both must time the same work on the same fixtures, so this bench
//! builds its pools from `FOUR_LANE_FIXTURE_SEED` exactly as the runner does
//! and calls the same lane adapters. Criterion keeps its own clock and its own
//! sampling, which is what makes the comparison independent.
//!
//! Group names and the `helius`, `arkworks`, `mcl` function names are the
//! contract `bench/scripts/cross_check.py` reads.

use std::time::Duration;

use criterion::{
    BenchmarkId, Criterion, black_box, criterion_group, criterion_main, measurement::WallTime,
};
use helius_narsil_bench::{
    MILLER_POOL_DOMAIN, MILLER_POOL_SIZE, MclFourLaneOp, MclFourLanePool, MillerFixture,
    ProductionGroth16Pool, ark_final_exponentiation, ark_full_pairing,
    ark_groth16_prepared_schedules, ark_groth16_verify_accumulated, ark_miller,
    ark_miller_prepared_replay, ark_miller_prepared_setup, assert_miller_equivalent,
    helius_final_exponentiation, helius_full_pairing, helius_groth16_verify_accumulated,
    helius_miller, helius_miller_prepared, seeded_pool,
};

/// The campaign passes one seed to every spawn. An unset variable keeps the
/// frozen pools, which is what a laptop smoke run wants.
fn fixture_seed() -> u64 {
    let Some(raw) = std::env::var_os("FOUR_LANE_FIXTURE_SEED") else {
        return 0;
    };
    let text = raw.to_string_lossy().trim().to_owned();
    let parsed = match text.strip_prefix("0x") {
        Some(hex) => u64::from_str_radix(hex, 16),
        None => text.parse::<u64>(),
    };
    parsed.expect("FOUR_LANE_FIXTURE_SEED must be a u64, decimal or 0x hex")
}

/// Register one lane under the name gate A looks for.
fn register<T, R>(
    group: &mut criterion::BenchmarkGroup<'_, WallTime>,
    lane: &'static str,
    pool: &[T],
    operation: impl Fn(&T) -> R,
) {
    let mut cursor = 0_usize;
    group.bench_with_input(BenchmarkId::new(lane, 1), &1, |bencher, _| {
        bencher.iter(|| {
            let input = &pool[cursor];
            cursor = (cursor + 1) % pool.len();
            black_box(operation(black_box(input)))
        });
    });
}

/// Register the mcl lane. The bridge runs one iteration per call, so Criterion
/// still owns the timing loop, and the row measures the same C++ body the
/// campaign measures.
fn register_mcl(
    group: &mut criterion::BenchmarkGroup<'_, WallTime>,
    pool: &MclFourLanePool<'_>,
    operation: MclFourLaneOp,
    length: usize,
) {
    let mut cursor = 0_usize;
    group.bench_with_input(BenchmarkId::new("mcl", 1), &1, |bencher, _| {
        bencher.iter(|| {
            let rotation = cursor;
            cursor = (cursor + 1) % length;
            black_box(pool.run(operation, 1, rotation))
        });
    });
}

fn anchors(c: &mut Criterion) {
    let seed = fixture_seed();
    let miller = seeded_pool(
        seed,
        MILLER_POOL_DOMAIN,
        MILLER_POOL_SIZE,
        MillerFixture::from_seed,
    );
    // Gate A compares two harnesses, so it is worth nothing unless both time a
    // correct computation. The runner proves this before every campaign spawn.
    for fixture in &miller {
        assert_miller_equivalent(fixture);
    }
    let mcl_miller = MclFourLanePool::miller(&miller);

    let mut group = c.benchmark_group("bn254_single_pair_miller_loop");
    register(&mut group, "helius", &miller, helius_miller);
    register(&mut group, "arkworks", &miller, ark_miller);
    register_mcl(
        &mut group,
        &mcl_miller,
        MclFourLaneOp::MillerLive,
        miller.len(),
    );
    group.finish();

    let mut group = c.benchmark_group("bn254_final_exponentiation");
    register(&mut group, "helius", &miller, helius_final_exponentiation);
    register(&mut group, "arkworks", &miller, ark_final_exponentiation);
    register_mcl(
        &mut group,
        &mcl_miller,
        MclFourLaneOp::FinalExponentiation,
        miller.len(),
    );
    group.finish();

    let mut group = c.benchmark_group("bn254_single_pair_full_pairing");
    register(&mut group, "helius", &miller, helius_full_pairing);
    register(&mut group, "arkworks", &miller, ark_full_pairing);
    register_mcl(
        &mut group,
        &mcl_miller,
        MclFourLaneOp::FullPairing,
        miller.len(),
    );
    group.finish();

    // ark-ec 0.5 consumes a prepared schedule by value, so the campaign builds
    // one per replay outside its timer. `iter_batched` is the same shape.
    let mut group = c.benchmark_group("bn254_single_pair_miller_loop_prepared_g2");
    register(&mut group, "helius", &miller, helius_miller_prepared);
    let mut cursor = 0_usize;
    group.bench_with_input(BenchmarkId::new("arkworks", 1), &1, |bencher, _| {
        bencher.iter_batched(
            || {
                let index = cursor;
                cursor = (cursor + 1) % miller.len();
                (index, ark_miller_prepared_setup(&miller[index]))
            },
            |(index, schedule)| {
                black_box(ark_miller_prepared_replay(
                    black_box(&miller[index]),
                    black_box(schedule),
                ))
            },
            criterion::BatchSize::SmallInput,
        );
    });
    register_mcl(
        &mut group,
        &mcl_miller,
        MclFourLaneOp::MillerPrepared,
        miller.len(),
    );
    group.finish();

    let production = ProductionGroth16Pool::from_seed(seed);
    let standard = production.standard();
    let mcl_production = MclFourLanePool::groth16(standard);
    let mut group = c.benchmark_group("bn254_groth16_verify_prepared_e2e");
    register(
        &mut group,
        "helius",
        standard,
        helius_groth16_verify_accumulated,
    );
    let mut cursor = 0_usize;
    group.bench_with_input(BenchmarkId::new("arkworks", 1), &1, |bencher, _| {
        bencher.iter_batched(
            || {
                let index = cursor;
                cursor = (cursor + 1) % standard.len();
                (index, ark_groth16_prepared_schedules(&standard[index]))
            },
            |(index, (gamma_neg, delta_neg))| {
                black_box(ark_groth16_verify_accumulated(
                    black_box(&standard[index]),
                    gamma_neg,
                    delta_neg,
                ))
            },
            criterion::BatchSize::SmallInput,
        );
    });
    register_mcl(
        &mut group,
        &mcl_production,
        MclFourLaneOp::Groth16Accumulated,
        standard.len(),
    );
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(50)
        .warm_up_time(Duration::from_secs(2))
        .measurement_time(Duration::from_secs(10));
    targets = anchors
}
criterion_main!(benches);
