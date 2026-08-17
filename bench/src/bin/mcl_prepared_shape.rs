//! Price mcl's two-pair ceiling on prepared Miller products.
//!
//! `mcl_prepared_shape ITERATIONS` prints one JSON line.
//!
//! The disclosures say mcl's widest prepared entry covers two pairs, so a
//! three-pair Groth16 product runs as one two-pair loop plus one one-pair loop
//! and one Fp12 multiply. The extra one-pair loop repeats a Miller squaring
//! chain that a single wider accumulator would run once. This measures that
//! penalty instead of asserting it.
//!
//! penalty = one_pair - (two_pair - one_pair) + fp12_mul
//!
//! `two_pair - one_pair` is what a further pair costs inside an accumulator
//! that already exists, so the difference from a standalone one-pair loop is
//! the chain mcl cannot share. The estimate is an upper bound on the true
//! penalty, because mcl publishes no wider entry to measure against.

use std::{
    hint::black_box,
    process::exit,
    time::{Duration, Instant},
};

use helius_narsil_bench::{
    FIELD_POOL_DOMAIN, FieldFixture, MILLER_POOL_DOMAIN, MclFourLaneOp, MclFourLanePool,
    MillerFixture, mcl_prepared_shape_run, seeded_pool,
};

fn main() {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let Some(iterations) = arguments
        .first()
        .and_then(|value| value.parse::<usize>().ok())
    else {
        eprintln!("usage: mcl_prepared_shape ITERATIONS");
        exit(2);
    };
    if iterations == 0 {
        eprintln!("ITERATIONS must be positive");
        exit(2);
    }

    let miller = seeded_pool(0, MILLER_POOL_DOMAIN, 1, MillerFixture::from_seed);
    let field = seeded_pool(0, FIELD_POOL_DOMAIN, 1, FieldFixture::from_seed);
    let mcl_field = MclFourLanePool::field(&field);

    // One untimed pass per shape, so the first timed call meets warm code and
    // a warm schedule.
    black_box(mcl_prepared_shape_run(&miller[0], 1, iterations / 8 + 1));
    black_box(mcl_prepared_shape_run(&miller[0], 2, iterations / 8 + 1));
    black_box(mcl_field.run(MclFourLaneOp::Fp12Mul, iterations / 8 + 1, 0));

    let one = time(|| mcl_prepared_shape_run(&miller[0], 1, iterations));
    let two = time(|| mcl_prepared_shape_run(&miller[0], 2, iterations));
    let fp12_mul = time(|| mcl_field.run(MclFourLaneOp::Fp12Mul, iterations, 0));

    let one_ns = per_call(one, iterations);
    let two_ns = per_call(two, iterations);
    let fp12_ns = per_call(fp12_mul, iterations);
    let marginal_ns = two_ns - one_ns;
    let penalty_ns = one_ns - marginal_ns + fp12_ns;

    println!(
        "{}",
        serde_json::json!({
            "schema": "mcl-prepared-shape-v1",
            "iterations": iterations,
            "prepared_miller_1_pair_ns": one_ns,
            "prepared_miller_2_pairs_ns": two_ns,
            "fp12_mul_ns": fp12_ns,
            "marginal_pair_inside_accumulator_ns": marginal_ns,
            "three_pair_split_penalty_ns": penalty_ns,
        })
    );
}

fn time<R>(mut operation: impl FnMut() -> R) -> Duration {
    let started = Instant::now();
    black_box(operation());
    started.elapsed()
}

fn per_call(elapsed: Duration, iterations: usize) -> f64 {
    elapsed.as_nanos() as f64 / iterations as f64
}
