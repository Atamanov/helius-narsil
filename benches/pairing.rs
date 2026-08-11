use criterion::{Criterion, black_box, criterion_group, criterion_main};
use helius_narsil::{
    G1Affine, G2Affine, pairing,
    pairing::{final_exponentiation, miller_loop, miller_loop_prepared, prepare_g2},
};

fn bench_pairing(c: &mut Criterion) {
    let p = G1Affine::generator();
    let q = G2Affine::generator();
    let prepared = prepare_g2(&q).expect("generator is in the G2 subgroup");
    let miller = miller_loop(&p, &q);

    c.bench_function("narsil/miller_loop", |bench| {
        bench.iter(|| miller_loop(black_box(&p), black_box(&q)));
    });
    c.bench_function("narsil/miller_loop_prepared", |bench| {
        bench.iter(|| miller_loop_prepared(black_box(&p), black_box(&prepared)));
    });
    c.bench_function("narsil/final_exponentiation", |bench| {
        bench.iter(|| final_exponentiation(black_box(&miller)));
    });
    c.bench_function("narsil/pairing", |bench| {
        bench.iter(|| pairing(black_box(&p), black_box(&q)));
    });
}

criterion_group!(benches, bench_pairing);
criterion_main!(benches);
