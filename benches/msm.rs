use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use helius_narsil::{Fr, G1Affine, G1Projective, msm_variable_time_affine};

fn bench_msm(c: &mut Criterion) {
    let base = G1Projective::generator();
    let mut group = c.benchmark_group("narsil/g1_msm");

    for count in [1usize, 8, 32, 128] {
        let points: Vec<G1Affine> = (1..=count)
            .map(|value| base.mul_scalar(Fr::from_u64(value as u64 + 1)).to_affine())
            .collect();
        let scalars: Vec<[u64; 4]> = (1..=count)
            .map(|value| Fr::from_u64(value as u64 + 17).to_raw())
            .collect();
        group.bench_with_input(BenchmarkId::from_parameter(count), &count, |bench, _| {
            bench.iter(|| msm_variable_time_affine(black_box(&points), black_box(&scalars)));
        });
    }

    group.finish();
}

criterion_group!(benches, bench_msm);
criterion_main!(benches);
