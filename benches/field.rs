use criterion::{Criterion, black_box, criterion_group, criterion_main};
use helius_narsil::{Fp, Fp2};

fn bench_field(c: &mut Criterion) {
    let a = Fp::from_u64(123_456_789);
    let b = Fp::from_u64(987_654_321);
    let a2 = Fp2::new(a, b);
    let b2 = Fp2::new(b, a);

    c.bench_function("narsil/fp_mul", |bench| {
        bench.iter(|| black_box(a) * black_box(b));
    });
    c.bench_function("narsil/fp_square", |bench| {
        bench.iter(|| black_box(a).square());
    });
    c.bench_function("narsil/fp2_mul", |bench| {
        bench.iter(|| black_box(a2) * black_box(b2));
    });
    c.bench_function("narsil/fp2_square", |bench| {
        bench.iter(|| black_box(a2).square());
    });
}

criterion_group!(benches, bench_field);
criterion_main!(benches);
