//! Tower operations that the pairing spends its time in.
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use helius_narsil::{Fp2, Fp6, G1Affine, G2Affine, G2Projective, pairing};

fn bench(c: &mut Criterion) {
    let m = pairing::miller_loop(&G1Affine::generator(), &G2Affine::generator());
    let f = pairing::final_exponentiation(&m);
    let a2 = Fp2::ONE + Fp2::ONE;
    let b2 = a2.square();
    let a6 = Fp6::new(a2, b2, a2);
    let b6 = Fp6::new(b2, a2, b2);
    let q = G2Projective::from(G2Affine::generator());

    c.bench_function("narsil/fp2_mul", |t| {
        t.iter(|| black_box(a2) * black_box(b2))
    });
    c.bench_function("narsil/fp2_square", |t| t.iter(|| black_box(a2).square()));
    c.bench_function("narsil/fp6_mul", |t| {
        t.iter(|| black_box(a6) * black_box(b6))
    });
    c.bench_function("narsil/fp6_square", |t| t.iter(|| black_box(a6).square()));
    c.bench_function("narsil/fp12_mul", |t| {
        t.iter(|| black_box(m) * black_box(f))
    });
    c.bench_function("narsil/fp12_square", |t| t.iter(|| black_box(m).square()));
    c.bench_function("narsil/fp12_cyclotomic_square", |t| {
        t.iter(|| black_box(f).cyclotomic_square())
    });
    c.bench_function("narsil/g2_double", |t| t.iter(|| black_box(q).double()));
}

criterion_group!(benches, bench);
criterion_main!(benches);
