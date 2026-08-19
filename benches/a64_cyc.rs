//! Interleaved cyclotomic-square comparison harness. Temporary.

use std::time::Instant;

use ark_bn254::{Bn254, G1Affine as ArkG1, G2Affine as ArkG2};
use ark_ec::AffineRepr;
use ark_ec::pairing::Pairing;
use helius_narsil::{Fp2, G1Affine, G2Affine, pairing};

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
    println!("{name} {per:.2}");
    per
}

fn main() {
    let a2 = Fp2::ONE + Fp2::ONE;
    let b2 = a2.square();
    let g1 = G1Affine::generator();
    let g2 = G2Affine::generator();
    let m = pairing::miller_loop(&g1, &g2);
    let f = pairing::final_exponentiation(&m);
    let ag1 = ArkG1::generator();
    let ag2 = ArkG2::generator();
    let am = Bn254::multi_miller_loop([ag1], [ag2]);

    time("fp2_mul", 200, 20000, &|| {
        std::hint::black_box(a2) * std::hint::black_box(b2)
    });
    time("fp12_mul", 200, 20000, &|| {
        std::hint::black_box(m) * std::hint::black_box(f)
    });
    time("fp12_cyclotomic_square", 200, 20000, &|| {
        std::hint::black_box(f).cyclotomic_square()
    });
    time("miller_loop", 40, 200, &|| {
        pairing::miller_loop(std::hint::black_box(&g1), std::hint::black_box(&g2))
    });
    time("final_exponentiation", 40, 200, &|| {
        pairing::final_exponentiation(std::hint::black_box(&m))
    });
    time("pairing", 40, 200, &|| {
        pairing::pairing(std::hint::black_box(&g1), std::hint::black_box(&g2))
    });
    time("ark_miller_loop", 40, 200, &|| {
        Bn254::multi_miller_loop([std::hint::black_box(ag1)], [std::hint::black_box(ag2)])
    });
    time("ark_final_exponentiation", 40, 200, &|| {
        Bn254::final_exponentiation(std::hint::black_box(am))
    });
    time("ark_pairing", 40, 200, &|| {
        Bn254::pairing(std::hint::black_box(ag1), std::hint::black_box(ag2))
    });
}
