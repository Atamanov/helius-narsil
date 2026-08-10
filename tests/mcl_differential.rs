#![cfg(feature = "mcl-oracle")]

use std::sync::Once;

use helius_narsil::{
    Fp, Fp12, Fr, G1Affine, G1Bytes, G1Projective, G2Affine, G2Bytes, G2Projective,
    msm_variable_time_affine, pairing,
    pairing::{final_exponentiation, miller_loop},
};

unsafe extern "C" {
    fn narsil_mcl_init() -> i32;
    fn narsil_mcl_pairing_matches(g1: *const u8, g2: *const u8, expected: *const u8) -> i32;
    fn narsil_mcl_miller_matches(g1: *const u8, g2: *const u8, expected: *const u8) -> i32;
    fn narsil_mcl_final_exp_matches(input: *const u8, expected: *const u8) -> i32;
    fn narsil_mcl_msm_matches(
        points: *const u8,
        scalars: *const u8,
        count: usize,
        expected: *const u8,
    ) -> i32;
}

fn initialize() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        // SAFETY: the bridge has no inputs and initializes MCL once.
        assert_eq!(unsafe { narsil_mcl_init() }, 0);
    });
}

fn copy_fp(output: &mut [u8], offset: usize, value: Fp) {
    output[offset..offset + 32].copy_from_slice(&value.to_bytes_be());
}

fn fp12_bytes(value: Fp12) -> [u8; 384] {
    let components = [
        value.c0.c0.c0,
        value.c0.c0.c1,
        value.c0.c1.c0,
        value.c0.c1.c1,
        value.c0.c2.c0,
        value.c0.c2.c1,
        value.c1.c0.c0,
        value.c1.c0.c1,
        value.c1.c1.c0,
        value.c1.c1.c1,
        value.c1.c2.c0,
        value.c1.c2.c1,
    ];
    let mut bytes = [0u8; 384];
    for (index, component) in components.into_iter().enumerate() {
        copy_fp(&mut bytes, index * 32, component);
    }
    bytes
}

fn test_pair() -> (G1Affine, G2Affine) {
    let p = G1Projective::from(G1Affine::generator())
        .mul_scalar(Fr::from_u64(3))
        .to_affine();
    let q = G2Projective::from(G2Affine::generator())
        .mul_scalar(Fr::from_u64(5))
        .to_affine();
    (p, q)
}

#[test]
fn pairing_and_miller_loop_match_mcl() {
    initialize();
    let (p, q) = test_pair();
    let pairing_bytes = fp12_bytes(pairing(&p, &q));
    let miller_bytes = fp12_bytes(miller_loop(&p, &q));
    let p = G1Bytes::from_affine(&p).0;
    let q = G2Bytes::from_affine(&q).0;

    // SAFETY: every pointer references a fixed-size live byte buffer for the call.
    assert_eq!(
        unsafe { narsil_mcl_pairing_matches(p.as_ptr(), q.as_ptr(), pairing_bytes.as_ptr()) },
        1
    );
    // SAFETY: every pointer references a fixed-size live byte buffer for the call.
    assert_eq!(
        unsafe { narsil_mcl_miller_matches(p.as_ptr(), q.as_ptr(), miller_bytes.as_ptr()) },
        1
    );
}

#[test]
fn final_exponentiation_matches_mcl() {
    initialize();
    let (p, q) = test_pair();
    let input = miller_loop(&p, &q);
    let input_bytes = fp12_bytes(input);
    let expected = fp12_bytes(final_exponentiation(&input));

    // SAFETY: both pointers reference complete live Fp12 encodings for the call.
    assert_eq!(
        unsafe { narsil_mcl_final_exp_matches(input_bytes.as_ptr(), expected.as_ptr()) },
        1
    );
}

#[test]
fn small_msm_matches_mcl() {
    initialize();
    let generator = G1Projective::from(G1Affine::generator());
    let points = [
        generator.to_affine(),
        generator.mul_scalar(Fr::from_u64(2)).to_affine(),
        generator.mul_scalar(Fr::from_u64(3)).to_affine(),
    ];
    let scalars = [
        Fr::from_u64(2).to_raw(),
        Fr::from_u64(3).to_raw(),
        Fr::from_u64(5).to_raw(),
    ];
    let expected = G1Bytes::from_affine(&msm_variable_time_affine(&points, &scalars)).0;
    let mut point_bytes = [0u8; 3 * 64];
    let mut scalar_bytes = [0u8; 3 * 32];
    for index in 0..3 {
        point_bytes[index * 64..(index + 1) * 64]
            .copy_from_slice(&G1Bytes::from_affine(&points[index]).0);
        scalar_bytes[index * 32..(index + 1) * 32]
            .copy_from_slice(&Fr::from_raw(scalars[index]).to_bytes_be());
    }

    // SAFETY: the buffers contain `count` complete points and scalars.
    assert_eq!(
        unsafe {
            narsil_mcl_msm_matches(
                point_bytes.as_ptr(),
                scalar_bytes.as_ptr(),
                points.len(),
                expected.as_ptr(),
            )
        },
        1
    );
}
