use helius_narsil::{
    Bn254, Fp, Fp2, Fp6, Fp12, Fr, G1Affine, G1Projective, G2Affine, G2Projective, multi_pairing,
    pairing,
    pairing::{final_exponentiation, miller_loop},
};

#[test]
fn zero_has_no_inverse_in_any_field() {
    assert_eq!(Fp::ZERO.invert(), None);
    assert_eq!(Fr::ZERO.invert(), None);
    assert_eq!(Fp2::ZERO.invert(), None);
    assert_eq!(Fp6::ZERO.invert(), None);
    assert_eq!(Fp12::ZERO.invert(), None);
}

#[test]
fn pairing_identity_is_the_target_group_identity() {
    let p = G1Affine::generator();
    let q = G2Affine::generator();

    assert_eq!(Bn254::pairing(&G1Affine::identity(), &q), Fp12::ONE);
    assert_eq!(Bn254::pairing(&p, &G2Affine::identity()), Fp12::ONE);
}

#[test]
fn pairing_is_bilinear_away_from_the_generators() {
    let p = G1Projective::from(G1Affine::generator())
        .mul_scalar(Fr::from_u64(3))
        .to_affine();
    let q = G2Projective::from(G2Affine::generator())
        .mul_scalar(Fr::from_u64(5))
        .to_affine();
    let scalar = Fr::from_u64(7);
    let expected = pairing(&p, &q).pow_u64(7);
    let scaled_p = G1Projective::from(p).mul_scalar(scalar).to_affine();
    let scaled_q = G2Projective::from(q).mul_scalar(scalar).to_affine();

    assert_eq!(pairing(&scaled_p, &q), expected);
    assert_eq!(pairing(&p, &scaled_q), expected);
}

#[test]
fn bn254_facade_delegates_to_the_native_engine() {
    let p = G1Affine::generator();
    let q = G2Affine::generator();
    let negative_p = p.negate();
    let pairs = [(&p, &q), (&negative_p, &q)];
    let miller = miller_loop(&p, &q);

    assert_eq!(Bn254::pairing(&p, &q), pairing(&p, &q));
    assert_eq!(Bn254::multi_pairing(&pairs), multi_pairing(&pairs));
    assert_eq!(Bn254::miller_loop(&p, &q), miller);
    assert_eq!(
        Bn254::final_exponentiation(&miller),
        final_exponentiation(&miller)
    );
}
