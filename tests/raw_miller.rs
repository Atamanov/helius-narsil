use ark_bn254::{
    Bn254, Fq as ArkFq, Fr as ArkFr, G1Projective as ArkG1Projective,
    G2Projective as ArkG2Projective,
};
use ark_ec::{CurveGroup, PrimeGroup, pairing::Pairing};
use ark_ff::PrimeField;
use helius_narsil::{Fp, Fp12, Fr, G1Projective, G2Affine, G2Projective, pairing::miller_loop};

fn fp(value: ArkFq) -> Fp {
    Fp::from_raw(value.into_bigint().0)
}

fn from_ark(value: ark_bn254::Fq12) -> Fp12 {
    let fp2 = |x: ark_bn254::Fq2| helius_narsil::Fp2::new(fp(x.c0), fp(x.c1));
    let fp6 = |x: ark_bn254::Fq6| helius_narsil::Fp6::new(fp2(x.c0), fp2(x.c1), fp2(x.c2));
    Fp12::new(fp6(value.c0), fp6(value.c1))
}

#[test]
fn raw_miller_matches_arkworks() {
    let hg1 = G1Projective::generator();
    let hg2 = G2Projective::from(G2Affine::generator());
    let ag1 = ArkG1Projective::generator();
    let ag2 = ArkG2Projective::generator();
    for scalar in [1u64, 2, 17, 0xdead_beef] {
        let other = scalar.rotate_left(19) | 1;
        let hp = hg1.mul_scalar(Fr::from_u64(scalar)).to_affine();
        let hq = hg2.mul_scalar(Fr::from_u64(other)).to_affine();
        let ap = (ag1 * ArkFr::from(scalar)).into_affine();
        let aq = (ag2 * ArkFr::from(other)).into_affine();
        assert_eq!(
            miller_loop(&hp, &hq),
            from_ark(Bn254::miller_loop(ap, aq).0),
            "raw Miller scalar {scalar}",
        );
    }
}
