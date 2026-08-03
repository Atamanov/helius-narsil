use helius_narsil::{Fr, G1Affine, G1Projective, msm_variable_time_affine};

fn main() {
    let p = G1Affine::generator();
    let points = [p, p];
    let scalars = [Fr::from_u64(1).to_raw(), Fr::from_u64(2).to_raw()];
    let expected = G1Projective::from(p)
        .mul_scalar(Fr::from_u64(3))
        .to_affine();

    assert_eq!(msm_variable_time_affine(&points, &scalars), expected);
}
