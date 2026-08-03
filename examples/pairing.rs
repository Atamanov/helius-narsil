use helius_narsil::{Bn254, G1Affine, G2Affine, pairing};

fn main() {
    let p = G1Affine::generator();
    let q = G2Affine::generator();

    assert_eq!(Bn254::pairing(&p, &q), pairing(&p, &q));
}
