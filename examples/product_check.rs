use helius_narsil::{G1Affine, G1Bytes, G2Affine, G2Bytes, InputError, PairBytes};

fn main() -> Result<(), InputError> {
    let p = G1Affine::generator();
    let q = G2Affine::generator();
    let pairs = [
        PairBytes {
            g1: G1Bytes::from_affine(&p),
            g2: G2Bytes::from_affine(&q),
        },
        PairBytes {
            g1: G1Bytes::from_affine(&p.negate()),
            g2: G2Bytes::from_affine(&q),
        },
    ];

    assert!(helius_narsil::pairing_product_is_one(&pairs)?);
    Ok(())
}
