//! A random-linear-combination batch must reject whenever any member fails.
//!
//! Every corruption below keeps the tampered value a valid curve point, so the
//! rejection has to come from the pairing product and never from decoding. The
//! coefficients are redrawn on every trial, because batch soundness rests on
//! the prover not knowing them, not on one fixed vector.

#[path = "support/groth16_batch.rs"]
mod fixture;

use fixture::{Batch, Member, Rng, g1};
use helius_narsil::Fr;

const SIZES: [usize; 6] = [1, 2, 3, 4, 8, 16];
const DRAWS: u64 = 4;

/// Every way one member can be wrong, applied at a chosen index.
#[derive(Clone, Copy, Debug)]
enum Corruption {
    NegateA,
    NegateC,
    ScaleA,
    ShiftInput,
    StealNeighbourA,
    StealNeighbourB,
}

const CORRUPTIONS: [Corruption; 6] = [
    Corruption::NegateA,
    Corruption::NegateC,
    Corruption::ScaleA,
    Corruption::ShiftInput,
    Corruption::StealNeighbourA,
    Corruption::StealNeighbourB,
];

impl Corruption {
    fn apply(self, members: &mut [Member], index: usize) {
        let neighbour = members[(index + 1) % members.len()].clone();
        let member = &mut members[index];
        match self {
            Self::NegateA => member.a = member.a.negate(),
            Self::NegateC => member.c = member.c.negate(),
            Self::ScaleA => member.a = member.a.to_curve().mul_scalar(Fr::from_u64(3)).to_affine(),
            Self::ShiftInput => member.input += Fr::ONE,
            Self::StealNeighbourA => member.a = neighbour.a,
            Self::StealNeighbourB => member.b = neighbour.b,
        }
    }

    /// Skip the borrowings that leave the batch unchanged.
    fn is_meaningful(self, size: usize) -> bool {
        !matches!(self, Self::StealNeighbourA | Self::StealNeighbourB) || size > 1
    }
}

#[test]
fn valid_batches_accept_at_every_size() {
    for size in SIZES {
        for keys in [1usize, 2] {
            if size % keys != 0 {
                continue;
            }
            let mut batch = Batch::new(0xa11c_0000 + (size * 8 + keys) as u64, size, keys);
            for draw in 0..DRAWS {
                batch.redraw_challenges(0x5150_0000 + draw);
                assert!(batch.verify(), "size {size}, keys {keys}, draw {draw}");
                assert!(batch.verify_sequential(), "size {size}, keys {keys}");
            }
        }
    }
}

#[test]
fn one_bad_member_is_rejected_at_every_index() {
    for size in SIZES {
        for keys in [1usize, 2] {
            if size % keys != 0 {
                continue;
            }
            let good = Batch::new(0xbad0_0000 + (size * 8 + keys) as u64, size, keys);
            assert!(good.verify());
            for index in 0..size {
                for corruption in CORRUPTIONS {
                    if !corruption.is_meaningful(size) {
                        continue;
                    }
                    let mut batch = Batch::new(0xbad0_0000 + (size * 8 + keys) as u64, size, keys);
                    corruption.apply(&mut batch.members, index);
                    for draw in 0..DRAWS {
                        batch.redraw_challenges(0x7e57_0000 + draw);
                        assert!(
                            !batch.verify(),
                            "size {size}, keys {keys}, index {index}, {corruption:?}, draw {draw}"
                        );
                    }
                }
            }
        }
    }
}

/// A forger who knows the other members still cannot cancel against them: the
/// coefficients weight each member independently, so a cancelling pair would
/// have to hold for a coefficient the forger never saw.
#[test]
fn a_cancelling_pair_cannot_hide_two_bad_members() {
    for size in [2usize, 4, 8] {
        let mut batch = Batch::new(0xca11_0000 + size as u64, size, 1);
        let offset = g1(Fr::from_u64(0x9e37_79b9));
        batch.members[0].a = batch.members[0].a.to_curve().add_mixed(offset).to_affine();
        batch.members[size - 1].a = batch.members[size - 1]
            .a
            .to_curve()
            .add_mixed(offset.negate())
            .to_affine();
        for draw in 0..DRAWS {
            batch.redraw_challenges(0xc0de_0000 + draw);
            assert!(!batch.verify(), "size {size}, draw {draw}");
        }
    }
}

/// Zero coefficients would drop a member from the product, so a batch that
/// draws them must never accept a bad proof in that slot. The draw below is
/// the adversarial worst case a real verifier must exclude.
#[test]
fn a_zero_coefficient_drops_its_member_from_the_product() {
    let mut batch = Batch::new(0x0000_1234, 4, 1);
    batch.redraw_challenges(0x0000_4321);
    batch.members[2].a = batch.members[2].a.negate();
    assert!(!batch.verify());
    batch.members[2].challenge = Fr::ZERO;
    assert!(
        batch.verify(),
        "a zero coefficient must be excluded by the caller's draw"
    );
}

/// Members of a mixed batch are bound to their own key. Moving a valid proof
/// to another key must fail even though the proof itself is genuine.
#[test]
fn a_member_cannot_migrate_to_another_key() {
    let mut batch = Batch::new(0x4b45_5900, 4, 2);
    assert!(batch.verify());
    batch.members[1].key = 0;
    let mut rng = Rng::new(0x4b45_5901);
    for _ in 0..DRAWS {
        batch.redraw_challenges(rng.word());
        assert!(!batch.verify());
    }
}
