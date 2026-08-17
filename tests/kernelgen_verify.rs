//! Verifies each generated kernel with a host-independent interpreter.

#[path = "support/kernel_contract.rs"]
mod kernel_contract;
#[path = "../build/mod.rs"]
mod kernelgen;

use kernelgen::{
    BN254_MU, BN254_P, BN254_P_INV, interpret_f2mul, interpret_f2sqr, interpret_g2_ysqr,
    interpret_mont4_a64, interpret_mont4_mul, interpret_mont4_mulpre, interpret_mont4_redc,
    interpret_mont4_sqr, interpret_sos, interpret_sosd2_a64, interpret_sosd2_small,
    interpret_sosd6_a64,
};

/// Independent CIOS Montgomery multiplication oracle (word-by-word, u128).
fn reference_mont_mul(a: [u64; 4], b: [u64; 4], p: [u64; 4], p_inv: u64) -> [u64; 4] {
    let mut t = [0u64; 6];
    for &bi in &b {
        let mut carry = 0u128;
        for j in 0..4 {
            let sum = t[j] as u128 + a[j] as u128 * bi as u128 + carry;
            t[j] = sum as u64;
            carry = sum >> 64;
        }
        let sum = t[4] as u128 + carry;
        t[4] = sum as u64;
        t[5] = (sum >> 64) as u64;

        let m = t[0].wrapping_mul(p_inv);
        let mut carry = (t[0] as u128 + m as u128 * p[0] as u128) >> 64;
        for j in 1..4 {
            let sum = t[j] as u128 + m as u128 * p[j] as u128 + carry;
            t[j - 1] = sum as u64;
            carry = sum >> 64;
        }
        let sum = t[4] as u128 + carry;
        t[3] = sum as u64;
        t[4] = t[5] + (sum >> 64) as u64;
        t[5] = 0;
    }

    let mut reduced = [0u64; 4];
    let mut borrow = 0i128;
    for j in 0..4 {
        let difference = t[j] as i128 - p[j] as i128 + borrow;
        reduced[j] = difference as u64;
        borrow = difference >> 64;
    }
    if t[4] != 0 || borrow == 0 {
        reduced
    } else {
        [t[0], t[1], t[2], t[3]]
    }
}

/// Independent sum-of-products Montgomery oracle (Longa Alg. 2 shape,
/// u128 words, six-limb accumulator): `(sum_i a_i*b_i) * R^{-1} mod p`.
fn reference_sos(pairs: &[([u64; 4], [u64; 4])], p: [u64; 4], p_inv: u64) -> [u64; 4] {
    let mut t = [0u64; 6];
    for j in 0..4 {
        for (a, b) in pairs {
            let mut carry = 0u128;
            for k in 0..4 {
                let sum = t[k] as u128 + a[j] as u128 * b[k] as u128 + carry;
                t[k] = sum as u64;
                carry = sum >> 64;
            }
            let sum = t[4] as u128 + carry;
            t[4] = sum as u64;
            t[5] += (sum >> 64) as u64;
        }
        let m = t[0].wrapping_mul(p_inv);
        let mut carry = 0u128;
        for k in 0..4 {
            let sum = t[k] as u128 + m as u128 * p[k] as u128 + carry;
            t[k] = sum as u64;
            carry = sum >> 64;
        }
        let sum = t[4] as u128 + carry;
        t[4] = sum as u64;
        t[5] += (sum >> 64) as u64;
        assert_eq!(t[0], 0, "Montgomery factor cancels the low limb");
        for k in 0..5 {
            t[k] = t[k + 1];
        }
        t[5] = 0;
    }
    assert_eq!(t[4], 0, "final value < 3p < 2^256 fits four limbs");
    let mut out = [t[0], t[1], t[2], t[3]];
    for _ in 0..2 {
        if gte(&out, &p) {
            let mut borrow = 0u64;
            for k in 0..4 {
                let (mid, b1) = out[k].overflowing_sub(p[k]);
                let (low, b2) = mid.overflowing_sub(borrow);
                out[k] = low;
                borrow = (b1 | b2) as u64;
            }
        }
    }
    out
}

mod lazy {
    use super::gte;

    /// 512-bit little-endian value.
    pub type U512 = [u64; 8];

    pub(super) fn lt(a: U512, b: U512) -> bool {
        for k in (0..8).rev() {
            if a[k] != b[k] {
                return a[k] < b[k];
            }
        }
        false
    }

    /// p * 2^256: the double-width guard modulus.
    fn pk(p: [u64; 4]) -> U512 {
        [0, 0, 0, 0, p[0], p[1], p[2], p[3]]
    }

    /// Raw 4x4 -> 512-bit product (never overflows by construction).
    /// Exact subtraction: the schedule's provably nonnegative rows. Asserts
    /// the mask never fires.
    /// Guarded subtraction mod p*2^256: p joins the high limbs on borrow.
    /// Asserts the guarded result lands below p*2^256.
    /// Guarded addition mod p*2^256: p leaves the high limbs when they
    /// reach p. Asserts no 2^512 overflow and the guarded bound.
    /// The nine-fold xi step: 9x + y mod p*2^256 with a canonical high half.
    /// Mirrors the kernel exactly: exact low half with its carry limb, high
    /// half 9*xH + yH + carry < 10p reduced by the mu quotient estimate.
    /// Raw 4x4 -> 512-bit product. Never overflows: both operands are below
    /// 2p and 4p^2 < 2^512 for BN254.
    pub(super) fn mul_pre(a: [u64; 4], b: [u64; 4]) -> U512 {
        let mut z: U512 = [0; 8];
        for i in 0..4 {
            let mut carry = 0u128;
            for j in 0..4 {
                let v = z[i + j] as u128 + a[i] as u128 * b[j] as u128 + carry;
                z[i + j] = v as u64;
                carry = v >> 64;
            }
            z[i + 4] = carry as u64;
        }
        z
    }

    /// Guarded subtraction mod p*2^256: p joins the high limbs on borrow.
    /// Asserts the guarded result lands below p*2^256.
    pub(super) fn guarded_sub(a: U512, b: U512, p: [u64; 4]) -> U512 {
        let mut out: U512 = [0; 8];
        let mut borrow = false;
        for k in 0..8 {
            let (mid, b1) = a[k].overflowing_sub(b[k]);
            let (low, b2) = mid.overflowing_sub(borrow as u64);
            out[k] = low;
            borrow = b1 | b2;
        }
        if borrow {
            let mut carry = false;
            for k in 0..4 {
                let (mid, c1) = out[4 + k].overflowing_add(p[k]);
                let (high, c2) = mid.overflowing_add(carry as u64);
                out[4 + k] = high;
                carry = c1 | c2;
            }
            assert!(carry, "the fix-up carry cancels the borrow");
        }
        assert!(lt(out, pk(p)), "guarded difference stays below p*2^256");
        out
    }

    pub(super) fn below_pk(a: U512, p: [u64; 4]) -> bool {
        lt(a, pk(p))
    }

    /// `p*2^256 - a`, the distance from the reduction precondition edge.
    pub(super) fn gap_to_pk(a: U512, p: [u64; 4]) -> U512 {
        let modulus = pk(p);
        let mut out: U512 = [0; 8];
        let mut borrow = false;
        for k in 0..8 {
            let (mid, b1) = modulus[k].overflowing_sub(a[k]);
            let (low, b2) = mid.overflowing_sub(borrow as u64);
            out[k] = low;
            borrow = b1 | b2;
        }
        assert!(!borrow, "the value stays below p*2^256");
        out
    }

    /// Montgomery reduction of T < p*2^256: four cancel rounds, result < 2p,
    /// one conditional subtraction, canonical.
    pub(super) fn mont_red(t_in: U512, p: [u64; 4], p_inv: u64) -> [u64; 4] {
        assert!(lt(t_in, pk(p)), "reduction precondition T < p*2^256");
        let mut t = t_in;
        for round in 0..4 {
            let m = t[round].wrapping_mul(p_inv);
            let mut carry: u128 = 0;
            for k in 0..4 {
                let v = t[round + k] as u128 + m as u128 * p[k] as u128 + carry;
                t[round + k] = v as u64;
                carry = v >> 64;
            }
            assert_eq!(t[round], 0, "the Montgomery factor cancels the low word");
            for word in t.iter_mut().skip(round + 4) {
                let v = *word as u128 + carry;
                *word = v as u64;
                carry = v >> 64;
            }
            assert_eq!(carry, 0, "T + m*p*2^(64r) < 2p*2^256 never leaves T7");
        }
        let mut out = [t[4], t[5], t[6], t[7]];
        // out < 2p: at most one subtraction reaches canonical.
        if gte(&out, &p) {
            let mut borrow = false;
            for k in 0..4 {
                let (mid, b1) = out[k].overflowing_sub(p[k]);
                let (low, b2) = mid.overflowing_sub(borrow as u64);
                out[k] = low;
                borrow = b1 | b2;
            }
            assert!(!borrow);
        }
        assert!(
            !gte(&out, &p),
            "reduced coefficient canonical (< 2p before csub)"
        );
        out
    }
}

fn gte(a: &[u64; 4], b: &[u64; 4]) -> bool {
    for j in (0..4).rev() {
        if a[j] != b[j] {
            return a[j] > b[j];
        }
    }
    true
}

fn next_raw(state: &mut u64) -> [u64; 4] {
    core::array::from_fn(|_| {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    })
}

fn next_residue(state: &mut u64) -> [u64; 4] {
    loop {
        let value = next_raw(state);
        if !gte(&value, &BN254_P) {
            return value;
        }
    }
}

/// Operands that maximize/minimize carries through both chains.
fn edge_corpus() -> Vec<[u64; 4]> {
    let p = BN254_P;
    let p_minus = |k: u64| {
        let mut value = p;
        value[0] -= k; // p[0] is odd and > 2, safe for k <= 2
        value
    };
    vec![
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [2, 0, 0, 0],
        p_minus(1),
        p_minus(2),
        // Densest sub-p values: all-ones low limbs under the top limb of p.
        [u64::MAX, u64::MAX, u64::MAX, p[3] - 1],
        [u64::MAX, 0, u64::MAX, 0],
        [0, u64::MAX, 0, p[3] - 1],
        // Montgomery R mod p and R^2 mod p (hot in production conversions).
        [
            0xd35d438dc58f0d9d,
            0x0a78eb28f5c70b3d,
            0x666ea36f7879462c,
            0x0e0a77c19a07df2f,
        ],
        [
            0xf32cfc5b538afa89,
            0xb5e71911d44501fb,
            0x47ab1eff0a417ff6,
            0x06d89f71cab8351f,
        ],
    ]
}

#[test]
fn mul_schedule_matches_reference_on_edge_corpus() {
    let corpus = edge_corpus();
    for (i, a) in corpus.iter().enumerate() {
        for (j, b) in corpus.iter().enumerate() {
            assert_eq!(
                interpret_mont4_mul(*a, *b, BN254_P, BN254_P_INV),
                reference_mont_mul(*a, *b, BN254_P, BN254_P_INV),
                "edge pair ({i}, {j})",
            );
        }
    }
}

#[test]
fn sqr_schedule_matches_mul_and_reference_on_edge_corpus() {
    for (i, a) in edge_corpus().iter().enumerate() {
        let squared = interpret_mont4_sqr(*a, BN254_P, BN254_P_INV);
        assert_eq!(
            squared,
            reference_mont_mul(*a, *a, BN254_P, BN254_P_INV),
            "edge square {i} vs reference",
        );
        assert_eq!(
            squared,
            interpret_mont4_mul(*a, *a, BN254_P, BN254_P_INV),
            "edge square {i} vs mul schedule",
        );
    }
}

#[test]
fn a64_schedule_matches_reference_on_edge_corpus() {
    let corpus = edge_corpus();
    for (i, a) in corpus.iter().enumerate() {
        for (j, b) in corpus.iter().enumerate() {
            assert_eq!(
                interpret_mont4_a64(*a, *b, BN254_P, BN254_P_INV),
                reference_mont_mul(*a, *b, BN254_P, BN254_P_INV),
                "edge pair ({i}, {j})",
            );
        }
    }
}

#[test]
fn schedules_match_reference_on_random_residues() {
    let mut state = 0x0123_4567_89ab_cdefu64;
    for case in 0..20_000 {
        let a = next_residue(&mut state);
        let b = next_residue(&mut state);
        let product = interpret_mont4_mul(a, b, BN254_P, BN254_P_INV);
        assert_eq!(
            product,
            reference_mont_mul(a, b, BN254_P, BN254_P_INV),
            "case {case}",
        );
        assert!(!gte(&product, &BN254_P), "unreduced product, case {case}");
        assert_eq!(
            interpret_mont4_sqr(a, BN254_P, BN254_P_INV),
            reference_mont_mul(a, a, BN254_P, BN254_P_INV),
            "square case {case}",
        );
        assert_eq!(
            interpret_mont4_a64(a, b, BN254_P, BN254_P_INV),
            product,
            "a64 case {case}",
        );
    }
}

/// The four-limb operands the dual-lane A64 leaf must survive: the shared
/// edge corpus plus p itself, which is the operand bound and the value the
/// in-kernel `p - y1` produces when `y1` is zero.
fn sosd2_a64_corpus() -> Vec<[u64; 4]> {
    let mut corpus = edge_corpus();
    corpus.push(BN254_P);
    corpus
}

/// Both lanes of `(x0 + x1*u)*(y0 + y1*u)` from the independent u128 sums of
/// products oracle. Lane 0 subtracts through `p - y1`, exactly as the kernel.
fn reference_sosd2(x0: [u64; 4], x1: [u64; 4], y0: [u64; 4], y1: [u64; 4]) -> ([u64; 4], [u64; 4]) {
    let mut ny1 = [0u64; 4];
    let mut borrow = 0u64;
    for k in 0..4 {
        let (mid, b1) = BN254_P[k].overflowing_sub(y1[k]);
        let (low, b2) = mid.overflowing_sub(borrow);
        ny1[k] = low;
        borrow = (b1 | b2) as u64;
    }
    assert_eq!(borrow, 0, "operands are at most p");
    (
        reference_sos(&[(x0, y0), (x1, ny1)], BN254_P, BN254_P_INV),
        reference_sos(&[(x0, y1), (x1, y0)], BN254_P, BN254_P_INV),
    )
}

/// The dual-lane A64 leaf against the independent oracle on every ordered
/// quadruple of the edge corpus, then on random residues. The interpreter
/// checks the kernel's own bound claims (no accumulator carry-out, no fifth
/// word between rounds) on each of these inputs.
#[test]
fn a64_sosd2_matches_reference_on_edge_corpus() {
    let corpus = sosd2_a64_corpus();
    for (i, x0) in corpus.iter().enumerate() {
        for (j, x1) in corpus.iter().enumerate() {
            for (k, y0) in corpus.iter().enumerate() {
                for (l, y1) in corpus.iter().enumerate() {
                    assert_eq!(
                        interpret_sosd2_a64(*x0, *x1, *y0, *y1, BN254_P, BN254_P_INV),
                        reference_sosd2(*x0, *x1, *y0, *y1),
                        "edge quadruple ({i}, {j}, {k}, {l})",
                    );
                }
            }
        }
    }
}

#[test]
fn a64_sosd2_matches_the_field_algebra_on_random_residues() {
    use helius_narsil::Fp;
    use helius_narsil::consts::{P, P_INV};

    let mut state = 0x736f_7364_325f_6136u64;
    for case in 0..100_000 {
        let x0 = next_residue(&mut state);
        let x1 = next_residue(&mut state);
        let y0 = next_residue(&mut state);
        let y1 = next_residue(&mut state);
        let (fx0, fx1) = (Fp::from_raw_unchecked(x0), Fp::from_raw_unchecked(x1));
        let (fy0, fy1) = (Fp::from_raw_unchecked(y0), Fp::from_raw_unchecked(y1));
        let (lane0, lane1) = interpret_sosd2_a64(x0, x1, y0, y1, P, P_INV);
        assert_eq!(
            Fp::from_raw_unchecked(lane0),
            fx0 * fy0 - fx1 * fy1,
            "lane 0, case {case}",
        );
        assert_eq!(
            Fp::from_raw_unchecked(lane1),
            fx0 * fy1 + fx1 * fy0,
            "lane 1, case {case}",
        );
        assert!(
            !gte(&lane0, &P) && !gte(&lane1, &P),
            "unreduced, case {case}"
        );
    }
}

/// `p - x` for `x <= p`, the image every subtracted lane0 term enters through.
fn negp(x: [u64; 4]) -> [u64; 4] {
    let mut image = [0u64; 4];
    let mut borrow = 0u64;
    for k in 0..4 {
        let (mid, b1) = BN254_P[k].overflowing_sub(x[k]);
        let (low, b2) = mid.overflowing_sub(borrow);
        image[k] = low;
        borrow = (b1 | b2) as u64;
    }
    assert_eq!(borrow, 0, "operands are at most p");
    image
}

/// Both lanes of a sum of three Fp2 products from the independent u128 sums of
/// products oracle. Operands are in the kernel's table order.
fn reference_sosd6(operands: [[u64; 4]; 12]) -> ([u64; 4], [u64; 4]) {
    let mut lane0 = Vec::with_capacity(6);
    let mut lane1 = Vec::with_capacity(6);
    for i in 0..3 {
        let [x0, x1, y0, y1] = [0, 1, 2, 3].map(|part| operands[4 * i + part]);
        lane0.push((x0, y0));
        lane0.push((x1, negp(y1)));
        lane1.push((x0, y1));
        lane1.push((x1, y0));
    }
    (
        reference_sos(&lane0, BN254_P, BN254_P_INV),
        reference_sos(&lane1, BN254_P, BN254_P_INV),
    )
}

/// Every operand slot swept over the edge corpus against three saturated
/// backgrounds: p itself (the operand bound, and the image `p - 0`), p - 1,
/// and zero, whose `p - y_i1` image is p. The interpreter checks the kernel's
/// own bound claims (no accumulate carry-out of word 5, no sixth word between
/// rounds, no fifth word at the end) on each of these inputs.
#[test]
fn a64_sosd6_matches_reference_on_edge_corpus() {
    let corpus = sosd2_a64_corpus();
    let p_minus_one = {
        let mut value = BN254_P;
        value[0] -= 1;
        value
    };
    for background in [BN254_P, p_minus_one, [0; 4]] {
        for slot in 0..12 {
            for (index, value) in corpus.iter().enumerate() {
                let mut operands = [background; 12];
                operands[slot] = *value;
                assert_eq!(
                    interpret_sosd6_a64(operands, BN254_P, BN254_P_INV),
                    reference_sosd6(operands),
                    "slot {slot}, corpus {index}",
                );
            }
        }
    }
}

/// All twelve slots drawn from the edge corpus at once, so carry patterns that
/// only appear when several operands are extreme together are covered too.
#[test]
fn a64_sosd6_matches_reference_on_mixed_edge_operands() {
    let corpus = sosd2_a64_corpus();
    let mut state = 0x736f_7364_365f_6d78u64;
    for case in 0..20_000 {
        let operands = core::array::from_fn(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            corpus[(state >> 33) as usize % corpus.len()]
        });
        assert_eq!(
            interpret_sosd6_a64(operands, BN254_P, BN254_P_INV),
            reference_sosd6(operands),
            "case {case}",
        );
    }
}

#[test]
fn a64_sosd6_matches_the_field_algebra_on_random_residues() {
    use helius_narsil::Fp;
    use helius_narsil::consts::{P, P_INV};

    let mut state = 0x736f_7364_365f_6136u64;
    for case in 0..100_000 {
        let operands: [[u64; 4]; 12] = core::array::from_fn(|_| next_residue(&mut state));
        let f = operands.map(Fp::from_raw_unchecked);
        let (lane0, lane1) = interpret_sosd6_a64(operands, P, P_INV);
        let mut want0 = Fp::from_raw_unchecked([0; 4]);
        let mut want1 = want0;
        for i in 0..3 {
            let [x0, x1, y0, y1] = [0, 1, 2, 3].map(|part| f[4 * i + part]);
            want0 = want0 + x0 * y0 - x1 * y1;
            want1 = want1 + x0 * y1 + x1 * y0;
        }
        assert_eq!(Fp::from_raw_unchecked(lane0), want0, "lane 0, case {case}");
        assert_eq!(Fp::from_raw_unchecked(lane1), want1, "lane 1, case {case}");
        assert!(
            !gte(&lane0, &P) && !gte(&lane1, &P),
            "unreduced, case {case}"
        );
    }
}

/// The AArch64 leaf contract allows the first operand to be any value below
/// 2^256 while `b` stays below p (`src/fp/aarch64.rs`. `Fp::from_raw` relies
/// on this). Its single-chain rows absorb the wider products: the fifth
/// accumulator word plus the CSET bit give the reduction row 2^321 of
/// headroom. The x86-64 dual-chain schedule is NOT in this test: its 2^320
/// bound needs both operands below p, and the interpreter's flags-clear
/// assertions demonstrably fire for widened `a` (see the report accompanying
/// the build-time-generation change).
#[test]
fn a64_schedule_accepts_widened_first_operand() {
    let mut state = 0xc0de_1234_5678_9abcu64;
    for case in 0..2_000 {
        let a = next_raw(&mut state);
        let b = next_residue(&mut state);
        assert_eq!(
            interpret_mont4_a64(a, b, BN254_P, BN254_P_INV),
            reference_mont_mul(a, b, BN254_P, BN254_P_INV),
            "widened case {case}",
        );
    }
}

#[test]
fn mul_schedule_is_commutative() {
    let mut state = 0xfeed_face_dead_beefu64;
    for _ in 0..2_000 {
        let a = next_residue(&mut state);
        let b = next_residue(&mut state);
        assert_eq!(
            interpret_mont4_mul(a, b, BN254_P, BN254_P_INV),
            interpret_mont4_mul(b, a, BN254_P, BN254_P_INV),
        );
    }
}

/// The production pair counts (sos2/sos4 single lane, sosd2..sosd8 lanes)
/// plus the largest count the kernel contract covers. Counts are even: the
/// inner walk takes two pairs per iteration, and the interpreter's trip
/// proof rejects odd counts.
const SOS_PAIR_COUNTS: [usize; 5] = [2, 4, 6, 8, 10];

#[test]
fn sos_schedule_matches_reference_on_random_pairs() {
    let mut state = 0x1f83_d9ab_fb41_bd6bu64;
    for &count in &SOS_PAIR_COUNTS {
        for case in 0..2_000 {
            let pairs: Vec<([u64; 4], [u64; 4])> = (0..count)
                .map(|_| (next_residue(&mut state), next_residue(&mut state)))
                .collect();
            let out = interpret_sos(&pairs, BN254_P, BN254_P_INV);
            assert_eq!(
                out,
                reference_sos(&pairs, BN254_P, BN254_P_INV),
                "T = {count}, case {case}",
            );
            assert!(!gte(&out, &BN254_P), "unreduced output, T = {count}");
        }
    }
}

/// Operands may equal p (the `negp` image of zero). Saturating every pair at
/// p maximizes the accumulator against the claimed 2^325 bound.
#[test]
fn sos_schedule_matches_reference_on_edge_pairs() {
    let corpus = {
        let mut corpus = edge_corpus();
        corpus.push(BN254_P);
        corpus
    };
    for (i, a) in corpus.iter().enumerate() {
        for (j, b) in corpus.iter().enumerate() {
            for &count in &SOS_PAIR_COUNTS {
                let pairs = vec![(*a, *b); count];
                assert_eq!(
                    interpret_sos(&pairs, BN254_P, BN254_P_INV),
                    reference_sos(&pairs, BN254_P, BN254_P_INV),
                    "edge pair ({i}, {j}), T = {count}",
                );
            }
        }
    }
}

/// A product padded with a zero pair is a Montgomery product: the rolled
/// loop must agree with the straight-line mul schedule everywhere.
#[test]
fn sos_schedule_with_zero_padded_pair_matches_mul_schedule() {
    let mut state = 0x4528_21e6_38d0_1377u64;
    for case in 0..5_000 {
        let a = next_residue(&mut state);
        let b = next_residue(&mut state);
        assert_eq!(
            interpret_sos(&[(a, b), ([0; 4], [0; 4])], BN254_P, BN254_P_INV),
            interpret_mont4_mul(a, b, BN254_P, BN254_P_INV),
            "case {case}",
        );
    }
}

/// Independent model of the fused Fp2 square: canonical modular add and
/// subtract, then the reference CIOS multiplication per lane.
fn reference_f2sqr(x0: [u64; 4], x1: [u64; 4], p: [u64; 4], p_inv: u64) -> ([u64; 4], [u64; 4]) {
    let add_mod = |a: [u64; 4], b: [u64; 4]| {
        let mut sum = [0u64; 4];
        let mut carry = false;
        for j in 0..4 {
            let (word, c0) = a[j].overflowing_add(b[j]);
            let (word, c1) = word.overflowing_add(carry as u64);
            sum[j] = word;
            carry = c0 | c1;
        }
        assert!(!carry, "a + b < 2p < 2^255 for BN254");
        if gte(&sum, &p) {
            let mut borrow = false;
            for j in 0..4 {
                let (word, b0) = sum[j].overflowing_sub(p[j]);
                let (word, b1) = word.overflowing_sub(borrow as u64);
                sum[j] = word;
                borrow = b0 | b1;
            }
        }
        sum
    };
    let sub_mod = |a: [u64; 4], b: [u64; 4]| {
        let mut difference = [0u64; 4];
        let mut borrow = false;
        for j in 0..4 {
            let (word, b0) = a[j].overflowing_sub(b[j]);
            let (word, b1) = word.overflowing_sub(borrow as u64);
            difference[j] = word;
            borrow = b0 | b1;
        }
        if borrow {
            let mut carry = false;
            for j in 0..4 {
                let (word, c0) = difference[j].overflowing_add(p[j]);
                let (word, c1) = word.overflowing_add(carry as u64);
                difference[j] = word;
                carry = c0 | c1;
            }
        }
        difference
    };
    (
        reference_mont_mul(add_mod(x0, x1), sub_mod(x0, x1), p, p_inv),
        reference_mont_mul(x0, add_mod(x1, x1), p, p_inv),
    )
}

/// Both f2sqr schedules must equal the model, each other, and the composed
/// two-multiply route the ADX Miller path used before them.
#[test]
fn f2sqr_schedules_match_the_complex_square_model() {
    let corpus = edge_corpus();
    let mut cases: Vec<([u64; 4], [u64; 4])> = Vec::new();
    for a in &corpus {
        for b in &corpus {
            cases.push((*a, *b));
        }
    }
    let mut state = 0x6632_7371_725f_6c61u64;
    for _ in 0..1_000 {
        cases.push((next_residue(&mut state), next_residue(&mut state)));
    }
    for (case, (x0, x1)) in cases.into_iter().enumerate() {
        let model = reference_f2sqr(x0, x1, BN254_P, BN254_P_INV);
        for rolled in [false, true] {
            assert_eq!(
                interpret_f2sqr(x0, x1, BN254_P, BN254_P_INV, rolled),
                model,
                "case {case}, rolled {rolled}",
            );
        }
    }
}

/// The leaves replace `mont4_mul(x0 + x1, x0 - x1)` and `2*mont4_mul(x0, x1)`
/// limb for limb, which is what keeps the raw Miller lines bit-exact.
#[test]
fn f2sqr_schedules_match_the_two_multiply_route() {
    use helius_narsil::Fp;
    use helius_narsil::consts::{P, P_INV};

    let mut state = 0x1eaf_5171_6172_6564u64;
    for case in 0..500 {
        let x0 = next_residue(&mut state);
        let x1 = next_residue(&mut state);
        let product = Fp::from_raw_unchecked(interpret_mont4_mul(x0, x1, P, P_INV));
        let composed = (
            interpret_mont4_mul(
                (Fp::from_raw_unchecked(x0) + Fp::from_raw_unchecked(x1)).mont_limbs(),
                (Fp::from_raw_unchecked(x0) - Fp::from_raw_unchecked(x1)).mont_limbs(),
                P,
                P_INV,
            ),
            (product + product).mont_limbs(),
        );
        for rolled in [false, true] {
            assert_eq!(
                interpret_f2sqr(x0, x1, P, P_INV, rolled),
                composed,
                "case {case}, rolled {rolled}",
            );
        }
    }
}

/// Everything the lazy leaf holds between the prologue and the store:
/// `(x, y)` per raw product, the four raw products, and the two combining
/// rows the Montgomery reductions consume.
struct YsqrStages {
    operands: [([u64; 4], [u64; 4]); 4],
    products: [lazy::U512; 4],
    rows: [lazy::U512; 2],
}

/// Independent double-width model of `y = g^2 - 3*e^2`: uncorrected
/// four-limb sums, four exact 512-bit products, one guarded difference per
/// output half. Every stage bound the schedule claims is asserted here, on
/// the same values the kernel sees.
fn ysqr_stages(g: &[[u64; 4]; 2], e: &[[u64; 4]; 2], f: &[[u64; 4]; 2], p: [u64; 4]) -> YsqrStages {
    // The bound the whole schedule rests on. mcl calls it `!isFullBit`
    // (indeed `isLtQuad`): p leaves two spare bits in four limbs.
    assert!(p[3] < 1 << 62, "4p < 2^256 is what admits uncorrected sums");
    let add_pre = |a: [u64; 4], b: [u64; 4]| {
        let mut sum = [0u64; 4];
        let mut carry = false;
        for j in 0..4 {
            let (word, c1) = a[j].overflowing_add(b[j]);
            let (word, c2) = word.overflowing_add(carry as u64);
            sum[j] = word;
            carry = c1 | c2;
        }
        assert!(!carry, "canonical + canonical < 2p < 2^256: no carry out");
        sum
    };
    let sub_mod = |a: [u64; 4], b: [u64; 4]| {
        let mut out = [0u64; 4];
        let mut borrow = false;
        for j in 0..4 {
            let (word, b1) = a[j].overflowing_sub(b[j]);
            let (word, b2) = word.overflowing_sub(borrow as u64);
            out[j] = word;
            borrow = b1 | b2;
        }
        if borrow {
            let mut carry = false;
            for j in 0..4 {
                let (word, c1) = out[j].overflowing_add(p[j]);
                let (word, c2) = word.overflowing_add(carry as u64);
                out[j] = word;
                carry = c1 | c2;
            }
        }
        assert!(!gte(&out, &p), "the guarded difference is canonical");
        out
    };

    let operands = [
        (add_pre(g[0], g[1]), sub_mod(g[0], g[1])),
        (add_pre(e[0], e[1]), sub_mod(f[0], f[1])),
        (g[0], add_pre(g[1], g[1])),
        (e[0], add_pre(f[1], f[1])),
    ];
    let products = operands.map(|(x, y)| {
        let product = lazy::mul_pre(x, y);
        // Every product multiplies a sub-2p value by a sub-p value, so it is
        // below 2p^2 < p*2^256, already a valid reduction input.
        assert!(lazy::below_pk(product, p), "raw product below p*2^256");
        product
    });
    YsqrStages {
        operands,
        products,
        rows: [
            lazy::guarded_sub(products[0], products[1], p),
            lazy::guarded_sub(products[2], products[3], p),
        ],
    }
}

/// The two Montgomery reductions that close the model.
fn reference_g2_ysqr(
    g: &[[u64; 4]; 2],
    e: &[[u64; 4]; 2],
    f: &[[u64; 4]; 2],
    p: [u64; 4],
    p_inv: u64,
) -> ([u64; 4], [u64; 4]) {
    let rows = ysqr_stages(g, e, f, p).rows;
    (
        lazy::mont_red(rows[0], p, p_inv),
        lazy::mont_red(rows[1], p, p_inv),
    )
}

/// Canonical `f = 3e` for a random `e`, so the corpus only ever exercises
/// the kernel's stated precondition.
fn triple(e: &[[u64; 4]; 2]) -> [[u64; 4]; 2] {
    use helius_narsil::Fp;
    e.map(|component| {
        let value = Fp::from_raw_unchecked(component);
        (value + value + value).mont_limbs()
    })
}

/// One `(g, e)` operand pair of the lazy-leaf corpus, with the stage it is
/// there to drive.
type YsqrCase = (&'static str, [[u64; 4]; 2], [[u64; 4]; 2]);

/// Exact division by three of a four-limb multiple of three.
fn div3(value: [u64; 4]) -> [u64; 4] {
    let mut out = [0u64; 4];
    let mut remainder = 0u128;
    for j in (0..4).rev() {
        let word = (remainder << 64) | value[j] as u128;
        out[j] = (word / 3) as u64;
        remainder = word % 3;
    }
    assert_eq!(remainder, 0, "the value is a multiple of three");
    out
}

/// Operand pairs solved for, not sampled. The cross-product corpus couples
/// the two Fp2 inputs as `g = (a, b)`, `e = (b, a)`, so it can never place
/// the g side and the e side at an extreme in the same call, and random
/// residues reach the last percent of the reduction bound about one time in
/// a hundred, the borrow branch's edge never.
fn crafted_ysqr_cases() -> Vec<YsqrCase> {
    use helius_narsil::Fp;

    let p = BN254_P;
    let add = |x: [u64; 4], y: [u64; 4]| {
        (Fp::from_raw_unchecked(x) + Fp::from_raw_unchecked(y)).mont_limbs()
    };
    let sub = |x: [u64; 4], y: [u64; 4]| {
        (Fp::from_raw_unchecked(x) - Fp::from_raw_unchecked(y)).mont_limbs()
    };
    let pow2 = |bit: usize| {
        let mut value = [0u64; 4];
        value[bit / 64] = 1 << (bit % 64);
        value
    };
    let p_minus = |k: u64| {
        let mut value = p;
        value[0] -= k; // p[0] is odd and > 2, safe for k <= 2
        value
    };
    let zero = [0u64; 4];
    let one = [1u64, 0, 0, 0];
    let two = [2u64, 0, 0, 0];
    let p_minus_one = p_minus(1);
    let p_minus_two = p_minus(2);
    // 3*third = p-1, so e.im = third is what drives f.im to p-1.
    let third = div3(p_minus_one);

    // g.re = 3*2^251 beside g.re - 1 gives gs = 3*2^252 - 1 and gd = 1.
    let g_edge = add(pow2(252), pow2(251));
    // g.im = 3*2^200 - 1 beside g.re = 1 gives 2*g.re*g.im = 6*2^200 - 2.
    let g_row_b = sub(add(pow2(201), pow2(200)), one);

    vec![
        // The g side and the e side at their independent maxima in one call:
        // gs = 2p-3 and e6 = 2p-2 are both at the four-limb ceiling that the
        // uncorrected sums rest on, and both rows carry a near-maximal
        // double-width value (row B lands 2p-2 below p*2^256).
        (
            "g and e maxima together",
            [p_minus_one, p_minus_two],
            [p_minus_one, third],
        ),
        // The largest raw product the schedule can form, the maximal raw sum
        // gs = 2p-3 times the maximal canonical gd = p-1, in the same call as
        // the maximal e6.
        (
            "maximal raw products",
            [p_minus_two, p_minus_one],
            [p_minus_one, third],
        ),
        // Both e-side maxima at once. ed = p-1 needs f.im = f.re + 1, and
        // e6 = 2p-2 needs 3*e.im = p-1 mod p; e = (2*third, third) solves
        // both, since 3*third = p-1 makes f = (p-2, p-1).
        (
            "e-side operand maxima",
            [p_minus_two, p_minus_one],
            [add(third, third), third],
        ),
        // Both rows a dozen units below p*2^256 at once: every product is
        // tiny and both differences borrow, so both reductions run against
        // the top of their precondition.
        ("both rows at the borrow edge", [zero, zero], [two, one]),
        // Row A one unit below p*2^256, the tightest input the reduction can
        // take: gs*gd = 3*2^252 - 1 against es*ed = 3*2^252.
        (
            "row A one unit below the edge",
            [g_edge, sub(g_edge, one)],
            [pow2(126), zero],
        ),
        // Row B two units below p*2^256. Both row-B products carry a doubled
        // operand, so both are even and no odd gap exists: two units is that
        // row's edge.
        (
            "row B two units below the edge",
            [one, g_row_b],
            [pow2(100), pow2(100)],
        ),
    ]
}

/// Every non-random pair the two lazy-leaf tests share: the coupled edge
/// cross product, then the crafted cases.
fn ysqr_cases() -> Vec<YsqrCase> {
    let corpus = edge_corpus();
    let mut cases: Vec<YsqrCase> = Vec::new();
    for a in &corpus {
        for b in &corpus {
            cases.push(("edge cross product", [*a, *b], [*b, *a]));
        }
    }
    cases.extend(crafted_ysqr_cases());
    cases
}

/// The crafted cases must keep reaching the stage edges they were solved
/// for. A drifted constant would leave a case that still passes but stresses
/// nothing.
#[test]
fn crafted_ysqr_cases_stay_on_their_edges() {
    let p = BN254_P;
    let cases = crafted_ysqr_cases();
    let stages = |index: usize| {
        let (_, g, e) = cases[index];
        ysqr_stages(&g, &e, &triple(&e), p)
    };
    let wide = |value: [u64; 4]| {
        let mut out = [0u64; 8];
        out[..4].copy_from_slice(&value);
        out
    };
    let small = |value: u64| wide([value, 0, 0, 0]);
    // 2p - k, the ceiling of an uncorrected sum of two canonical values.
    let double_p_minus = |k: u64| {
        let mut out = [0u64; 4];
        let mut carry = 0u128;
        for j in 0..4 {
            let word = 2 * p[j] as u128 + carry;
            out[j] = word as u64;
            carry = word >> 64;
        }
        assert_eq!(carry, 0, "2p fits four limbs");
        let (low, borrow) = out[0].overflowing_sub(k);
        out[0] = low;
        assert!(!borrow, "p is odd and larger than k");
        out
    };
    let p_minus_one = {
        let mut value = p;
        value[0] -= 1;
        value
    };

    let ceiling = stages(0);
    assert_eq!(
        ceiling.operands[0].0,
        double_p_minus(3),
        "gs at its ceiling"
    );
    assert_eq!(
        ceiling.operands[3].1,
        double_p_minus(2),
        "e6 at its ceiling"
    );
    assert_eq!(
        lazy::gap_to_pk(ceiling.rows[1], p),
        wide(double_p_minus(2)),
        "row B within 2p-2 of the reduction edge",
    );

    let products = stages(1);
    assert_eq!(products.operands[0].0, double_p_minus(3), "maximal raw sum");
    assert_eq!(products.operands[0].1, p_minus_one, "maximal canonical gd");
    assert_eq!(products.operands[3].1, double_p_minus(2), "maximal e6");
    // (2p-3)*(p-1) is the largest raw product the operand bounds admit, and
    // it still leaves more than its own size under p*2^256. That headroom is
    // what lets a difference of two such products stay a valid REDC input.
    let widest = products.products[0];
    assert!(
        lazy::lt(widest, lazy::gap_to_pk(widest, p)),
        "the worst-case raw product keeps a factor of two of headroom",
    );

    let e_side = stages(2);
    assert_eq!(e_side.operands[1].1, p_minus_one, "ed at its ceiling");
    assert_eq!(e_side.operands[3].1, double_p_minus(2), "e6 at its ceiling");

    let both = stages(3);
    assert_eq!(
        lazy::gap_to_pk(both.rows[0], p),
        small(9),
        "row A near edge"
    );
    assert_eq!(
        lazy::gap_to_pk(both.rows[1], p),
        small(12),
        "row B near edge"
    );

    assert_eq!(
        lazy::gap_to_pk(stages(4).rows[0], p),
        small(1),
        "row A one unit below p*2^256",
    );
    assert_eq!(
        lazy::gap_to_pk(stages(5).rows[1], p),
        small(2),
        "row B two units below p*2^256",
    );
}

/// The lazy leaf must equal the independent double-width model on the edge
/// corpus, on the crafted cases and on random residues.
#[test]
fn g2_ysqr_matches_the_double_width_model() {
    let mut cases = ysqr_cases();
    let mut state = 0x7932_7371_725f_6732u64;
    for _ in 0..1_000 {
        cases.push((
            "random",
            [next_residue(&mut state), next_residue(&mut state)],
            [next_residue(&mut state), next_residue(&mut state)],
        ));
    }
    for (case, (what, g, e)) in cases.into_iter().enumerate() {
        let f = triple(&e);
        assert_eq!(
            interpret_g2_ysqr(&g, &e, &f, BN254_P, BN254_P_INV),
            reference_g2_ysqr(&g, &e, &f, BN254_P, BN254_P_INV),
            "case {case} ({what})",
        );
    }
}

/// The pin that keeps the raw Miller lines bit-exact: the leaf must equal
/// the multiply-then-reduce route it replaces, limb for limb, on the same
/// corpus the double-width model runs.
#[test]
fn g2_ysqr_matches_the_composed_square_route() {
    use helius_narsil::Fp;
    use helius_narsil::consts::{P, P_INV};

    let mut cases = ysqr_cases();
    let mut state = 0x6c61_7a79_5f79_7371u64;
    for _ in 0..500 {
        cases.push((
            "random",
            [next_residue(&mut state), next_residue(&mut state)],
            [next_residue(&mut state), next_residue(&mut state)],
        ));
    }
    for (case, (what, g, e)) in cases.into_iter().enumerate() {
        let f = triple(&e);
        // The composed route: two fused Fp2 squares, then y = g^2 - 3*e^2.
        let g_square = interpret_f2sqr(g[0], g[1], P, P_INV, true);
        let e_square = interpret_f2sqr(e[0], e[1], P, P_INV, true);
        let composed = {
            let lane = |gs: [u64; 4], es: [u64; 4]| {
                let square = Fp::from_raw_unchecked(es);
                let triple_square = square + square + square;
                (Fp::from_raw_unchecked(gs) - triple_square).mont_limbs()
            };
            (lane(g_square.0, e_square.0), lane(g_square.1, e_square.1))
        };
        assert_eq!(
            interpret_g2_ysqr(&g, &e, &f, P, P_INV),
            composed,
            "case {case} ({what})",
        );
    }
}

/// Every stage of the lazy Karatsuba Fp2 product, with the bound the
/// schedule claims for it asserted on the values the kernel sees.
struct F2MulStages {
    sums: [[u64; 4]; 2],
    products: [lazy::U512; 3],
    subtrahend: lazy::U512,
    rows: [lazy::U512; 2],
}

/// Independent double-width model of `(a0 + a1*u)(b0 + b1*u)`: two
/// uncorrected four-limb sums, three exact 512-bit products, one raw
/// eight-limb sum, one guarded difference per output half.
fn f2mul_stages(x: &[[u64; 4]; 2], y: &[[u64; 4]; 2], p: [u64; 4]) -> F2MulStages {
    // mcl's `!isFullBit` for BN254: p leaves two spare bits in four limbs.
    assert!(p[3] < 1 << 62, "4p < 2^256 is what admits uncorrected sums");
    let add_pre = |a: [u64; 4], b: [u64; 4]| {
        let mut sum = [0u64; 4];
        let mut carry = false;
        for j in 0..4 {
            let (word, c1) = a[j].overflowing_add(b[j]);
            let (word, c2) = word.overflowing_add(carry as u64);
            sum[j] = word;
            carry = c1 | c2;
        }
        assert!(!carry, "canonical + canonical < 2p < 2^256: no carry out");
        sum
    };
    let add_pre8 = |a: lazy::U512, b: lazy::U512| {
        let mut sum = [0u64; 8];
        let mut carry = false;
        for j in 0..8 {
            let (word, c1) = a[j].overflowing_add(b[j]);
            let (word, c2) = word.overflowing_add(carry as u64);
            sum[j] = word;
            carry = c1 | c2;
        }
        assert!(!carry, "a0*b0 + a1*b1 < 2p^2 < 2^512: no carry out");
        sum
    };

    let sums = [add_pre(x[0], x[1]), add_pre(y[0], y[1])];
    let operands = [(x[0], y[0]), (x[1], y[1]), (sums[0], sums[1])];
    let products = operands.map(|(a, b)| {
        let product = lazy::mul_pre(a, b);
        // The middle product is the largest: (2p-2)^2 < 4p^2 < p*2^256.
        assert!(lazy::below_pk(product, p), "raw product below p*2^256");
        product
    });
    let subtrahend = add_pre8(products[0], products[1]);
    assert!(
        !lazy::lt(products[2], subtrahend),
        "s*t dominates a0*b0 + a1*b1, so the imaginary row never borrows",
    );
    F2MulStages {
        sums,
        rows: [
            lazy::guarded_sub(products[0], products[1], p),
            lazy::guarded_sub(products[2], subtrahend, p),
        ],
        products,
        subtrahend,
    }
}

/// The two Montgomery reductions that close the model.
fn reference_f2mul(
    x: &[[u64; 4]; 2],
    y: &[[u64; 4]; 2],
    p: [u64; 4],
    p_inv: u64,
) -> ([u64; 4], [u64; 4]) {
    let rows = f2mul_stages(x, y, p).rows;
    (
        lazy::mont_red(rows[0], p, p_inv),
        lazy::mont_red(rows[1], p, p_inv),
    )
}

/// One `(x, y)` operand pair of the f2mul corpus with the stage it drives.
type F2MulCase = (&'static str, [[u64; 4]; 2], [[u64; 4]; 2]);

/// Operand pairs solved for, not sampled. The real row's borrow branch ends
/// one unit below `p*2^256` only when the two raw products are adjacent
/// integers, which random residues never produce.
fn crafted_f2mul_cases() -> Vec<F2MulCase> {
    let p = BN254_P;
    let p_minus_one = {
        let mut value = p;
        value[0] -= 1;
        value
    };
    let small = |v: u64| [v, 0, 0, 0];
    vec![
        (
            "maximal sums, products and imaginary row",
            [p_minus_one, p_minus_one],
            [p_minus_one, p_minus_one],
        ),
        (
            "real row one unit below p*2^256",
            [small(0), small(1)],
            [small(0), small(1)],
        ),
        (
            "real row two units below p*2^256",
            [small(0), small(2)],
            [small(0), small(1)],
        ),
        (
            "real row at its no-borrow ceiling",
            [p_minus_one, small(0)],
            [p_minus_one, small(0)],
        ),
    ]
}

/// Edge cross product plus the crafted cases.
fn f2mul_cases() -> Vec<F2MulCase> {
    let corpus = edge_corpus();
    let mut cases: Vec<F2MulCase> = Vec::new();
    for a in &corpus {
        for b in &corpus {
            cases.push(("edge cross product", [*a, *b], [*b, *a]));
        }
    }
    cases.extend(crafted_f2mul_cases());
    cases
}

/// The crafted cases must keep reaching the stage edges they were solved
/// for. A drifted constant would leave a case that still passes but stresses
/// nothing.
#[test]
fn crafted_f2mul_cases_stay_on_their_edges() {
    let p = BN254_P;
    let cases = crafted_f2mul_cases();
    let stages = |index: usize| {
        let (_, x, y) = cases[index];
        f2mul_stages(&x, &y, p)
    };
    let wide = |value: [u64; 4]| {
        let mut out = [0u64; 8];
        out[..4].copy_from_slice(&value);
        out
    };
    let small = |value: u64| wide([value, 0, 0, 0]);
    // 2p - 2, the ceiling of an uncorrected sum of two canonical values.
    let double_p_minus_two = {
        let mut out = [0u64; 4];
        let mut carry = 0u128;
        for j in 0..4 {
            let word = 2 * p[j] as u128 + carry;
            out[j] = word as u64;
            carry = word >> 64;
        }
        assert_eq!(carry, 0, "2p fits four limbs");
        out[0] -= 2;
        out
    };

    let maximal = stages(0);
    assert_eq!(maximal.sums, [double_p_minus_two; 2], "both sums at 2p-2");
    assert_eq!(
        maximal.products[2],
        lazy::mul_pre(double_p_minus_two, double_p_minus_two),
        "the middle product at its (2p-2)^2 ceiling",
    );
    // (2p-2)^2 is the largest raw product the operand bounds admit and it is
    // still a valid reduction input. That is the whole isFullBit argument.
    assert!(
        lazy::below_pk(maximal.products[2], p),
        "4p^2 < p*2^256 admits the middle product unreduced",
    );
    assert_eq!(
        maximal.rows[1], maximal.subtrahend,
        "x = y makes the imaginary row equal 2*a0*b0, its ceiling",
    );

    assert_eq!(
        lazy::gap_to_pk(stages(1).rows[0], p),
        small(1),
        "real row one unit below p*2^256",
    );
    assert_eq!(
        lazy::gap_to_pk(stages(2).rows[0], p),
        small(2),
        "real row two units below p*2^256",
    );
    let ceiling = stages(3);
    assert_eq!(
        ceiling.rows[0], ceiling.products[0],
        "no borrow: the real row is the bare product",
    );
}

/// The lazy Karatsuba leaf must equal the independent double-width model on
/// the edge corpus, on the crafted cases and on random residues.
#[test]
fn f2mul_matches_the_double_width_model() {
    let mut cases = f2mul_cases();
    let mut state = 0x6632_6d75_6c5f_6d6du64;
    for _ in 0..1_000 {
        cases.push((
            "random",
            [next_residue(&mut state), next_residue(&mut state)],
            [next_residue(&mut state), next_residue(&mut state)],
        ));
    }
    for (case, (what, x, y)) in cases.into_iter().enumerate() {
        assert_eq!(
            interpret_f2mul(&x, &y, BN254_P, BN254_P_INV),
            reference_f2mul(&x, &y, BN254_P, BN254_P_INV),
            "case {case} ({what})",
        );
    }
}

/// The pin that keeps every Fp2 product bit-exact: the leaf must equal the
/// fused sums-of-products route it replaces, limb for limb.
#[test]
fn f2mul_matches_the_sos_route() {
    use helius_narsil::consts::{P, P_INV};

    let mut cases = f2mul_cases();
    let mut state = 0x6b61_7261_7473_7562u64;
    for _ in 0..500 {
        cases.push((
            "random",
            [next_residue(&mut state), next_residue(&mut state)],
            [next_residue(&mut state), next_residue(&mut state)],
        ));
    }
    for (case, (what, x, y)) in cases.into_iter().enumerate() {
        assert_eq!(
            interpret_f2mul(&x, &y, P, P_INV),
            interpret_sosd2_small(x[0], x[1], y[0], y[1], P, P_INV),
            "case {case} ({what})",
        );
    }
}

#[test]
fn kernelgen_constants_match_production_constants() {
    assert_eq!(BN254_P, helius_narsil::consts::P);
    assert_eq!(BN254_P_INV, helius_narsil::consts::P_INV);
    assert_eq!(BN254_MU, helius_narsil::consts::P_MU_310);
}

/// The generated schedules must agree with the production portable oracle on
/// this host, whatever its architecture: the interpreters execute the exact
/// emitted operation sequences with bit-accurate carry-flag semantics.
#[test]
fn interpreted_schedules_match_portable_oracle() {
    use helius_narsil::Fp;
    use helius_narsil::consts::{P, P_INV};

    let mut state = 0x5851_f42d_4c95_7f2du64;
    for _ in 0..1_000 {
        let a = next_residue(&mut state);
        let b = next_residue(&mut state);
        let oracle = Fp::from_raw_unchecked(a) * Fp::from_raw_unchecked(b);
        let interpreted = interpret_mont4_mul(a, b, P, P_INV);
        assert_eq!(Fp::from_raw_unchecked(interpreted), oracle);
        let interpreted_a64 = interpret_mont4_a64(a, b, P, P_INV);
        assert_eq!(Fp::from_raw_unchecked(interpreted_a64), oracle);
        let square_oracle = Fp::from_raw_unchecked(a).square();
        let interpreted_square = interpret_mont4_sqr(a, P, P_INV);
        assert_eq!(Fp::from_raw_unchecked(interpreted_square), square_oracle);
    }
    assert_direct_mulpre_and_redc_match_independent_oracles();
}

fn reference_mulpre(a: [u64; 4], b: [u64; 4]) -> [u64; 8] {
    let mut z = [0u64; 8];
    for i in 0..4 {
        let mut carry = 0u128;
        for j in 0..4 {
            let sum = z[i + j] as u128 + a[i] as u128 * b[j] as u128 + carry;
            z[i + j] = sum as u64;
            carry = sum >> 64;
        }
        let mut k = i + 4;
        while carry != 0 {
            let sum = z[k] as u128 + carry;
            z[k] = sum as u64;
            carry = sum >> 64;
            k += 1;
        }
    }
    z
}

fn assert_direct_mulpre_and_redc_match_independent_oracles() {
    assert!(BN254_P[3] < 1 << 62, "BN254 p < 2^254 = R/4");
    let max = [BN254_P[0] - 1, BN254_P[1], BN254_P[2], BN254_P[3]];
    let mut state = 0x6d75_6c34_3033_7265;
    // One thousand independent residues plus the explicit boundary corpus
    // below is the minimum admission gate for any future wide G2 kernel.
    for case in 0..1_000 {
        let a = if case == 0 {
            max
        } else {
            next_residue(&mut state)
        };
        let b = if case == 0 {
            max
        } else {
            next_residue(&mut state)
        };
        let wide = interpret_mont4_mulpre(a, b, BN254_P, BN254_P_INV);
        assert_eq!(wide, reference_mulpre(a, b), "raw product {case}");
        assert_eq!(
            interpret_mont4_redc(wide, BN254_P, BN254_P_INV),
            reference_mont_mul(a, b, BN254_P, BN254_P_INV),
            "REDC {case}",
        );
    }

    // `mulPreA<false>` consumes raw Fp component sums below 2p. Maximal
    // canonical inputs reach 2p-2, and (2p-2)^2 remains below pR because
    // p < R/4. This is the exact MCL BN254 pre-add bound.
    let mut twice_max = [0; 4];
    let mut carry = false;
    for i in 0..4 {
        (twice_max[i], carry) = max[i].carrying_add(max[i], carry);
    }
    assert!(!carry);
    let wide = interpret_mont4_mulpre(twice_max, twice_max, BN254_P, BN254_P_INV);
    assert_eq!(wide, reference_mulpre(twice_max, twice_max));
    assert_eq!(
        interpret_mont4_redc(wide, BN254_P, BN254_P_INV),
        lazy::mont_red(wide, BN254_P, BN254_P_INV),
    );

    // REDC's full contractual edge, pR-1, exercises a high half at p-1
    // rather than only raw products near p^2.
    let pr_minus_one = [
        u64::MAX,
        u64::MAX,
        u64::MAX,
        u64::MAX,
        max[0],
        max[1],
        max[2],
        max[3],
    ];
    assert_eq!(
        interpret_mont4_redc(pr_minus_one, BN254_P, BN254_P_INV),
        lazy::mont_red(pr_minus_one, BN254_P, BN254_P_INV),
    );
}

/// Regeneration determinism: the build script must render the identical text
/// on every host and every run (no timestamps, paths, or host data).
/// The emitted text of one A64 kernel: from its label to the register map
/// comment that opens the next one.
fn a64_kernel_body<'a>(rendered: &'a str, symbol: &str) -> &'a str {
    let label = format!("\n{symbol}:\n");
    let start = rendered
        .find(&label)
        .unwrap_or_else(|| panic!("{symbol} body present"));
    let body = &rendered[start..];
    match body.find("\n/* ") {
        Some(end) => &body[..end],
        None => body,
    }
}

#[test]
fn rendered_files_are_deterministic_and_sized() {
    let rendered = kernelgen::render::render_mont4_x86_64();
    assert_eq!(
        rendered,
        kernelgen::render::render_mont4_x86_64(),
        "x86-64 rendering must be pure",
    );
    let [
        (fp2_xi_compact_instructions, fp2_xi_compact_bytes),
        (fp12_sqr_mcl_instructions, fp12_sqr_mcl_bytes),
        (mul_instructions, mul_bytes),
        (sqr_instructions, sqr_bytes),
        (sos_instructions, sos_bytes),
        (sosd2_instructions, sosd2_bytes),
        (sosd2_small_instructions, sosd2_small_bytes),
        (fp6_instructions, fp6_bytes),
        (fp12_034_instructions, fp12_034_bytes),
        (fp4_sqr_instructions, fp4_sqr_bytes),
        (fp12_sqr_instructions, fp12_sqr_bytes),
        (fp12_mul_instructions, fp12_mul_bytes),
        (cyc_sqr_instructions, cyc_sqr_bytes),
        (fp12_034l_instructions, fp12_034l_bytes),
        (fp12_034k_instructions, fp12_034k_bytes),
        (sosd6_instructions, sosd6_bytes),
    ] = kernelgen::render::kernel_sizes();
    let [
        (mulpre_instructions, mulpre_bytes),
        (redc_instructions, redc_bytes),
    ] = kernelgen::render::mul403_primitive_sizes();
    let [
        (f2sqr_instructions, f2sqr_bytes),
        (f2sqr_small_instructions, f2sqr_small_bytes),
    ] = kernelgen::render::f2sqr_sizes();
    let (f2mul_instructions, f2mul_bytes) = kernelgen::render::f2mul_size();
    for (symbol, instructions, bytes) in [
        (kernelgen::render::MUL_SYMBOL, mul_instructions, mul_bytes),
        (kernelgen::render::SQR_SYMBOL, sqr_instructions, sqr_bytes),
        (kernelgen::render::SOS_SYMBOL, sos_instructions, sos_bytes),
        (
            kernelgen::render::SOSD2_SYMBOL,
            sosd2_instructions,
            sosd2_bytes,
        ),
        (
            kernelgen::render::SOSD2_SMALL_SYMBOL,
            sosd2_small_instructions,
            sosd2_small_bytes,
        ),
        (kernelgen::render::FP6_SYMBOL, fp6_instructions, fp6_bytes),
        (
            kernelgen::render::FP12_034_SYMBOL,
            fp12_034_instructions,
            fp12_034_bytes,
        ),
        (
            kernelgen::render::FP4_SQR_SYMBOL,
            fp4_sqr_instructions,
            fp4_sqr_bytes,
        ),
        (
            kernelgen::render::FP12_SQR_SYMBOL,
            fp12_sqr_instructions,
            fp12_sqr_bytes,
        ),
        (
            kernelgen::render::FP12_MUL_SYMBOL,
            fp12_mul_instructions,
            fp12_mul_bytes,
        ),
        (
            kernelgen::render::CYC_SQR_SYMBOL,
            cyc_sqr_instructions,
            cyc_sqr_bytes,
        ),
        (
            kernelgen::render::FP12_034L_SYMBOL,
            fp12_034l_instructions,
            fp12_034l_bytes,
        ),
        (
            kernelgen::render::FP12_034K_SYMBOL,
            fp12_034k_instructions,
            fp12_034k_bytes,
        ),
        (
            kernelgen::render::SOSD6_SYMBOL,
            sosd6_instructions,
            sosd6_bytes,
        ),
        (
            kernelgen::render::MULPRE_SYMBOL,
            mulpre_instructions,
            mulpre_bytes,
        ),
        (
            kernelgen::render::REDC_SYMBOL,
            redc_instructions,
            redc_bytes,
        ),
        (
            kernelgen::render::FP2_XI_COMPACT_SYMBOL,
            fp2_xi_compact_instructions,
            fp2_xi_compact_bytes,
        ),
        (
            kernelgen::render::FP12_SQR_MCL_SYMBOL,
            fp12_sqr_mcl_instructions,
            fp12_sqr_mcl_bytes,
        ),
        (
            kernelgen::render::F2SQR_SYMBOL,
            f2sqr_instructions,
            f2sqr_bytes,
        ),
        (
            kernelgen::render::F2SQR_SMALL_SYMBOL,
            f2sqr_small_instructions,
            f2sqr_small_bytes,
        ),
        (
            kernelgen::render::F2MUL_SYMBOL,
            f2mul_instructions,
            f2mul_bytes,
        ),
    ] {
        assert!(rendered.contains(symbol), "{symbol} missing");
        assert!(
            rendered.contains(&format!("{instructions} instructions, {bytes} bytes")),
            "{symbol} header sizes missing",
        );
        assert!(!rendered.contains("jmp"), "only conditional back edges");
    }
    assert_eq!(
        rendered.matches("    call ").count(),
        4,
        "only the compact Fp12 wrapper may call its two fixed helpers twice",
    );

    // Reusable admission contracts for the Miller campaign. Existing direct
    // primitives prove the checker itself is live. Future G2 kernels inherit
    // the same deterministic text/stack/call checks as soon as their symbols
    // appear in the common renderer. Absence is allowed while lane A is still
    // the readable Rust model.
    for budget in kernel_contract::EXISTING_WIDE_PRIMITIVES {
        kernel_contract::assert_kernel_contract(&rendered, budget);
    }
    for budget in kernel_contract::FUTURE_MILLER_KERNELS {
        kernel_contract::assert_kernel_contract_if_present(&rendered, budget);
    }
    let fp12_sqr_mcl_body = rendered
        .split(&format!("\n{}:\n", kernelgen::render::FP12_SQR_MCL_SYMBOL))
        .nth(1)
        .expect("compact Fp12 square body present");
    let fp12_sqr_mcl_leaf = fp12_sqr_mcl_body
        .split(&format!(".size {}", kernelgen::render::FP12_SQR_MCL_SYMBOL))
        .next()
        .expect("compact Fp12 square size directive present");
    assert!(
        fp12_sqr_mcl_leaf.contains("sub rsp, 584"),
        "compact wrapper must preserve 16-byte System V call-site alignment",
    );
    let mulpre_body = rendered
        .split(&format!("\n{}:\n", kernelgen::render::MULPRE_SYMBOL))
        .nth(1)
        .expect("mulpre body present");
    let mulpre_leaf = mulpre_body
        .split(&format!(".size {}", kernelgen::render::MULPRE_SYMBOL))
        .next()
        .expect("mulpre size directive present");
    assert!(!mulpre_leaf.contains("call"));
    assert!(
        !mulpre_leaf.contains("(%rip)"),
        "mulpre must not walk tables"
    );
    assert_eq!(mulpre_leaf.matches("jne").count(), 1);
    let redc_body = rendered
        .split(&format!("\n{}:\n", kernelgen::render::REDC_SYMBOL))
        .nth(1)
        .expect("redc body present");
    let redc_leaf = redc_body
        .split(&format!(".size {}", kernelgen::render::REDC_SYMBOL))
        .next()
        .expect("redc size directive present");
    assert!(!redc_leaf.contains("call"));
    assert!(!redc_leaf.contains("(%rip)"), "redc must not walk tables");
    let sosd6_body = rendered
        .split(&format!("\n{}:\n", kernelgen::render::SOSD6_SYMBOL))
        .nth(1)
        .expect("sosd6 kernel body present");
    assert_eq!(
        sosd6_body.matches("jne").count() - mulpre_body.matches("jne").count(),
        2,
        "sosd6 kernel has exactly its two counted back edges",
    );
    let fp12_034k_body = rendered
        .split(&format!("\n{}:\n", kernelgen::render::FP12_034K_SYMBOL))
        .nth(1)
        .expect("fp12_034k kernel body present");
    assert_eq!(
        fp12_034k_body.matches("jne").count() - sosd6_body.matches("jne").count(),
        13,
        "fp12_034k kernel has exactly its thirteen counted back edges",
    );
    let fp12_034l_body = rendered
        .split(&format!("\n{}:\n", kernelgen::render::FP12_034L_SYMBOL))
        .nth(1)
        .expect("fp12_034l kernel body present");
    assert_eq!(
        fp12_034l_body.matches("jne").count() - fp12_034k_body.matches("jne").count(),
        11,
        "fp12_034l kernel has exactly its eleven counted back edges",
    );
    let cyc_sqr_body = rendered
        .split(&format!("\n{}:\n", kernelgen::render::CYC_SQR_SYMBOL))
        .nth(1)
        .expect("cyc_sqr kernel body present");
    assert_eq!(
        cyc_sqr_body.matches("jne").count() - fp12_034l_body.matches("jne").count(),
        10,
        "cyc_sqr kernel has exactly its ten counted back edges",
    );
    let sos_body = rendered
        .split(&format!("\n{}:\n", kernelgen::render::SOS_SYMBOL))
        .nth(1)
        .expect("sos kernel body present");
    assert_eq!(
        rendered.matches("jne").count() - sos_body.matches("jne").count(),
        8,
        "only the compact wrapper's eight fixed-row walks precede mont4",
    );
    let fp12_mul_body = rendered
        .split(&format!("\n{}:\n", kernelgen::render::FP12_MUL_SYMBOL))
        .nth(1)
        .expect("fp12_mul kernel body present");
    assert_eq!(
        fp12_mul_body.matches("jne").count() - cyc_sqr_body.matches("jne").count(),
        17,
        "fp12_mul kernel has exactly its seventeen counted back edges",
    );
    let fp12_sqr_body = rendered
        .split(&format!("\n{}:\n", kernelgen::render::FP12_SQR_SYMBOL))
        .nth(1)
        .expect("fp12_sqr kernel body present");
    assert_eq!(
        fp12_sqr_body.matches("jne").count() - fp12_mul_body.matches("jne").count(),
        17,
        "fp12_sqr kernel has exactly its seventeen counted back edges",
    );
    let fp4_sqr_body = rendered
        .split(&format!("\n{}:\n", kernelgen::render::FP4_SQR_SYMBOL))
        .nth(1)
        .expect("fp4_sqr kernel body present");
    assert_eq!(
        fp4_sqr_body.matches("jne").count() - fp12_sqr_body.matches("jne").count(),
        7,
        "fp4_sqr kernel has exactly its seven counted back edges",
    );
    let fp12_034_body = rendered
        .split(&format!("\n{}:\n", kernelgen::render::FP12_034_SYMBOL))
        .nth(1)
        .expect("fp12_034 kernel body present");
    assert_eq!(
        fp12_034_body.matches("jne").count() - fp4_sqr_body.matches("jne").count(),
        9,
        "fp12_034 kernel has exactly its nine counted back edges",
    );
    let fp6_body = rendered
        .split(&format!("\n{}:\n", kernelgen::render::FP6_SYMBOL))
        .nth(1)
        .expect("fp6 kernel body present");
    assert_eq!(
        fp6_body.matches("jne").count() - fp12_034_body.matches("jne").count(),
        7,
        "fp6 kernel has exactly its seven counted back edges",
    );
    // sos's two counted back edges plus rolled sosd2's one.
    assert_eq!(
        sos_body.matches("jne").count() - fp6_body.matches("jne").count(),
        3,
    );
    let sosd2_small_body = rendered
        .split(&format!("\n{}:\n", kernelgen::render::SOSD2_SMALL_SYMBOL))
        .nth(1)
        .expect("rolled sosd2 kernel body present");
    assert_eq!(
        sosd2_small_body.matches("jne").count() - fp6_body.matches("jne").count(),
        1,
        "rolled sosd2 has exactly its one counted back edge",
    );
    // The frontend thesis in numbers: the rolled kernels must stay well under
    // typical op-cache reach (and far under either mont4 straight line).
    assert!(
        sos_bytes < 768,
        "sos kernel must stay compact ({sos_bytes} bytes)",
    );
    assert!(
        sosd2_small_bytes < 1024,
        "rolled sosd2 must stay op-cache compact ({sosd2_small_bytes} bytes)",
    );
    assert!(
        fp6_bytes <= 1536,
        "fp6 kernel must stay within ~1.5 KiB ({fp6_bytes} bytes)",
    );
    assert!(
        fp12_034_bytes <= 1792,
        "fp12_034 kernel must stay within ~1.7 KiB ({fp12_034_bytes} bytes)",
    );
    assert!(
        fp4_sqr_bytes <= 1280,
        "fp4_sqr kernel must stay within 1.25 KiB ({fp4_sqr_bytes} bytes)",
    );
    assert!(
        fp12_sqr_bytes <= 3072,
        "fp12_sqr kernel must stay within 3 KiB of text ({fp12_sqr_bytes} bytes)",
    );
    assert!(
        fp12_mul_bytes <= 3072,
        "fp12_mul kernel must stay within 3 KiB of text ({fp12_mul_bytes} bytes)",
    );
    assert!(
        cyc_sqr_bytes <= 2112,
        "cyc_sqr kernel must stay within ~2 KiB of text ({cyc_sqr_bytes} bytes)",
    );
    assert!(
        fp12_034l_bytes <= 2560,
        "fp12_034l kernel must stay within 2.5 KiB of text ({fp12_034l_bytes} bytes)",
    );
    assert!(
        fp12_034k_bytes <= 2252,
        "fp12_034k kernel must stay within 2.2 KiB of text ({fp12_034k_bytes} bytes)",
    );
    assert!(
        sosd6_bytes < 1536,
        "rolled sosd6 must stay within 1.5 KiB of text ({sosd6_bytes} bytes)",
    );
    // Both f2sqr leaves sit in the Miller loop beside the accumulator
    // kernels, so their text stays inside the same frontend budget.
    assert!(
        f2sqr_bytes <= 2048,
        "f2sqr must stay within 2 KiB of text ({f2sqr_bytes} bytes)",
    );
    assert!(
        f2sqr_small_bytes < 1024,
        "rolled f2sqr must stay op-cache compact ({f2sqr_small_bytes} bytes)",
    );
    // The lazy y leaf is text added to the same line path, so its budget is
    // the one that decides whether its two saved reductions can pay.
    let (_, g2_ysqr_bytes) = kernelgen::render::g2_ysqr_size();
    assert!(
        g2_ysqr_bytes <= 2048,
        "g2_ysqr must stay within 2 KiB of text ({g2_ysqr_bytes} bytes)",
    );

    let rendered_a64 = kernelgen::a64::render::render_aarch64();
    assert_eq!(
        rendered_a64,
        kernelgen::a64::render::render_aarch64(),
        "aarch64 rendering must be pure",
    );
    let sizes = kernelgen::a64::render::kernel_sizes();
    assert_eq!(sizes.len(), kernelgen::a64::render::A64_KERNELS.len());
    for (kernel, (instructions, bytes)) in kernelgen::a64::render::A64_KERNELS.iter().zip(sizes) {
        let symbol = kernel.symbol;
        assert_eq!(bytes, 4 * instructions, "A64 instructions are 4 bytes");
        assert!(
            rendered_a64.contains(&format!("{symbol}:")),
            "{symbol} absent"
        );
        assert!(
            rendered_a64.contains(&format!("{instructions} instructions, {bytes} bytes")),
            "a64 header sizes missing for {symbol}",
        );
        // L1I is 192 KiB, so the budget is about text discipline, not the
        // instruction cache: a kernel over it has grown a structure the
        // generator was not meant to produce.
        assert!(
            bytes <= 8192,
            "{symbol} must stay within 8 KiB of text ({bytes} bytes)",
        );
    }
    // 4 rounds x (product rows + 2 reduction rows) x 4 limbs, times two for
    // MUL and UMULH, plus the eight Montgomery factors: 4 rows per round for
    // sosd2 and 12 for sosd6.
    // The frame carries ten spilled callee-saved registers (5 STP, 5 LDP) and
    // one `p - y` image per product (2 STP each, reread once per round). A
    // spilled accumulator word would show up here as a larger count.
    for (symbol, products, stack) in [
        (kernelgen::a64::render::SOSD2_SYMBOL, 200, 10 + 2 + 4 * 2),
        (
            kernelgen::a64::render::SOSD6_SYMBOL,
            456,
            10 + 3 * 2 + 4 * 3 * 2,
        ),
    ] {
        let body = a64_kernel_body(&rendered_a64, symbol);
        assert_eq!(
            body.matches("mul ").count() + body.matches("umulh ").count(),
            products,
            "{symbol} must issue exactly the schoolbook product count",
        );
        assert_eq!(
            body.matches("[sp").count(),
            stack,
            "{symbol} touches the stack outside its frame contract",
        );
        assert!(!body.contains("movk"), "{symbol} must not rematerialize p");
    }
    // Leaf property, and the rounds are unrolled, so the kernel is straight
    // line. (Match with surrounding spaces: `.globl` would otherwise hit the
    // "bl" pattern.)
    assert!(!rendered_a64.contains(" bl "), "kernel must remain a leaf");
    assert!(!rendered_a64.contains(" blr "), "kernel must remain a leaf");
    assert_eq!(rendered_a64.matches("b.ne").count(), 0);
}

#[test]
fn runtime_ifma_follows_the_base_target_and_the_override() {
    use kernelgen::policy::runtime_ifma_enabled;

    // A host CPUID check cannot see the base target, so the build must gate
    // the dispatch itself. Below `avx512f` the caller has no ZMM register
    // class and the IFMA route loses to the scalar one.
    assert_eq!(runtime_ifma_enabled(None, false), Some(false));
    assert_eq!(runtime_ifma_enabled(None, true), Some(true));
    assert_eq!(runtime_ifma_enabled(Some("1"), false), Some(true));
    assert_eq!(runtime_ifma_enabled(Some("0"), true), Some(false));
    assert_eq!(runtime_ifma_enabled(Some("yes"), true), None);
}
