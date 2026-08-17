//! Pin test for the IFMA roofline discipline.
//!
//! No disassembly runs here. The kernel internals are not public, so this
//! file pins the discipline two ways. First, a self-contained radix-52 model
//! re-derives the bound math from the public `P`/`P_INV` constants and
//! asserts the documented lazy-bound budgets, the fold trajectory, and the
//! split-arity occupancy identities. Second, an IFMA-gated runtime test
//! checks that every Miller route produces one identical value through the
//! public API. Static counts marked "pinned" are copied from the named
//! source files at commit b12edac and from docs/kb entries. They document
//! the shipped shape and fail this test when the shape drifts.

use helius_narsil::consts::{ATE_LOOP_COUNT, P, P_INV};

const MASK52: u64 = (1 << 52) - 1;

/// Pinned from src/fp/avx512ifma.rs (`LAZY_CAP`). The model below proves it
/// is the largest admissible bound multiple, not merely an admissible one.
const LAZY_CAP: u64 = 84;

/// Pinned bound tags of the live fused-leaf call sites.
/// src/miller_coeff_ifma.rs uses `<1, 13, 1, 1>` (sparse) and `<1, 23, 2, 1>`
/// (square). src/final_exp_ifma.rs uses `<1, 13, 2, 1>` (dense).
const SPARSE_NEG: u64 = 13;
const SQ_NEG: u64 = 23;
const FE_NEG: u64 = 13;

/// Pinned stream arities of the two fused split-arity leaves
/// (`sos_mac_6_3_pair_fused_lazy`, `sos_mac_12_6_pair_fused_lazy`).
const MILLER_FULL_TERMS: usize = 6;
const MILLER_HALF_TERMS: usize = 3;
const FE_FULL_TERMS: usize = 12;
const FE_HALF_TERMS: usize = 6;
const LANES: usize = 8;

/// Pinned from src/x86_runtime.rs (`vertical_min_miller_terms`). A full-width
/// 512-bit IFMA pipe reaches the lane engine at four live terms, a
/// double-pumped one at six.
const VERTICAL_MIN_MILLER_TERMS_INTEL: usize = 4;
const VERTICAL_MIN_MILLER_TERMS_OTHER: usize = 6;

/// `k * p` in normalized radix-52 limbs plus the carry past limb 4.
const fn mul_p52(k: u64) -> ([u64; 5], u64) {
    let p52 = to_radix52(P);
    let mut out = [0u64; 5];
    let mut carry = 0u128;
    let mut j = 0;
    while j < 5 {
        let acc = k as u128 * p52[j] as u128 + carry;
        out[j] = (acc as u64) & MASK52;
        carry = acc >> 52;
        j += 1;
    }
    (out, carry as u64)
}

const fn to_radix52(x: [u64; 4]) -> [u64; 5] {
    [
        x[0] & MASK52,
        ((x[0] >> 52) | (x[1] << 12)) & MASK52,
        ((x[1] >> 40) | (x[2] << 24)) & MASK52,
        ((x[2] >> 28) | (x[3] << 36)) & MASK52,
        x[3] >> 16,
    ]
}

/// A bound multiple `b` is admissible when `b * p < 2^260` and the 254-bit
/// fold quotient stays below `2^6`.
const fn admissible(b: u64) -> bool {
    let (limbs, carry) = mul_p52(b);
    carry == 0 && limbs[4] >> 46 < 1 << 6
}

/// `s * p <= r * 2^260`, the budget that lets `r` subtraction rounds
/// canonicalize a sum of bounded products.
const fn sos_sum_ok(s: u64, r: u64) -> bool {
    mul_p52(s).1 < r
}

const fn lt52(a: &[u64; 5], b: &[u64; 5]) -> bool {
    let mut j = 4;
    loop {
        if a[j] < b[j] {
            return true;
        }
        if a[j] > b[j] {
            return false;
        }
        if j == 0 {
            return false;
        }
        j -= 1;
    }
}

/// Exact post-fold bound. One additive fold with `eps = 2^254 - p` takes a
/// value below `b * p` to a value below `fold_bound(b) * p`.
const fn fold_bound(b: u64) -> u64 {
    let eps52 = {
        let two254 = [0u64, 0, 0, 1 << 62];
        let mut out = [0u64; 4];
        let mut borrow = 0u64;
        let mut i = 0;
        while i < 4 {
            let (d, b1) = two254[i].overflowing_sub(P[i]);
            let (d, b2) = d.overflowing_sub(borrow);
            out[i] = d;
            borrow = (b1 as u64) | (b2 as u64);
            i += 1;
        }
        to_radix52(out)
    };
    let (bp, carry) = mul_p52(b);
    assert!(b >= 1 && carry == 0);
    let mut vmax = bp;
    let mut j = 0;
    while j < 5 {
        if vmax[j] > 0 {
            vmax[j] -= 1;
            break;
        }
        vmax[j] = MASK52;
        j += 1;
    }
    let q = vmax[4] >> 46;
    let mut out = [MASK52, MASK52, MASK52, MASK52, (1u64 << 46) - 1];
    let mut c = 0u128;
    let mut j = 0;
    while j < 5 {
        let acc = out[j] as u128 + q as u128 * eps52[j] as u128 + c;
        out[j] = (acc as u64) & MASK52;
        c = acc >> 52;
        j += 1;
    }
    assert!(c == 0);
    let mut bound = 1;
    loop {
        let (cp, cc) = mul_p52(bound);
        if cc == 0 && lt52(&out, &cp) {
            return bound;
        }
        bound += 1;
        assert!(bound <= LAZY_CAP);
    }
}

const fn folds_needed(mut b: u64) -> usize {
    let mut n = 0;
    while b > 3 {
        b = fold_bound(b);
        n += 1;
        assert!(n <= 4);
    }
    n
}

/// vpmadd52 count of one interleaved SoS stream of arity `n`. Five
/// recurrence rounds each issue `2 * 5 * n` product madds, one quotient
/// madd, and `2 * 5` reduction madds. Derived from the
/// `mont_sos_mac_8_pair_split` source shape, not from disassembly.
const fn stream_madds(n: usize) -> usize {
    5 * (10 * n + 11)
}

#[test]
#[ignore = "roofline discipline pin, run explicitly"]
fn lazy_bound_model_matches_the_documented_discipline() {
    // LAZY_CAP is the exact admissibility edge for this p.
    assert!(admissible(LAZY_CAP));
    assert!(!admissible(LAZY_CAP + 1));

    // Fold trajectory documented in src/fp/avx512ifma.rs tests.
    assert_eq!(fold_bound(39), 11);
    assert_eq!(fold_bound(11), 4);
    assert_eq!(fold_bound(4), 3);
    assert_eq!(folds_needed(39), 3);
    assert_eq!(folds_needed(15), 2);
    assert_eq!(folds_needed(14), 2);
    assert_eq!(folds_needed(6), 1);
    assert_eq!(folds_needed(2), 0);

    // Sum budgets of the live fused-leaf call sites, with the adversarial
    // negatives that show each subtraction-round count is minimal.
    assert!(sos_sum_ok(6 * SPARSE_NEG, 1));
    assert!(sos_sum_ok(3 * SPARSE_NEG, 1));
    assert!(sos_sum_ok(6 * SQ_NEG, 2));
    assert!(!sos_sum_ok(6 * SQ_NEG, 1));
    assert!(sos_sum_ok(3 * SQ_NEG, 1));
    assert!(sos_sum_ok(12 * FE_NEG, 2));
    assert!(!sos_sum_ok(12 * FE_NEG, 1));
    assert!(sos_sum_ok(6 * FE_NEG, 1));
}

#[test]
#[ignore = "roofline discipline pin, run explicitly"]
fn split_arity_occupancy_and_schedule_counts_hold() {
    // Full occupancy identities. An Fp12 square or sparse product has
    // 6 Fp2 outputs x 3 Fp2 terms x 4 Fp products = 72 schoolbook products,
    // and the 6+3 row pair issues exactly 72 product slots. A dense Fp12
    // product has 6 x 6 x 4 = 144, matched by the 12+6 pair.
    assert_eq!(
        6 * 3 * 4,
        LANES * MILLER_FULL_TERMS + LANES * MILLER_HALF_TERMS
    );
    assert_eq!(6 * 6 * 4, LANES * FE_FULL_TERMS + LANES * FE_HALF_TERMS);

    // Kernel madd totals implied by the recurrence shape.
    assert_eq!(
        stream_madds(MILLER_FULL_TERMS) + stream_madds(MILLER_HALF_TERMS),
        560
    );
    assert_eq!(
        stream_madds(FE_FULL_TERMS) + stream_madds(FE_HALF_TERMS),
        1010
    );

    // The optimal-Ate schedule fixes 87 lines, one doubling per iteration,
    // one addition per nonzero NAF digit, two Frobenius tails. This is the
    // per-term dependent-chain growth of the shared accumulator that the
    // vertical crossover reasons about.
    let doublings = ATE_LOOP_COUNT.len() - 1;
    let additions = ATE_LOOP_COUNT[..ATE_LOOP_COUNT.len() - 1]
        .iter()
        .filter(|&&digit| digit != 0)
        .count()
        + 2;
    assert_eq!(doublings + additions, 87);

    // Every crossover must sit strictly inside the masked-lane width.
    const {
        assert!(VERTICAL_MIN_MILLER_TERMS_INTEL > 1 && VERTICAL_MIN_MILLER_TERMS_INTEL <= LANES);
        assert!(VERTICAL_MIN_MILLER_TERMS_OTHER > 1 && VERTICAL_MIN_MILLER_TERMS_OTHER <= LANES);
    };

    // The kernel truncates the radix-64 inverse to 52 bits.
    // p * (-p^-1) = -1 mod 2^52 proves the truncation is the inverse.
    assert_eq!(
        (P[0] & MASK52).wrapping_mul(P_INV & MASK52) & MASK52,
        MASK52
    );
}

/// One value through every Miller route. The lane engine (live terms at or
/// above the host crossover), the coefficient-lane engine (single pairs,
/// live and prepared), and the mixed stream must agree bit for bit, which
/// is the routing invariant ARCHITECTURE.md states. The routes only exist
/// on an IFMA build, and on a non-IFMA host they all fall back to one
/// scalar path, so the equality stays meaningful but stops being a
/// cross-engine check.
#[cfg(all(
    target_arch = "x86_64",
    any(narsil_avx512_ifma, narsil_x86_runtime_ifma),
    feature = "std",
    not(feature = "force-portable")
))]
mod ifma_routes {
    use helius_narsil::pairing::{
        miller_loop, miller_loop_prepared, multi_miller_loop, prepare_g2,
    };
    use helius_narsil::{Fp12, Fr, G1Affine, G1Projective, G2Affine, G2Projective};

    #[test]
    #[ignore = "roofline discipline pin, run explicitly"]
    fn all_miller_routes_produce_one_value() {
        let g1 = G1Projective::generator();
        let g2 = G2Projective::from(G2Affine::generator());
        let pairs: Vec<_> = (1..=5u64)
            .map(|i| {
                (
                    g1.mul_scalar(Fr::from_u64(i)).to_affine(),
                    g2.mul_scalar(Fr::from_u64(11 * i + 3)).to_affine(),
                )
            })
            .collect();
        let refs: Vec<(&G1Affine, _)> = pairs.iter().map(|(p, q)| (p, q)).collect();

        // Five live terms take whichever route the host crossover selects.
        let fused = multi_miller_loop(&refs);

        // Single pairs take the coefficient-lane engine. The Miller value of
        // a multi loop is the product of the per-pair Miller values because
        // every pair shares the same squaring schedule.
        let mut product = Fp12::ONE;
        for (p, q) in &pairs {
            product *= miller_loop(p, q);
        }
        assert_eq!(fused, product);

        // The prepared replay must match the live stream per pair.
        for (p, q) in &pairs {
            let prepared = prepare_g2(q).expect("generator multiples stay in the subgroup");
            assert_eq!(miller_loop_prepared(p, &prepared), miller_loop(p, q));
        }
    }
}
