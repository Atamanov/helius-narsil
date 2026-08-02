//! Differential and range tests for sums-of-products kernels.

use alloc::vec::Vec;

use rand::RngCore;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::consts::P;
use crate::fp::Fp;
use crate::fp::sos::{negp, sos2, sos4, sos6, sos8, sosd2, sosd4, sosd6, sosd8};
use crate::fp2::Fp2;
use crate::fp2_fast::{f2_from, f2_mul, f2_mul_karatsuba, f2_sqr, f2_sqr_lazy, f2_to};
use crate::fp6::Fp6;
use crate::fp12::Fp12;
use crate::limb::sub_noborrow;

const N_FP2: usize = 100_000;
#[cfg(debug_assertions)]
const N_BIG: usize = 20_000;
#[cfg(not(debug_assertions))]
const N_BIG: usize = 100_000;

pub(crate) fn random_fp(rng: &mut StdRng) -> Fp {
    let mut l = [0u64; 4];
    for w in l.iter_mut() {
        *w = rng.next_u64();
    }
    l[3] &= (1 << 61) - 1; // < 2^253 < p
    Fp::from_raw(l)
}

pub(crate) fn random_fp2(rng: &mut StdRng) -> Fp2 {
    Fp2::new(random_fp(rng), random_fp(rng))
}

pub(crate) fn random_fp6(rng: &mut StdRng) -> Fp6 {
    Fp6::new(random_fp2(rng), random_fp2(rng), random_fp2(rng))
}

pub(crate) fn random_fp12(rng: &mut StdRng) -> Fp12 {
    Fp12::new(random_fp6(rng), random_fp6(rng))
}

/// Montgomery residue whose limb pattern is p-1: maximizes every product and
/// therefore the interleaved accumulator.
fn max_limb_fp() -> Fp {
    Fp::from_raw_unchecked(sub_noborrow(&P, &[1, 0, 0, 0]))
}

/// Edge palette: zero (drives `negp(0) = p` rows), one, Montgomery R,
/// max-limb pattern, and small values.
fn edge_fps() -> [Fp; 6] {
    [
        Fp::ZERO,
        Fp::ONE,
        Fp::from_raw_unchecked([u64::MAX, u64::MAX, u64::MAX, 0]), // dense low limbs
        max_limb_fp(),
        Fp::ZERO - Fp::ONE, // canonical p-1 value (Montgomery form of -1)
        Fp::from_u64(u64::MAX),
    ]
}

/// Reference for the kernels themselves: per-product Montgomery semantics via
/// the untouched `Fp` multiplier. Operands equal to p (from `negp(0)`) are
/// congruent to zero.
fn fp_of(limbs: &[u64; 4]) -> Fp {
    if *limbs == P {
        Fp::ZERO
    } else {
        Fp::from_raw_unchecked(*limbs)
    }
}

fn sos_ref(pairs: &[([u64; 4], [u64; 4])]) -> [u64; 4] {
    let mut acc = Fp::ZERO;
    for (a, b) in pairs {
        acc += fp_of(a) * fp_of(b);
    }
    acc.0
}

#[test]
fn kernels_match_reference_random() {
    let mut rng = StdRng::seed_from_u64(0x50505);
    for _ in 0..N_FP2 {
        let v: [[u64; 4]; 16] = core::array::from_fn(|_| random_fp(&mut rng).0);
        let pairs: Vec<([u64; 4], [u64; 4])> = (0..8).map(|i| (v[2 * i], v[2 * i + 1])).collect();
        assert_eq!(
            sos2(&pairs[0].0, &pairs[0].1, &pairs[1].0, &pairs[1].1),
            sos_ref(&pairs[..2])
        );
        assert_eq!(
            sos4!(
                &pairs[0].0,
                &pairs[0].1,
                &pairs[1].0,
                &pairs[1].1,
                &pairs[2].0,
                &pairs[2].1,
                &pairs[3].0,
                &pairs[3].1
            ),
            sos_ref(&pairs[..4])
        );
        assert_eq!(
            sos6!(
                &pairs[0].0,
                &pairs[0].1,
                &pairs[1].0,
                &pairs[1].1,
                &pairs[2].0,
                &pairs[2].1,
                &pairs[3].0,
                &pairs[3].1,
                &pairs[4].0,
                &pairs[4].1,
                &pairs[5].0,
                &pairs[5].1
            ),
            sos_ref(&pairs[..6])
        );
        assert_eq!(
            sos8!(
                &pairs[0].0,
                &pairs[0].1,
                &pairs[1].0,
                &pairs[1].1,
                &pairs[2].0,
                &pairs[2].1,
                &pairs[3].0,
                &pairs[3].1,
                &pairs[4].0,
                &pairs[4].1,
                &pairs[5].0,
                &pairs[5].1,
                &pairs[6].0,
                &pairs[6].1,
                &pairs[7].0,
                &pairs[7].1
            ),
            sos_ref(&pairs[..8])
        );
    }
}

/// Worst-case accumulation: every operand at the p-1 limb pattern (largest
/// admissible row/digit values) and the `negp(0) = p` row. In debug builds
/// this drives the in-round peaks toward the documented bounds with the
/// bound `debug_assert`s active.
#[test]
fn kernels_worst_case_accumulation() {
    let m = max_limb_fp().0;
    let pm = negp(&[0, 0, 0, 0]); // == p, admissible row
    let mm = [(m, m); 8];
    assert_eq!(sos2(&m, &m, &m, &m), sos_ref(&mm[..2]));
    assert_eq!(sos4!(&m, &m, &m, &m, &m, &m, &m, &m), sos_ref(&mm[..4]));
    assert_eq!(
        sos6!(&m, &m, &m, &m, &m, &m, &m, &m, &m, &m, &m, &m),
        sos_ref(&mm[..6])
    );
    assert_eq!(
        sos8!(
            &m, &m, &m, &m, &m, &m, &m, &m, &m, &m, &m, &m, &m, &m, &m, &m
        ),
        sos_ref(&mm[..8])
    );
    // p-valued rows (= 0 mod p) mixed with maximal rows.
    assert_eq!(sos2(&m, &pm, &m, &m), sos_ref(&[(m, [0; 4]), (m, m)]));
    assert_eq!(
        sos6!(&m, &pm, &m, &pm, &m, &pm, &m, &m, &m, &m, &m, &m),
        sos_ref(&[
            (m, [0; 4]),
            (m, [0; 4]),
            (m, [0; 4]),
            (m, m),
            (m, m),
            (m, m)
        ])
    );
}

/// Dual-lane kernels must match the single-lane kernels bit-for-bit on both
/// lanes (lane0 gets `negp` of each second y component, as the callers did).
#[test]
fn dual_kernels_match_single_lane() {
    let mut rng = StdRng::seed_from_u64(0xD0D0);
    let m = max_limb_fp().0;
    let z = [0u64; 4];
    for i in 0..N_FP2 {
        // Random rows, plus edge rows (all-max, zeros) on early iterations.
        let v: [[u64; 4]; 16] = if i == 0 {
            [m; 16]
        } else if i == 1 {
            core::array::from_fn(|k| if k % 2 == 0 { m } else { z })
        } else {
            core::array::from_fn(|_| random_fp(&mut rng).0)
        };
        let x: Vec<([u64; 4], [u64; 4])> = (0..4).map(|i| (v[4 * i], v[4 * i + 1])).collect();
        let y: Vec<([u64; 4], [u64; 4])> = (0..4).map(|i| (v[4 * i + 2], v[4 * i + 3])).collect();
        let ny: Vec<[u64; 4]> = y.iter().map(|p| negp(&p.1)).collect();

        let d2 = sosd2(&x[0].0, &x[0].1, &y[0].0, &y[0].1);
        assert_eq!(d2.0, sos2(&x[0].0, &y[0].0, &x[0].1, &ny[0]));
        assert_eq!(d2.1, sos2(&x[0].0, &y[0].1, &x[0].1, &y[0].0));

        let d4 = sosd4!(
            &x[0].0, &x[0].1, &y[0].0, &y[0].1, &x[1].0, &x[1].1, &y[1].0, &y[1].1,
        );
        assert_eq!(
            d4.0,
            sos4!(
                &x[0].0, &y[0].0, &x[0].1, &ny[0], &x[1].0, &y[1].0, &x[1].1, &ny[1]
            )
        );
        assert_eq!(
            d4.1,
            sos4!(
                &x[0].0, &y[0].1, &x[0].1, &y[0].0, &x[1].0, &y[1].1, &x[1].1, &y[1].0
            )
        );

        let d6 = sosd6!(
            &x[0].0, &x[0].1, &y[0].0, &y[0].1, &x[1].0, &x[1].1, &y[1].0, &y[1].1, &x[2].0,
            &x[2].1, &y[2].0, &y[2].1,
        );
        assert_eq!(
            d6.0,
            sos6!(
                &x[0].0, &y[0].0, &x[0].1, &ny[0], &x[1].0, &y[1].0, &x[1].1, &ny[1], &x[2].0,
                &y[2].0, &x[2].1, &ny[2]
            )
        );
        assert_eq!(
            d6.1,
            sos6!(
                &x[0].0, &y[0].1, &x[0].1, &y[0].0, &x[1].0, &y[1].1, &x[1].1, &y[1].0, &x[2].0,
                &y[2].1, &x[2].1, &y[2].0
            )
        );

        let d8 = sosd8!(
            &x[0].0, &x[0].1, &y[0].0, &y[0].1, &x[1].0, &x[1].1, &y[1].0, &y[1].1, &x[2].0,
            &x[2].1, &y[2].0, &y[2].1, &x[3].0, &x[3].1, &y[3].0, &y[3].1,
        );
        assert_eq!(
            d8.0,
            sos8!(
                &x[0].0, &y[0].0, &x[0].1, &ny[0], &x[1].0, &y[1].0, &x[1].1, &ny[1], &x[2].0,
                &y[2].0, &x[2].1, &ny[2], &x[3].0, &y[3].0, &x[3].1, &ny[3]
            )
        );
        assert_eq!(
            d8.1,
            sos8!(
                &x[0].0, &y[0].1, &x[0].1, &y[0].0, &x[1].0, &y[1].1, &x[1].1, &y[1].0, &x[2].0,
                &y[2].1, &x[2].1, &y[2].0, &x[3].0, &y[3].1, &x[3].1, &y[3].0
            )
        );
    }
}

#[test]
fn fp2_mul_sqr_differential() {
    let mut rng = StdRng::seed_from_u64(0xF2F2);
    for _ in 0..N_FP2 {
        let a = random_fp2(&mut rng);
        let b = random_fp2(&mut rng);
        let (fa, fb) = (f2_from(a), f2_from(b));
        assert_eq!(f2_mul(fa, fb), f2_mul_karatsuba(fa, fb));
        assert_eq!(f2_sqr(fa), f2_sqr_lazy(fa));
        // Operator API routes through the same kernels. Cross-check anyway.
        assert_eq!(f2_to(f2_mul(fa, fb)), a * b);
        assert_eq!(f2_to(f2_sqr(fa)), a.square());
    }
}

#[test]
fn fp2_mul_sqr_edges() {
    let edges = edge_fps();
    for &c0 in &edges {
        for &c1 in &edges {
            for &d0 in &edges {
                for &d1 in &edges {
                    let a = Fp2::new(c0, c1);
                    let b = Fp2::new(d0, d1);
                    let (fa, fb) = (f2_from(a), f2_from(b));
                    assert_eq!(f2_mul(fa, fb), f2_mul_karatsuba(fa, fb));
                    assert_eq!(f2_sqr(fa), f2_sqr_lazy(fa));
                }
            }
        }
    }
}

#[test]
fn fp6_mul_differential() {
    let mut rng = StdRng::seed_from_u64(0xF6F6);
    for _ in 0..N_BIG {
        let a = random_fp6(&mut rng);
        let b = random_fp6(&mut rng);
        assert_eq!(a * b, a.mul_karatsuba(b));
    }
}

#[test]
fn fp6_mul_by_01_differential() {
    let mut rng = StdRng::seed_from_u64(0x0101);
    for _ in 0..N_BIG {
        let a = random_fp6(&mut rng);
        let c0 = random_fp2(&mut rng);
        let c1 = random_fp2(&mut rng);
        assert_eq!(a.mul_by_01(c0, c1), a.mul_by_01_karatsuba(c0, c1));
    }
}

#[test]
fn fp6_edges() {
    let edges = edge_fps();
    let pats: Vec<Fp2> = edges
        .iter()
        .flat_map(|&x| edges.iter().map(move |&y| Fp2::new(x, y)))
        .collect();
    // Exhaustive over a reduced palette for the 6-slot shapes.
    let small = [
        Fp2::ZERO,
        Fp2::ONE,
        Fp2::new(max_limb_fp(), max_limb_fp()),
        Fp2::new(Fp::ZERO - Fp::ONE, Fp::ZERO),
    ];
    for &x in &small {
        for &y in &small {
            for &z in &small {
                let a = Fp6::new(x, y, z);
                for &w in &small {
                    let b = Fp6::new(w, z, x);
                    assert_eq!(a * b, a.mul_karatsuba(b));
                    assert_eq!(a.mul_by_01(w, z), a.mul_by_01_karatsuba(w, z));
                }
            }
        }
    }
    // Every Fp2 edge pattern in each coordinate against a worst-case operand.
    let worst = Fp6::new(
        Fp2::new(max_limb_fp(), max_limb_fp()),
        Fp2::new(max_limb_fp(), max_limb_fp()),
        Fp2::new(max_limb_fp(), max_limb_fp()),
    );
    for &p0 in &pats {
        let a = Fp6::new(p0, p0, p0);
        assert_eq!(a * worst, a.mul_karatsuba(worst));
        assert_eq!(worst * a, worst.mul_karatsuba(a));
        assert_eq!(a.mul_by_01(p0, p0), a.mul_by_01_karatsuba(p0, p0));
        assert_eq!(worst.mul_by_01(p0, p0), worst.mul_by_01_karatsuba(p0, p0));
    }
}

#[test]
fn fp12_mul_by_034_differential() {
    let mut rng = StdRng::seed_from_u64(0x0340);
    for _ in 0..N_BIG {
        let a = random_fp12(&mut rng);
        let c0 = random_fp2(&mut rng);
        let c3 = random_fp2(&mut rng);
        let c4 = random_fp2(&mut rng);
        assert_eq!(a.mul_by_034(c0, c3, c4), a.mul_by_034_karatsuba(c0, c3, c4));
    }
}

#[test]
fn fp12_square_differential() {
    let mut rng = StdRng::seed_from_u64(0x50);
    for _ in 0..N_BIG {
        let a = random_fp12(&mut rng);
        assert_eq!(a.square(), a.square_karatsuba());
    }
}

/// Compare both square formulas on random and maximum field values.
#[test]
fn fp12_square_formulas_match() {
    let compare = |a: Fp12| {
        let mut new = a;
        new.square_in_place_sos();
        let mut old = a;
        old.square_in_place_sos_d8();
        assert_eq!(new, old);
    };
    let mut rng = StdRng::seed_from_u64(0x5084);
    for _ in 0..N_BIG {
        compare(random_fp12(&mut rng));
    }
    let m2 = Fp2::new(max_limb_fp(), max_limb_fp());
    let m6 = Fp6::new(m2, m2, m2);
    compare(Fp12::new(m6, m6));
    compare(Fp12::ZERO);
    compare(Fp12::ONE);
}

#[test]
fn fp12_cyclotomic_square_differential() {
    // The formula identity holds for all Fp12 inputs, cyclotomic or not.
    let mut rng = StdRng::seed_from_u64(0xC5C5);
    for _ in 0..N_BIG {
        let a = random_fp12(&mut rng);
        assert_eq!(a.cyclotomic_square(), a.cyclotomic_square_karatsuba());
    }
}

#[test]
fn fp12_edges() {
    let m2 = Fp2::new(max_limb_fp(), max_limb_fp());
    let small = [
        Fp2::ZERO,
        Fp2::ONE,
        m2,
        Fp2::new(Fp::ZERO - Fp::ONE, Fp::ZERO),
    ];
    for &x in &small {
        for &y in &small {
            let a = Fp12::new(Fp6::new(x, y, x), Fp6::new(y, x, y));
            assert_eq!(a.square(), a.square_karatsuba());
            assert_eq!(a.cyclotomic_square(), a.cyclotomic_square_karatsuba());
            for &z in &small {
                assert_eq!(a.mul_by_034(x, y, z), a.mul_by_034_karatsuba(x, y, z));
            }
        }
    }
}

/// x86-64 ADX tier: the rolled `helius_sos_x86` leaf behind the public SoS
/// entry points must agree with the portable kernels bit for bit. Random
/// residues, the edge palette (zero drives `negp(0) = p` operands through
/// the leaf), and a million-case release stress gate.
#[cfg(all(helius_mont4_x86_64_adx, not(feature = "force-portable")))]
mod leaf_differential {
    use super::{edge_fps, random_fp};
    use crate::fp::Fp;
    use crate::fp::sos::{
        sos2, sos2_portable, sos4, sos4_portable, sosd2, sosd2_portable, sosd4, sosd4_portable,
        sosd6, sosd6_portable, sosd8, sosd8_portable,
    };
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn compare_all(f: &[Fp; 16], case: usize) {
        let w: [&[u64; 4]; 16] = core::array::from_fn(|i| &f[i].0);
        let products4 = [(w[0], w[1]), (w[2], w[3]), (w[4], w[5]), (w[6], w[7])];
        let fp2_products2 = [(w[0], w[1], w[2], w[3]), (w[4], w[5], w[6], w[7])];
        let fp2_products3 = [
            (w[0], w[1], w[2], w[3]),
            (w[4], w[5], w[6], w[7]),
            (w[8], w[9], w[10], w[11]),
        ];
        let fp2_products4 = [
            fp2_products3[0],
            fp2_products3[1],
            fp2_products3[2],
            (w[12], w[13], w[14], w[15]),
        ];
        assert_eq!(
            sos2(w[0], w[1], w[2], w[3]),
            sos2_portable(w[0], w[1], w[2], w[3]),
            "sos2 case {case}",
        );
        assert_eq!(
            sos4!(w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]),
            sos4_portable(products4),
            "sos4 case {case}",
        );
        assert_eq!(
            sosd2(w[0], w[1], w[2], w[3]),
            sosd2_portable(w[0], w[1], w[2], w[3]),
            "sosd2 case {case}",
        );
        // The asm sosd2 leaf is verified on silicon regardless of whether the
        // production dispatch links it (HELIUS_SOSD2_ASM).
        #[cfg(all(helius_mont4_x86_64_adx, not(feature = "force-portable")))]
        assert_eq!(
            crate::fp::x86_64::sosd2(w[0], w[1], w[2], w[3]),
            sosd2_portable(w[0], w[1], w[2], w[3]),
            "sosd2 asm leaf case {case}",
        );
        assert_eq!(
            sosd4!(w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]),
            sosd4_portable(fp2_products2),
            "sosd4 case {case}",
        );
        assert_eq!(
            sosd6!(
                w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], w[8], w[9], w[10], w[11],
            ),
            sosd6_portable(fp2_products3),
            "sosd6 case {case}",
        );
        // Both x86 sosd6 routes are verified on silicon regardless of which
        // one the production dispatch links (HELIUS_SOSD6_ASM).
        assert_eq!(
            crate::fp::x86_64::sosd6_leaf(fp2_products3),
            sosd6_portable(fp2_products3),
            "sosd6 asm leaf case {case}",
        );
        assert_eq!(
            crate::fp::x86_64::sosd6(fp2_products3),
            sosd6_portable(fp2_products3),
            "sosd6 composed case {case}",
        );
        assert_eq!(
            sosd8!(
                w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], w[8], w[9], w[10], w[11], w[12],
                w[13], w[14], w[15],
            ),
            sosd8_portable(fp2_products4),
            "sosd8 case {case}",
        );
    }

    fn random_case(rng: &mut StdRng) -> [Fp; 16] {
        core::array::from_fn(|_| random_fp(rng))
    }

    #[test]
    fn leaf_matches_portable_on_random_residues() {
        let mut rng = StdRng::seed_from_u64(0x50F7);
        for case in 0..super::N_BIG {
            compare_all(&random_case(&mut rng), case);
        }
    }

    #[test]
    fn leaf_matches_portable_on_edge_palette() {
        let palette = edge_fps();
        let mut rng = StdRng::seed_from_u64(0x50F8);
        // Saturate all sixteen slots with each palette value, and mixed
        // palette/random fills, so every operand position sees the edges.
        for (i, &e) in palette.iter().enumerate() {
            compare_all(&[e; 16], i);
            for round in 0..64 {
                let mut f = random_case(&mut rng);
                for slot in 0..16 {
                    if (round >> (slot % 6)) & 1 == 1 {
                        f[slot] = palette[(slot + i) % palette.len()];
                    }
                }
                compare_all(&f, 1000 * i + round);
            }
        }
    }

    #[test]
    #[ignore = "million-case release stress gate; run explicitly before changing field backends"]
    fn million_random_cases_match_portable() {
        let mut rng = StdRng::seed_from_u64(0x50F9);
        for case in 0..1_000_000 {
            compare_all(&random_case(&mut rng), case);
        }
    }
}

/// x86-64 ADX tier: the whole-Fp6 leaf must agree with the composed sosd6
/// path bit for bit, whatever HELIUS_FP6_ASM selects for production
/// dispatch -- the leaf wrapper is called directly. Random operands, the
/// edge palette rotated through every Fp2 slot, and targeted xi edges
/// (b1/b2 components where 9*re - im underflows without the +p route).
#[cfg(all(helius_mont4_x86_64_adx, not(feature = "force-portable")))]
mod fp6_leaf_differential {
    use super::{edge_fps, random_fp2, random_fp6};
    use crate::fp::Fp;
    use crate::fp2::Fp2;
    use crate::fp6::Fp6;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn compare(a: Fp6, b: Fp6, case: usize) {
        assert_eq!(
            crate::fp::x86_64::fp6_mul(&a, &b),
            a.mul_sosd6(b),
            "fp6 leaf case {case}",
        );
    }

    #[test]
    fn fp6_leaf_matches_composed_on_random_operands() {
        let mut rng = StdRng::seed_from_u64(0xF6F6);
        for case in 0..super::N_BIG / 10 {
            compare(random_fp6(&mut rng), random_fp6(&mut rng), case);
        }
    }

    #[test]
    fn fp6_leaf_matches_composed_on_edge_palette() {
        let palette = edge_fps();
        let mut rng = StdRng::seed_from_u64(0xF6F7);
        for (i, &e) in palette.iter().enumerate() {
            let uniform = Fp6::new(Fp2::new(e, e), Fp2::new(e, e), Fp2::new(e, e));
            compare(uniform, uniform, i);
            // Rotate the edge through every Fp slot of both operands.
            for slot in 0..12 {
                let mut a = random_fp6(&mut rng);
                let mut b = random_fp6(&mut rng);
                {
                    let target = if slot < 6 { &mut a } else { &mut b };
                    let fp2 = match (slot / 2) % 3 {
                        0 => &mut target.c0,
                        1 => &mut target.c1,
                        _ => &mut target.c2,
                    };
                    if slot % 2 == 0 {
                        fp2.c0 = e;
                    } else {
                        fp2.c1 = e;
                    }
                }
                compare(a, b, 100 * i + slot);
            }
        }
        // xi edges: re = 0, im = p-1 maximizes the (p - im) route. Im = 0
        // sends the negp rows to exactly p. Both extremes together.
        let p_minus_one = Fp::ZERO - Fp::ONE;
        for (case, edge) in [
            Fp2::new(Fp::ZERO, p_minus_one),
            Fp2::new(p_minus_one, Fp::ZERO),
            Fp2::new(Fp::ZERO, Fp::ZERO),
            Fp2::new(p_minus_one, p_minus_one),
        ]
        .into_iter()
        .enumerate()
        {
            let a = random_fp6(&mut rng);
            let b = Fp6::new(random_fp2(&mut rng), edge, edge);
            compare(a, b, 10_000 + case);
        }
    }

    #[test]
    #[ignore = "hundred-thousand-case release stress gate; run explicitly before changing field backends"]
    fn stress_random_cases_match_composed() {
        let mut rng = StdRng::seed_from_u64(0xF6F8);
        for case in 0..100_000 {
            compare(random_fp6(&mut rng), random_fp6(&mut rng), case);
        }
    }
}

/// x86-64 ADX tier: the whole-op fp12_034 leaf must agree with the composed
/// sosd6 path bit for bit, whatever HELIUS_FP12_034_ASM selects for
/// production dispatch -- the leaf wrapper is called directly. Random
/// operands, the edge palette rotated through every Fp slot, zero
/// coefficients (the line shapes ell() can degenerate to), line_value-shaped
/// sparse accumulators (the second Miller iteration's input), and targeted
/// xi edges on c3/c4.
#[cfg(all(helius_mont4_x86_64_adx, not(feature = "force-portable")))]
mod fp12_034_leaf_differential {
    use super::{edge_fps, random_fp2, random_fp12};
    use crate::fp::Fp;
    use crate::fp2::Fp2;
    use crate::fp6::Fp6;
    use crate::fp12::Fp12;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn compare(f: Fp12, c0: Fp2, c3: Fp2, c4: Fp2, case: usize) {
        let mut leaf = f;
        crate::fp::x86_64::fp12_034_assign(&mut leaf, &c0, &c3, &c4);
        let mut composed = f;
        composed.mul_by_034_assign_sosd6(c0, c3, c4);
        assert_eq!(leaf, composed, "fp12_034 leaf case {case}");
    }

    #[test]
    fn fp12_034_leaf_matches_composed_on_random_operands() {
        let mut rng = StdRng::seed_from_u64(0x0341);
        for case in 0..super::N_BIG / 10 {
            compare(
                random_fp12(&mut rng),
                random_fp2(&mut rng),
                random_fp2(&mut rng),
                random_fp2(&mut rng),
                case,
            );
        }
    }

    #[test]
    fn fp12_034_leaf_matches_composed_on_edge_palette() {
        let palette = edge_fps();
        let mut rng = StdRng::seed_from_u64(0x0342);
        for (i, &e) in palette.iter().enumerate() {
            let e2 = Fp2::new(e, e);
            let uniform = Fp12::new(Fp6::new(e2, e2, e2), Fp6::new(e2, e2, e2));
            compare(uniform, e2, e2, e2, i);
            // Rotate the edge through every Fp slot of f and the coefficients.
            for slot in 0..18 {
                let mut f = random_fp12(&mut rng);
                let mut c = [
                    random_fp2(&mut rng),
                    random_fp2(&mut rng),
                    random_fp2(&mut rng),
                ];
                {
                    let fp2 = if slot < 12 {
                        let half = if slot < 6 { &mut f.c0 } else { &mut f.c1 };
                        match (slot / 2) % 3 {
                            0 => &mut half.c0,
                            1 => &mut half.c1,
                            _ => &mut half.c2,
                        }
                    } else {
                        &mut c[(slot - 12) / 2]
                    };
                    if slot % 2 == 0 {
                        fp2.c0 = e;
                    } else {
                        fp2.c1 = e;
                    }
                }
                compare(f, c[0], c[1], c[2], 100 * i + slot);
            }
        }
        // Zero coefficients: every subset a degenerate line could produce
        // (zero Fp2s also drive negp(0) = p rows through every block).
        for mask in 1..8usize {
            let f = random_fp12(&mut rng);
            let pick = |bit: usize, rng: &mut StdRng| {
                if mask & (1 << bit) != 0 {
                    Fp2::ZERO
                } else {
                    random_fp2(rng)
                }
            };
            let c0 = pick(0, &mut rng);
            let c3 = pick(1, &mut rng);
            let c4 = pick(2, &mut rng);
            compare(f, c0, c3, c4, 1000 + mask);
        }
        // The second Miller iteration's accumulator: a line_value image
        // (only the c0/c3/c4 slots of f populated).
        let sparse = Fp12::new(
            Fp6::new(random_fp2(&mut rng), Fp2::ZERO, Fp2::ZERO),
            Fp6::new(random_fp2(&mut rng), random_fp2(&mut rng), Fp2::ZERO),
        );
        compare(
            sparse,
            random_fp2(&mut rng),
            random_fp2(&mut rng),
            random_fp2(&mut rng),
            2000,
        );
        // xi edges on the scaled coefficients: re = 0, im = p-1 maximizes
        // the (p - im) route. Im = 0 sends the negp rows to exactly p.
        let p_minus_one = Fp::ZERO - Fp::ONE;
        for (case, edge) in [
            Fp2::new(Fp::ZERO, p_minus_one),
            Fp2::new(p_minus_one, Fp::ZERO),
            Fp2::new(Fp::ZERO, Fp::ZERO),
            Fp2::new(p_minus_one, p_minus_one),
        ]
        .into_iter()
        .enumerate()
        {
            let f = random_fp12(&mut rng);
            compare(f, random_fp2(&mut rng), edge, edge, 3000 + case);
        }
    }

    #[test]
    #[ignore = "hundred-thousand-case release stress gate; run explicitly before changing field backends"]
    fn stress_random_cases_match_composed() {
        let mut rng = StdRng::seed_from_u64(0x0343);
        for case in 0..100_000 {
            compare(
                random_fp12(&mut rng),
                random_fp2(&mut rng),
                random_fp2(&mut rng),
                random_fp2(&mut rng),
                case,
            );
        }
    }
}

/// x86-64 ADX tier: the lazy double-width 034 leaf (39 products + 12
/// reductions, mcl mul_403's shape) must agree with the composed sosd6
/// path bit for bit, whatever the 034 toggles select for production
/// dispatch -- the leaf wrapper is called directly. Random operands, the
/// edge palette rotated through every Fp slot of f and the coefficients,
/// degenerate-line coefficient subsets, xi edges, and the exact
/// (accumulator, line coefficient) pairs ell() feeds mid-Miller.
#[cfg(all(helius_mont4_x86_64_adx, not(feature = "force-portable")))]
mod fp12_034l_leaf_differential {
    use super::{edge_fps, random_fp2, random_fp12};
    use crate::fp::Fp;
    use crate::fp2::Fp2;
    use crate::fp6::Fp6;
    use crate::fp12::Fp12;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn compare(f: Fp12, c0: Fp2, c3: Fp2, c4: Fp2, case: usize) {
        let mut leaf = f;
        crate::fp::x86_64::fp12_034l_assign(&mut leaf, &c0, &c3, &c4);
        let mut composed = f;
        composed.mul_by_034_assign_sosd6(c0, c3, c4);
        assert_eq!(leaf, composed, "fp12_034l leaf case {case}");
    }

    #[test]
    fn fp12_034l_leaf_matches_composed_on_random_operands() {
        let mut rng = StdRng::seed_from_u64(0x034A);
        for case in 0..super::N_BIG / 10 {
            compare(
                random_fp12(&mut rng),
                random_fp2(&mut rng),
                random_fp2(&mut rng),
                random_fp2(&mut rng),
                case,
            );
        }
    }

    #[test]
    fn fp12_034l_leaf_matches_composed_on_edge_palette() {
        let palette = edge_fps();
        let mut rng = StdRng::seed_from_u64(0x034B);
        for (i, &e) in palette.iter().enumerate() {
            let e2 = Fp2::new(e, e);
            let uniform = Fp12::new(Fp6::new(e2, e2, e2), Fp6::new(e2, e2, e2));
            compare(uniform, e2, e2, e2, i);
            // Rotate the edge through every Fp slot of f and the coefficients.
            for slot in 0..18 {
                let mut f = random_fp12(&mut rng);
                let mut c = [
                    random_fp2(&mut rng),
                    random_fp2(&mut rng),
                    random_fp2(&mut rng),
                ];
                {
                    let fp2 = if slot < 12 {
                        let half = if slot < 6 { &mut f.c0 } else { &mut f.c1 };
                        match (slot / 2) % 3 {
                            0 => &mut half.c0,
                            1 => &mut half.c1,
                            _ => &mut half.c2,
                        }
                    } else {
                        &mut c[(slot - 12) / 2]
                    };
                    if slot % 2 == 0 {
                        fp2.c0 = e;
                    } else {
                        fp2.c1 = e;
                    }
                }
                compare(f, c[0], c[1], c[2], 100 * i + slot);
            }
        }
        // Zero coefficients: every subset a degenerate line could produce
        // (zeros also drive the guarded negations through exact p*2^256).
        for mask in 1..8usize {
            let f = random_fp12(&mut rng);
            let pick = |bit: usize, rng: &mut StdRng| {
                if mask & (1 << bit) != 0 {
                    Fp2::ZERO
                } else {
                    random_fp2(rng)
                }
            };
            let c0 = pick(0, &mut rng);
            let c3 = pick(1, &mut rng);
            let c4 = pick(2, &mut rng);
            compare(f, c0, c3, c4, 1000 + mask);
        }
        // The second Miller iteration's accumulator: a line_value image
        // (only the c0/c3/c4 slots of f populated).
        let sparse = Fp12::new(
            Fp6::new(random_fp2(&mut rng), Fp2::ZERO, Fp2::ZERO),
            Fp6::new(random_fp2(&mut rng), random_fp2(&mut rng), Fp2::ZERO),
        );
        compare(
            sparse,
            random_fp2(&mut rng),
            random_fp2(&mut rng),
            random_fp2(&mut rng),
            2000,
        );
        // xi edges on the CE/CE' sites (a1.c2 * c4) and the BD.c fold:
        // re = 0, im = p-1 maximizes the negation rows. Im = 0 sends them
        // to exactly p*2^256.
        let p_minus_one = Fp::ZERO - Fp::ONE;
        for (case, edge) in [
            Fp2::new(Fp::ZERO, p_minus_one),
            Fp2::new(p_minus_one, Fp::ZERO),
            Fp2::new(Fp::ZERO, Fp::ZERO),
            Fp2::new(p_minus_one, p_minus_one),
        ]
        .into_iter()
        .enumerate()
        {
            let mut f = random_fp12(&mut rng);
            f.c1.c2 = edge;
            compare(f, random_fp2(&mut rng), edge, edge, 3000 + case);
        }
    }

    /// Homogeneous projective D-twist point of the replayed Miller
    /// recurrence.
    struct Hom {
        x: Fp2,
        y: Fp2,
        z: Fp2,
    }

    /// The miller.rs doubling step (eprint 2013/722): advances `r` and
    /// returns the D-type line coefficients (-h, 3j, i).
    fn dbl_step(r: &mut Hom, twist_b: Fp2, half: Fp2) -> (Fp2, Fp2, Fp2) {
        let a = r.x * r.y * half;
        let b = r.y.square();
        let c = r.z.square();
        let e = twist_b * (c.double() + c);
        let f = e.double() + e;
        let g = (b + f) * half;
        let h = (r.y + r.z).square() - (b + c);
        let i = e - b;
        let j = r.x.square();
        let e2 = e.square();
        r.x = a * (b - f);
        r.y = g.square() - (e2.double() + e2);
        r.z = b * h;
        (-h, j.double() + j, i)
    }

    /// The miller.rs mixed-addition step: advances `r` and returns the
    /// D-type line coefficients (lambda, -theta, j).
    fn add_step(r: &mut Hom, qx: Fp2, qy: Fp2) -> (Fp2, Fp2, Fp2) {
        let theta = r.y - qy * r.z;
        let lambda = r.x - qx * r.z;
        let c = theta.square();
        let d = lambda.square();
        let e = lambda * d;
        let f = r.z * c;
        let g = r.x * d;
        let h = e + f - g.double();
        r.x = lambda * h;
        r.y = theta * (g - h) - e * r.y;
        r.z *= e;
        let j = theta * qx - lambda * qy;
        (lambda, -theta, j)
    }

    /// The production shape: the exact (accumulator, coefficient) pairs
    /// ell() feeds the leaf, harvested by replaying the Miller recurrence
    /// (miller.rs line formulas over the public Fp2 API) with the leaf and
    /// the composed path in lockstep at every line update. The replay's
    /// final value is pinned against the production miller_loop, so the
    /// harvested sequence is provably the real one.
    #[test]
    fn fp12_034l_leaf_matches_composed_mid_miller() {
        use crate::consts::ATE_LOOP_COUNT;
        use crate::fp2_fast::f2_to;
        use crate::fr::Fr;
        use crate::g1::{G1Affine, G1Projective};
        use crate::g2::{G2Affine, G2Projective};
        use crate::pairing::miller::{mul_by_char, twist_b_f2};

        let half = Fp2::from(2u64).invert().unwrap();
        let twist_b = f2_to(twist_b_f2());
        let generator_p = G1Projective::generator();
        let generator_q = G2Projective::from(G2Affine::generator());
        for point_case in 0..2u64 {
            let p: G1Affine = generator_p
                .mul_scalar(Fr::from_u64(2 * point_case + 3))
                .to_affine();
            let q: G2Affine = generator_q
                .mul_scalar(Fr::from_u64(5 * point_case + 7))
                .to_affine();
            let mut r = Hom {
                x: q.x,
                y: q.y,
                z: Fp2::ONE,
            };
            let mut f = Fp12::ONE;
            let mut started = false;
            let mut step = 2000 * point_case as usize;
            let mut push_line = |f: &mut Fp12, coeffs: (Fp2, Fp2, Fp2), step: &mut usize| {
                let c0 = coeffs.0.mul_by_fp(p.y);
                let c3 = coeffs.1.mul_by_fp(p.x);
                let c4 = coeffs.2;
                if !started {
                    // First iteration: the accumulator IS the line value.
                    started = true;
                    *f = Fp12::new(
                        Fp6::new(c0, Fp2::ZERO, Fp2::ZERO),
                        Fp6::new(c3, c4, Fp2::ZERO),
                    );
                } else {
                    compare(*f, c0, c3, c4, *step);
                    f.mul_by_034_assign_sosd6(c0, c3, c4);
                }
                *step += 1;
            };
            for i in (1..ATE_LOOP_COUNT.len()).rev() {
                if i != ATE_LOOP_COUNT.len() - 1 {
                    f.square_in_place_sos();
                }
                let coeffs = dbl_step(&mut r, twist_b, half);
                push_line(&mut f, coeffs, &mut step);
                match ATE_LOOP_COUNT[i - 1] {
                    1 => {
                        let coeffs = add_step(&mut r, q.x, q.y);
                        push_line(&mut f, coeffs, &mut step);
                    }
                    -1 => {
                        let coeffs = add_step(&mut r, q.x, -q.y);
                        push_line(&mut f, coeffs, &mut step);
                    }
                    _ => {}
                }
            }
            let q1 = mul_by_char(q);
            let q2 = mul_by_char(q1).negate();
            for corrector in [q1, q2] {
                let coeffs = add_step(&mut r, corrector.x, corrector.y);
                push_line(&mut f, coeffs, &mut step);
            }
            assert_eq!(
                f,
                crate::pairing::miller_loop(&p, &q),
                "replayed recurrence diverged from the production Miller loop, case {point_case}",
            );
        }
    }

    #[test]
    #[ignore = "hundred-thousand-case release stress gate; run explicitly before changing field backends"]
    fn stress_random_cases_match_composed() {
        let mut rng = StdRng::seed_from_u64(0x034C);
        for case in 0..100_000 {
            compare(
                random_fp12(&mut rng),
                random_fp2(&mut rng),
                random_fp2(&mut rng),
                random_fp2(&mut rng),
                case,
            );
        }
    }
}

/// x86-64 ADX tier: the Karatsuba uniform-walk 034 leaf (54 products + 12
/// reductions, the v1 walk shape) must agree with the composed sosd6 path
/// bit for bit, whatever the 034 toggles select for production dispatch --
/// the leaf wrapper is called directly. Random operands, the edge palette
/// rotated through every Fp slot of f and the coefficients,
/// degenerate-line coefficient subsets, xi edges on c3/c4 (the X3/X4
/// sites), and the exact (accumulator, line coefficient) pairs ell() feeds
/// mid-Miller.
#[cfg(all(helius_mont4_x86_64_adx, not(feature = "force-portable")))]
mod fp12_034k_leaf_differential {
    use super::{edge_fps, random_fp2, random_fp12};
    use crate::fp::Fp;
    use crate::fp2::Fp2;
    use crate::fp6::Fp6;
    use crate::fp12::Fp12;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn compare(f: Fp12, c0: Fp2, c3: Fp2, c4: Fp2, case: usize) {
        let mut leaf = f;
        crate::fp::x86_64::fp12_034k_assign(&mut leaf, &c0, &c3, &c4);
        let mut composed = f;
        composed.mul_by_034_assign_sosd6(c0, c3, c4);
        assert_eq!(leaf, composed, "fp12_034k leaf case {case}");
    }

    #[test]
    fn fp12_034k_leaf_matches_composed_on_random_operands() {
        let mut rng = StdRng::seed_from_u64(0x034D);
        for case in 0..super::N_BIG / 10 {
            compare(
                random_fp12(&mut rng),
                random_fp2(&mut rng),
                random_fp2(&mut rng),
                random_fp2(&mut rng),
                case,
            );
        }
    }

    #[test]
    fn fp12_034k_leaf_matches_composed_on_edge_palette() {
        let palette = edge_fps();
        let mut rng = StdRng::seed_from_u64(0x034E);
        for (i, &e) in palette.iter().enumerate() {
            let e2 = Fp2::new(e, e);
            let uniform = Fp12::new(Fp6::new(e2, e2, e2), Fp6::new(e2, e2, e2));
            compare(uniform, e2, e2, e2, i);
            // Rotate the edge through every Fp slot of f and the coefficients.
            for slot in 0..18 {
                let mut f = random_fp12(&mut rng);
                let mut c = [
                    random_fp2(&mut rng),
                    random_fp2(&mut rng),
                    random_fp2(&mut rng),
                ];
                {
                    let fp2 = if slot < 12 {
                        let half = if slot < 6 { &mut f.c0 } else { &mut f.c1 };
                        match (slot / 2) % 3 {
                            0 => &mut half.c0,
                            1 => &mut half.c1,
                            _ => &mut half.c2,
                        }
                    } else {
                        &mut c[(slot - 12) / 2]
                    };
                    if slot % 2 == 0 {
                        fp2.c0 = e;
                    } else {
                        fp2.c1 = e;
                    }
                }
                compare(f, c[0], c[1], c[2], 100 * i + slot);
            }
        }
        // Zero coefficients: every subset a degenerate line could produce
        // (zero Fp2s drive the pre-sums, guards and negp rows to zero/p).
        for mask in 1..8usize {
            let f = random_fp12(&mut rng);
            let pick = |bit: usize, rng: &mut StdRng| {
                if mask & (1 << bit) != 0 {
                    Fp2::ZERO
                } else {
                    random_fp2(rng)
                }
            };
            let c0 = pick(0, &mut rng);
            let c3 = pick(1, &mut rng);
            let c4 = pick(2, &mut rng);
            compare(f, c0, c3, c4, 1000 + mask);
        }
        // The second Miller iteration's accumulator: a line_value image
        // (only the c0/c3/c4 slots of f populated).
        let sparse = Fp12::new(
            Fp6::new(random_fp2(&mut rng), Fp2::ZERO, Fp2::ZERO),
            Fp6::new(random_fp2(&mut rng), random_fp2(&mut rng), Fp2::ZERO),
        );
        compare(
            sparse,
            random_fp2(&mut rng),
            random_fp2(&mut rng),
            random_fp2(&mut rng),
            2000,
        );
        // xi edges on the scaled coefficients c3, c4: re = 0, im = p-1
        // maximizes the (p - im) route. Im = 0 sends the negp rows to
        // exactly p.
        let p_minus_one = Fp::ZERO - Fp::ONE;
        for (case, edge) in [
            Fp2::new(Fp::ZERO, p_minus_one),
            Fp2::new(p_minus_one, Fp::ZERO),
            Fp2::new(Fp::ZERO, Fp::ZERO),
            Fp2::new(p_minus_one, p_minus_one),
        ]
        .into_iter()
        .enumerate()
        {
            let f = random_fp12(&mut rng);
            compare(f, random_fp2(&mut rng), edge, edge, 3000 + case);
        }
    }

    /// Homogeneous projective D-twist point of the replayed Miller
    /// recurrence.
    struct Hom {
        x: Fp2,
        y: Fp2,
        z: Fp2,
    }

    /// The miller.rs doubling step (eprint 2013/722): advances `r` and
    /// returns the D-type line coefficients (-h, 3j, i).
    fn dbl_step(r: &mut Hom, twist_b: Fp2, half: Fp2) -> (Fp2, Fp2, Fp2) {
        let a = r.x * r.y * half;
        let b = r.y.square();
        let c = r.z.square();
        let e = twist_b * (c.double() + c);
        let f = e.double() + e;
        let g = (b + f) * half;
        let h = (r.y + r.z).square() - (b + c);
        let i = e - b;
        let j = r.x.square();
        let e2 = e.square();
        r.x = a * (b - f);
        r.y = g.square() - (e2.double() + e2);
        r.z = b * h;
        (-h, j.double() + j, i)
    }

    /// The miller.rs mixed-addition step: advances `r` and returns the
    /// D-type line coefficients (lambda, -theta, j).
    fn add_step(r: &mut Hom, qx: Fp2, qy: Fp2) -> (Fp2, Fp2, Fp2) {
        let theta = r.y - qy * r.z;
        let lambda = r.x - qx * r.z;
        let c = theta.square();
        let d = lambda.square();
        let e = lambda * d;
        let f = r.z * c;
        let g = r.x * d;
        let h = e + f - g.double();
        r.x = lambda * h;
        r.y = theta * (g - h) - e * r.y;
        r.z *= e;
        let j = theta * qx - lambda * qy;
        (lambda, -theta, j)
    }

    /// The production shape: the exact (accumulator, coefficient) pairs
    /// ell() feeds the leaf, harvested by replaying the Miller recurrence
    /// (miller.rs line formulas over the public Fp2 API) with the leaf and
    /// the composed path in lockstep at every line update. The replay's
    /// final value is pinned against the production miller_loop, so the
    /// harvested sequence is provably the real one.
    #[test]
    fn fp12_034k_leaf_matches_composed_mid_miller() {
        use crate::consts::ATE_LOOP_COUNT;
        use crate::fp2_fast::f2_to;
        use crate::fr::Fr;
        use crate::g1::{G1Affine, G1Projective};
        use crate::g2::{G2Affine, G2Projective};
        use crate::pairing::miller::{mul_by_char, twist_b_f2};

        let half = Fp2::from(2u64).invert().unwrap();
        let twist_b = f2_to(twist_b_f2());
        let generator_p = G1Projective::generator();
        let generator_q = G2Projective::from(G2Affine::generator());
        for point_case in 0..2u64 {
            let p: G1Affine = generator_p
                .mul_scalar(Fr::from_u64(2 * point_case + 3))
                .to_affine();
            let q: G2Affine = generator_q
                .mul_scalar(Fr::from_u64(5 * point_case + 7))
                .to_affine();
            let mut r = Hom {
                x: q.x,
                y: q.y,
                z: Fp2::ONE,
            };
            let mut f = Fp12::ONE;
            let mut started = false;
            let mut step = 2000 * point_case as usize;
            let mut push_line = |f: &mut Fp12, coeffs: (Fp2, Fp2, Fp2), step: &mut usize| {
                let c0 = coeffs.0.mul_by_fp(p.y);
                let c3 = coeffs.1.mul_by_fp(p.x);
                let c4 = coeffs.2;
                if !started {
                    // First iteration: the accumulator IS the line value.
                    started = true;
                    *f = Fp12::new(
                        Fp6::new(c0, Fp2::ZERO, Fp2::ZERO),
                        Fp6::new(c3, c4, Fp2::ZERO),
                    );
                } else {
                    compare(*f, c0, c3, c4, *step);
                    f.mul_by_034_assign_sosd6(c0, c3, c4);
                }
                *step += 1;
            };
            for i in (1..ATE_LOOP_COUNT.len()).rev() {
                if i != ATE_LOOP_COUNT.len() - 1 {
                    f.square_in_place_sos();
                }
                let coeffs = dbl_step(&mut r, twist_b, half);
                push_line(&mut f, coeffs, &mut step);
                match ATE_LOOP_COUNT[i - 1] {
                    1 => {
                        let coeffs = add_step(&mut r, q.x, q.y);
                        push_line(&mut f, coeffs, &mut step);
                    }
                    -1 => {
                        let coeffs = add_step(&mut r, q.x, -q.y);
                        push_line(&mut f, coeffs, &mut step);
                    }
                    _ => {}
                }
            }
            let q1 = mul_by_char(q);
            let q2 = mul_by_char(q1).negate();
            for corrector in [q1, q2] {
                let coeffs = add_step(&mut r, corrector.x, corrector.y);
                push_line(&mut f, coeffs, &mut step);
            }
            assert_eq!(
                f,
                crate::pairing::miller_loop(&p, &q),
                "replayed recurrence diverged from the production Miller loop, case {point_case}",
            );
        }
    }

    #[test]
    #[ignore = "hundred-thousand-case release stress gate; run explicitly before changing field backends"]
    fn stress_random_cases_match_composed() {
        let mut rng = StdRng::seed_from_u64(0x034F);
        for case in 0..100_000 {
            compare(
                random_fp12(&mut rng),
                random_fp2(&mut rng),
                random_fp2(&mut rng),
                random_fp2(&mut rng),
                case,
            );
        }
    }
}

/// x86-64 ADX tier: the whole Fp4 square leaf must agree with the composed
/// sos4/sos2 path bit for bit, whatever HELIUS_FP4_SQR_ASM selects for
/// production dispatch -- the leaf wrapper is called directly. Random
/// operands, the edge palette rotated through every Fp slot, targeted xi
/// and doubling edges, and the cyclotomic-subgroup-shaped inputs pow_x
/// actually feeds (generated through the full final exponentiation).
#[cfg(all(helius_mont4_x86_64_adx, not(feature = "force-portable")))]
mod fp4_leaf_differential {
    use super::{edge_fps, random_fp2, random_fp12};
    use crate::fp::Fp;
    use crate::fp2::Fp2;
    use crate::fp12::Fp12;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn compare(r0: Fp2, r1: Fp2, case: usize) {
        assert_eq!(
            crate::fp::x86_64::fp4_sqr(&r0, &r1),
            Fp12::fp4_square_sos_composed(r0, r1),
            "fp4 leaf case {case}",
        );
    }

    #[test]
    fn fp4_leaf_matches_composed_on_random_operands() {
        let mut rng = StdRng::seed_from_u64(0xF4F4);
        for case in 0..super::N_BIG / 10 {
            compare(random_fp2(&mut rng), random_fp2(&mut rng), case);
        }
    }

    #[test]
    fn fp4_leaf_matches_composed_on_edge_palette() {
        let palette = edge_fps();
        let mut rng = StdRng::seed_from_u64(0xF4F5);
        for (i, &e) in palette.iter().enumerate() {
            let e2 = Fp2::new(e, e);
            compare(e2, e2, i);
            // Rotate the edge through every Fp slot of both operands.
            for slot in 0..4 {
                let mut r0 = random_fp2(&mut rng);
                let mut r1 = random_fp2(&mut rng);
                {
                    let target = if slot < 2 { &mut r0 } else { &mut r1 };
                    if slot % 2 == 0 {
                        target.c0 = e;
                    } else {
                        target.c1 = e;
                    }
                }
                compare(r0, r1, 100 * i + slot);
            }
        }
        // xi edges on r1 (the scaled operand): re = 0, im = p-1 maximizes
        // the (p - im) route. Im = 0 sends the negp rows to exactly p.
        // Doubling edges on r0: p-1 maximizes 2*r0.
        let p_minus_one = Fp::ZERO - Fp::ONE;
        for (case, edge) in [
            Fp2::new(Fp::ZERO, p_minus_one),
            Fp2::new(p_minus_one, Fp::ZERO),
            Fp2::new(Fp::ZERO, Fp::ZERO),
            Fp2::new(p_minus_one, p_minus_one),
        ]
        .into_iter()
        .enumerate()
        {
            compare(random_fp2(&mut rng), edge, 1000 + case);
            compare(edge, random_fp2(&mut rng), 2000 + case);
        }
    }

    /// The production shape: cyclotomic-subgroup elements (final-exp
    /// images), squared through the exact (r0, r1) pairings
    /// `cyclotomic_square` dispatches.
    #[test]
    fn fp4_leaf_matches_composed_on_cyclotomic_inputs() {
        let mut rng = StdRng::seed_from_u64(0xF4F6);
        for case in 0..4 {
            let f = crate::pairing::final_exponentiation(&random_fp12(&mut rng));
            for (pair, (r0, r1)) in [(f.c0.c0, f.c1.c1), (f.c1.c0, f.c0.c2), (f.c0.c1, f.c1.c2)]
                .into_iter()
                .enumerate()
            {
                compare(r0, r1, 10 * case + pair);
            }
        }
    }

    #[test]
    #[ignore = "hundred-thousand-case release stress gate; run explicitly before changing field backends"]
    fn stress_random_cases_match_composed() {
        let mut rng = StdRng::seed_from_u64(0xF4F7);
        for case in 0..100_000 {
            compare(random_fp2(&mut rng), random_fp2(&mut rng), case);
        }
    }
}

/// x86-64 ADX tier: the whole Fp12 square leaf (lazy double-width, 36
/// products) must agree with the composed 72-product SoS path bit for bit,
/// whatever HELIUS_FP12_SQR_ASM selects for production dispatch -- the leaf
/// wrapper is called directly. Random operands, the edge palette rotated
/// through every Fp slot, targeted xi edges on the scaled components,
/// line_value-shaped sparse accumulators, and actual Miller-loop
/// accumulators (the production input shape).
#[cfg(all(helius_mont4_x86_64_adx, not(feature = "force-portable")))]
mod fp12_sqr_leaf_differential {
    use super::{edge_fps, random_fp2, random_fp12};
    use crate::fp::Fp;
    use crate::fp2::Fp2;
    use crate::fp6::Fp6;
    use crate::fp12::Fp12;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn compare(f: Fp12, case: usize) {
        let mut leaf = f;
        crate::fp::x86_64::fp12_sqr_assign(&mut leaf);
        let mut composed = f;
        composed.square_in_place_sos();
        assert_eq!(leaf, composed, "fp12_sqr leaf case {case}");
    }

    #[test]
    fn fp12_sqr_leaf_matches_composed_on_random_operands() {
        let mut rng = StdRng::seed_from_u64(0x50A1);
        for case in 0..super::N_BIG / 10 {
            compare(random_fp12(&mut rng), case);
        }
    }

    #[test]
    fn fp12_sqr_leaf_matches_composed_on_edge_palette() {
        let palette = edge_fps();
        let mut rng = StdRng::seed_from_u64(0x50A2);
        for (i, &e) in palette.iter().enumerate() {
            let e2 = Fp2::new(e, e);
            let uniform = Fp12::new(Fp6::new(e2, e2, e2), Fp6::new(e2, e2, e2));
            compare(uniform, i);
            // Rotate the edge through every Fp slot of f.
            for slot in 0..12 {
                let mut f = random_fp12(&mut rng);
                {
                    let half = if slot < 6 { &mut f.c0 } else { &mut f.c1 };
                    let fp2 = match (slot / 2) % 3 {
                        0 => &mut half.c0,
                        1 => &mut half.c1,
                        _ => &mut half.c2,
                    };
                    if slot % 2 == 0 {
                        fp2.c0 = e;
                    } else {
                        fp2.c1 = e;
                    }
                }
                compare(f, 100 * i + slot);
            }
        }
        // xi edges on the scaled components b.c2 (the t1 site) and, via V,
        // both mulVadd sites: re = 0, im = p-1 maximizes the (p - im)
        // route. Im = 0 sends the negp row to exactly p.
        let p_minus_one = Fp::ZERO - Fp::ONE;
        for (case, edge) in [
            Fp2::new(Fp::ZERO, p_minus_one),
            Fp2::new(p_minus_one, Fp::ZERO),
            Fp2::new(Fp::ZERO, Fp::ZERO),
            Fp2::new(p_minus_one, p_minus_one),
        ]
        .into_iter()
        .enumerate()
        {
            let mut f = random_fp12(&mut rng);
            f.c1.c2 = edge;
            compare(f, 1000 + case);
        }
        // Line-value-shaped sparse accumulator (early Miller iterations)
        // and the ladder's initial value.
        let sparse = Fp12::new(
            Fp6::new(random_fp2(&mut rng), Fp2::ZERO, Fp2::ZERO),
            Fp6::new(random_fp2(&mut rng), random_fp2(&mut rng), Fp2::ZERO),
        );
        compare(sparse, 2000);
        compare(Fp12::ONE, 2001);
        compare(Fp12::ZERO, 2002);
    }

    /// The production shape: actual Miller-loop accumulators (the values
    /// square_in_place sees 63 times per loop), generated through the real
    /// pairing pipeline.
    #[test]
    fn fp12_sqr_leaf_matches_composed_on_miller_shaped_inputs() {
        use crate::fr::Fr;
        use crate::g1::{G1Affine, G1Projective};
        use crate::g2::{G2Affine, G2Projective};
        let generator_p = G1Projective::generator();
        let generator_q = G2Projective::from(G2Affine::generator());
        for case in 0..4u64 {
            let p: G1Affine = generator_p
                .mul_scalar(Fr::from_u64(2 * case + 3))
                .to_affine();
            let q: G2Affine = generator_q
                .mul_scalar(Fr::from_u64(5 * case + 7))
                .to_affine();
            let f = crate::pairing::miller_loop(&p, &q);
            compare(f, 10 + case as usize);
            compare(f.square(), 20 + case as usize);
        }
    }

    #[test]
    #[ignore = "hundred-thousand-case release stress gate; run explicitly before changing field backends"]
    fn stress_random_cases_match_composed() {
        let mut rng = StdRng::seed_from_u64(0x50A3);
        for case in 0..100_000 {
            compare(random_fp12(&mut rng), case);
        }
    }
}

#[cfg(all(helius_mont4_x86_64_adx, not(feature = "force-portable")))]
mod fp12_sqr_mcl_leaf_differential {
    use super::{edge_fps, random_fp12};
    use crate::fp2::Fp2;
    use crate::fp6::Fp6;
    use crate::fp12::Fp12;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn compare(f: Fp12, case: usize) {
        let mut compact = f;
        crate::fp::x86_64::fp12_sqr_mcl_assign(&mut compact);
        let mut composed = f;
        composed.square_in_place_sos();
        assert_eq!(compact, composed, "compact fp12 square case {case}");
    }

    #[test]
    fn fp12_sqr_mcl_leaf_differential() {
        let mut rng = StdRng::seed_from_u64(0x4d43_4c5f_5351_525f);
        for case in 0..200 {
            compare(random_fp12(&mut rng), case);
        }
        for (case, &edge) in edge_fps().iter().enumerate() {
            let e = Fp2::new(edge, edge);
            compare(Fp12::new(Fp6::new(e, e, e), Fp6::new(e, e, e)), 1000 + case);
        }
        compare(Fp12::ZERO, 2000);
        compare(Fp12::ONE, 2001);

        use crate::fr::Fr;
        use crate::g1::{G1Affine, G1Projective};
        use crate::g2::{G2Affine, G2Projective};
        let gp = G1Projective::generator();
        let gq = G2Projective::from(G2Affine::generator());
        for case in 0..4u64 {
            let p: G1Affine = gp.mul_scalar(Fr::from_u64(2 * case + 3)).to_affine();
            let q: G2Affine = gq.mul_scalar(Fr::from_u64(5 * case + 7)).to_affine();
            let raw = crate::pairing::miller_loop(&p, &q);
            compare(raw, 3000 + case as usize);
            let mut squared = raw;
            squared.square_in_place_sos();
            compare(squared, 4000 + case as usize);
        }
    }
}

/// x86-64 ADX tier: the whole Fp12 product leaf (lazy double-width, 54
/// products) must agree with the composed Fp6-Karatsuba path (108 products)
/// bit for bit, whatever HELIUS_FP12_MUL_ASM selects for production
/// dispatch -- the leaf wrapper is called directly in its z == a shape.
/// Random operands, the edge palette rotated through the slots of both
/// operands, targeted xi edges on the a1.c2/b1.c2 sites (the mulVadd BD.c
/// route), the a == b diagonal against the square, sparse line-shaped and
/// identity operands, and final-exp-shaped inputs (cyclotomic elements and
/// their ladder products, the values Fp12 multiplication sees 60 times per
/// final exponentiation).
#[cfg(all(helius_mont4_x86_64_adx, not(feature = "force-portable")))]
mod fp12_mul_leaf_differential {
    use super::{edge_fps, random_fp2, random_fp12};
    use crate::fp::Fp;
    use crate::fp2::Fp2;
    use crate::fp6::Fp6;
    use crate::fp12::Fp12;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn compare(a: Fp12, b: Fp12, case: usize) {
        let mut leaf = a;
        crate::fp::x86_64::fp12_mul_assign(&mut leaf, &b);
        let composed = a.mul_composed(b);
        assert_eq!(leaf, composed, "fp12_mul leaf case {case}");
    }

    #[test]
    fn fp12_mul_leaf_matches_composed_on_random_operands() {
        let mut rng = StdRng::seed_from_u64(0x50B1);
        for case in 0..super::N_BIG / 10 {
            compare(random_fp12(&mut rng), random_fp12(&mut rng), case);
        }
    }

    #[test]
    fn fp12_mul_leaf_matches_composed_on_edge_palette() {
        let palette = edge_fps();
        let mut rng = StdRng::seed_from_u64(0x50B2);
        for (i, &e) in palette.iter().enumerate() {
            let e2 = Fp2::new(e, e);
            let uniform = Fp12::new(Fp6::new(e2, e2, e2), Fp6::new(e2, e2, e2));
            compare(uniform, uniform, i);
            compare(uniform, random_fp12(&mut rng), 10 + i);
            // Rotate the edge through every Fp slot of either operand.
            for slot in 0..12 {
                let mut a = random_fp12(&mut rng);
                let mut b = random_fp12(&mut rng);
                {
                    let f = if slot % 2 == 0 { &mut a } else { &mut b };
                    let half = if slot < 6 { &mut f.c0 } else { &mut f.c1 };
                    let fp2 = match (slot / 2) % 3 {
                        0 => &mut half.c0,
                        1 => &mut half.c1,
                        _ => &mut half.c2,
                    };
                    if slot % 2 == 0 {
                        fp2.c0 = e;
                    } else {
                        fp2.c1 = e;
                    }
                }
                compare(a, b, 100 * i + slot);
            }
        }
        // xi edges on the a1.c2/b1.c2 sites: their product is BD.c, the
        // mulVadd xi route. Re = 0, im = p-1 maximizes the negation row,
        // im = 0 sends it to exactly p*2^256 - 0.
        let p_minus_one = Fp::ZERO - Fp::ONE;
        for (case, edge) in [
            Fp2::new(Fp::ZERO, p_minus_one),
            Fp2::new(p_minus_one, Fp::ZERO),
            Fp2::new(Fp::ZERO, Fp::ZERO),
            Fp2::new(p_minus_one, p_minus_one),
        ]
        .into_iter()
        .enumerate()
        {
            let mut a = random_fp12(&mut rng);
            let mut b = random_fp12(&mut rng);
            a.c1.c2 = edge;
            b.c1.c2 = edge;
            compare(a, b, 1000 + case);
        }
        // The a == b diagonal: the product must equal the square.
        let d = random_fp12(&mut rng);
        compare(d, d, 2000);
        // Sparse line-value shape times dense, and the identities.
        let sparse = Fp12::new(
            Fp6::new(random_fp2(&mut rng), Fp2::ZERO, Fp2::ZERO),
            Fp6::new(random_fp2(&mut rng), random_fp2(&mut rng), Fp2::ZERO),
        );
        compare(sparse, random_fp12(&mut rng), 2001);
        compare(Fp12::ONE, random_fp12(&mut rng), 2002);
        compare(random_fp12(&mut rng), Fp12::ONE, 2003);
        compare(Fp12::ZERO, random_fp12(&mut rng), 2004);
    }

    /// The production shape: cyclotomic elements and their ladder products,
    /// the operands MulAssign sees inside the final exponentiation's hard
    /// part, generated through the real pairing pipeline.
    #[test]
    fn fp12_mul_leaf_matches_composed_on_final_exp_shaped_inputs() {
        use crate::fr::Fr;
        use crate::g1::{G1Affine, G1Projective};
        use crate::g2::{G2Affine, G2Projective};
        let generator_p = G1Projective::generator();
        let generator_q = G2Projective::from(G2Affine::generator());
        for case in 0..3u64 {
            let p: G1Affine = generator_p
                .mul_scalar(Fr::from_u64(2 * case + 3))
                .to_affine();
            let q: G2Affine = generator_q
                .mul_scalar(Fr::from_u64(5 * case + 7))
                .to_affine();
            let f = crate::pairing::miller_loop(&p, &q);
            // Easy part: f^{(p^6-1)(p^2+1)}, a cyclotomic element.
            let r = f.conjugate() * f.invert().unwrap();
            let r = r.frobenius_map_squared() * r;
            compare(f, r, 10 + case as usize);
            compare(r, r.cyclotomic_square(), 20 + case as usize);
            compare(r.conjugate(), r, 30 + case as usize);
        }
    }

    #[test]
    #[ignore = "hundred-thousand-case release stress gate; run explicitly before changing field backends"]
    fn stress_random_cases_match_composed() {
        let mut rng = StdRng::seed_from_u64(0x50B3);
        for case in 0..100_000 {
            compare(random_fp12(&mut rng), random_fp12(&mut rng), case);
        }
    }
}

/// x86-64 ADX tier: the cyclotomic-square leaf (lazy double-width, 18
/// products) must agree with the composed 36-product path bit for bit,
/// whatever HELIUS_CYC_SQR_ASM selects for production dispatch -- the leaf
/// wrapper is called directly in its in-place (z == f) shape. The formula
/// identity leaf == composed holds on ARBITRARY canonical inputs (both
/// compute the same polynomial), so random and edge operands are compared
/// everywhere. The Granger-Scott square semantics (result == f^2) only hold
/// on the cyclotomic subgroup, so real final-exp images are additionally
/// squared through pow_x-shaped chains.
#[cfg(all(helius_mont4_x86_64_adx, not(feature = "force-portable")))]
mod cyc_sqr_leaf_differential {
    use super::{edge_fps, random_fp12};
    use crate::fp::Fp;
    use crate::fp2::Fp2;
    use crate::fp6::Fp6;
    use crate::fp12::Fp12;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn compare(f: Fp12, case: usize) {
        let mut leaf = f;
        crate::fp::x86_64::cyc_sqr_assign(&mut leaf);
        let composed = f.cyclotomic_square_composed();
        assert_eq!(leaf, composed, "cyc_sqr leaf case {case}");
    }

    #[test]
    fn cyc_sqr_leaf_matches_composed_on_random_operands() {
        let mut rng = StdRng::seed_from_u64(0xC5A1);
        for case in 0..super::N_BIG / 10 {
            compare(random_fp12(&mut rng), case);
        }
    }

    #[test]
    fn cyc_sqr_leaf_matches_composed_on_edge_palette() {
        let palette = edge_fps();
        let mut rng = StdRng::seed_from_u64(0xC5A2);
        for (i, &e) in palette.iter().enumerate() {
            let e2 = Fp2::new(e, e);
            let uniform = Fp12::new(Fp6::new(e2, e2, e2), Fp6::new(e2, e2, e2));
            compare(uniform, i);
            // Rotate the edge through every Fp slot of f.
            for slot in 0..12 {
                let mut f = random_fp12(&mut rng);
                {
                    let half = if slot < 6 { &mut f.c0 } else { &mut f.c1 };
                    let fp2 = match (slot / 2) % 3 {
                        0 => &mut half.c0,
                        1 => &mut half.c1,
                        _ => &mut half.c2,
                    };
                    if slot % 2 == 0 {
                        fp2.c0 = e;
                    } else {
                        fp2.c1 = e;
                    }
                }
                compare(f, 100 * i + slot);
            }
        }
        // xi edges on the x1 operands r1 = c1.c1, r3 = c0.c2, r5 = c1.c2
        // (their squares feed the nine-fold walk): re = 0, im = p-1
        // maximizes the negation rows. Im = 0 sends them to exactly
        // p*2^256 - 0.
        let p_minus_one = Fp::ZERO - Fp::ONE;
        for (case, edge) in [
            Fp2::new(Fp::ZERO, p_minus_one),
            Fp2::new(p_minus_one, Fp::ZERO),
            Fp2::new(Fp::ZERO, Fp::ZERO),
            Fp2::new(p_minus_one, p_minus_one),
        ]
        .into_iter()
        .enumerate()
        {
            for site in 0..3 {
                let mut f = random_fp12(&mut rng);
                match site {
                    0 => f.c1.c1 = edge,
                    1 => f.c0.c2 = edge,
                    _ => f.c1.c2 = edge,
                }
                compare(f, 1000 + 10 * case + site);
            }
        }
        // The identity (pow_x's frequent accumulator) and zero.
        compare(Fp12::ONE, 2000);
        compare(Fp12::ZERO, 2001);
    }

    /// The production shape: REAL cyclotomic-subgroup elements (full
    /// final-exp images) squared through pow_x-length dependent chains --
    /// the exact values the leaf sees 192 times per final exponentiation.
    /// Also checks the subgroup semantics: the leaf output equals the
    /// plain square there.
    #[test]
    fn cyc_sqr_leaf_matches_composed_on_cyclotomic_inputs() {
        let mut rng = StdRng::seed_from_u64(0xC5A3);
        for case in 0..3 {
            let f = crate::pairing::final_exponentiation(&random_fp12(&mut rng));
            let mut acc = f;
            for step in 0..8 {
                compare(acc, 100 * case + step);
                let mut leaf = acc;
                crate::fp::x86_64::cyc_sqr_assign(&mut leaf);
                assert_eq!(
                    leaf,
                    acc.square(),
                    "Granger-Scott square on the cyclotomic subgroup, case {case} step {step}",
                );
                acc = leaf;
            }
            compare(f.conjugate(), 1000 + case);
        }
    }

    #[test]
    #[ignore = "hundred-thousand-case release stress gate; run explicitly before changing field backends"]
    fn stress_random_cases_match_composed() {
        let mut rng = StdRng::seed_from_u64(0xC5A4);
        for case in 0..100_000 {
            compare(random_fp12(&mut rng), case);
        }
    }
}
