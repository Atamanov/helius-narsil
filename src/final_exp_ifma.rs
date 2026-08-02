//! Persistent AVX-512IFMA lane for the hard final exponentiation.
//!
//! Two vectors hold one Fp12 value. The low vector holds coefficients 0 to 3.
//! The high vector holds coefficients 4 and 5, followed by zero padding. The
//! hard part uses this layout without a scalar store.

use crate::fp::Fp;
use crate::fp::avx512ifma::{FpVec8, FpVec8L};
use crate::fp2::Fp2;
use crate::fp6::Fp6;
use crate::fp12::{Fp12, X_W4};

const REAL_LANES: u8 = 0x55;
/// Real-component lanes of the split half layout (lane `4e + 2c + h`).
const HALF_REAL_LANES: u8 = 0x33;

/// Tracked bound of a lazy `xi` product of canonical pairs.
const XI1: u64 = 11;
/// Row bound after the odd-row lazy negation of an `XI1` row.
const NEG: u64 = 13;

#[derive(Clone, Copy)]
struct PairSlot(u8);

#[derive(Clone, Copy)]
struct Term {
    left: PairSlot,
    right: PairSlot,
    right_xi: bool,
}

/// Six Fp2 products: all schoolbook terms of one dense output coefficient.
type Equation = [Term; 6];

#[derive(Clone, Copy)]
struct Plan12 {
    left: [[u64; 8]; 12],
    right: [[u64; 8]; 12],
    right_xi: [u8; 12],
}

/// Half rows for the short SoS stream: output lane `4e + 2c + h` holds Fp
/// terms `6h..6h + 6` of component `c` of equation `e`, so all 48 products
/// of two dense output coefficients fill eight six-term lanes exactly.
#[derive(Clone, Copy)]
struct HalfPlan6 {
    left: [[u64; 8]; 6],
    right: [[u64; 8]; 6],
    right_xi: [u8; 6],
}

const fn term(left: PairSlot, right: PairSlot) -> Term {
    Term {
        left,
        right,
        right_xi: false,
    }
}

const fn xi_term(left: PairSlot, right: PairSlot) -> Term {
    Term {
        left,
        right,
        right_xi: true,
    }
}

/// Lower six Fp2 products per equation to twelve base-field product rows.
const fn lower(equations: &[Equation; 4]) -> Plan12 {
    let mut left = [[0; 8]; 12];
    let mut right = [[0; 8]; 12];
    let mut right_xi = [0; 12];
    let mut output = 0;
    while output < 8 {
        let component = output & 1;
        let mut product = 0;
        while product < 6 {
            let row = 2 * product;
            let value = equations[output / 2][product];
            assert!(value.left.0 < 8 && value.right.0 < 8);
            left[row][output] = (2 * value.left.0) as u64;
            left[row + 1][output] = (2 * value.left.0 + 1) as u64;
            right[row][output] = (2 * value.right.0 + component as u8) as u64;
            right[row + 1][output] = (2 * value.right.0 + (component as u8 ^ 1)) as u64;
            if value.right_xi {
                right_xi[row] |= 1 << output;
                right_xi[row + 1] |= 1 << output;
            }
            product += 1;
        }
        output += 1;
    }
    Plan12 {
        left,
        right,
        right_xi,
    }
}

/// Lower two dense output coefficients to split half rows. Row parity equals
/// Fp-term parity, so the shared odd-row negation applies under
/// [`HALF_REAL_LANES`].
const fn lower_half(equations: &[Equation; 2]) -> HalfPlan6 {
    let mut left = [[0u64; 8]; 6];
    let mut right = [[0u64; 8]; 6];
    let mut right_xi = [0u8; 6];
    let mut lane = 0;
    while lane < 8 {
        let (e, c, h) = (lane / 4, (lane / 2) % 2, lane % 2);
        let mut row = 0;
        while row < 6 {
            let value = equations[e][3 * h + row / 2];
            assert!(value.left.0 < 8 && value.right.0 < 8);
            let odd = (row % 2) as u8;
            left[row][lane] = (2 * value.left.0 + odd) as u64;
            right[row][lane] = (2 * value.right.0 + (odd ^ c as u8)) as u64;
            if value.right_xi {
                right_xi[row] |= 1 << lane;
            }
            row += 1;
        }
        lane += 1;
    }
    HalfPlan6 {
        left,
        right,
        right_xi,
    }
}

const A0: PairSlot = PairSlot(0);
const A1: PairSlot = PairSlot(1);
const A2: PairSlot = PairSlot(2);
const B0: PairSlot = PairSlot(3);
const B1: PairSlot = PairSlot(4);
const B2: PairSlot = PairSlot(5);

/// All twelve schoolbook Fp2 products per dense output coefficient, split as
/// the four full-row outputs (`c0` triple and `c1.c0`) and the two half-row
/// outputs (`c1.c1` and `c1.c2`).
const fn dense_full() -> Plan12 {
    lower(&[
        [
            term(A0, A0),
            xi_term(A1, A2),
            xi_term(A2, A1),
            xi_term(B0, B2),
            xi_term(B1, B1),
            xi_term(B2, B0),
        ],
        [
            term(A0, A1),
            term(A1, A0),
            xi_term(A2, A2),
            term(B0, B0),
            xi_term(B1, B2),
            xi_term(B2, B1),
        ],
        [
            term(A0, A2),
            term(A1, A1),
            term(A2, A0),
            term(B0, B1),
            term(B1, B0),
            xi_term(B2, B2),
        ],
        [
            term(A0, B0),
            xi_term(A1, B2),
            xi_term(A2, B1),
            term(B0, A0),
            xi_term(B1, A2),
            xi_term(B2, A1),
        ],
    ])
}

const fn dense_half() -> HalfPlan6 {
    lower_half(&[
        [
            term(A0, B1),
            term(A1, B0),
            xi_term(A2, B2),
            term(B0, A1),
            term(B1, A0),
            xi_term(B2, A2),
        ],
        [
            term(A0, B2),
            term(A1, B1),
            term(A2, B0),
            term(B0, A2),
            term(B1, A1),
            term(B2, A0),
        ],
    ])
}

const DENSE_FULL: Plan12 = dense_full();
const DENSE_HALF: HalfPlan6 = dense_half();

#[derive(Clone, Copy)]
struct Sources<'a, const B: u64> {
    low: &'a FpVec8L<B>,
    high: &'a FpVec8L<B>,
}

impl<'a, const B: u64> Sources<'a, B> {
    const fn new(low: &'a FpVec8L<B>, high: &'a FpVec8L<B>) -> Self {
        Self { low, high }
    }
}

/// The final four high lanes stay zero. Static plans use them as padding.
#[derive(Clone, Copy)]
struct PackedFp12 {
    low: FpVec8,
    high: FpVec8,
}

impl PackedFp12 {
    #[inline]
    unsafe fn load(value: Fp12) -> Self {
        unsafe {
            Self::load_coefficients([
                value.c0.c0,
                value.c0.c1,
                value.c0.c2,
                value.c1.c0,
                value.c1.c1,
                value.c1.c2,
            ])
        }
    }

    #[inline]
    unsafe fn load_coefficients(c: [Fp2; 6]) -> Self {
        Self {
            low: unsafe {
                FpVec8::load(&[
                    c[0].c0, c[0].c1, c[1].c0, c[1].c1, c[2].c0, c[2].c1, c[3].c0, c[3].c1,
                ])
            },
            high: unsafe {
                FpVec8::load(&[
                    c[4].c0,
                    c[4].c1,
                    c[5].c0,
                    c[5].c1,
                    Fp::ZERO,
                    Fp::ZERO,
                    Fp::ZERO,
                    Fp::ZERO,
                ])
            },
        }
    }

    #[inline]
    unsafe fn load_real_coefficients(c: [Fp; 6]) -> Self {
        Self {
            low: unsafe { FpVec8::load(&[c[0], c[0], c[1], c[1], c[2], c[2], c[3], c[3]]) },
            high: unsafe {
                FpVec8::load(&[
                    c[4],
                    c[4],
                    c[5],
                    c[5],
                    Fp::ZERO,
                    Fp::ZERO,
                    Fp::ZERO,
                    Fp::ZERO,
                ])
            },
        }
    }

    #[inline]
    fn store(self) -> Fp12 {
        let low = self.low.store();
        let high = self.high.store();
        Fp12::new(
            Fp6::new(
                Fp2::new(low[0], low[1]),
                Fp2::new(low[2], low[3]),
                Fp2::new(low[4], low[5]),
            ),
            Fp6::new(
                Fp2::new(low[6], low[7]),
                Fp2::new(high[0], high[1]),
                Fp2::new(high[2], high[3]),
            ),
        )
    }

    #[inline(always)]
    fn conjugate(self) -> Self {
        Self {
            low: self.low.negate_lanes(0xc0),
            high: self.high.negate_lanes(0x0f),
        }
    }

    #[inline(always)]
    fn conjugate_fp2(self) -> Self {
        Self {
            low: self.low.negate_lanes(0xaa),
            high: self.high.negate_lanes(0x0a),
        }
    }

    #[inline(never)]
    fn mul(self, rhs: Self) -> Self {
        let self_low = self.low.lazy();
        let self_high = self.high.lazy();
        let rhs_low = rhs.low.lazy();
        let rhs_high = rhs.high.lazy();
        let left = Sources::new(&self_low, &self_high);
        let right = Sources::new(&rhs_low, &rhs_high);
        let right_xi_low = rhs_low.xi_lazy::<XI1>();
        let right_xi_high = rhs_high.xi_lazy::<XI1>();
        let right_xi = Sources::new(&right_xi_low, &right_xi_high);
        // SAFETY: `PackedFp12` values exist only below an IFMA-authorized entry.
        let [low, halves] = unsafe { execute_dense_pair(left, right, right_xi) };
        // Adjacent canonical halves sum below `2p`, and the cleared high
        // lanes keep the padding invariant of `high`.
        let even = halves.permute2(&halves, [0, 2, 4, 6, 0, 0, 0, 0]);
        let odd = halves.permute2(&halves, [1, 3, 5, 7, 0, 0, 0, 0]);
        let high = even.add(&odd).keep_lanes(0x0f);
        Self { low, high }
    }

    /// Apply three Granger-Scott Fp4 squares in three packed product waves.
    #[inline(never)]
    fn cyclotomic_square(self) -> Self {
        // The active pairs are r0+r1, r2+r3, and r4+r5.
        let sum_left = self.low.permute2(&self.high, [0, 1, 6, 7, 2, 3, 12, 12]);
        let sum_right = self.low.permute2(&self.high, [8, 9, 4, 5, 10, 11, 12, 12]);
        let sum = sum_left.lazy().add_lazy::<1, 2>(&sum_right.lazy());
        let [square_low, square_high, square_sum] = fp2_square_pairs_three(
            &self.low.lazy().relax::<2>(),
            &self.high.lazy().relax::<2>(),
            &sum,
        );

        let square_a = square_low.permute2(&square_high, [0, 1, 6, 7, 2, 3, 12, 12]);
        let square_b = square_low.permute2(&square_high, [8, 9, 4, 5, 10, 11, 12, 12]);
        let even = square_a
            .lazy()
            .add_lazy::<XI1, 12>(&square_b.lazy().xi_lazy::<XI1>());
        let odd = square_sum
            .lazy()
            .sub2_lazy::<1, 1, 4>(&square_a.lazy(), &square_b.lazy())
            .reduce_full();

        // Output order is z0, z4, z3, z2, z1, z5.
        let xi_odd = odd.lazy().xi_lazy::<XI1>();
        let t_low = even.permute2::<XI1, 12>(&xi_odd, [0, 1, 2, 3, 4, 5, 12, 13]);
        let low = t_low
            .triple_pm_double::<1, 39>(&self.low.lazy(), 0xc0)
            .reduce_full();

        let t_high = odd.permute2(&self.high, [0, 1, 2, 3, 12, 12, 12, 12]);
        let high = t_high
            .lazy()
            .triple_pm_double::<1, 6>(&self.high.lazy(), 0xff)
            .reduce_full();
        Self { low, high }
    }

    #[inline(always)]
    fn mul_fp2_coefficients(self, rhs: Self) -> Self {
        Self {
            low: mul_fp2_pairs(&self.low, &rhs.low),
            high: mul_fp2_pairs(&self.high, &rhs.high),
        }
    }

    #[inline(never)]
    fn frobenius_p(self) -> Self {
        // Coefficients follow 1, v, v^2, w, wv, wv^2.
        let constants = unsafe {
            Self::load_coefficients([
                Fp2::ONE,
                crate::fp12::gamma1(2),
                crate::fp12::gamma1(4),
                crate::fp12::gamma1(1),
                crate::fp12::gamma1(3),
                crate::fp12::gamma1(5),
            ])
        };
        self.conjugate_fp2().mul_fp2_coefficients(constants)
    }

    #[inline(never)]
    fn frobenius_p2(self) -> Self {
        let constants = unsafe {
            Self::load_real_coefficients([
                Fp::ONE,
                crate::fp12::gamma2(2).c0,
                crate::fp12::gamma2(4).c0,
                crate::fp12::gamma2(1).c0,
                crate::fp12::gamma2(3).c0,
                crate::fp12::gamma2(5).c0,
            ])
        };
        Self {
            low: self.low.mul(&constants.low),
            high: self.high.mul(&constants.high),
        }
    }

    #[inline(never)]
    fn frobenius_p3(self) -> Self {
        let constants = unsafe {
            Self::load_coefficients([
                Fp2::ONE,
                gamma3(2),
                gamma3(4),
                gamma3(1),
                gamma3(3),
                gamma3(5),
            ])
        };
        self.conjugate_fp2().mul_fp2_coefficients(constants)
    }

    #[inline]
    fn pow_x(self) -> Self {
        let x2 = self.cyclotomic_square();
        let x3 = x2.mul(self);
        let x5 = x3.mul(x2);
        let x7 = x5.mul(x2);
        let table = [self, x3, x5, x7];
        let mut acc = table[0];
        for &digit in X_W4.iter().rev().skip(1) {
            acc = acc.cyclotomic_square();
            if digit != 0 {
                let value = table[(digit.unsigned_abs() / 2) as usize];
                acc = acc.mul(if digit < 0 { value.conjugate() } else { value });
            }
        }
        acc
    }

    #[inline(always)]
    fn exp_by_neg_x(self) -> Self {
        self.pow_x().conjugate()
    }
}

/// One right row: the xi-mask select, then the lazy odd-row negation whose
/// sign convention the real-lane mask fixes.
#[inline(always)]
fn right_row(
    right: Sources<'_, 1>,
    right_xi: Sources<'_, XI1>,
    indices: [u64; 8],
    xi_mask: u8,
    odd: bool,
    real_lanes: u8,
) -> FpVec8L<NEG> {
    let regular = right.low.permute2::<1, XI1>(right.high, indices);
    let twisted = right_xi.low.permute2::<XI1, XI1>(right_xi.high, indices);
    let selected = FpVec8L::<XI1>::select(xi_mask, &twisted, &regular);
    if odd {
        selected.negate_lanes_lazy::<NEG>(real_lanes)
    } else {
        selected.relax::<NEG>()
    }
}

#[inline(always)]
fn plan_operands(
    left: Sources<'_, 1>,
    right: Sources<'_, 1>,
    right_xi: Sources<'_, XI1>,
    plan: &Plan12,
) -> ([FpVec8L<1>; 12], [FpVec8L<NEG>; 12]) {
    let a: [FpVec8L<1>; 12] =
        core::array::from_fn(|row| left.low.permute2::<1, 1>(left.high, plan.left[row]));
    let b: [FpVec8L<NEG>; 12] = core::array::from_fn(|row| {
        right_row(
            right,
            right_xi,
            plan.right[row],
            plan.right_xi[row],
            row & 1 == 1,
            REAL_LANES,
        )
    });
    (a, b)
}

#[inline(always)]
fn plan_operands_half(
    left: Sources<'_, 1>,
    right: Sources<'_, 1>,
    right_xi: Sources<'_, XI1>,
    plan: &HalfPlan6,
) -> ([FpVec8L<1>; 6], [FpVec8L<NEG>; 6]) {
    let a: [FpVec8L<1>; 6] =
        core::array::from_fn(|row| left.low.permute2::<1, 1>(left.high, plan.left[row]));
    let b: [FpVec8L<NEG>; 6] = core::array::from_fn(|row| {
        right_row(
            right,
            right_xi,
            plan.right[row],
            plan.right_xi[row],
            row & 1 == 1,
            HALF_REAL_LANES,
        )
    });
    (a, b)
}

/// Run the full-row and the half-row reductions of one dense product as two
/// interleaved streams in one body, so no operand row crosses an
/// outlined-call boundary through the stack.
///
/// # Safety
///
/// The CPU and operating system must support AVX-512F and AVX-512IFMA.
#[target_feature(enable = "avx512f,avx512ifma")]
#[inline(never)]
unsafe fn execute_dense_pair(
    left: Sources<'_, 1>,
    right: Sources<'_, 1>,
    right_xi: Sources<'_, XI1>,
) -> [FpVec8; 2] {
    let (a0, b0) = plan_operands(left, right, right_xi, &DENSE_FULL);
    let (a1, b1) = plan_operands_half(left, right, right_xi, &DENSE_HALF);
    // SAFETY: the enclosing target features satisfy the ISA precondition.
    unsafe { FpVec8::sos_mac_12_6_pair_fused_lazy::<1, NEG, 2, 1>(&a0, &b0, &a1, &b1) }
}

/// Karatsuba-shaped Fp2 square operands: `(2r, i)` on odd lanes and
/// `(r - i, r + i)` on even lanes.
#[inline(always)]
fn fp2_square_operands(value: &FpVec8L<2>) -> (FpVec8L<5>, FpVec8L<4>) {
    let real = value.permute2::<2, 2>(value, [0, 0, 2, 2, 4, 4, 6, 6]);
    let imag = value.permute2::<2, 2>(value, [1, 1, 3, 3, 5, 5, 7, 7]);
    let left = FpVec8L::<5>::select(
        0xaa,
        &real.double_lazy::<4>().relax::<5>(),
        &real.sub_lazy::<2, 5>(&imag),
    );
    let right = FpVec8L::<4>::select(0xaa, &imag.relax::<4>(), &real.add_lazy::<2, 4>(&imag));
    (left, right)
}

#[inline(never)]
fn fp2_square_pairs_three(
    first: &FpVec8L<2>,
    second: &FpVec8L<2>,
    third: &FpVec8L<2>,
) -> [FpVec8; 3] {
    let (a0, b0) = fp2_square_operands(first);
    let (a1, b1) = fp2_square_operands(second);
    let (a2, b2) = fp2_square_operands(third);
    FpVec8::mul_three_lazy::<5, 4>(&a0, &b0, &a1, &b1, &a2, &b2)
}

/// Multiply four adjacent Fp2 pairs in one two-term SoS reduction.
#[inline(never)]
fn mul_fp2_pairs(left: &FpVec8, right: &FpVec8) -> FpVec8 {
    let real = left.permute2(left, [0, 0, 2, 2, 4, 4, 6, 6]).lazy();
    let imag = left.permute2(left, [1, 1, 3, 3, 5, 5, 7, 7]).lazy();
    let cross = right.permute2(right, [1, 0, 3, 2, 5, 4, 7, 6]).lazy();
    let signed_cross = cross.negate_lanes_lazy::<3>(REAL_LANES);
    FpVec8::sos_mac_2_lazy::<1, 3>(&[real, imag], &[right.lazy().relax::<3>(), signed_cross])
}

#[inline(always)]
const fn fp2_mont(c0: [u64; 4], c1: [u64; 4]) -> Fp2 {
    Fp2::new(Fp::from_raw_unchecked(c0), Fp::from_raw_unchecked(c1))
}

/// gamma3 is gamma1 times gamma2. The constants are in Montgomery form.
#[inline(always)]
const fn gamma3(j: usize) -> Fp2 {
    match j {
        1 => fp2_mont(
            [
                3914496794763385213,
                790120733010914719,
                7322192392869644725,
                581366264293887267,
            ],
            [
                12817045492518885689,
                4440270538777280383,
                11178533038884588256,
                2767537931541304486,
            ],
        ),
        2 => fp2_mont(
            [
                14532872967180610477,
                12903226530429559474,
                1868623743233345524,
                2316889217940299650,
            ],
            [
                12447993766991532972,
                4121872836076202828,
                7630813605053367399,
                740282956577754197,
            ],
        ),
        3 => fp2_mont(
            [
                6297350639395948318,
                15875321927225446337,
                9702569988553770230,
                805825149519570764,
            ],
            [
                11117433864585119104,
                10363184613815941297,
                5420513773305887730,
                278429812070195549,
            ],
        ),
        4 => fp2_mont(
            [
                4938922280314430175,
                13823286637238282975,
                15589480384090068090,
                481952561930628184,
            ],
            [
                3105754162722846417,
                11647802298615474591,
                13057042392041828081,
                1660844386505564338,
            ],
        ),
        5 => fp2_mont(
            [
                16193900971494954399,
                13995139551301264911,
                9239559758168096094,
                1571199014989505406,
            ],
            [
                3254114329011132839,
                11171599147282597747,
                10965492220518093659,
                2657556514797346915,
            ],
        ),
        _ => Fp2::ONE,
    }
}

/// Run the hard part after runtime IFMA authorization.
pub(crate) fn hard_part(_ifma: crate::x86_runtime::Avx512Ifma, value: Fp12) -> Fp12 {
    // SAFETY: `_ifma` proves CPU and operating-system support.
    unsafe { hard_part_ifma(value) }
}

#[target_feature(enable = "avx512f,avx512ifma")]
#[inline(never)]
unsafe fn hard_part_ifma(value: Fp12) -> Fp12 {
    // This is the only live-value entry into the radix-52 domain.
    let r = unsafe { PackedFp12::load(value) };
    let y0 = r.exp_by_neg_x();
    let y1 = y0.cyclotomic_square();
    let y2 = y1.cyclotomic_square();
    let y3 = y2.mul(y1);
    let y4 = y3.exp_by_neg_x();
    let y5 = y4.cyclotomic_square();
    let y6 = y5.exp_by_neg_x();
    let y7 = y6.conjugate().mul(y4);
    let y8 = y7.mul(y3.conjugate());
    let y9 = y8.mul(y1);
    let y10 = y8.mul(y4);
    let y11 = y10.mul(r);
    let y12 = y9.frobenius_p();
    let y13 = y12.mul(y11);
    let y14 = y8.frobenius_p2().mul(y13);
    let y15 = r.conjugate().mul(y9).frobenius_p3();
    // This is the only live-value exit from the radix-52 domain.
    y15.mul(y14).store()
}

#[cfg(all(test, helius_avx512_ifma))]
mod tests {
    use super::*;

    fn sample(seed: u64) -> Fp12 {
        let mut x = seed;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            Fp::from_u64(x)
        };
        Fp12::new(
            Fp6::new(
                Fp2::new(next(), next()),
                Fp2::new(next(), next()),
                Fp2::new(next(), next()),
            ),
            Fp6::new(
                Fp2::new(next(), next()),
                Fp2::new(next(), next()),
                Fp2::new(next(), next()),
            ),
        )
    }

    fn easy(value: Fp12) -> Fp12 {
        let t = value.conjugate() * value.invert().unwrap();
        t.frobenius_map_squared() * t
    }

    #[test]
    fn dense_square_and_frobenius_leaves_match_scalar() {
        for index in 1..=5 {
            assert_eq!(
                gamma3(index),
                crate::fp12::gamma1(index) * crate::fp12::gamma2(index)
            );
        }
        for index in 1..=128 {
            let a = sample(index);
            let b = sample(index ^ 0x9e37_79b9);
            assert_eq!(unsafe { PackedFp12::load(a) }.store(), a);
            assert_eq!(
                unsafe { PackedFp12::load(a) }
                    .mul(unsafe { PackedFp12::load(b) })
                    .store(),
                a * b
            );
            let unitary = easy(a);
            assert_eq!(
                unsafe { PackedFp12::load(unitary) }
                    .cyclotomic_square()
                    .store(),
                unitary.cyclotomic_square()
            );
            let packed = unsafe { PackedFp12::load(a) };
            assert_eq!(packed.frobenius_p().store(), a.frobenius_map());
            assert_eq!(packed.frobenius_p2().store(), a.frobenius_map_squared());
            assert_eq!(packed.frobenius_p3().store(), a.frobenius_map_cubed());
        }
    }

    #[test]
    fn resident_hard_part_matches_scalar_exactly() {
        for index in 1..=32 {
            let value = easy(sample(index));
            assert_eq!(
                unsafe { hard_part_ifma(value) },
                crate::pairing::final_exp::hard_part_scalar(value)
            );
        }
    }

    #[test]
    fn raw_final_exponentiation_matches_the_exact_scalar_output() {
        assert_eq!(
            crate::pairing::final_exponentiation(&Fp12::ZERO),
            Fp12::ZERO
        );
        assert_eq!(unsafe { hard_part_ifma(Fp12::ONE) }, Fp12::ONE);
        for index in 1..=128 {
            let raw = sample(index ^ 0xd1ff_1200);
            let value = easy(raw);
            assert_eq!(
                unsafe { hard_part_ifma(value) },
                crate::pairing::final_exp::hard_part_scalar(value)
            );
        }
    }
}
