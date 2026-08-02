//! Output-coefficient AVX-512IFMA backend for one Fp12 Miller accumulator.
//!
//! Lanes hold the base field coefficients of one Fp12 value. Each square and
//! each sparse product runs one interleaved two-stream pair of eight-lane
//! Montgomery reductions: eight full outputs at six terms and four split
//! outputs at three terms, so the 72 schoolbook base-field products of one
//! operation fill all 72 issued product slots.
//! Inputs must be public.

use alloc::vec::Vec;
use core::mem::MaybeUninit;

use crate::fp::Fp;
use crate::fp::avx512ifma::{FpVec8, FpVec8L};
use crate::fp2::Fp2;
use crate::fp6::Fp6;
use crate::fp12::Fp12;
use crate::g1::G1Affine;
use crate::g2::G2Affine;
use crate::pairing::miller::{
    ActiveMillerLineSink, EllCoeff, G2Miller, G2Prepared, MillerLineSink, MillerPump,
    stream_line_coeffs, stream_mixed_line_coeffs, stream_prepared_line_coeffs,
};

const ZERO_STATE: u64 = 12;
const REAL_LANES: u8 = 0x55;

#[derive(Clone, Copy)]
struct PairSlot(u8);

#[derive(Clone, Copy)]
struct Term(PairSlot, PairSlot);

type Equation = [Term; 3];

#[derive(Clone, Copy)]
struct Plan {
    left: [[u64; 8]; 6],
    right: [[u64; 8]; 6],
}

/// Half rows for the short SoS stream: output lane `4e + 2c + h` holds
/// schoolbook Fp terms `3h..3h + 3` of component `c` of equation `e`, so all
/// 24 products of two Fp2 equations fill eight three-term lanes exactly.
#[derive(Clone, Copy)]
struct HalfPlan {
    left: [[u64; 8]; 3],
    right: [[u64; 8]; 3],
    negate: [u8; 3],
}

struct PackedPlans {
    full: Plan,
    half: HalfPlan,
}

/// Lower four Fp2 equations to the six schoolbook-product vectors consumed
/// by the six-term SoS stream.
const fn lower(equations: [Equation; 4]) -> Plan {
    let (mut left, mut right) = ([[0; 8]; 6], [[0; 8]; 6]);
    let mut output = 0;
    while output < 8 {
        let component = output & 1;
        let mut product = 0;
        while product < 3 {
            let odd = product * 2 + 1;
            let Term(PairSlot(x), PairSlot(y)) = equations[output / 2][product];
            assert!(x < 8 && y < 8);
            left[odd - 1][output] = (2 * x) as u64;
            left[odd][output] = (2 * x + 1) as u64;
            right[odd - 1][output] = (2 * y + component as u8) as u64;
            right[odd][output] = (2 * y + (component ^ 1) as u8) as u64;
            product += 1;
        }
        output += 1;
    }
    Plan { left, right }
}

/// Lower two Fp2 equations to split half rows. The negate mask marks the
/// minus-signed schoolbook terms (the odd Fp term of a real component), so
/// the sign convention matches [`lower`] rows under the `REAL_LANES` mask.
const fn lower_half(equations: [Equation; 2]) -> HalfPlan {
    let mut left = [[0u64; 8]; 3];
    let mut right = [[0u64; 8]; 3];
    let mut negate = [0u8; 3];
    let mut lane = 0;
    while lane < 8 {
        let (e, c, h) = (lane / 4, (lane / 2) % 2, lane % 2);
        let mut row = 0;
        while row < 3 {
            let term = 3 * h + row;
            let Term(PairSlot(x), PairSlot(y)) = equations[e][term / 2];
            assert!(x < 8 && y < 8);
            let odd = (term % 2) as u8;
            left[row][lane] = (2 * x + odd) as u64;
            right[row][lane] = (2 * y + (odd ^ c as u8)) as u64;
            if odd == 1 && c == 0 {
                negate[row] |= 1 << lane;
            }
            row += 1;
        }
        lane += 1;
    }
    HalfPlan {
        left,
        right,
        negate,
    }
}

/// Express one Fp2 coefficient as three Fp2 products.
macro_rules! sum3 {
    ($a:ident * $x:ident + $b:ident * $y:ident + $c:ident * $z:ident) => {
        [Term($a, $x), Term($b, $y), Term($c, $z)]
    };
}

macro_rules! slots {
    ($first:literal => $c0:ident, $c1:ident, $c2:ident) => {
        let ($c0, $c1, $c2) = (PairSlot($first), PairSlot($first + 1), PairSlot($first + 2));
    };
}

macro_rules! packed_plans {
    (full [$($full:expr),+ $(,)?]; half [$($half:expr),+ $(,)?]) => {
        PackedPlans {
            full: lower([$($full),+]),
            half: lower_half([$($half),+]),
        }
    };
}

/// Both permute sources of one plan half share the same tracked bound.
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

/// Tracked bound of a lazy `xi` product of canonical pairs.
const XI1: u64 = 11;
/// Bound of the doubling-stage right rows: `xi` of sums below `2p`.
const SQ_RIGHT: u64 = 21;
/// Row bounds after the odd-row lazy negation.
const SPARSE_NEG: u64 = 13;
const SQ_NEG: u64 = 23;

/// The unused high lanes must remain zero because row plans use them as padding.
#[derive(Clone, Copy)]
struct PackedFp12 {
    lo: FpVec8,
    hi: FpVec8,
}

/// The last two lanes must remain zero because row plans use them as padding.
#[derive(Clone, Copy)]
struct PackedLine {
    base: FpVec8,
}

impl PackedFp12 {
    #[cfg(all(test, helius_avx512_ifma))]
    #[inline]
    fn load(value: &Fp12) -> Self {
        let c = coefficients(value);
        Self {
            lo: unsafe { FpVec8::load(&[c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) },
            hi: unsafe {
                FpVec8::load(&[
                    c[8],
                    c[9],
                    c[10],
                    c[11],
                    Fp::ZERO,
                    Fp::ZERO,
                    Fp::ZERO,
                    Fp::ZERO,
                ])
            },
        }
    }

    #[inline]
    fn store(&self) -> Fp12 {
        let lo = self.lo.store();
        let hi = self.hi.store();
        from_coefficients([
            lo[0], lo[1], lo[2], lo[3], lo[4], lo[5], lo[6], lo[7], hi[0], hi[1], hi[2], hi[3],
        ])
    }

    #[inline]
    fn mul_by_034_assign(&mut self, line: &PackedLine) {
        let xi = line.xi();
        let lo = self.lo.lazy();
        let hi = self.hi.lazy();
        let base = line.base.lazy().relax::<XI1>();
        let state = Sources::new(&lo, &hi);
        let coefficients = Sources::new(&base, &xi);
        // SAFETY: `PackedFp12` values exist only below an IFMA-authorized entry.
        let [lo, halves] = unsafe {
            execute_row_pair_63::<1, XI1, SPARSE_NEG, 1, 1>(
                state,
                coefficients,
                state,
                coefficients,
                &SPARSE,
            )
        };
        // Adjacent canonical halves sum below `2p`, and the cleared high
        // lanes keep the padding invariant of `hi`.
        let even = halves.permute2(&halves, [0, 2, 4, 6, 0, 0, 0, 0]);
        let odd = halves.permute2(&halves, [1, 3, 5, 7, 0, 0, 0, 0]);
        let hi = even.add(&odd).keep_lanes(0x0f);
        self.lo.copy_from(&lo);
        self.hi.copy_from(&hi);
    }

    #[inline]
    fn from_line(line: &PackedLine) -> Self {
        Self {
            lo: line.base.permute2(&line.base, [0, 1, 6, 6, 6, 6, 2, 3]),
            hi: line.base.permute2(&line.base, [4, 5, 6, 6, 6, 6, 6, 6]),
        }
    }

    /// Square by `ab = a*b`, `s = a+b`, and `t = a+v*b`.
    #[inline]
    fn square_assign(&mut self) {
        let a = self
            .lo
            .permute2(&self.hi, [0, 1, 2, 3, 4, 5, ZERO_STATE, ZERO_STATE]);
        let b = self
            .lo
            .permute2(&self.hi, [6, 7, 8, 9, 10, 11, ZERO_STATE, ZERO_STATE]);
        let s = a.add(&b);
        let xi_b = b.lazy().xi_lazy::<XI1>();

        // v*b = (xi*b2, b0, b1)
        let vb = b
            .lazy()
            .relax::<XI1>()
            .permute2::<XI1, XI1>(&xi_b, [12, 13, 0, 1, 2, 3, 6, 7]);
        let t = a.lazy().add_lazy::<XI1, 12>(&vb);
        // Lanes 2..=5 of t hold sums of two canonical values, the rest are
        // zeroed here, so the bound 2 holds per lane.
        let t_small = unsafe { t.keep_lanes(0x3c).retag::<2>() };
        let xi_t = t_small.xi_lazy::<SQ_RIGHT>();

        let left0 = self.lo.permute2(&s, [0, 1, 2, 3, 4, 5, 8, 9]).lazy();
        let left1 = s.permute2(&s, [2, 3, 4, 5, 6, 7, 6, 7]).lazy();
        let s_rows = s.lazy();
        let right0 = b
            .lazy()
            .relax::<SQ_RIGHT>()
            .permute2::<SQ_RIGHT, SQ_RIGHT>(&xi_b.relax::<SQ_RIGHT>(), [0, 1, 2, 3, 4, 5, 10, 11]);
        let u = t
            .relax::<SQ_RIGHT>()
            .permute2::<SQ_RIGHT, SQ_RIGHT>(&xi_t, [0, 1, 10, 11, 12, 13, 6, 7]);
        let right1 = xi_b
            .relax::<SQ_RIGHT>()
            .permute2::<SQ_RIGHT, SQ_RIGHT>(&u, [4, 5, 8, 9, 10, 11, 12, 13]);
        // Padding lanes of the left rows remain zero.
        let right_hi = t
            .relax::<SQ_RIGHT>()
            .permute2::<SQ_RIGHT, SQ_RIGHT>(&xi_t, [0, 1, 2, 3, 4, 5, 12, 13]);
        // SAFETY: `PackedFp12` values exist only below an IFMA-authorized entry.
        let [mid_lo, halves] = unsafe {
            execute_row_pair_63::<1, SQ_RIGHT, SQ_NEG, 2, 1>(
                Sources::new(&left0, &left1),
                Sources::new(&right0, &right1),
                Sources::new(&s_rows, &s_rows),
                Sources::new(&right_hi, &right_hi),
                &SQUARE,
            )
        };
        // SAFETY: the enclosing IFMA entry authorizes the zero constructor.
        let zero = unsafe { FpVec8::zero() };

        let ab = mid_lo.permute2(&mid_lo, [0, 1, 2, 3, 4, 5, 0, 1]);
        // st lanes 2..=5 sum two canonical halves. Lanes 6 and 7 stay merely
        // bounded because the final select overwrites them.
        let st_even = mid_lo.permute2(&halves, [6, 7, 8, 10, 12, 14, 6, 7]);
        let st_odd = halves
            .permute2(&halves, [0, 0, 1, 3, 5, 7, 0, 0])
            .lazy()
            .keep_lanes(0x3c);
        let st = st_even.lazy().add_lazy::<1, 2>(&st_odd);
        let vab = mid_lo.permute2(&zero, [4, 5, 0, 1, 2, 3, 8, 8]);
        let xi_ab2 = vab.lazy().xi_lazy::<XI1>();
        let vab = FpVec8L::<XI1>::select(0x03, &xi_ab2, &vab.lazy().relax::<XI1>());
        let c0 = st.sub2_lazy::<1, XI1, 15>(&ab.lazy(), &vab);

        let doubled_ab = ab.lazy().double_lazy::<2>();
        let lo = FpVec8L::<15>::select(0xc0, &doubled_ab.relax::<15>(), &c0);
        self.lo = lo.reduce_full();
        self.hi = mid_lo.permute2(&zero, [2, 3, 4, 5, 8, 8, 8, 8]).double();
    }
}

impl PackedLine {
    /// Load one unscaled line and apply the anchor's `(py, px)` scaling
    /// inside the domain-entry multiplication, so no scalar scaling remains.
    #[inline]
    fn load(coeffs: &EllCoeff, scale: &FpVec8) -> Self {
        let mut out = MaybeUninit::uninit();
        Self::load_into(&mut out, coeffs, scale);
        // SAFETY: `load_into` initialized every field.
        unsafe { out.assume_init() }
    }

    #[inline(always)]
    fn load_into(out: &mut MaybeUninit<Self>, coeffs: &EllCoeff, scale: &FpVec8) {
        let ((a0, a1), (b0, b1), (c0, c1)) = *coeffs;
        // SAFETY: `base` is the only field. `load_scaled_into` initializes it.
        let base = unsafe { core::ptr::addr_of_mut!((*out.as_mut_ptr()).base) };
        unsafe {
            FpVec8::load_scaled_into(
                base,
                &[
                    Fp(a0),
                    Fp(a1),
                    Fp(b0),
                    Fp(b1),
                    Fp(c0),
                    Fp(c1),
                    Fp::ZERO,
                    Fp::ZERO,
                ],
                scale,
            )
        };
    }

    #[inline(always)]
    fn xi(&self) -> FpVec8L<XI1> {
        let all = self.base.lazy().xi_lazy::<XI1>();
        all.permute2::<XI1, XI1>(&all, [2, 3, 4, 5, 6, 7, 6, 7])
    }
}

struct IfmaSink;

/// One G1 anchor's line-load scale pair. Doubling records `(h, j, i)` scale
/// by `(-py, 3px)` and addition records `(lambda, theta, j)` by `(py, -px)`,
/// so the record negations and the tripling cost nothing extra here.
#[derive(Clone, Copy)]
struct AnchorScales {
    doubling: FpVec8,
    addition: FpVec8,
}

impl AnchorScales {
    /// # Safety
    ///
    /// The current CPU and OS must support AVX-512F and AVX-512IFMA.
    #[inline(always)]
    unsafe fn new(p: &G1Affine) -> Self {
        let neg_py = -p.y;
        let triple_px = p.x + p.x + p.x;
        let neg_px = -p.x;
        unsafe {
            Self {
                doubling: FpVec8::line_scale_pair(neg_py, triple_px),
                addition: FpVec8::line_scale_pair(p.y, neg_px),
            }
        }
    }
}

/// `has_pending` marks an outstanding sparse application of the line in
/// `lines[next_line ^ 1]`. `square_pending` marks the accumulator square that
/// must precede it. Both stages drain through `pump_stage` during the next
/// line's scalar generation, or through `drain` when the next line arrives.
/// `scales` holds one line-load scale pair per registered anchor.
struct ActiveIfmaSink {
    f: PackedFp12,
    lines: [MaybeUninit<PackedLine>; 2],
    next_line: usize,
    has_pending: bool,
    square_pending: bool,
    scales: Vec<AnchorScales>,
}

impl MillerLineSink for IfmaSink {
    type Output = Fp12;
    type Active = ActiveIfmaSink;

    #[inline(always)]
    fn start(self, p: &G1Affine, line: impl FnOnce() -> EllCoeff) -> (Self::Active, u32) {
        // SAFETY: the sink is constructed only below an IFMA-authorized entry.
        let scales = unsafe { AnchorScales::new(p) };
        let first = PackedLine::load(&line(), &scales.doubling);
        (
            ActiveIfmaSink {
                f: PackedFp12::from_line(&first),
                lines: [MaybeUninit::uninit(), MaybeUninit::uninit()],
                next_line: 0,
                has_pending: false,
                square_pending: false,
                scales: alloc::vec![scales],
            },
            0,
        )
    }
}

impl ActiveIfmaSink {
    #[inline(always)]
    unsafe fn pending_line(&self, index: usize) -> &PackedLine {
        // SAFETY: `push` writes each slot before it reads it. The index stays in 0..2.
        unsafe { self.lines.get_unchecked(index).assume_init_ref() }
    }

    #[inline(always)]
    fn apply_pending(&mut self) {
        // SAFETY: The `push` that set `has_pending` wrote the other slot.
        let pending = unsafe { self.pending_line(self.next_line ^ 1) } as *const PackedLine;
        // SAFETY: `pending` points into `lines`. The called function does not change `lines`.
        self.f.mul_by_034_assign(unsafe { &*pending });
        self.has_pending = false;
    }

    /// Execute at most one deferred stage, so one ~700-madd row pair lands
    /// inside each scalar slice of the line generator.
    #[target_feature(enable = "avx512f,avx512ifma")]
    unsafe fn pump_stage(&mut self) {
        if self.square_pending {
            self.f.square_assign();
            self.square_pending = false;
        } else if self.has_pending {
            self.apply_pending();
        }
    }

    #[inline(always)]
    fn drain(&mut self) {
        if self.square_pending {
            self.f.square_assign();
            self.square_pending = false;
        }
        if self.has_pending {
            self.apply_pending();
        }
    }

    #[inline(always)]
    fn push(
        &mut self,
        anchor: u32,
        square: bool,
        doubling: bool,
        line: impl FnOnce(&mut dyn MillerPump) -> EllCoeff,
    ) {
        let coeffs = line(&mut *self);
        // SAFETY: the sink exists only below an IFMA-authorized entry.
        unsafe { self.commit_line(anchor, square, doubling, &coeffs) };
    }

    /// Drain the leftover stages and stage the new line. One outlined body
    /// serves every push site, so the drain operations and the scaled line
    /// load stay out of the entry bodies.
    ///
    /// # Safety
    ///
    /// The CPU and operating system must support AVX-512F and AVX-512IFMA.
    #[target_feature(enable = "avx512f,avx512ifma")]
    #[inline(never)]
    unsafe fn commit_line(&mut self, anchor: u32, square: bool, doubling: bool, coeffs: &EllCoeff) {
        self.drain();
        let scales = &self.scales[anchor as usize];
        let scale = if doubling {
            &scales.doubling
        } else {
            &scales.addition
        };
        // SAFETY: `next_line` starts at zero. XOR with one keeps the index in 0..2.
        let next = unsafe { self.lines.get_unchecked_mut(self.next_line) };
        PackedLine::load_into(next, coeffs, scale);
        self.next_line ^= 1;
        self.has_pending = true;
        self.square_pending = square;
    }
}

impl MillerPump for ActiveIfmaSink {
    fn pump(&mut self) {
        // SAFETY: the sink exists only below an IFMA-authorized entry.
        unsafe { self.pump_stage() }
    }
}

impl ActiveMillerLineSink for ActiveIfmaSink {
    type Output = Fp12;
    type Anchor = u32;

    #[inline(always)]
    fn anchor(&mut self, p: &G1Affine) -> u32 {
        // SAFETY: the sink exists only below an IFMA-authorized entry.
        let scales = unsafe { AnchorScales::new(p) };
        let index = self.scales.len() as u32;
        self.scales.push(scales);
        index
    }

    #[inline(always)]
    fn doubling(&mut self, anchor: u32, line: impl FnOnce(&mut dyn MillerPump) -> EllCoeff) {
        self.push(anchor, true, true, line);
    }

    #[inline(always)]
    fn merged_doubling(&mut self, anchor: u32, line: impl FnOnce(&mut dyn MillerPump) -> EllCoeff) {
        self.push(anchor, false, true, line);
    }

    #[inline(always)]
    fn addition(&mut self, anchor: u32, line: impl FnOnce(&mut dyn MillerPump) -> EllCoeff) {
        self.push(anchor, false, false, line);
    }

    #[inline(always)]
    fn tail(&mut self, anchor: u32, line: impl FnOnce(&mut dyn MillerPump) -> EllCoeff) {
        self.push(anchor, false, false, line);
    }

    #[inline(always)]
    fn finish(&mut self) -> Self::Output {
        self.drain();
        self.f.store()
    }
}

/// Compute one Miller loop after runtime IFMA authorization.
pub(crate) fn miller_loop(
    _ifma: crate::x86_runtime::Avx512Ifma,
    p: &G1Affine,
    q: &G2Affine,
) -> Fp12 {
    debug_assert!(!p.is_identity() && !q.is_identity());
    // SAFETY `_ifma` proves CPU and operating-system support.
    unsafe { miller_loop_ifma(p, q) }
}

/// Replay one prepared schedule after runtime IFMA authorization.
pub(crate) fn miller_loop_prepared(
    _ifma: crate::x86_runtime::Avx512Ifma,
    p: &G1Affine,
    q: &G2Prepared,
) -> Fp12 {
    debug_assert!(!p.is_identity());
    // SAFETY: `_ifma` proves CPU and operating-system support.
    unsafe { miller_loop_prepared_ifma(p, q) }
}

pub(crate) fn multi_miller_loop_mixed(
    _ifma: crate::x86_runtime::Avx512Ifma,
    pairs: &[(&G1Affine, G2Miller<'_>)],
) -> Fp12 {
    // SAFETY: `_ifma` proves CPU and operating-system support.
    unsafe { multi_miller_loop_mixed_ifma(pairs) }
}

#[target_feature(enable = "avx512f,avx512ifma")]
#[inline(never)]
unsafe fn miller_loop_ifma(p: &G1Affine, q: &G2Affine) -> Fp12 {
    stream_line_coeffs(p, q, IfmaSink)
}

#[target_feature(enable = "avx512f,avx512ifma")]
#[inline(never)]
unsafe fn miller_loop_prepared_ifma(p: &G1Affine, q: &G2Prepared) -> Fp12 {
    stream_prepared_line_coeffs(p, q, IfmaSink)
}

#[target_feature(enable = "avx512f,avx512ifma")]
#[inline(never)]
unsafe fn multi_miller_loop_mixed_ifma(pairs: &[(&G1Affine, G2Miller<'_>)]) -> Fp12 {
    stream_mixed_line_coeffs(pairs, IfmaSink).unwrap_or(Fp12::ONE)
}

/// `NB` also bounds the even rows, so one row type feeds the recurrence.
/// The negation assert inside `negate_lanes_lazy` pins `NB` against `RB`.
#[inline(always)]
fn plan_operands<const LB: u64, const RB: u64, const NB: u64>(
    left: Sources<'_, LB>,
    right: Sources<'_, RB>,
    plan: &Plan,
) -> ([FpVec8L<LB>; 6], [FpVec8L<NB>; 6]) {
    let a: [FpVec8L<LB>; 6] =
        core::array::from_fn(|term| left.low.permute2::<LB, LB>(left.high, plan.left[term]));
    let b: [FpVec8L<NB>; 6] = core::array::from_fn(|term| {
        let selected = right.low.permute2::<RB, RB>(right.high, plan.right[term]);
        if term & 1 == 0 {
            selected.relax::<NB>()
        } else {
            selected.negate_lanes_lazy::<NB>(REAL_LANES)
        }
    });
    (a, b)
}

/// Operand rows of the short stream. `NB` matches the full-row bound, so one
/// bound tag feeds the shared recurrence. Every half row carries lazy signs,
/// so every right row passes through the masked negation.
#[inline(always)]
fn plan_operands_half<const LB: u64, const RB: u64, const NB: u64>(
    left: Sources<'_, LB>,
    right: Sources<'_, RB>,
    plan: &HalfPlan,
) -> ([FpVec8L<LB>; 3], [FpVec8L<NB>; 3]) {
    let a: [FpVec8L<LB>; 3] =
        core::array::from_fn(|row| left.low.permute2::<LB, LB>(left.high, plan.left[row]));
    let b: [FpVec8L<NB>; 3] = core::array::from_fn(|row| {
        right
            .low
            .permute2::<RB, RB>(right.high, plan.right[row])
            .negate_lanes_lazy::<NB>(plan.negate[row])
    });
    (a, b)
}

/// Run the full-row and the half-row reductions of one Fp12 operation as two
/// interleaved streams. The operand builds and the paired recurrence must
/// stay in one body, or every operand row crosses an outlined-call boundary
/// through the stack.
///
/// # Safety
///
/// The CPU and operating system must support AVX-512F and AVX-512IFMA.
#[target_feature(enable = "avx512f,avx512ifma")]
#[inline(never)]
unsafe fn execute_row_pair_63<
    const LB: u64,
    const RB: u64,
    const NB: u64,
    const R0: usize,
    const R1: usize,
>(
    full_left: Sources<'_, LB>,
    full_right: Sources<'_, RB>,
    half_left: Sources<'_, LB>,
    half_right: Sources<'_, RB>,
    plans: &PackedPlans,
) -> [FpVec8; 2] {
    let (a0, b0) = plan_operands::<LB, RB, NB>(full_left, full_right, &plans.full);
    let (a1, b1) = plan_operands_half::<LB, RB, NB>(half_left, half_right, &plans.half);
    // SAFETY: the enclosing target features satisfy the ISA precondition.
    unsafe { FpVec8::sos_mac_6_3_pair_fused_lazy::<LB, NB, R0, R1>(&a0, &b0, &a1, &b1) }
}

const fn sparse_plans() -> PackedPlans {
    slots!(0 => a0, a1, a2);
    slots!(3 => b0, b1, b2);
    slots!(0 => c0, c3, c4);
    let xi_c3 = PairSlot(4);
    let xi_c4 = PairSlot(5);

    packed_plans! {
        full [
            sum3!(a0 * c0 + b1 * xi_c4 + b2 * xi_c3),
            sum3!(a1 * c0 + b0 * c3 + b2 * xi_c4),
            sum3!(a2 * c0 + b0 * c4 + b1 * c3),
            sum3!(a0 * c3 + a2 * xi_c4 + b0 * c0),
        ];
        half [
            sum3!(a0 * c4 + a1 * c3 + b1 * c0),
            sum3!(a1 * c4 + a2 * c3 + b2 * c0),
        ]
    }
}

const fn square_plans() -> PackedPlans {
    slots!(0 => a0, a1, a2);
    slots!(3 => s0, s1, s2);
    slots!(0 => b0, b1, b2);
    let xi_b1 = PairSlot(3);
    let xi_b2 = PairSlot(4);
    let t0 = PairSlot(5);
    let xi_t1 = PairSlot(6);
    let xi_t2 = PairSlot(7);
    slots!(0 => high_s0, high_s1, high_s2);
    slots!(0 => high_t0, high_t1, high_t2);
    let high_xi_t2 = PairSlot(3);

    packed_plans! {
        full [
            sum3!(a0 * b0 + a1 * xi_b2 + a2 * xi_b1),
            sum3!(a0 * b1 + a1 * b0 + a2 * xi_b2),
            sum3!(a0 * b2 + a1 * b1 + a2 * b0),
            sum3!(s0 * t0 + s1 * xi_t2 + s2 * xi_t1),
        ];
        half [
            sum3!(high_s0 * high_t1 + high_s1 * high_t0 + high_s2 * high_xi_t2),
            sum3!(high_s0 * high_t2 + high_s1 * high_t1 + high_s2 * high_t0),
        ]
    }
}

const SPARSE: PackedPlans = sparse_plans();
const SQUARE: PackedPlans = square_plans();

#[cfg(all(test, helius_avx512_ifma))]
#[inline]
fn coefficients(value: &Fp12) -> [Fp; 12] {
    [
        value.c0.c0.c0,
        value.c0.c0.c1,
        value.c0.c1.c0,
        value.c0.c1.c1,
        value.c0.c2.c0,
        value.c0.c2.c1,
        value.c1.c0.c0,
        value.c1.c0.c1,
        value.c1.c1.c0,
        value.c1.c1.c1,
        value.c1.c2.c0,
        value.c1.c2.c1,
    ]
}

#[inline]
fn from_coefficients(c: [Fp; 12]) -> Fp12 {
    Fp12::new(
        Fp6::new(
            Fp2::new(c[0], c[1]),
            Fp2::new(c[2], c[3]),
            Fp2::new(c[4], c[5]),
        ),
        Fp6::new(
            Fp2::new(c[6], c[7]),
            Fp2::new(c[8], c[9]),
            Fp2::new(c[10], c[11]),
        ),
    )
}

// Direct tests require IFMA in the compile-time target.
#[cfg(all(test, helius_avx512_ifma))]
mod tests {
    use super::*;
    use crate::consts::P;
    use crate::g1::G1Affine;
    use crate::g2::G2Affine;
    use crate::pairing::miller::{miller_loop_scalar_for_test, prepare};

    #[rustfmt::skip]
    #[test]
    fn equations_reproduce_winning_shuffle_plans() {
        assert_eq!(SPARSE.full.left, [[0,0,2,2,4,4,0,0],[1,1,3,3,5,5,1,1],[8,8,6,6,6,6,4,4],[9,9,7,7,7,7,5,5],[10,10,10,10,8,8,6,6],[11,11,11,11,9,9,7,7]]);
        assert_eq!(SPARSE.full.right, [[0,1,0,1,0,1,2,3],[1,0,1,0,1,0,3,2],[10,11,2,3,4,5,10,11],[11,10,3,2,5,4,11,10],[8,9,10,11,2,3,0,1],[9,8,11,10,3,2,1,0]]);
        assert_eq!(SPARSE.half.left, [[0,3,0,3,2,5,2,5],[1,8,1,8,3,10,3,10],[2,9,2,9,4,11,4,11]]);
        assert_eq!(SPARSE.half.right, [[4,3,5,2,4,3,5,2],[5,0,4,1,5,0,4,1],[2,1,3,0,2,1,3,0]]);
        assert_eq!(SPARSE.half.negate, [0x22,0x11,0x22]);
        assert_eq!(SQUARE.full.left, [[0,0,0,0,0,0,6,6],[1,1,1,1,1,1,7,7],[2,2,2,2,2,2,8,8],[3,3,3,3,3,3,9,9],[4,4,4,4,4,4,10,10],[5,5,5,5,5,5,11,11]]);
        assert_eq!(SQUARE.full.right, [[0,1,2,3,4,5,10,11],[1,0,3,2,5,4,11,10],[8,9,0,1,2,3,14,15],[9,8,1,0,3,2,15,14],[6,7,8,9,0,1,12,13],[7,6,9,8,1,0,13,12]]);
        assert_eq!(SQUARE.half.left, [[0,3,0,3,0,3,0,3],[1,4,1,4,1,4,1,4],[2,5,2,5,2,5,2,5]]);
        assert_eq!(SQUARE.half.right, [[2,1,3,0,4,3,5,2],[3,6,2,7,5,0,4,1],[0,7,1,6,2,1,3,0]]);
        assert_eq!(SQUARE.half.negate, [0x22,0x11,0x22]);
    }

    fn next(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    fn fp(state: &mut u64) -> Fp {
        Fp::from_raw([next(state), next(state), next(state), next(state)])
    }

    fn fp2(state: &mut u64) -> Fp2 {
        Fp2::new(fp(state), fp(state))
    }

    fn packed_line(c0: Fp2, c3: Fp2, c4: Fp2) -> PackedLine {
        let coeffs = ((c0.c0.0, c0.c1.0), (c3.c0.0, c3.c1.0), (c4.c0.0, c4.c1.0));
        // SAFETY: the compile-time test target enables IFMA.
        let identity = unsafe { FpVec8::line_scale_pair(Fp::ONE, Fp::ONE) };
        PackedLine::load(&coeffs, &identity)
    }

    fn fp12(state: &mut u64) -> Fp12 {
        Fp12::new(
            Fp6::new(fp2(state), fp2(state), fp2(state)),
            Fp6::new(fp2(state), fp2(state), fp2(state)),
        )
    }

    #[test]
    fn sparse_rows_match_scalar_edges_and_random() {
        let pm1 = Fp::from_raw([P[0] - 1, P[1], P[2], P[3]]);
        let edge = Fp12::new(
            Fp6::new(Fp2::new(Fp::ZERO, pm1), Fp2::ONE, Fp2::new(pm1, Fp::ZERO)),
            Fp6::new(Fp2::new(pm1, pm1), Fp2::ZERO, Fp2::ONE),
        );
        let edge_lines = [
            (Fp2::ZERO, Fp2::ZERO, Fp2::ZERO),
            (Fp2::ONE, Fp2::ONE, Fp2::ONE),
            (
                Fp2::new(pm1, Fp::ZERO),
                Fp2::new(Fp::ZERO, pm1),
                Fp2::new(pm1, pm1),
            ),
        ];
        for (c0, c3, c4) in edge_lines {
            let mut want = edge;
            want.mul_by_034_assign_sosd6(c0, c3, c4);
            let mut got = PackedFp12::load(&edge);
            got.mul_by_034_assign(&packed_line(c0, c3, c4));
            let got = got.store();
            assert_eq!(got, want);
        }

        let mut state = 0x9e37_79b9_7f4a_7c15;
        for _ in 0..256 {
            let value = fp12(&mut state);
            let (c0, c3, c4) = (fp2(&mut state), fp2(&mut state), fp2(&mut state));
            let mut want = value;
            want.mul_by_034_assign_sosd6(c0, c3, c4);
            let mut got = PackedFp12::load(&value);
            got.mul_by_034_assign(&packed_line(c0, c3, c4));
            let got = got.store();
            assert_eq!(got, want);
        }
    }

    #[test]
    fn square_rows_match_scalar_edges_and_random() {
        let pm1 = Fp::from_raw([P[0] - 1, P[1], P[2], P[3]]);
        let edges = [
            Fp12::ZERO,
            Fp12::ONE,
            Fp12::new(
                Fp6::new(Fp2::new(Fp::ZERO, pm1), Fp2::ONE, Fp2::new(pm1, Fp::ZERO)),
                Fp6::new(Fp2::new(pm1, pm1), Fp2::ZERO, Fp2::ONE),
            ),
        ];
        for value in edges {
            let mut want = value;
            want.square_in_place_sos();
            let mut got = PackedFp12::load(&value);
            got.square_assign();
            assert_eq!(got.store(), want);
        }

        let mut state = 0x243f_6a88_85a3_08d3;
        for _ in 0..256 {
            let value = fp12(&mut state);
            let mut want = value;
            want.square_in_place_sos();
            let mut got = PackedFp12::load(&value);
            got.square_assign();
            assert_eq!(got.store(), want);
        }
    }

    #[test]
    fn compile_time_ifma_stream_matches_uncached_live_raw_miller() {
        let p = G1Affine::generator();
        let q = G2Affine::generator();
        assert_eq!(
            unsafe { miller_loop_ifma(&p, &q) },
            miller_loop_scalar_for_test(&p, &q)
        );

        let np = p.negate();
        let nq = q.negate();
        assert_eq!(
            unsafe { miller_loop_ifma(&np, &nq) },
            miller_loop_scalar_for_test(&np, &nq)
        );
    }

    #[test]
    fn compile_time_ifma_prepared_matches_scalar_live_raw_miller() {
        let p = G1Affine::generator();
        let q = G2Affine::generator();
        let prepared = prepare(&q);
        assert_eq!(
            unsafe { miller_loop_prepared_ifma(&p, &prepared) },
            miller_loop_scalar_for_test(&p, &q)
        );

        let np = p.negate();
        let nq = q.negate();
        let prepared = prepare(&nq);
        assert_eq!(
            unsafe { miller_loop_prepared_ifma(&np, &prepared) },
            miller_loop_scalar_for_test(&np, &nq)
        );
    }
}
