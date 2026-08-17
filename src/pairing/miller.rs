//! Miller loop for optimal Ate pairing on BN254 (D-type twist).
//! Line formulas: eprint 2013/722. G2 arithmetic via monomorphized Fp2 limbs.

use alloc::vec::Vec;

use crate::consts::ATE_LOOP_COUNT;
use crate::fp::Fp;
use crate::fp2::Fp2;
use crate::fp2_fast::f2_mul as f2_mul_g2;
use crate::fp2_fast::f2_square as f2_sqr_g2;
use crate::fp2_fast::{F2, f2_add, f2_dbl, f2_from, f2_mul_fp, f2_neg, f2_one, f2_sub, f2_to};
use crate::fp12::Fp12;
use crate::g1::G1Affine;
use crate::g2::G2Affine;

/// Homogeneous projective point on the twist.
struct G2Hom {
    x: F2,
    y: F2,
    z: F2,
}

pub(crate) type EllCoeff = (F2, F2, F2);

/// `mul_by_034` operands: line coefficients with the G1 scalings applied.
pub(crate) type ScaledCoeffs = (Fp2, Fp2, Fp2);

/// One loop iteration's lines: the doubling line plus the NAF digit's
/// optional addition line, scaled and ready for `mul_by_034`.
type Lines = (ScaledCoeffs, Option<ScaledCoeffs>);

/// Hook a line generator calls between its scalar work chunks. Each call may
/// execute at most one deferred accumulator stage, so vector work lands inside
/// the scalar slices. Pumping is optional: unpumped stages run when the next
/// line arrives, and the accumulator op sequence is identical for every pump
/// schedule.
///
/// `dyn` keeps one generator body for every consumer.
pub(crate) trait MillerPump {
    fn pump(&mut self);
}

/// Pump for consumers with no deferred stages.
pub(crate) struct NoPump;

impl MillerPump for NoPump {
    #[inline(always)]
    fn pump(&mut self) {}
}

impl G2Hom {
    #[inline(always)]
    fn from_affine(q: &G2Affine) -> Self {
        Self {
            x: f2_from(q.x),
            y: f2_from(q.y),
            z: f2_one(),
        }
    }

    /// Double. Returns the raw record `(h, j, i)`. The D-type line is
    /// `(-h, 3j, i)`; the negation and the tripling ride the per-pair G1
    /// scalars, see [`G1LineScalars`].
    #[inline(never)]
    fn double_in_place(&mut self, pump: &mut dyn MillerPump) -> EllCoeff {
        let mut a = f2_mul_g2(self.x, self.y);
        a = f2_half(a);
        let b = f2_sqr_g2(self.y);
        let c = f2_sqr_g2(self.z);
        // e = twist_b * 3c
        let e = f2_mul_g2(twist_b_f2(), f2_add(f2_dbl(c), c));
        pump.pump();
        let f = f2_add(f2_dbl(e), e); // 3e
        let mut g = f2_add(b, f);
        g = f2_half(g);
        let h = f2_sub(f2_sqr_g2(f2_add(self.y, self.z)), f2_add(b, c));
        let i = f2_sub(e, b);
        let j = f2_sqr_g2(self.x);
        pump.pump();

        self.x = f2_mul_g2(a, f2_sub(b, f));
        // y = g^2 - 3e^2. The lazy leaf holds both squares at double width
        // and reduces once per output half.
        #[cfg(all(narsil_g2_ysqr_asm, not(feature = "force-portable")))]
        {
            self.y = crate::fp2_fast::f2_ysqr(g, e, f);
        }
        #[cfg(not(all(narsil_g2_ysqr_asm, not(feature = "force-portable"))))]
        {
            let e_square = f2_sqr_g2(e);
            self.y = f2_sub(f2_sqr_g2(g), f2_add(f2_dbl(e_square), e_square));
        }
        self.z = f2_mul_g2(b, h);

        (h, j, i)
    }

    /// Add affine q. Returns the raw record `(lambda, theta, j)`. The D-type
    /// line is `(lambda, -theta, j)`; the negation rides the per-pair G1
    /// scalars, see [`G1LineScalars`].
    #[inline(never)]
    fn add_in_place(&mut self, qx: F2, qy: F2, pump: &mut dyn MillerPump) -> EllCoeff {
        let theta = f2_sub(self.y, f2_mul_g2(qy, self.z));
        let lambda = f2_sub(self.x, f2_mul_g2(qx, self.z));
        let c = f2_sqr_g2(theta);
        let d = f2_sqr_g2(lambda);
        let e = f2_mul_g2(lambda, d);
        pump.pump();
        let f = f2_mul_g2(self.z, c);
        let g = f2_mul_g2(self.x, d);
        let h = f2_sub(f2_add(e, f), f2_dbl(g));
        self.x = f2_mul_g2(lambda, h);
        pump.pump();
        self.y = f2_sub(f2_mul_g2(theta, f2_sub(g, h)), f2_mul_g2(e, self.y));
        self.z = f2_mul_g2(self.z, e);
        let j = f2_sub(f2_mul_g2(theta, qx), f2_mul_g2(lambda, qy));
        (lambda, theta, j)
    }
}

/// D-twist coefficient b' = 3/xi = (27 - 3u)/82 in Montgomery form
/// (E': y^2 = x^3 + b'). Pinned below to `const_tower::TWIST_B`.
#[inline(always)]
pub(crate) const fn twist_b_f2() -> F2 {
    (
        [
            0x3bf938e377b802a8,
            0x020b1b273633535d,
            0x26b7edf049755260,
            0x2514c6324384a86d,
        ],
        [
            0x38e7ecccd1dcff67,
            0x65f0b37d93ce0d3e,
            0xd749d0dd22ac00aa,
            0x0141b9ce4a688d4d,
        ],
    )
}

/// Per-pair G1 scalars for raw line records. A doubling record `(h, j, i)`
/// scales to the `mul_by_034` operands `(h*(-py), j*(3px), i)`, an addition
/// record `(lambda, theta, j)` to `(lambda*py, theta*(-px), j)`, so the
/// per-line Fp2 negations and the tripling collapse into these constants.
#[derive(Clone, Copy)]
pub(crate) struct G1LineScalars {
    neg_py: [u64; 4],
    triple_px: [u64; 4],
    py: [u64; 4],
    neg_px: [u64; 4],
}

impl G1LineScalars {
    #[inline]
    pub(crate) fn new(p: &G1Affine) -> Self {
        use crate::consts::P;
        use crate::limb::{add_mod, sub_mod};
        let px = p.x.0;
        let py = p.y.0;
        Self {
            neg_py: sub_mod(&[0; 4], &py, &P),
            triple_px: add_mod(&add_mod(&px, &px, &P), &px, &P),
            py,
            neg_px: sub_mod(&[0; 4], &px, &P),
        }
    }
}

/// The exact `mul_by_034` operands of one raw doubling record.
#[cfg_attr(not(narsil_miller_inline), inline(never))]
#[cfg_attr(narsil_miller_inline, inline(always))]
fn scale_doubling(coeffs: &EllCoeff, s: &G1LineScalars) -> ScaledCoeffs {
    (
        f2_to(f2_mul_fp(coeffs.0, &s.neg_py)),
        f2_to(f2_mul_fp(coeffs.1, &s.triple_px)),
        f2_to(coeffs.2),
    )
}

/// The exact `mul_by_034` operands of one raw addition record.
#[cfg_attr(not(narsil_miller_inline), inline(never))]
#[cfg_attr(narsil_miller_inline, inline(always))]
fn scale_addition(coeffs: &EllCoeff, s: &G1LineScalars) -> ScaledCoeffs {
    (
        f2_to(f2_mul_fp(coeffs.0, &s.py)),
        f2_to(f2_mul_fp(coeffs.1, &s.neg_px)),
        f2_to(coeffs.2),
    )
}

#[inline(never)]
fn ell_doubling(f: &mut Fp12, coeffs: &EllCoeff, s: &G1LineScalars) {
    let (c0, c3, c4) = scale_doubling(coeffs, s);
    f.mul_by_034_assign(c0, c3, c4);
}

#[inline(never)]
fn ell_addition(f: &mut Fp12, coeffs: &EllCoeff, s: &G1LineScalars) {
    let (c0, c3, c4) = scale_addition(coeffs, s);
    f.mul_by_034_assign(c0, c3, c4);
}

/// Materialize a D-twist line as the sparse Fp12 element consumed by
/// `mul_by_034`. The first Miller iteration starts from one, so constructing
/// that line directly avoids a complete sparse multiplication by one.
#[inline(always)]
fn line_value(l: &ScaledCoeffs) -> Fp12 {
    use crate::fp6::Fp6;

    Fp12 {
        c0: Fp6::new(l.0, Fp2::ZERO, Fp2::ZERO),
        c1: Fp6::new(l.1, l.2, Fp2::ZERO),
    }
}

/// psi (untwist-Frobenius-twist) x-scale gamma_{1,2} = xi^((p-1)/3),
/// Montgomery. Pinned below to `const_tower::GAMMA1[1]`.
const FROB_TWIST_X: Fp2 = Fp2::new(
    Fp::from_raw_unchecked([
        13075984984163199792,
        3782902503040509012,
        8791150885551868305,
        1825854335138010348,
    ]),
    Fp::from_raw_unchecked([
        7963664994991228759,
        12257807996192067905,
        13179524609921305146,
        2767831111890561987,
    ]),
);

/// psi y-scale gamma_{1,3} = xi^((p-1)/2), Montgomery.
/// Pinned below to `const_tower::GAMMA1[2]`.
const FROB_TWIST_Y: Fp2 = Fp2::new(
    Fp::from_raw_unchecked([
        16482010305593259561,
        13488546290961988299,
        3578621962720924518,
        2681173117283399901,
    ]),
    Fp::from_raw_unchecked([
        11661927080404088775,
        553939530661941723,
        7860678177968807019,
        3208568454732775116,
    ]),
);

/// Compile-time pin: the twist and psi constants above are exactly their
/// first-principles derivations from P and xi (see `const_tower`).
const _: () = {
    use crate::const_tower::{GAMMA1, TWIST_B};
    use crate::consts::derive::eq4;
    let b = twist_b_f2();
    assert!(eq4(b.0, TWIST_B.0) && eq4(b.1, TWIST_B.1));
    assert!(eq4(FROB_TWIST_X.c0.0, GAMMA1[1].0) && eq4(FROB_TWIST_X.c1.0, GAMMA1[1].1));
    assert!(eq4(FROB_TWIST_Y.c0.0, GAMMA1[2].0) && eq4(FROB_TWIST_Y.c1.0, GAMMA1[2].1));
};

/// psi(x, y) = (conj(x)*gamma_{1,2}, conj(y)*gamma_{1,3}): the p-power
/// Frobenius endomorphism carried through the D-twist isomorphism.
#[inline(always)]
pub(crate) fn mul_by_char(q: G2Affine) -> G2Affine {
    G2Affine {
        x: q.x.conjugate() * FROB_TWIST_X,
        y: q.y.conjugate() * FROB_TWIST_Y,
        infinity: false,
    }
}

/// Exact halving mod p: even values shift. Odd values shift `(v + p)`, whose
/// carry survives into the top bit. Valid on Montgomery form (division by 2 is
/// a field operation) and far cheaper than multiplying by `2^{-1}`.
///
/// The parity that selects the two cases is uniform over the Miller lines, so
/// the branch would mispredict half the time. Masking `p` with it keeps the
/// same two results without one.
#[inline(always)]
fn half4(v: &[u64; 4]) -> [u64; 4] {
    use crate::consts::P;
    let odd = 0u64.wrapping_sub(v[0] & 1);
    let (s0, c0) = v[0].carrying_add(P[0] & odd, false);
    let (s1, c1) = v[1].carrying_add(P[1] & odd, c0);
    let (s2, c2) = v[2].carrying_add(P[2] & odd, c1);
    let (s3, c3) = v[3].carrying_add(P[3] & odd, c2);
    [
        (s0 >> 1) | (s1 << 63),
        (s1 >> 1) | (s2 << 63),
        (s2 >> 1) | (s3 << 63),
        (s3 >> 1) | ((c3 as u64) << 63),
    ]
}

/// x86 uses one outlined body to limit instruction-cache use.
#[cfg_attr(all(target_arch = "x86_64", not(narsil_miller_inline)), inline(never))]
#[cfg_attr(any(not(target_arch = "x86_64"), narsil_miller_inline), inline(always))]
fn f2_half(a: F2) -> F2 {
    (half4(&a.0), half4(&a.1))
}

/// Multiply `f` by one iteration's lines (doubling line, then the digit's
/// addition line if present).
#[inline(always)]
fn apply_lines(f: &mut Fp12, lines: &Lines) {
    let (d, a) = lines;
    f.mul_by_034_assign(d.0, d.1, d.2);
    if let Some(a) = a {
        f.mul_by_034_assign(a.0, a.1, a.2);
    }
}

/// Optimal Ate Miller loop.
///
/// On an x86 host where CPUID and XCR0 authorize AVX-512IFMA, `std` builds use
/// a cold coefficient-parallel path: every call regenerates and scales the G2
/// schedule, then evaluates one resident Fp12 chain. No preparation cache is
/// involved. Other `std` hosts retain the four-entry exact-coordinate
/// prepared-G2 LRU. `no_std` and `force-portable` execute the uncached scalar
/// live path.
///
/// This typed function does not validate points. Use
/// [`crate::batch::pairing_product_is_one`] for untrusted input.
pub fn miller_loop(p: &G1Affine, q: &G2Affine) -> Fp12 {
    if p.is_identity() || q.is_identity() {
        return Fp12::ONE;
    }

    // The capability proves CPU and OS support before the IFMA entry.
    #[cfg(all(
        feature = "std",
        any(narsil_avx512_ifma, narsil_x86_runtime_ifma),
        not(feature = "force-portable")
    ))]
    if let Some(ifma) = crate::x86_runtime::avx512_ifma() {
        return crate::miller_coeff_ifma::miller_loop(ifma, p, q);
    }

    #[cfg(all(feature = "std", not(feature = "force-portable")))]
    return typed_prepared_cache::miller_loop(p, q);

    #[cfg(not(all(feature = "std", not(feature = "force-portable"))))]
    miller_loop_live_nonidentity(p, q)
}

/// Uncached live G2 arithmetic.
#[cfg(any(test, not(all(feature = "std", not(feature = "force-portable")))))]
fn miller_loop_live_nonidentity(p: &G1Affine, q: &G2Affine) -> Fp12 {
    debug_assert!(!p.is_identity() && !q.is_identity());

    #[cfg(all(
        feature = "std",
        any(narsil_avx512_ifma, narsil_x86_runtime_ifma),
        not(feature = "force-portable")
    ))]
    if let Some(ifma) = crate::x86_runtime::avx512_ifma() {
        return crate::miller_coeff_ifma::miller_loop(ifma, p, q);
    }

    miller_loop_live_scalar_nonidentity(p, q)
}

/// Scalar live path.
#[cfg(any(test, not(all(feature = "std", not(feature = "force-portable")))))]
#[inline(never)]
fn miller_loop_live_scalar_nonidentity(p: &G1Affine, q: &G2Affine) -> Fp12 {
    debug_assert!(!p.is_identity() && !q.is_identity());

    // Produce each line before the accumulator consumes the prior line.
    let mut r = G2Hom::from_affine(q);
    let qx = f2_from(q.x);
    let qy = f2_from(q.y);
    let nqy = f2_neg(qy);

    // Scaled lines for one loop iteration given its NAF digit.
    let scalars = G1LineScalars::new(p);
    let lines = |r: &mut G2Hom, digit: i8| -> Lines {
        let d = scale_doubling(&r.double_in_place(&mut NoPump), &scalars);
        let a = match digit {
            1 => Some(scale_addition(
                &r.add_in_place(qx, qy, &mut NoPump),
                &scalars,
            )),
            -1 => Some(scale_addition(
                &r.add_in_place(qx, nqy, &mut NoPump),
                &scalars,
            )),
            _ => None,
        };
        (d, a)
    };

    let ate = &ATE_LOOP_COUNT;
    let n = ate.len();

    // The first doubling line is the initial accumulator value.
    let (first_d, first_a) = lines(&mut r, ate[n - 2]);
    let mut f = line_value(&first_d);
    if let Some(a) = &first_a {
        f.mul_by_034_assign(a.0, a.1, a.2);
    }

    let mut pending = lines(&mut r, ate[n - 3]);
    for i in (2..n - 1).rev() {
        f.square_in_place();
        let next = lines(&mut r, ate[i - 2]);
        apply_lines(&mut f, &pending);
        pending = next;
    }

    // The final iteration includes both Frobenius additions.
    let q1 = mul_by_char(*q);
    let q2 = mul_by_char(q1).negate();
    f.square_in_place();
    let l1 = scale_addition(
        &r.add_in_place(f2_from(q1.x), f2_from(q1.y), &mut NoPump),
        &scalars,
    );
    apply_lines(&mut f, &pending);
    let l2 = scale_addition(
        &r.add_in_place(f2_from(q2.x), f2_from(q2.y), &mut NoPump),
        &scalars,
    );
    f.mul_by_034_assign(l1.0, l1.1, l1.2);
    f.mul_by_034_assign(l2.0, l2.1, l2.2);

    f
}

/// Exact uncached production path for adversarial differential tests. This
/// prevents an `std` prepared-cache hit from accidentally standing in for the
/// live G2 backend under test.
#[cfg(test)]
pub(crate) fn miller_loop_live_for_test(p: &G1Affine, q: &G2Affine) -> Fp12 {
    if p.is_identity() || q.is_identity() {
        Fp12::ONE
    } else {
        miller_loop_live_nonidentity(p, q)
    }
}

#[cfg(all(test, feature = "std", narsil_avx512_ifma))]
pub(crate) fn miller_loop_scalar_for_test(p: &G1Affine, q: &G2Affine) -> Fp12 {
    if p.is_identity() || q.is_identity() {
        Fp12::ONE
    } else {
        miller_loop_live_scalar_nonidentity(p, q)
    }
}

/// Terms the eight-lane engine takes from a group of `terms` live pairs.
///
/// The engine always runs a whole eight-lane pass, masked or not, so a tail
/// shorter than the host crossover is cheaper on the shared accumulator.
#[cfg(all(
    feature = "std",
    any(narsil_avx512_ifma, narsil_x86_runtime_ifma),
    not(feature = "force-portable")
))]
pub(crate) fn vertical_prefix(terms: usize) -> usize {
    let tail = terms % 8;
    if tail >= crate::x86_runtime::vertical_min_miller_terms() {
        terms
    } else {
        terms - tail
    }
}

/// Fused Miller loop over several pairs.
/// Callers filter identity pairs and apply the final exponentiation.
///
/// Authorized IFMA hosts split the live terms between the masked eight-lane
/// engine (independent per-lane chains, then one lane product) and the
/// coefficient-parallel accumulator. Every route fully reduces, so the result
/// is the shared-accumulator product bit for bit.
pub fn multi_miller_loop(pairs: &[(&G1Affine, &G2Affine)]) -> Fp12 {
    #[cfg(all(
        any(narsil_avx512_ifma, narsil_x86_runtime_ifma),
        not(feature = "force-portable")
    ))]
    if crate::batch8::ifma_available() {
        let live: Vec<(G1Affine, G2Affine)> = pairs
            .iter()
            .filter(|(p, q)| !p.is_identity() && !q.is_identity())
            .map(|&(p, q)| (*p, *q))
            .collect();
        #[cfg(feature = "std")]
        {
            let (lanes, tail) = live.split_at(vertical_prefix(live.len()));
            let mut product = crate::batch8::multi_miller_product8(lanes).unwrap_or(Fp12::ONE);
            if !tail.is_empty() {
                let sources: Vec<(&G1Affine, G2Miller<'_>)> =
                    tail.iter().map(|(p, q)| (p, G2Miller::Live(q))).collect();
                product *= multi_miller_loop_mixed(&sources);
            }
            return product;
        }
        // Without std there is no prepared-schedule tail route, so a group
        // below the crossover goes to the shared accumulator whole.
        #[cfg(not(feature = "std"))]
        if live.len() >= crate::x86_runtime::vertical_min_miller_terms()
            && let Some(product) = crate::batch8::multi_miller_product8(&live)
        {
            return product;
        }
    }

    multi_miller_loop_shared(pairs)
}

/// Shared-accumulator scalar tier of [`multi_miller_loop`]. Crate-visible
/// so lane-engine tests can pin against an engine-independent reference.
pub(crate) fn multi_miller_loop_shared(pairs: &[(&G1Affine, &G2Affine)]) -> Fp12 {
    struct State {
        scalars: G1LineScalars,
        r: G2Hom,
        qx: F2,
        qy: F2,
        nqy: F2,
        q: G2Affine,
    }

    let mut states = Vec::with_capacity(pairs.len());
    for &(p, q) in pairs {
        if p.is_identity() || q.is_identity() {
            continue;
        }
        let qx = f2_from(q.x);
        let qy = f2_from(q.y);
        states.push(State {
            scalars: G1LineScalars::new(p),
            r: G2Hom::from_affine(q),
            qx,
            qy,
            nqy: f2_neg(qy),
            q: *q,
        });
    }
    if states.is_empty() {
        return Fp12::ONE;
    }

    let mut f = Fp12::ONE;
    let ate = &ATE_LOOP_COUNT;
    for i in (1..ate.len()).rev() {
        if i != ate.len() - 1 {
            f.square_in_place();
        }
        for (state_index, state) in states.iter_mut().enumerate() {
            let coeffs = state.r.double_in_place(&mut NoPump);
            if i == ate.len() - 1 && state_index == 0 {
                f = line_value(&scale_doubling(&coeffs, &state.scalars));
            } else {
                ell_doubling(&mut f, &coeffs, &state.scalars);
            }
        }
        match ate[i - 1] {
            1 => {
                for state in &mut states {
                    let coeffs = state.r.add_in_place(state.qx, state.qy, &mut NoPump);
                    ell_addition(&mut f, &coeffs, &state.scalars);
                }
            }
            -1 => {
                for state in &mut states {
                    let coeffs = state.r.add_in_place(state.qx, state.nqy, &mut NoPump);
                    ell_addition(&mut f, &coeffs, &state.scalars);
                }
            }
            _ => {}
        }
    }

    for state in &mut states {
        let q1 = mul_by_char(state.q);
        let q2 = mul_by_char(q1).negate();
        let coeffs = state
            .r
            .add_in_place(f2_from(q1.x), f2_from(q1.y), &mut NoPump);
        ell_addition(&mut f, &coeffs, &state.scalars);
        let coeffs = state
            .r
            .add_in_place(f2_from(q2.x), f2_from(q2.y), &mut NoPump);
        ell_addition(&mut f, &coeffs, &state.scalars);
    }
    f
}

/// Number of lines in the fixed optimal-Ate schedule: one doubling per
/// iteration, the nonzero NAF additions, and the two Frobenius-tail lines.
#[cfg(feature = "std")]
pub(crate) const G2_PREPARED_COEFFS: usize = {
    let ate = &ATE_LOOP_COUNT;
    let doublings = ate.len() - 1;
    let mut additions = 2;
    let mut i = 0;
    while i < ate.len() - 1 {
        assert!(ate[i] >= -1 && ate[i] <= 1);
        if ate[i] != 0 {
            additions += 1;
        }
        i += 1;
    }
    doublings + additions
};
#[cfg(feature = "std")]
const _: () = assert!(G2_PREPARED_COEFFS == 87);

#[cfg(feature = "std")]
const F2_ZERO: F2 = ([0; 4], [0; 4]);
#[cfg(feature = "std")]
const ELL_ZERO: EllCoeff = (F2_ZERO, F2_ZERO, F2_ZERO);

/// A validated fixed-G2 optimal-Ate line schedule.
///
/// Construct this with [`prepare_g2`]. The schedule can be reused with
/// [`miller_loop_prepared`] without repeating G2 line generation. Its contents
/// stay private so callers cannot forge a schedule that does not correspond to
/// a validated curve point.
#[cfg(feature = "std")]
pub struct G2Prepared {
    coeffs: [EllCoeff; G2_PREPARED_COEFFS],
    identity: bool,
}

#[cfg(feature = "std")]
impl G2Prepared {
    /// Raw limbs of the last schedule line. Harness digest hook, the limbs
    /// name the internal line representation, not canonical field values.
    #[doc(hidden)]
    pub fn last_line_limbs(&self) -> [u64; 24] {
        let (c0, c1, c2) = self.coeffs[G2_PREPARED_COEFFS - 1];
        let mut limbs = [0_u64; 24];
        for (chunk, half) in limbs
            .chunks_exact_mut(4)
            .zip([c0.0, c0.1, c1.0, c1.1, c2.0, c2.1])
        {
            chunk.copy_from_slice(&half);
        }
        limbs
    }
}

/// Replay the live G2 arithmetic once and retain its exact raw record
/// sequence (see [`G1LineScalars`] for the record conventions).
#[cfg(feature = "std")]
pub(crate) fn prepare(q: &G2Affine) -> G2Prepared {
    debug_assert!(!q.is_identity());
    let mut coeffs = [ELL_ZERO; G2_PREPARED_COEFFS];
    let mut r = G2Hom::from_affine(q);
    let qx = f2_from(q.x);
    let qy = f2_from(q.y);
    let nqy = f2_neg(qy);
    let ate = &ATE_LOOP_COUNT;
    let mut next = 0;
    for i in (1..ate.len()).rev() {
        coeffs[next] = r.double_in_place(&mut NoPump);
        next += 1;
        match ate[i - 1] {
            1 => {
                coeffs[next] = r.add_in_place(qx, qy, &mut NoPump);
                next += 1;
            }
            -1 => {
                coeffs[next] = r.add_in_place(qx, nqy, &mut NoPump);
                next += 1;
            }
            _ => {}
        }
    }
    let q1 = mul_by_char(*q);
    let q2 = mul_by_char(q1).negate();
    coeffs[next] = r.add_in_place(f2_from(q1.x), f2_from(q1.y), &mut NoPump);
    coeffs[next + 1] = r.add_in_place(f2_from(q2.x), f2_from(q2.y), &mut NoPump);
    debug_assert_eq!(next + 2, G2_PREPARED_COEFFS);
    G2Prepared {
        coeffs,
        identity: false,
    }
}

/// Schedule lines in the classic `(-h, 3j, i)` / `(lambda, -theta, j)`
/// convention, so independent reference derivations stay comparable to the
/// raw stored records.
#[cfg(all(test, feature = "std"))]
pub(crate) fn scalar_line_coeffs_for_model_test(q: &G2Affine) -> Vec<(Fp2, Fp2, Fp2)> {
    let prepared = prepare(q);
    let classic_doubling =
        |&(h, j, i): &EllCoeff| (f2_to(f2_neg(h)), f2_to(f2_add(f2_dbl(j), j)), f2_to(i));
    let classic_addition = |&(l, t, j): &EllCoeff| (f2_to(l), f2_to(f2_neg(t)), f2_to(j));
    let ate = &ATE_LOOP_COUNT;
    let mut out = Vec::with_capacity(G2_PREPARED_COEFFS);
    let mut next = 0;
    for i in (1..ate.len()).rev() {
        out.push(classic_doubling(&prepared.coeffs[next]));
        next += 1;
        if ate[i - 1] != 0 {
            out.push(classic_addition(&prepared.coeffs[next]));
            next += 1;
        }
    }
    out.push(classic_addition(&prepared.coeffs[next]));
    out.push(classic_addition(&prepared.coeffs[next + 1]));
    out
}

#[cfg(all(test, feature = "std"))]
pub(crate) fn prepared_line_coeffs_for_test(q: &G2Affine) -> Vec<(Fp2, Fp2, Fp2)> {
    scalar_line_coeffs_for_model_test(q)
}

/// Static consumer for a streamed Miller line schedule.
///
/// Lines arrive unscaled. Each line names the G1 anchor whose `(py, px)`
/// scaling the sink must apply, so a sink can fold the scaling into its own
/// representation. The first line initializes the active sink and its
/// accumulator. Closures let the sink select the producer order.
#[cfg(any(test, feature = "std"))]
pub(crate) trait MillerLineSink {
    /// Result produced after the final tail line is consumed.
    type Output;
    /// Sink state after the first doubling initialized the accumulator.
    type Active: ActiveMillerLineSink<Output = Self::Output>;

    /// Initialize the sink from the first pair's anchor and its first
    /// doubling line. Also returns that anchor's handle.
    fn start(
        self,
        p: &G1Affine,
        line: impl FnOnce() -> EllCoeff,
    ) -> (Self::Active, <Self::Active as ActiveMillerLineSink>::Anchor);
}

/// Initialized half of [`MillerLineSink`].
///
/// Each `line` closure receives a [`MillerPump`] so a live generator can
/// interleave the sink's deferred stages into its own scalar work.
#[cfg(any(test, feature = "std"))]
pub(crate) trait ActiveMillerLineSink {
    /// Result produced after the final tail line is consumed.
    type Output;
    /// Handle of one registered G1 anchor.
    type Anchor: Copy;

    /// Register one G1 anchor for later lines.
    fn anchor(&mut self, p: &G1Affine) -> Self::Anchor;

    /// Retain a doubling record to be consumed after an accumulator square.
    fn doubling(
        &mut self,
        anchor: Self::Anchor,
        line: impl FnOnce(&mut dyn MillerPump) -> EllCoeff,
    );

    /// Retain a doubling record without an accumulator square: a non-lead
    /// term of a shared multi-pair iteration.
    fn merged_doubling(
        &mut self,
        anchor: Self::Anchor,
        line: impl FnOnce(&mut dyn MillerPump) -> EllCoeff,
    );

    /// Retain a signed-NAF addition record.
    fn addition(
        &mut self,
        anchor: Self::Anchor,
        line: impl FnOnce(&mut dyn MillerPump) -> EllCoeff,
    );

    /// Retain a Frobenius-tail addition record.
    fn tail(&mut self, anchor: Self::Anchor, line: impl FnOnce(&mut dyn MillerPump) -> EllCoeff);

    /// Consume the last pending line and return the accumulator.
    fn finish(&mut self) -> Self::Output;
}

#[cfg(any(
    test,
    all(
        feature = "std",
        any(narsil_avx512_ifma, narsil_x86_runtime_ifma),
        not(feature = "force-portable")
    )
))]
#[inline(always)]
fn push_signed_addition<S: ActiveMillerLineSink>(
    sink: &mut S,
    anchor: S::Anchor,
    r: &mut G2Hom,
    qx: F2,
    qy: F2,
    nqy: F2,
    digit: i8,
) {
    if digit > 0 {
        sink.addition(anchor, |pump| r.add_in_place(qx, qy, pump));
    } else if digit < 0 {
        sink.addition(anchor, |pump| r.add_in_place(qx, nqy, pump));
    }
}

/// Generate and scale the fixed optimal-Ate schedule into a static sink.
///
/// Production sinks hold one logical pending line while generating the next.
#[cfg(any(
    test,
    all(
        feature = "std",
        any(narsil_avx512_ifma, narsil_x86_runtime_ifma),
        not(feature = "force-portable")
    )
))]
#[inline(always)]
pub(crate) fn stream_line_coeffs<S: MillerLineSink>(
    p: &G1Affine,
    q: &G2Affine,
    sink: S,
) -> S::Output {
    let mut r = G2Hom::from_affine(q);
    let qx = f2_from(q.x);
    let qy = f2_from(q.y);
    let nqy = f2_neg(qy);

    // The top stored digit is the folded leading digit, not a Miller step.
    const FIRST_DIGIT: usize = ATE_LOOP_COUNT.len() - 2;
    let (mut sink, anchor) = sink.start(p, || r.double_in_place(&mut NoPump));
    push_signed_addition(
        &mut sink,
        anchor,
        &mut r,
        qx,
        qy,
        nqy,
        ATE_LOOP_COUNT[FIRST_DIGIT],
    );

    for &digit in ATE_LOOP_COUNT[..FIRST_DIGIT].iter().rev() {
        sink.doubling(anchor, |pump| r.double_in_place(pump));
        push_signed_addition(&mut sink, anchor, &mut r, qx, qy, nqy, digit);
    }

    let q1 = mul_by_char(*q);
    let q2 = mul_by_char(q1).negate();
    sink.tail(anchor, |pump| {
        r.add_in_place(f2_from(q1.x), f2_from(q1.y), pump)
    });
    sink.tail(anchor, |pump| {
        r.add_in_place(f2_from(q2.x), f2_from(q2.y), pump)
    });
    sink.finish()
}

/// Scale one prepared schedule into a static sink in optimal-Ate order.
#[cfg(all(
    feature = "std",
    any(
        test,
        all(
            any(narsil_avx512_ifma, narsil_x86_runtime_ifma),
            not(feature = "force-portable")
        )
    )
))]
#[inline(always)]
pub(crate) fn stream_prepared_line_coeffs<S: MillerLineSink>(
    p: &G1Affine,
    q: &G2Prepared,
    sink: S,
) -> S::Output {
    let mut next = 0usize;
    const FIRST_DIGIT: usize = ATE_LOOP_COUNT.len() - 2;

    let line = q.coeffs[next];
    next += 1;
    let (mut sink, anchor) = sink.start(p, || line);
    if ATE_LOOP_COUNT[FIRST_DIGIT] != 0 {
        let line = q.coeffs[next];
        next += 1;
        sink.addition(anchor, move |_| line);
    }

    for &digit in ATE_LOOP_COUNT[..FIRST_DIGIT].iter().rev() {
        let line = q.coeffs[next];
        next += 1;
        sink.doubling(anchor, move |_| line);
        if digit != 0 {
            let line = q.coeffs[next];
            next += 1;
            sink.addition(anchor, move |_| line);
        }
    }

    for _ in 0..2 {
        let line = q.coeffs[next];
        next += 1;
        sink.tail(anchor, move |_| line);
    }
    debug_assert_eq!(next, G2_PREPARED_COEFFS);
    sink.finish()
}

/// Validate a typed G2 point and build its reusable Miller line schedule.
///
/// Returns `None` for an off-curve point or a point outside the prime-order
/// subgroup. The identity is valid and produces an identity schedule.
#[cfg(feature = "std")]
pub fn prepare_g2(q: &G2Affine) -> Option<G2Prepared> {
    if q.is_identity() {
        return Some(G2Prepared {
            coeffs: [ELL_ZERO; G2_PREPARED_COEFFS],
            identity: true,
        });
    }
    if !q.is_on_curve() || !q.is_in_correct_subgroup_assuming_on_curve() {
        return None;
    }
    Some(prepare(q))
}

/// Build the Miller line schedule without curve or subgroup validation.
///
/// A schedule from an invalid point proves nothing, so every verifying path
/// must use [`prepare_g2`]. The benchmark harness uses this entry to time
/// schedule construction only.
#[cfg(feature = "std")]
pub fn prepare_g2_unchecked(q: &G2Affine) -> G2Prepared {
    if q.is_identity() {
        return G2Prepared {
            coeffs: [ELL_ZERO; G2_PREPARED_COEFFS],
            identity: true,
        };
    }
    prepare(q)
}

/// Evaluate one Miller loop using a previously validated G2 schedule.
///
/// Preparation is deliberately outside this call, making the boundary useful
/// for fixed verifier keys. Supported x86 hosts use coefficient-parallel IFMA.
/// Other hosts use scalar replay.
#[cfg(feature = "std")]
pub fn miller_loop_prepared(p: &G1Affine, q: &G2Prepared) -> Fp12 {
    if p.is_identity() || q.identity {
        return Fp12::ONE;
    }

    // The capability proves CPU and OS support before the IFMA entry.
    #[cfg(all(
        any(narsil_avx512_ifma, narsil_x86_runtime_ifma),
        not(feature = "force-portable")
    ))]
    if let Some(ifma) = crate::x86_runtime::avx512_ifma() {
        return crate::miller_coeff_ifma::miller_loop_prepared(ifma, p, q);
    }

    let ate = &ATE_LOOP_COUNT;
    let n = ate.len();
    let scalars = G1LineScalars::new(p);
    let mut next = 0;
    let mut lines = |digit: i8| -> Lines {
        let d = scale_doubling(&q.coeffs[next], &scalars);
        next += 1;
        let a = if digit == 0 {
            None
        } else {
            let scaled = scale_addition(&q.coeffs[next], &scalars);
            next += 1;
            Some(scaled)
        };
        (d, a)
    };

    // Match the live single-pair schedule exactly, but replay line coefficients
    // instead of regenerating them with G2 arithmetic.
    let (first_d, first_a) = lines(ate[n - 2]);
    let mut f = line_value(&first_d);
    if let Some(a) = &first_a {
        f.mul_by_034_assign(a.0, a.1, a.2);
    }

    let mut pending = lines(ate[n - 3]);
    for i in (2..n - 1).rev() {
        f.square_in_place();
        let following = lines(ate[i - 2]);
        apply_lines(&mut f, &pending);
        pending = following;
    }

    f.square_in_place();
    apply_lines(&mut f, &pending);
    let first_tail = scale_addition(&q.coeffs[next], &scalars);
    let second_tail = scale_addition(&q.coeffs[next + 1], &scalars);
    next += 2;
    f.mul_by_034_assign(first_tail.0, first_tail.1, first_tail.2);
    f.mul_by_034_assign(second_tail.0, second_tail.1, second_tail.2);
    debug_assert_eq!(next, G2_PREPARED_COEFFS);
    f
}

/// Transparent prepared-line cache for the typed single-pair Miller entry.
///
/// This is deliberately separate from the byte facade's cache. The facade
/// owns validation and exact wire-key semantics. This cache receives typed
/// affine points and keys the exact internal coordinate representation.
/// Neither cache stores a pairing result or a validation verdict.
#[cfg(all(feature = "std", not(feature = "force-portable")))]
mod typed_prepared_cache {
    use alloc::{boxed::Box, vec::Vec};
    use core::cell::RefCell;

    use super::{F2, Fp12, G1Affine, G2Affine, G2Prepared, f2_from, miller_loop_prepared, prepare};

    /// Four complete schedules plus exact affine keys stay below 70 KiB per
    /// thread. Hits move to the back. A full miss evicts the front entry.
    pub(super) const CAPACITY: usize = 4;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(super) struct AffineKey {
        x: F2,
        y: F2,
    }

    impl AffineKey {
        #[inline]
        pub(super) fn from_point(q: &G2Affine) -> Self {
            debug_assert!(!q.is_identity());
            Self {
                x: f2_from(q.x),
                y: f2_from(q.y),
            }
        }
    }

    struct Entry {
        key: AffineKey,
        prepared: G2Prepared,
    }

    const _: () = assert!(core::mem::size_of::<Entry>() * CAPACITY < 70 * 1024);

    #[derive(Default)]
    struct Cache {
        entries: Vec<Box<Entry>>,
        hits: usize,
        misses: usize,
        evictions: usize,
    }

    std::thread_local! {
        static CACHE: RefCell<Cache> = RefCell::new(Cache::default());
    }

    pub(super) fn miller_loop(p: &G1Affine, q: &G2Affine) -> Fp12 {
        debug_assert!(!p.is_identity() && !q.is_identity());
        let key = AffineKey::from_point(q);
        CACHE.with(|cell| {
            let mut cache = cell.borrow_mut();
            if let Some(index) = cache.entries.iter().position(|entry| entry.key == key) {
                cache.hits += 1;
                let entry = cache.entries.remove(index);
                let result = miller_loop_prepared(p, &entry.prepared);
                cache.entries.push(entry);
                return result;
            }

            cache.misses += 1;
            let prepared = prepare(q);
            let result = miller_loop_prepared(p, &prepared);
            if cache.entries.len() == CAPACITY {
                cache.entries.remove(0);
                cache.evictions += 1;
            }
            cache.entries.push(Box::new(Entry { key, prepared }));
            result
        })
    }

    #[cfg(test)]
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub(super) struct Snapshot {
        pub keys: Vec<AffineKey>,
        pub hits: usize,
        pub misses: usize,
        pub evictions: usize,
    }

    #[cfg(test)]
    pub(super) fn reset() {
        CACHE.with(|cell| *cell.borrow_mut() = Cache::default());
    }

    #[cfg(test)]
    pub(super) fn snapshot() -> Snapshot {
        CACHE.with(|cell| {
            let cache = cell.borrow();
            Snapshot {
                keys: cache.entries.iter().map(|entry| entry.key).collect(),
                hits: cache.hits,
                misses: cache.misses,
                evictions: cache.evictions,
            }
        })
    }
}

/// One fixed or live G2 source for a shared multi-Miller accumulator.
#[cfg(feature = "std")]
#[derive(Clone, Copy)]
pub(crate) enum G2Miller<'a> {
    /// Generate line coefficients online.
    Live(&'a G2Affine),
    /// Replay a previously validated fixed-G2 schedule.
    Prepared(&'a G2Prepared),
}

#[cfg(feature = "std")]
enum MixedSource<'a> {
    Live,
    Prepared {
        schedule: &'a G2Prepared,
        next: usize,
    },
}

#[cfg(feature = "std")]
struct MixedState<'a> {
    p: &'a G1Affine,
    scalars: G1LineScalars,
    source: MixedSource<'a>,
    r: G2Hom,
    qx: F2,
    qy: F2,
    nqy: F2,
    q: G2Affine,
}

#[cfg(feature = "std")]
impl MixedState<'_> {
    #[inline(always)]
    fn doubling_line(&mut self, pump: &mut dyn MillerPump) -> EllCoeff {
        match &mut self.source {
            MixedSource::Live => self.r.double_in_place(pump),
            MixedSource::Prepared { schedule, next } => {
                let coeffs = schedule.coeffs[*next];
                *next += 1;
                coeffs
            }
        }
    }

    #[inline(always)]
    fn addition_line(&mut self, digit: i8, pump: &mut dyn MillerPump) -> EllCoeff {
        debug_assert!(digit == -1 || digit == 1);
        match &mut self.source {
            MixedSource::Live => {
                self.r
                    .add_in_place(self.qx, if digit == 1 { self.qy } else { self.nqy }, pump)
            }
            MixedSource::Prepared { schedule, next } => {
                let coeffs = schedule.coeffs[*next];
                *next += 1;
                coeffs
            }
        }
    }

    #[inline(always)]
    fn tail_lines(&mut self) -> (EllCoeff, EllCoeff) {
        match &mut self.source {
            MixedSource::Live => {
                let q1 = mul_by_char(self.q);
                let q2 = mul_by_char(q1).negate();
                (
                    self.r
                        .add_in_place(f2_from(q1.x), f2_from(q1.y), &mut NoPump),
                    self.r
                        .add_in_place(f2_from(q2.x), f2_from(q2.y), &mut NoPump),
                )
            }
            MixedSource::Prepared { schedule, next } => {
                let first = schedule.coeffs[*next];
                let second = schedule.coeffs[*next + 1];
                *next += 2;
                (first, second)
            }
        }
    }

    #[inline(always)]
    fn assert_consumed(&self) {
        if let MixedSource::Prepared { next, .. } = self.source {
            debug_assert_eq!(next, G2_PREPARED_COEFFS);
        }
    }
}

#[cfg(feature = "std")]
fn mixed_states<'a>(pairs: &[(&'a G1Affine, G2Miller<'a>)]) -> Vec<MixedState<'a>> {
    let mut states = Vec::with_capacity(pairs.len());
    for &(p, source) in pairs {
        if p.is_identity() {
            continue;
        }
        let state = match source {
            G2Miller::Live(q) => {
                if q.is_identity() {
                    continue;
                }
                let qx = f2_from(q.x);
                let qy = f2_from(q.y);
                MixedState {
                    p,
                    scalars: G1LineScalars::new(p),
                    source: MixedSource::Live,
                    r: G2Hom::from_affine(q),
                    qx,
                    qy,
                    nqy: f2_neg(qy),
                    q: *q,
                }
            }
            G2Miller::Prepared(schedule) => {
                if schedule.identity {
                    continue;
                }
                MixedState {
                    p,
                    scalars: G1LineScalars::new(p),
                    source: MixedSource::Prepared { schedule, next: 0 },
                    r: G2Hom {
                        x: F2_ZERO,
                        y: F2_ZERO,
                        z: f2_one(),
                    },
                    qx: F2_ZERO,
                    qy: F2_ZERO,
                    nqy: F2_ZERO,
                    q: G2Affine::identity(),
                }
            }
        };
        states.push(state);
    }
    states
}

#[cfg(feature = "std")]
struct ScalarSink;

#[cfg(feature = "std")]
struct ActiveScalarSink(Fp12);

#[cfg(feature = "std")]
impl MillerLineSink for ScalarSink {
    type Output = Fp12;
    type Active = ActiveScalarSink;

    #[inline(always)]
    fn start(self, p: &G1Affine, line: impl FnOnce() -> EllCoeff) -> (Self::Active, G1LineScalars) {
        let scalars = G1LineScalars::new(p);
        (
            ActiveScalarSink(line_value(&scale_doubling(&line(), &scalars))),
            scalars,
        )
    }
}

#[cfg(feature = "std")]
impl ActiveMillerLineSink for ActiveScalarSink {
    type Output = Fp12;
    type Anchor = G1LineScalars;

    #[inline(always)]
    fn anchor(&mut self, p: &G1Affine) -> G1LineScalars {
        G1LineScalars::new(p)
    }

    #[inline(always)]
    fn doubling(
        &mut self,
        anchor: G1LineScalars,
        line: impl FnOnce(&mut dyn MillerPump) -> EllCoeff,
    ) {
        self.0.square_in_place();
        self.merged_doubling(anchor, line);
    }

    #[inline(always)]
    fn merged_doubling(
        &mut self,
        anchor: G1LineScalars,
        line: impl FnOnce(&mut dyn MillerPump) -> EllCoeff,
    ) {
        let (c0, c3, c4) = scale_doubling(&line(&mut NoPump), &anchor);
        self.0.mul_by_034_assign(c0, c3, c4);
    }

    #[inline(always)]
    fn addition(
        &mut self,
        anchor: G1LineScalars,
        line: impl FnOnce(&mut dyn MillerPump) -> EllCoeff,
    ) {
        let (c0, c3, c4) = scale_addition(&line(&mut NoPump), &anchor);
        self.0.mul_by_034_assign(c0, c3, c4);
    }

    #[inline(always)]
    fn tail(&mut self, anchor: G1LineScalars, line: impl FnOnce(&mut dyn MillerPump) -> EllCoeff) {
        self.addition(anchor, line);
    }

    #[inline(always)]
    fn finish(&mut self) -> Self::Output {
        self.0
    }
}

#[cfg(feature = "std")]
#[inline(always)]
pub(crate) fn stream_mixed_line_coeffs<S: MillerLineSink>(
    pairs: &[(&G1Affine, G2Miller<'_>)],
    sink: S,
) -> Option<S::Output> {
    let mut states = mixed_states(pairs);
    let (first, rest) = states.split_first_mut()?;
    let (mut sink, first_anchor) = sink.start(first.p, || first.doubling_line(&mut NoPump));

    let mut anchors = Vec::with_capacity(rest.len() + 1);
    anchors.push(first_anchor);
    for state in rest {
        let anchor = sink.anchor(state.p);
        anchors.push(anchor);
        sink.merged_doubling(anchor, |pump| state.doubling_line(pump));
    }

    const FIRST_DIGIT: usize = ATE_LOOP_COUNT.len() - 2;
    let top_digit = ATE_LOOP_COUNT[FIRST_DIGIT];
    if top_digit != 0 {
        for (state, &anchor) in states.iter_mut().zip(&anchors) {
            sink.addition(anchor, |pump| state.addition_line(top_digit, pump));
        }
    }

    for &digit in ATE_LOOP_COUNT[..FIRST_DIGIT].iter().rev() {
        for (index, (state, &anchor)) in states.iter_mut().zip(&anchors).enumerate() {
            if index == 0 {
                sink.doubling(anchor, |pump| state.doubling_line(pump));
            } else {
                sink.merged_doubling(anchor, |pump| state.doubling_line(pump));
            }
        }
        if digit != 0 {
            for (state, &anchor) in states.iter_mut().zip(&anchors) {
                sink.addition(anchor, |pump| state.addition_line(digit, pump));
            }
        }
    }

    for (state, &anchor) in states.iter_mut().zip(&anchors) {
        let (first, second) = state.tail_lines();
        sink.tail(anchor, |_| first);
        sink.tail(anchor, |_| second);
        state.assert_consumed();
    }
    Some(sink.finish())
}

/// Mixed prepared/live multi-Miller loop with exactly one shared Fp12 state.
///
/// Prepared terms remove only fixed-G2 line generation. All terms still share
/// every Miller square and the caller still performs exactly one final
/// exponentiation. This is deliberately not implemented as independent Miller
/// loops followed by a product, which would duplicate the serial squarings.
#[cfg(feature = "std")]
pub(crate) fn multi_miller_loop_mixed(pairs: &[(&G1Affine, G2Miller<'_>)]) -> Fp12 {
    #[cfg(all(
        any(narsil_avx512_ifma, narsil_x86_runtime_ifma),
        not(feature = "force-portable")
    ))]
    if let Some(ifma) = crate::x86_runtime::avx512_ifma() {
        return crate::miller_coeff_ifma::multi_miller_loop_mixed(ifma, pairs);
    }

    stream_mixed_line_coeffs(pairs, ScalarSink).unwrap_or(Fp12::ONE)
}

/// Return `Miller(pairs) * root^(-lambda)` without a separate square chain.
#[cfg(feature = "std")]
pub(crate) fn miller_residue_relation_mixed(
    pairs: &[(&G1Affine, G2Miller<'_>)],
    root: Fp12,
    inverse: Fp12,
) -> Fp12 {
    let mut states = mixed_states(pairs);
    let mut f = inverse;
    let ate = &ATE_LOOP_COUNT;
    for i in (1..ate.len()).rev() {
        f.square_in_place();
        for state in &mut states {
            let coeffs = state.doubling_line(&mut NoPump);
            ell_doubling(&mut f, &coeffs, &state.scalars);
        }
        match ate[i - 1] {
            1 => {
                f *= inverse;
                for state in &mut states {
                    let coeffs = state.addition_line(1, &mut NoPump);
                    ell_addition(&mut f, &coeffs, &state.scalars);
                }
            }
            -1 => {
                f *= root;
                for state in &mut states {
                    let coeffs = state.addition_line(-1, &mut NoPump);
                    ell_addition(&mut f, &coeffs, &state.scalars);
                }
            }
            _ => {}
        }
    }

    f *= inverse.frobenius_map();
    f *= root.frobenius_map_squared();
    f *= inverse.frobenius_map_cubed();
    for state in &mut states {
        let (first, second) = state.tail_lines();
        ell_addition(&mut f, &first, &state.scalars);
        ell_addition(&mut f, &second, &state.scalars);
        state.assert_consumed();
    }
    f
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fr::Fr;
    use crate::g2::G2Projective;

    #[derive(Default, Debug, PartialEq, Eq)]
    struct ScheduleCounts {
        doublings: usize,
        signed_additions: usize,
        tails: usize,
        squares: usize,
    }

    struct ScheduleSink;

    impl MillerLineSink for ScheduleSink {
        type Output = ScheduleCounts;
        type Active = ScheduleCounts;

        fn start(self, _p: &G1Affine, line: impl FnOnce() -> EllCoeff) -> (Self::Active, ()) {
            let _ = line();
            (
                ScheduleCounts {
                    doublings: 1,
                    ..ScheduleCounts::default()
                },
                (),
            )
        }
    }

    impl ActiveMillerLineSink for ScheduleCounts {
        type Output = Self;
        type Anchor = ();

        fn anchor(&mut self, _p: &G1Affine) -> Self::Anchor {}

        fn doubling(&mut self, _anchor: (), line: impl FnOnce(&mut dyn MillerPump) -> EllCoeff) {
            let _ = line(&mut NoPump);
            self.doublings += 1;
            self.squares += 1;
        }

        fn merged_doubling(
            &mut self,
            _anchor: (),
            line: impl FnOnce(&mut dyn MillerPump) -> EllCoeff,
        ) {
            let _ = line(&mut NoPump);
            self.doublings += 1;
        }

        fn addition(&mut self, _anchor: (), line: impl FnOnce(&mut dyn MillerPump) -> EllCoeff) {
            let _ = line(&mut NoPump);
            self.signed_additions += 1;
        }

        fn tail(&mut self, _anchor: (), line: impl FnOnce(&mut dyn MillerPump) -> EllCoeff) {
            let _ = line(&mut NoPump);
            self.tails += 1;
        }

        fn finish(&mut self) -> Self::Output {
            core::mem::take(self)
        }
    }

    #[test]
    fn streamed_schedule_has_fixed_operation_counts() {
        let counts =
            stream_line_coeffs(&G1Affine::generator(), &G2Affine::generator(), ScheduleSink);
        assert_eq!(
            counts,
            ScheduleCounts {
                doublings: 64,
                signed_additions: 21,
                tails: 2,
                squares: 63,
            }
        );
    }

    #[cfg(feature = "std")]
    #[test]
    fn unchecked_prepare_matches_validated_prepare_schedule() {
        let q = G2Projective::from(G2Affine::generator())
            .mul_scalar(Fr::from_u64(7))
            .to_affine();
        let validated = prepare_g2(&q).unwrap();
        let unchecked = prepare_g2_unchecked(&q);
        assert_eq!(unchecked.identity, validated.identity);
        assert_eq!(unchecked.coeffs, validated.coeffs);
        assert_eq!(unchecked.last_line_limbs(), validated.last_line_limbs());
        assert!(prepare_g2_unchecked(&G2Affine::identity()).identity);
    }

    #[cfg(feature = "std")]
    #[test]
    fn prepared_stream_has_live_operation_counts() {
        let p = G1Affine::generator();
        let q = G2Affine::generator();
        let prepared = prepare(&q);
        let live = stream_line_coeffs(&p, &q, ScheduleSink);
        let replay = stream_prepared_line_coeffs(&p, &prepared, ScheduleSink);
        assert_eq!(replay, live);
    }

    #[cfg(all(feature = "std", not(feature = "force-portable")))]
    fn typed_miller_bypasses_cache() -> bool {
        #[cfg(any(narsil_avx512_ifma, narsil_x86_runtime_ifma))]
        {
            crate::batch8::ifma_available()
        }
        #[cfg(not(any(narsil_avx512_ifma, narsil_x86_runtime_ifma)))]
        {
            false
        }
    }

    /// psi acts on G2 as multiplication by p. For BN curves p - r = 6x^2
    /// (p = 36x^4+36x^3+24x^2+6x+1, r = 36x^4+36x^3+18x^2+6x+1), so
    /// psi(Q) = [6x^2 mod r]Q, recomputed here from the seed alone.
    #[test]
    fn mul_by_char_is_multiplication_by_p_on_g2() {
        let x = u128::from(crate::consts::BN_X);
        let s = 6 * x * x;
        let scalar = Fr::from_raw([s as u64, (s >> 64) as u64, 0, 0]);
        let q = G2Affine::generator();
        let want = G2Projective::from(q).mul_scalar(scalar).to_affine();
        assert_eq!(mul_by_char(q), want);
    }

    /// psi output stays on the twist y^2 = x^3 + 3/xi with the derived twist_b.
    #[test]
    fn mul_by_char_lands_on_twist_curve() {
        let b = f2_to(twist_b_f2());
        let q = mul_by_char(G2Affine::generator());
        assert_eq!(q.y.square(), q.x.square() * q.x + b);
    }

    #[cfg(feature = "std")]
    #[test]
    fn mixed_prepared_and_live_schedules_match_the_live_shared_accumulator() {
        use crate::g1::G1Projective;

        let g1 = G1Projective::generator();
        let g2 = G2Projective::from(G2Affine::generator());
        let ps: Vec<_> = (1..=5)
            .map(|scalar| g1.mul_scalar(Fr::from_u64(scalar + 1)).to_affine())
            .collect();
        let qs: Vec<_> = (1..=5)
            .map(|scalar| g2.mul_scalar(Fr::from_u64(scalar * 3 + 1)).to_affine())
            .collect();
        let prepared: Vec<_> = qs.iter().map(prepare).collect();

        for count in 1..=5 {
            let live: Vec<_> = ps[..count].iter().zip(&qs[..count]).collect();
            let expected = multi_miller_loop(&live);
            for prepared_mask in 0..(1usize << count) {
                let mixed: Vec<_> = (0..count)
                    .map(|index| {
                        let source = if prepared_mask & (1 << index) == 0 {
                            G2Miller::Live(&qs[index])
                        } else {
                            G2Miller::Prepared(&prepared[index])
                        };
                        (&ps[index], source)
                    })
                    .collect();
                assert_eq!(multi_miller_loop_mixed(&mixed), expected);
            }
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn residue_recurrence_matches_direct_short_exponent() {
        let p = G1Affine::generator();
        let q = G2Affine::generator();
        let prepared = prepare(&q);
        let pairs = [(&p, G2Miller::Prepared(&prepared))];
        let raw = multi_miller_loop_mixed(&pairs);
        let root = raw;
        let inverse = root.invert().unwrap();
        let short = u128::from(crate::consts::BN_X) * 6 + 2;
        let bits: Vec<_> = (0..128).map(|bit| short & (1 << bit) != 0).collect();
        let expected = raw
            * inverse.pow_bits(&bits)
            * inverse.frobenius_map()
            * root.frobenius_map_squared()
            * inverse.frobenius_map_cubed();
        assert_eq!(
            miller_residue_relation_mixed(&pairs, root, inverse),
            expected
        );
    }

    #[cfg(all(feature = "std", not(feature = "force-portable")))]
    #[test]
    fn typed_prepared_cache_distinguishes_cold_miss_and_warm_hit() {
        use crate::g1::G1Projective;

        typed_prepared_cache::reset();
        let g1 = G1Projective::generator();
        let g2 = G2Projective::from(G2Affine::generator());
        let p1 = g1.mul_scalar(Fr::from_u64(17)).to_affine();
        let p2 = g1.mul_scalar(Fr::from_u64(19)).to_affine();
        let q1 = g2.mul_scalar(Fr::from_u64(23)).to_affine();
        let q2 = g2.mul_scalar(Fr::from_u64(29)).to_affine();

        let want_cold = miller_loop_live_nonidentity(&p1, &q1);
        assert_eq!(miller_loop(&p1, &q1), want_cold);
        let cold = typed_prepared_cache::snapshot();
        if typed_miller_bypasses_cache() {
            assert_eq!((cold.hits, cold.misses, cold.evictions), (0, 0, 0));
            assert!(cold.keys.is_empty());
        } else {
            assert_eq!((cold.hits, cold.misses, cold.evictions), (0, 1, 0));
            assert_eq!(
                cold.keys,
                vec![typed_prepared_cache::AffineKey::from_point(&q1)]
            );
        }

        // Same exact G2, different live G1: only the line schedule can have
        // been reused. A cached Miller result would fail this equality.
        let want_warm = miller_loop_live_nonidentity(&p2, &q1);
        assert_eq!(miller_loop(&p2, &q1), want_warm);
        let warm = typed_prepared_cache::snapshot();
        if typed_miller_bypasses_cache() {
            assert_eq!((warm.hits, warm.misses, warm.evictions), (0, 0, 0));
            assert!(warm.keys.is_empty());
        } else {
            assert_eq!((warm.hits, warm.misses, warm.evictions), (1, 1, 0));
        }

        assert_eq!(
            miller_loop(&p1, &q2),
            miller_loop_live_nonidentity(&p1, &q2)
        );
        let distinct = typed_prepared_cache::snapshot();
        if typed_miller_bypasses_cache() {
            assert_eq!((distinct.hits, distinct.misses), (0, 0));
            assert!(distinct.keys.is_empty());
        } else {
            assert_eq!((distinct.hits, distinct.misses), (1, 2));
            assert_eq!(
                distinct.keys,
                vec![
                    typed_prepared_cache::AffineKey::from_point(&q1),
                    typed_prepared_cache::AffineKey::from_point(&q2),
                ]
            );
        }

        assert_eq!(miller_loop(&G1Affine::identity(), &q1), Fp12::ONE);
        assert_eq!(miller_loop(&p1, &G2Affine::identity()), Fp12::ONE);
        assert_eq!(typed_prepared_cache::snapshot(), distinct);
    }

    /// Benchmark-shape control. A harness that rotates a pool larger than
    /// [`typed_prepared_cache::CAPACITY`] must never serve a timed call from
    /// the cache: every call pays live G2 line generation.
    #[cfg(all(feature = "std", not(feature = "force-portable")))]
    #[test]
    fn rotating_more_keys_than_the_cache_holds_never_warms_it() {
        use crate::g1::G1Projective;

        const POOL: u64 = 16;
        assert!(
            POOL as usize > typed_prepared_cache::CAPACITY,
            "the control needs a pool the cache cannot hold"
        );
        typed_prepared_cache::reset();
        let g1 = G1Projective::generator();
        let g2 = G2Projective::from(G2Affine::generator());
        let points: Vec<_> = (1..=POOL)
            .map(|scalar| g2.mul_scalar(Fr::from_u64(scalar)).to_affine())
            .collect();
        let p = g1.mul_scalar(Fr::from_u64(7)).to_affine();

        // Four passes over the pool, exactly a four-round benchmark.
        for _ in 0..4 {
            for q in &points {
                assert_eq!(miller_loop(&p, q), miller_loop_live_nonidentity(&p, q));
            }
        }

        let state = typed_prepared_cache::snapshot();
        assert_eq!(state.hits, 0, "a rotated pool must not hit the cache");
        if typed_miller_bypasses_cache() {
            assert_eq!((state.misses, state.evictions), (0, 0));
            assert!(state.keys.is_empty());
        } else {
            // Every access misses; the first CAPACITY inserts fill the cache
            // and each later miss evicts exactly one entry.
            assert_eq!(state.misses as u64, 4 * POOL);
            assert_eq!(
                state.evictions as u64,
                4 * POOL - typed_prepared_cache::CAPACITY as u64
            );
        }
    }

    #[cfg(all(feature = "std", not(feature = "force-portable")))]
    #[test]
    fn typed_prepared_cache_has_exact_four_entry_lru() {
        use crate::g1::G1Affine;

        typed_prepared_cache::reset();
        let p = G1Affine::generator();
        let g2 = G2Projective::from(G2Affine::generator());
        let points: Vec<_> = (1..=typed_prepared_cache::CAPACITY as u64 + 1)
            .map(|scalar| g2.mul_scalar(Fr::from_u64(scalar)).to_affine())
            .collect();
        for q in &points[..typed_prepared_cache::CAPACITY] {
            miller_loop(&p, q);
        }
        miller_loop(&p, &points[0]);
        miller_loop(&p, &points[typed_prepared_cache::CAPACITY]);

        let state = typed_prepared_cache::snapshot();
        if typed_miller_bypasses_cache() {
            assert_eq!((state.hits, state.misses, state.evictions), (0, 0, 0));
            assert!(state.keys.is_empty());
        } else {
            assert_eq!((state.hits, state.misses, state.evictions), (1, 5, 1));
            assert_eq!(
                state.keys,
                [&points[2], &points[3], &points[0], &points[4]]
                    .map(typed_prepared_cache::AffineKey::from_point)
                    .to_vec()
            );
        }
    }

    #[cfg(all(feature = "std", not(feature = "force-portable")))]
    #[test]
    fn typed_prepared_cache_is_thread_local() {
        use crate::g1::G1Affine;

        typed_prepared_cache::reset();
        let p = G1Affine::generator();
        let g2 = G2Projective::from(G2Affine::generator());
        let q1 = g2.mul_scalar(Fr::from_u64(31)).to_affine();
        let q2 = g2.mul_scalar(Fr::from_u64(37)).to_affine();
        miller_loop(&p, &q1);
        let outer = typed_prepared_cache::snapshot();

        let inner = std::thread::spawn(move || {
            assert!(typed_prepared_cache::snapshot().keys.is_empty());
            miller_loop(&p, &q2);
            miller_loop(&p, &q2);
            typed_prepared_cache::snapshot()
        })
        .join()
        .unwrap();
        if typed_miller_bypasses_cache() {
            assert_eq!((inner.hits, inner.misses, inner.evictions), (0, 0, 0));
            assert!(inner.keys.is_empty());
        } else {
            assert_eq!((inner.hits, inner.misses, inner.evictions), (1, 1, 0));
            assert_eq!(
                inner.keys,
                vec![typed_prepared_cache::AffineKey::from_point(&q2)]
            );
        }
        assert_eq!(typed_prepared_cache::snapshot(), outer);
    }
}
