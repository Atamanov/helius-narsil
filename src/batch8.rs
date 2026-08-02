use crate::consts::ATE_LOOP_COUNT;
use crate::fp::Fp;
use crate::fp::avx512ifma::FpVec8;
#[cfg(all(test, helius_avx512_ifma))]
use crate::fp12::X_W4;
use crate::pairing::miller::{mul_by_char, twist_b_f2};
use crate::{Fp2, Fp6, Fp12, G1Affine, G2Affine};

/// `c0 + c1 u`, `u^2 = -1`, eight lanes.
#[derive(Clone, Copy)]
pub(crate) struct Fp2x8 {
    c0: FpVec8,
    c1: FpVec8,
}

impl Fp2x8 {
    pub(crate) fn load(v: &[Fp2; 8]) -> Self {
        Self {
            c0: unsafe { FpVec8::load(&core::array::from_fn(|i| v[i].c0)) },
            c1: unsafe { FpVec8::load(&core::array::from_fn(|i| v[i].c1)) },
        }
    }

    pub(crate) fn store(&self) -> [Fp2; 8] {
        let (c0, c1) = (self.c0.store(), self.c1.store());
        core::array::from_fn(|i| Fp2::new(c0[i], c1[i]))
    }

    #[inline(always)]
    fn add(&self, o: &Self) -> Self {
        Self {
            c0: self.c0.add(&o.c0),
            c1: self.c1.add(&o.c1),
        }
    }

    #[inline(always)]
    fn sub(&self, o: &Self) -> Self {
        Self {
            c0: self.c0.sub(&o.c0),
            c1: self.c1.sub(&o.c1),
        }
    }

    #[inline(always)]
    fn negate(&self) -> Self {
        Self {
            c0: self.c0.negate(),
            c1: self.c1.negate(),
        }
    }

    #[inline(always)]
    fn double(&self) -> Self {
        Self {
            c0: self.c0.double(),
            c1: self.c1.double(),
        }
    }

    #[inline(always)]
    fn triple(&self) -> Self {
        Self {
            c0: self.c0.triple(),
            c1: self.c1.triple(),
        }
    }

    #[inline(always)]
    fn half(&self) -> Self {
        Self {
            c0: self.c0.half(),
            c1: self.c1.half(),
        }
    }

    #[inline(always)]
    #[cfg(helius_avx512_ifma)]
    fn conjugate(&self) -> Self {
        Self {
            c0: self.c0,
            c1: self.c1.negate(),
        }
    }

    #[inline(always)]
    #[cfg(helius_avx512_ifma)]
    fn zero_mask(&self) -> u8 {
        self.c0.is_zero_mask() & self.c1.is_zero_mask()
    }

    #[inline(always)]
    #[cfg(helius_avx512_ifma)]
    fn select(mask: u8, if_true: &Self, if_false: &Self) -> Self {
        Self {
            c0: FpVec8::select(mask, &if_true.c0, &if_false.c0),
            c1: FpVec8::select(mask, &if_true.c1, &if_false.c1),
        }
    }

    /// `c0 = a0*b0 - a1*b1`, `c1 = a0*b1 + a1*b0`. Two folded reductions.
    #[inline(always)]
    fn mul(&self, o: &Self) -> Self {
        let nb1 = o.c1.negate();
        Self {
            c0: FpVec8::sos_mac_2(&[self.c0, self.c1], &[o.c0, nb1]),
            c1: FpVec8::sos_mac_2(&[self.c0, self.c1], &[o.c1, o.c0]),
        }
    }

    /// `c0 = a0^2 - a1^2`, `c1 = 2 a0 a1`.
    #[inline(always)]
    fn square(&self) -> Self {
        let na1 = self.c1.negate();
        Self {
            c0: FpVec8::sos_mac_2(&[self.c0, self.c1], &[self.c0, na1]),
            c1: FpVec8::sos_mac_2(&[self.c0, self.c1], &[self.c1, self.c0]),
        }
    }

    /// Multiply by `xi = 9 + u`: `c0 = 9 a0 - a1`, `c1 = 9 a1 + a0`.
    #[inline(always)]
    fn mul_by_nonresidue(&self) -> Self {
        let nine = |x: &FpVec8| x.triple().triple();
        Self {
            c0: nine(&self.c0).sub(&self.c1),
            c1: nine(&self.c1).add(&self.c0),
        }
    }

    #[inline(always)]
    fn mul_by_fp(&self, f: &FpVec8) -> Self {
        Self {
            c0: self.c0.mul(f),
            c1: self.c1.mul(f),
        }
    }
}

/// Evaluate the twist equation for eight already-canonical affine points.
/// Identity lanes pass. Bit `i` corresponds to lane `i`.
#[cfg(all(test, helius_avx512_ifma))]
pub(crate) fn g2_on_curve_mask8(points: &[G2Affine; 8]) -> u8 {
    let x = Fp2x8::load(&core::array::from_fn(|i| points[i].x));
    let y = Fp2x8::load(&core::array::from_fn(|i| points[i].y));
    g2_on_curve_mask8_resident(points, &x, &y)
}

#[cfg(helius_avx512_ifma)]
fn g2_on_curve_mask8_resident(points: &[G2Affine; 8], x: &Fp2x8, y: &Fp2x8) -> u8 {
    let b = Fp2x8::broadcast({
        let value = twist_b_f2();
        Fp2::new(Fp(value.0), Fp(value.1))
    });
    let residual = y.square().sub(&x.square().mul(x).add(&b)).store();
    points.iter().enumerate().fold(0_u8, |mask, (lane, point)| {
        mask | (u8::from(point.infinity || residual[lane].is_zero()) << lane)
    })
}

/// Compute both validation masks from one initial canonical-to-radix52 load.
#[cfg(helius_avx512_ifma)]
pub(crate) fn g2_validation_masks8(points: &[G2Affine; 8]) -> (u8, u8) {
    let x = Fp2x8::load(&core::array::from_fn(|i| points[i].x));
    let y = Fp2x8::load(&core::array::from_fn(|i| points[i].y));
    let on_curve = g2_on_curve_mask8_resident(points, &x, &y);
    let subgroup = g2_subgroup_mask8_resident(points, on_curve, x, y);
    (on_curve, subgroup)
}

/// One Fp2 sum of three products `l0*r0 + l1*r1 + l2*r2` as two `sos_mac(6)`
/// (real folds the imag*imag terms with a negated operand. Imag is the
/// all-positive cross terms). One folded reduction per Fp output component --
/// the scalar `sosd6` lazy reduction, lane-parallel. The tower's fused muls and
/// the sparse `mul_by_034` route through this.
#[inline(always)]
fn fp2_sum3(l0: &Fp2x8, r0: &Fp2x8, l1: &Fp2x8, r1: &Fp2x8, l2: &Fp2x8, r2: &Fp2x8) -> Fp2x8 {
    let a = [l0.c0, l0.c1, l1.c0, l1.c1, l2.c0, l2.c1];
    Fp2x8 {
        c0: FpVec8::sos_mac_6(
            &a,
            &[
                r0.c0,
                r0.c1.negate(),
                r1.c0,
                r1.c1.negate(),
                r2.c0,
                r2.c1.negate(),
            ],
        ),
        c1: FpVec8::sos_mac_6(&a, &[r0.c1, r0.c0, r1.c1, r1.c0, r2.c1, r2.c0]),
    }
}

/// `c0 + c1 v + c2 v^2`, `v^3 = xi`, eight lanes.
#[derive(Clone, Copy)]
pub(crate) struct Fp6x8 {
    c0: Fp2x8,
    c1: Fp2x8,
    c2: Fp2x8,
}

impl Fp6x8 {
    #[cfg(all(test, helius_avx512_ifma))]
    fn load(v: &[Fp6; 8]) -> Self {
        Self {
            c0: Fp2x8::load(&core::array::from_fn(|i| v[i].c0)),
            c1: Fp2x8::load(&core::array::from_fn(|i| v[i].c1)),
            c2: Fp2x8::load(&core::array::from_fn(|i| v[i].c2)),
        }
    }

    fn store(&self) -> [Fp6; 8] {
        let (c0, c1, c2) = (self.c0.store(), self.c1.store(), self.c2.store());
        core::array::from_fn(|i| Fp6::new(c0[i], c1[i], c2[i]))
    }

    #[inline(always)]
    fn add(&self, o: &Self) -> Self {
        Self {
            c0: self.c0.add(&o.c0),
            c1: self.c1.add(&o.c1),
            c2: self.c2.add(&o.c2),
        }
    }

    #[inline(always)]
    fn sub(&self, o: &Self) -> Self {
        Self {
            c0: self.c0.sub(&o.c0),
            c1: self.c1.sub(&o.c1),
            c2: self.c2.sub(&o.c2),
        }
    }

    #[inline(always)]
    #[cfg(all(test, helius_avx512_ifma))]
    fn negate(&self) -> Self {
        Self {
            c0: self.c0.negate(),
            c1: self.c1.negate(),
            c2: self.c2.negate(),
        }
    }

    /// `v * (c0 + c1 v + c2 v^2) = xi c2 + c0 v + c1 v^2`.
    #[inline(always)]
    fn mul_by_nonresidue(&self) -> Self {
        Self {
            c0: self.c2.mul_by_nonresidue(),
            c1: self.c0,
            c2: self.c1,
        }
    }

    /// Fused sum-of-products (matches the scalar `Fp6` product, `xi` folded):
    /// `c0 = a0*b0 + a1*(xi b2) + a2*(xi b1)`, `c1 = a0*b1 + a1*b0 + a2*(xi b2)`,
    /// `c2 = a0*b2 + a1*b1 + a2*b0`.
    ///
    /// Each output Fp component is one `sos_mac` over six Fp products (the
    /// scalar `sosd6` lazy reduction, now lane-parallel), so an Fp6 mul costs
    /// six `sos_mac(6)` instead of the eighteen `sos_mac(2)` a composed Fp2
    /// product tree would pay -- a third of the Montgomery reductions.
    #[inline(always)]
    fn mul(&self, o: &Self) -> Self {
        let x1 = o.c1.mul_by_nonresidue();
        let x2 = o.c2.mul_by_nonresidue();
        let (a0, a1, a2) = (&self.c0, &self.c1, &self.c2);
        Self {
            c0: fp2_sum3(a0, &o.c0, a1, &x2, a2, &x1),
            c1: fp2_sum3(a0, &o.c1, a1, &o.c0, a2, &x2),
            c2: fp2_sum3(a0, &o.c2, a1, &o.c1, a2, &o.c0),
        }
    }

    /// Devegili cubic squaring (matches the scalar `Fp6::square`).
    #[inline(always)]
    #[cfg(all(test, helius_avx512_ifma))]
    fn square(&self) -> Self {
        let s0 = self.c0.square();
        let s1 = self.c0.mul(&self.c1).double();
        let s2 = self.c0.sub(&self.c1).add(&self.c2).square();
        let s3 = self.c1.mul(&self.c2).double();
        let s4 = self.c2.square();
        Self {
            c0: s0.add(&s3.mul_by_nonresidue()),
            c1: s1.add(&s4.mul_by_nonresidue()),
            c2: s1.add(&s2).add(&s3).sub(&s0).sub(&s4),
        }
    }
}

/// `c0 + c1 w`, `w^2 = v`, eight lanes.
#[derive(Clone, Copy)]
pub(crate) struct Fp12x8 {
    c0: Fp6x8,
    c1: Fp6x8,
}

impl Fp12x8 {
    #[cfg(all(test, helius_avx512_ifma))]
    pub(crate) fn load(v: &[Fp12; 8]) -> Self {
        Self {
            c0: Fp6x8::load(&core::array::from_fn(|i| v[i].c0)),
            c1: Fp6x8::load(&core::array::from_fn(|i| v[i].c1)),
        }
    }

    pub(crate) fn store(&self) -> [Fp12; 8] {
        let (c0, c1) = (self.c0.store(), self.c1.store());
        core::array::from_fn(|i| Fp12::new(c0[i], c1[i]))
    }

    #[inline(always)]
    #[cfg(all(test, helius_avx512_ifma))]
    fn conjugate(&self) -> Self {
        Self {
            c0: self.c0,
            c1: self.c1.negate(),
        }
    }

    /// Karatsuba over Fp6: `c0 = a0 b0 + v a1 b1`,
    /// `c1 = (a0+a1)(b0+b1) - a0 b0 - a1 b1`.
    #[inline(always)]
    fn mul(&self, o: &Self) -> Self {
        let t0 = self.c0.mul(&o.c0);
        let t1 = self.c1.mul(&o.c1);
        let cross = self.c0.add(&self.c1).mul(&o.c0.add(&o.c1));
        Self {
            c0: t0.add(&t1.mul_by_nonresidue()),
            c1: cross.sub(&t0).sub(&t1),
        }
    }

    /// Complex squaring: `c0 = (a0+a1)(a0 + v a1) - a0 a1 - v a0 a1`,
    /// `c1 = 2 a0 a1`.
    #[inline(always)]
    fn square(&self) -> Self {
        let ab = self.c0.mul(&self.c1);
        let c0 = self
            .c0
            .add(&self.c1)
            .mul(&self.c0.add(&self.c1.mul_by_nonresidue()))
            .sub(&ab)
            .sub(&ab.mul_by_nonresidue());
        Self {
            c0,
            c1: ab.add(&ab),
        }
    }
}

// --- Constructors and constants the Miller loop needs ---------------------

impl Fp2x8 {
    fn zero() -> Self {
        Self {
            c0: unsafe { FpVec8::zero() },
            c1: unsafe { FpVec8::zero() },
        }
    }
    fn one() -> Self {
        Self {
            c0: unsafe { FpVec8::load(&[Fp::ONE; 8]) },
            c1: unsafe { FpVec8::zero() },
        }
    }
    /// Broadcast one Fp2 constant into all eight lanes.
    fn broadcast(c: Fp2) -> Self {
        Self::load(&[c; 8])
    }

    #[inline(always)]
    fn select_lanes(&self, active_mask: u8, fallback: &Self) -> Self {
        Self {
            c0: self.c0.select_lanes(active_mask, &fallback.c0),
            c1: self.c1.select_lanes(active_mask, &fallback.c1),
        }
    }
}

impl Fp6x8 {
    fn zero() -> Self {
        Self {
            c0: Fp2x8::zero(),
            c1: Fp2x8::zero(),
            c2: Fp2x8::zero(),
        }
    }

    #[inline(always)]
    fn select_lanes(&self, active_mask: u8, fallback: &Self) -> Self {
        Self {
            c0: self.c0.select_lanes(active_mask, &fallback.c0),
            c1: self.c1.select_lanes(active_mask, &fallback.c1),
            c2: self.c2.select_lanes(active_mask, &fallback.c2),
        }
    }
}

impl Fp12x8 {
    fn one() -> Self {
        Self {
            c0: Fp6x8 {
                c0: Fp2x8::one(),
                c1: Fp2x8::zero(),
                c2: Fp2x8::zero(),
            },
            c1: Fp6x8::zero(),
        }
    }

    /// Keep active Miller outputs and replace every inactive lane with the
    /// multiplicative identity before this vector participates in a product.
    #[inline(always)]
    fn neutralize_inactive(&self, active_mask: u8) -> Self {
        let one = Self::one();
        Self {
            c0: self.c0.select_lanes(active_mask, &one.c0),
            c1: self.c1.select_lanes(active_mask, &one.c1),
        }
    }

    /// Sparse multiply by `c0 + c3 w + c4 w v` (mirrors the scalar
    /// `mul_by_034`): with `self = a + b w`, `a = (a0,a1,a2)`, `b = (b0,b1,b2)`,
    /// `xic3 = xi c3`, `xic4 = xi c4`.
    #[inline(always)]
    fn mul_by_034(&self, c0: &Fp2x8, c3: &Fp2x8, c4: &Fp2x8) -> Self {
        let xic3 = c3.mul_by_nonresidue();
        let xic4 = c4.mul_by_nonresidue();
        let (a0, a1, a2) = (&self.c0.c0, &self.c0.c1, &self.c0.c2);
        let (b0, b1, b2) = (&self.c1.c0, &self.c1.c1, &self.c1.c2);
        Self {
            c0: Fp6x8 {
                c0: fp2_sum3(a0, c0, b1, &xic4, b2, &xic3),
                c1: fp2_sum3(a1, c0, b0, c3, b2, &xic4),
                c2: fp2_sum3(a2, c0, b0, c4, b1, c3),
            },
            c1: Fp6x8 {
                c0: fp2_sum3(a0, c3, a2, &xic4, b0, c0),
                c1: fp2_sum3(a0, c4, a1, c3, b1, c0),
                c2: fp2_sum3(a1, c4, a2, c3, b2, c0),
            },
        }
    }
}

/// Eight homogeneous projective points on the twist, one per pairing lane.
#[derive(Clone, Copy)]
struct G2x8 {
    x: Fp2x8,
    y: Fp2x8,
    z: Fp2x8,
}

/// One line's three sparse Fp2 coefficients, eight lanes.
type EllCoeff8 = (Fp2x8, Fp2x8, Fp2x8);

impl G2x8 {
    #[cfg(helius_avx512_ifma)]
    fn identity() -> Self {
        Self {
            x: Fp2x8::zero(),
            y: Fp2x8::one(),
            z: Fp2x8::zero(),
        }
    }

    #[inline(always)]
    #[cfg(helius_avx512_ifma)]
    fn select(mask: u8, if_true: &Self, if_false: &Self) -> Self {
        Self {
            x: Fp2x8::select(mask, &if_true.x, &if_false.x),
            y: Fp2x8::select(mask, &if_true.y, &if_false.y),
            z: Fp2x8::select(mask, &if_true.z, &if_false.z),
        }
    }
    fn from_affine(qx: Fp2x8, qy: Fp2x8) -> Self {
        Self {
            x: qx,
            y: qy,
            z: Fp2x8::one(),
        }
    }

    /// Doubling with D-type line coefficients `(-h, 3j, i)`.
    fn double_in_place(&mut self, twist_b: &Fp2x8) -> EllCoeff8 {
        let a = self.x.mul(&self.y).half();
        let b = self.y.square();
        let c = self.z.square();
        let e = twist_b.mul(&c.triple());
        let f = e.triple();
        let g = b.add(&f).half();
        let h = self.y.add(&self.z).square().sub(&b.add(&c));
        let i = e.sub(&b);
        let j = self.x.square();
        let e_sq = e.square();
        let new_x = a.mul(&b.sub(&f));
        let new_y = g.square().sub(&e_sq.triple());
        let new_z = b.mul(&h);
        self.x = new_x;
        self.y = new_y;
        self.z = new_z;
        (h.negate(), j.triple(), i)
    }

    /// Mixed addition of an affine `q`, line coefficients `(lambda, -theta, j)`.
    fn add_in_place(&mut self, qx: &Fp2x8, qy: &Fp2x8) -> EllCoeff8 {
        let theta = self.y.sub(&qy.mul(&self.z));
        let lambda = self.x.sub(&qx.mul(&self.z));
        let c = theta.square();
        let d = lambda.square();
        let e = lambda.mul(&d);
        let f = self.z.mul(&c);
        let g = self.x.mul(&d);
        let h = e.add(&f).sub(&g.double());
        let new_y = theta.mul(&g.sub(&h)).sub(&e.mul(&self.y));
        self.x = lambda.mul(&h);
        self.y = new_y;
        self.z = self.z.mul(&e);
        let jj = theta.mul(qx).sub(&lambda.mul(qy));
        (lambda, theta.negate(), jj)
    }

    #[inline(always)]
    #[cfg(helius_avx512_ifma)]
    fn double_point(self) -> Self {
        let x2 = self.x.square();
        let mut y2 = self.y.square();
        let xy = self.x.mul(&y2).double().double();
        y2 = y2.square();
        let t = x2.triple();
        let x3 = t.square().sub(&xy.double());
        let y3 = t.mul(&xy.sub(&x3)).sub(&y2.double().double().double());
        Self {
            x: x3,
            y: y3,
            z: self.y.mul(&self.z).double(),
        }
    }

    #[inline(always)]
    #[cfg(helius_avx512_ifma)]
    fn add_mixed_point(self, qx: &Fp2x8, qy: &Fp2x8) -> Self {
        let z2 = self.z.square();
        let h = qx.mul(&z2).sub(&self.x);
        let r = qy.mul(&z2.mul(&self.z)).sub(&self.y);
        let h2 = h.square();
        let h3 = h2.mul(&h);
        let u1h2 = self.x.mul(&h2);
        let x3 = r.square().sub(&u1h2.double()).sub(&h3);
        let y3 = r.mul(&u1h2.sub(&x3)).sub(&self.y.mul(&h3));
        let general = Self {
            x: x3,
            y: y3,
            z: self.z.mul(&h),
        };
        let h_zero = h.zero_mask();
        let r_zero = r.zero_mask();
        let doubled = self.double_point();
        let same = Self::select(h_zero & r_zero, &doubled, &general);
        let exceptional = Self::select(h_zero & !r_zero, &Self::identity(), &same);
        Self::select(
            self.z.zero_mask(),
            &Self::from_affine(*qx, *qy),
            &exceptional,
        )
    }

    #[inline(always)]
    #[cfg(helius_avx512_ifma)]
    fn add_point(self, other: Self) -> Self {
        let z1z1 = self.z.square();
        let z2z2 = other.z.square();
        let u1 = self.x.mul(&z2z2);
        let u2 = other.x.mul(&z1z1);
        let s1 = self.y.mul(&other.z).mul(&z2z2);
        let s2 = other.y.mul(&self.z).mul(&z1z1);
        let h = u2.sub(&u1);
        let i = h.double().square();
        let j = h.mul(&i);
        let r = s2.sub(&s1).double();
        let v = u1.mul(&i);
        let x3 = r.square().sub(&j).sub(&v.double());
        let y3 = r.mul(&v.sub(&x3)).sub(&s1.mul(&j).double());
        let z3 = self.z.add(&other.z).square().sub(&z1z1).sub(&z2z2).mul(&h);
        let general = Self {
            x: x3,
            y: y3,
            z: z3,
        };
        let equal_x = u2.sub(&u1).zero_mask();
        let equal_y = s2.sub(&s1).zero_mask();
        let doubled = self.double_point();
        let same = Self::select(equal_x & equal_y, &doubled, &general);
        let exceptional = Self::select(equal_x & !equal_y, &Self::identity(), &same);
        let with_other_identity = Self::select(other.z.zero_mask(), &self, &exceptional);
        Self::select(self.z.zero_mask(), &other, &with_other_identity)
    }

    #[inline(always)]
    #[cfg(helius_avx512_ifma)]
    fn psi(self) -> Self {
        let px = Fp2x8::broadcast(crate::g2::PSI_X);
        let py = Fp2x8::broadcast(crate::g2::PSI_Y);
        Self {
            x: self.x.conjugate().mul(&px),
            y: self.y.conjugate().mul(&py),
            z: self.z.conjugate(),
        }
    }

    #[inline(always)]
    #[cfg(helius_avx512_ifma)]
    fn psi_squared(self) -> Self {
        let coefficient = unsafe { FpVec8::load(&[crate::g2::PSI2_X; 8]) };
        Self {
            x: self.x.mul_by_fp(&coefficient),
            y: self.y.negate(),
            z: self.z,
        }
    }

    #[cfg(helius_avx512_ifma)]
    fn mul_by_bn_x(qx: &Fp2x8, qy: &Fp2x8) -> Self {
        const NAF: [i8; 62] = crate::consts::derive::bn_x_naf_msb(crate::consts::BN_X);
        let nqy = qy.negate();
        let mut acc = Self::from_affine(*qx, *qy);
        for digit in NAF {
            acc = acc.double_point();
            if digit == 1 {
                acc = acc.add_mixed_point(qx, qy);
            } else if digit == -1 {
                acc = acc.add_mixed_point(qx, &nqy);
            }
        }
        acc
    }
}

/// El Housni-Guillevic-Piellard subgroup predicate in eight independent lanes.
/// `on_curve` identifies usable lanes. Identities pass without entering the
/// projective schedule. False/off-curve lanes are replaced by a valid lane.
#[cfg(all(test, helius_avx512_ifma))]
pub(crate) fn g2_subgroup_mask8(points: &[G2Affine; 8], on_curve: u8) -> u8 {
    let qx = Fp2x8::load(&core::array::from_fn(|lane| points[lane].x));
    let qy = Fp2x8::load(&core::array::from_fn(|lane| points[lane].y));
    g2_subgroup_mask8_resident(points, on_curve, qx, qy)
}

#[cfg(helius_avx512_ifma)]
fn g2_subgroup_mask8_resident(
    points: &[G2Affine; 8],
    on_curve: u8,
    resident_x: Fp2x8,
    resident_y: Fp2x8,
) -> u8 {
    let identity_mask = points.iter().enumerate().fold(0_u8, |mask, (lane, point)| {
        mask | (u8::from(point.infinity && on_curve & (1 << lane) != 0) << lane)
    });
    let Some(fallback) = points.iter().enumerate().find_map(|(lane, point)| {
        (!point.infinity && on_curve & (1 << lane) != 0).then_some(*point)
    }) else {
        return identity_mask;
    };
    let safe: [G2Affine; 8] = core::array::from_fn(|lane| {
        if !points[lane].infinity && on_curve & (1 << lane) != 0 {
            points[lane]
        } else {
            fallback
        }
    });
    let all_active = points
        .iter()
        .enumerate()
        .all(|(lane, point)| !point.infinity && on_curve & (1 << lane) != 0);
    let qx = if all_active {
        resident_x
    } else {
        Fp2x8::load(&core::array::from_fn(|lane| safe[lane].x))
    };
    let qy = if all_active {
        resident_y
    } else {
        Fp2x8::load(&core::array::from_fn(|lane| safe[lane].y))
    };
    let s = G2x8::mul_by_bn_x(&qx, &qy);
    let lhs = s
        .add_mixed_point(&qx, &qy)
        .add_point(s.psi())
        .add_point(s.psi_squared());
    let rhs = s.double_point().psi_squared().psi();
    let z1z1 = lhs.z.square();
    let z2z2 = rhs.z.square();
    let dx = lhs.x.mul(&z2z2).sub(&rhs.x.mul(&z1z1)).store();
    let dy = lhs
        .y
        .mul(&z2z2)
        .mul(&rhs.z)
        .sub(&rhs.y.mul(&z1z1).mul(&lhs.z))
        .store();
    safe.iter()
        .enumerate()
        .fold(identity_mask, |mask, (lane, _)| {
            let active = on_curve & (1 << lane) != 0 && !points[lane].infinity;
            mask | (u8::from(active && dx[lane].is_zero() && dy[lane].is_zero()) << lane)
        })
}

/// Sparse Fp12 line materialized directly (first Miller iteration).
fn line_value8(coeffs: &EllCoeff8, px: &FpVec8, py: &FpVec8) -> Fp12x8 {
    let c0 = coeffs.0.mul_by_fp(py);
    let c3 = coeffs.1.mul_by_fp(px);
    let c4 = coeffs.2;
    Fp12x8 {
        c0: Fp6x8 {
            c0,
            c1: Fp2x8::zero(),
            c2: Fp2x8::zero(),
        },
        c1: Fp6x8 {
            c0: c3,
            c1: c4,
            c2: Fp2x8::zero(),
        },
    }
}

#[inline(always)]
fn ell8(f: &mut Fp12x8, coeffs: &EllCoeff8, px: &FpVec8, py: &FpVec8) {
    let c0 = coeffs.0.mul_by_fp(py);
    let c3 = coeffs.1.mul_by_fp(px);
    let c4 = coeffs.2;
    *f = f.mul_by_034(&c0, &c3, &c4);
}

/// 8-wide optimal-ate Miller loop: eight independent pairings, one per lane,
/// all following the same ate schedule in lockstep. Inputs must be
/// non-identity (the batch-verify caller filters identities).
pub(crate) fn miller8(p: &[G1Affine; 8], q: &[G2Affine; 8]) -> Fp12x8 {
    let px = unsafe { FpVec8::load(&core::array::from_fn(|i| p[i].x)) };
    let py = unsafe { FpVec8::load(&core::array::from_fn(|i| p[i].y)) };
    let qx = Fp2x8::load(&core::array::from_fn(|i| q[i].x));
    let qy = Fp2x8::load(&core::array::from_fn(|i| q[i].y));
    let nqy = qy.negate();

    let twist_b = {
        let tb = twist_b_f2();
        Fp2x8::broadcast(Fp2::new(Fp(tb.0), Fp(tb.1)))
    };
    // Frobenius endpoints, computed scalar-side per lane.
    let q1: [G2Affine; 8] = core::array::from_fn(|i| mul_by_char(q[i]));
    let q2: [G2Affine; 8] = core::array::from_fn(|i| mul_by_char(q1[i]).negate());
    let q1x = Fp2x8::load(&core::array::from_fn(|i| q1[i].x));
    let q1y = Fp2x8::load(&core::array::from_fn(|i| q1[i].y));
    let q2x = Fp2x8::load(&core::array::from_fn(|i| q2[i].x));
    let q2y = Fp2x8::load(&core::array::from_fn(|i| q2[i].y));

    let mut r = G2x8::from_affine(qx, qy);
    let mut f = Fp12x8::one();
    let ate = &ATE_LOOP_COUNT;
    for i in (1..ate.len()).rev() {
        if i != ate.len() - 1 {
            f = f.square();
        }
        let coeffs = r.double_in_place(&twist_b);
        if i == ate.len() - 1 {
            f = line_value8(&coeffs, &px, &py);
        } else {
            ell8(&mut f, &coeffs, &px, &py);
        }
        match ate[i - 1] {
            1 => {
                let coeffs = r.add_in_place(&qx, &qy);
                ell8(&mut f, &coeffs, &px, &py);
            }
            -1 => {
                let coeffs = r.add_in_place(&qx, &nqy);
                ell8(&mut f, &coeffs, &px, &py);
            }
            _ => {}
        }
    }
    let coeffs = r.add_in_place(&q1x, &q1y);
    ell8(&mut f, &coeffs, &px, &py);
    let coeffs = r.add_in_place(&q2x, &q2y);
    ell8(&mut f, &coeffs, &px, &py);
    f
}

/// Run the eight-lane Miller schedule, then make inactive lanes neutral for a
/// later lane-wise or horizontal product. Inputs in inactive lanes must still
/// be valid non-identity points because masking happens after the schedule.
#[inline]
fn miller8_masked(p: &[G1Affine; 8], q: &[G2Affine; 8], active_mask: u8) -> Fp12x8 {
    miller8(p, q).neutralize_inactive(active_mask)
}

// --- Layer 3: final exponentiation ----------------------------------------

/// SoS Fp4 square `(r0 + r1 y)^2`, `y^2 = xi`: `t0 = r0^2 + xi r1^2`,
/// `t1 = 2 r0 r1`.
#[inline(always)]
#[cfg(all(test, helius_avx512_ifma))]
fn fp4_square8(r0: &Fp2x8, r1: &Fp2x8) -> (Fp2x8, Fp2x8) {
    let x = r1.mul_by_nonresidue();
    let t0 = r0.square().add(&x.mul(r1));
    let t1 = r0.double().mul(r1);
    (t0, t1)
}

#[cfg(all(test, helius_avx512_ifma))]
impl Fp12x8 {
    /// Apply a scalar Fp12 map per lane (store, map, load). Used for the
    /// constant-heavy Frobenius maps and the one-time inversion, which are cold
    /// (a handful of calls per pairing, none in the pow_x hot loop).
    #[inline]
    fn map_scalar(&self, f: impl Fn(Fp12) -> Fp12) -> Self {
        let s = self.store();
        Self::load(&core::array::from_fn(|i| f(s[i])))
    }

    /// Granger-Scott cyclotomic square (valid after the easy part). Mirrors the
    /// scalar `cyclotomic_square`. `fp4_square8` replaces the scalar SoS kernels.
    #[inline(always)]
    fn cyclotomic_square(&self) -> Self {
        let r0 = self.c0.c0;
        let r4 = self.c0.c1;
        let r3 = self.c0.c2;
        let r2 = self.c1.c0;
        let r1 = self.c1.c1;
        let r5 = self.c1.c2;
        let (t0, t1) = fp4_square8(&r0, &r1);
        let (t2, t3) = fp4_square8(&r2, &r3);
        let (t4, t5) = fp4_square8(&r4, &r5);
        let z0 = t0.sub(&r0).double().add(&t0); // 3 t0 - 2 r0
        let z1 = t1.add(&r1).double().add(&t1); // 3 t1 + 2 r1
        let tmp = t5.mul_by_nonresidue();
        let z2 = r2.add(&tmp).double().add(&tmp); // 2 r2 + 3 xi t5
        let z3 = t4.sub(&r3).double().add(&t4);
        let z4 = t2.sub(&r4).double().add(&t2);
        let z5 = r5.add(&t3).double().add(&t3);
        Self {
            c0: Fp6x8 {
                c0: z0,
                c1: z4,
                c2: z3,
            },
            c1: Fp6x8 {
                c0: z2,
                c1: z1,
                c2: z5,
            },
        }
    }

    /// `self^x`, `x = BN_X`, over signed 4-bit windows of cyclotomic squares
    /// (mirrors the scalar `pow_x`).
    fn pow_x(&self) -> Self {
        let x2 = self.cyclotomic_square();
        let x3 = x2.mul(self);
        let x5 = x3.mul(&x2);
        let x7 = x5.mul(&x2);
        let tab = [*self, x3, x5, x7];
        let mut acc = tab[0];
        for &d in X_W4.iter().rev().skip(1) {
            acc = acc.cyclotomic_square();
            if d != 0 {
                let t = tab[(d.unsigned_abs() / 2) as usize];
                acc = acc.mul(&if d < 0 { t.conjugate() } else { t });
            }
        }
        acc
    }

    /// `f^{-x}`. X is positive for BN_SNARK1, so this is the unitary inverse of
    /// `f^x`.
    #[inline]
    fn exp_by_neg_x(&self) -> Self {
        self.pow_x().conjugate()
    }
}

/// 8-wide final exponentiation `f^{(p^12-1)/r}`: easy part then the
/// Fuentes-Castaneda hard part, mirroring the scalar `final_exponentiation`.
/// The inversion and Frobenius maps run scalar-side per lane (cold path).
#[cfg(all(test, helius_avx512_ifma))]
pub(crate) fn final_exp8(f: &Fp12x8) -> Fp12x8 {
    // Easy: f^{(p^6-1)(p^2+1)}.
    let f1 = f.conjugate();
    let f2 = f.map_scalar(|x| x.invert().expect("final exp input is nonzero"));
    let t = f1.mul(&f2); // f^{p^6-1}
    let r = t.map_scalar(Fp12::frobenius_map_squared).mul(&t);

    // Hard part (Fuentes-Castaneda).
    let y0 = r.exp_by_neg_x();
    let y1 = y0.cyclotomic_square();
    let y2 = y1.cyclotomic_square();
    let y3 = y2.mul(&y1);
    let y4 = y3.exp_by_neg_x();
    let y5 = y4.cyclotomic_square();
    let y6 = y5.exp_by_neg_x();
    let y3c = y3.conjugate();
    let y6c = y6.conjugate();
    let y7 = y6c.mul(&y4);
    let y8 = y7.mul(&y3c);
    let y9 = y8.mul(&y1);
    let y10 = y8.mul(&y4);
    let y11 = y10.mul(&r);
    let y12 = y9.map_scalar(Fp12::frobenius_map);
    let y13 = y12.mul(&y11);
    let y8f = y8.map_scalar(Fp12::frobenius_map_squared);
    let y14 = y8f.mul(&y13);
    let y15 = r.conjugate().mul(&y9).map_scalar(Fp12::frobenius_map_cubed);
    y15.mul(&y14)
}

/// 8-wide full pairing: `miller8` then `final_exp8`.
#[cfg(all(test, helius_avx512_ifma))]
pub(crate) fn pairing8(p: &[G1Affine; 8], q: &[G2Affine; 8]) -> Fp12x8 {
    final_exp8(&miller8(p, q))
}

#[target_feature(enable = "avx512f,avx512ifma")]
pub(crate) unsafe fn multi_pairing8_ifma(pairs: &[(G1Affine, G2Affine)]) -> Fp12 {
    use crate::pairing::final_exponentiation;
    use alloc::vec::Vec;

    let live: Vec<(G1Affine, G2Affine)> = pairs
        .iter()
        .copied()
        .filter(|(p, q)| !p.is_identity() && !q.is_identity())
        .collect();
    if live.is_empty() {
        return Fp12::ONE;
    }
    final_exponentiation(&miller_product8_ifma(&live))
}

/// Product of per-pair Miller values through eight-lane chunks. Pairs must be
/// non-identity. Full reduction on every path keeps the result equal to the
/// shared-accumulator Miller product.
#[target_feature(enable = "avx512f,avx512ifma")]
fn miller_product8_ifma(live: &[(G1Affine, G2Affine)]) -> Fp12 {
    use crate::pairing::miller::miller_loop;

    // Lane i accumulates pair i from every complete chunk. Keeping that product
    // in radix-52 SoA form removes one store plus seven scalar Fp12 products per
    // additional chunk. There is no safe cross-lane permutation primitive in
    // FpVec8 yet, so reduce the eight lanes once after all complete chunks.
    let mut vector_acc: Option<Fp12x8> = None;
    let mut chunks = live.chunks_exact(8);
    for chunk in &mut chunks {
        let p = core::array::from_fn(|i| chunk[i].0);
        let q = core::array::from_fn(|i| chunk[i].1);
        let product = miller8(&p, &q);
        vector_acc = Some(match vector_acc {
            Some(acc) => acc.mul(&product),
            None => product,
        });
    }

    let remainder = chunks.remainder();
    if remainder.len() >= 2 {
        // Miller formulas require non-identity affine inputs in every lane.
        // Duplicate the final live lane as harmless work, then replace every
        // inactive output by one while still in SoA form. This also preserves
        // inactive lanes of any preceding complete chunk when the masked
        // product is multiplied into `vector_acc`. Two or more remaining pairs
        // favor one masked pass because sequential single-pair loops pay one
        // full dependent chain each.
        let final_live = remainder.len() - 1;
        let p = core::array::from_fn(|i| remainder[i.min(final_live)].0);
        let q = core::array::from_fn(|i| remainder[i.min(final_live)].1);
        let active_mask = ((1_u16 << remainder.len()) - 1) as u8;
        let product = miller8_masked(&p, &q, active_mask);
        vector_acc = Some(match vector_acc {
            Some(acc) => acc.mul(&product),
            None => product,
        });
    }

    let mut acc = Fp12::ONE;
    if let Some(product) = vector_acc {
        for lane in product.store() {
            acc *= lane;
        }
    }
    if remainder.len() == 1 {
        let (p, q) = &remainder[0];
        acc *= miller_loop(p, q);
    }
    acc
}

/// Miller product of non-identity pairs through the masked eight-lane engine,
/// or `None` when runtime CPUID and XCR0 state deny IFMA.
pub(crate) fn multi_miller_product8(live: &[(G1Affine, G2Affine)]) -> Option<Fp12> {
    if !ifma_available() {
        return None;
    }
    // SAFETY: `ifma_available` uses the shared capability boundary, which
    // checks both CPUID and XCR0 before returning true.
    Some(unsafe { miller_product8_ifma(live) })
}

/// Safely dispatch batch multi-pairing to IFMA when CPU and OS state allow it.
///
/// Generic x86-64 binaries use runtime dispatch. Typed vector operations need
/// IFMA in the compile-time target.
pub fn multi_pairing8(pairs: &[(G1Affine, G2Affine)]) -> Fp12 {
    if ifma_available() {
        // SAFETY: `ifma_available` uses the shared capability boundary, which
        // checks both CPUID and XCR0 before returning true.
        unsafe { multi_pairing8_ifma(pairs) }
    } else {
        let refs: alloc::vec::Vec<_> = pairs.iter().map(|(p, q)| (p, q)).collect();
        crate::pairing::multi_pairing(&refs)
    }
}

/// Whether runtime CPUID and OS register state authorize the isolated IFMA
/// batch entry. Callers check this before bypassing warm prepared routes.
#[inline]
pub(crate) fn ifma_available() -> bool {
    crate::x86_runtime::avx512_ifma().is_some()
}

// These white-box tests call IFMA helpers directly and therefore require an
// IFMA target baseline. Generic x86 test binaries cover the runtime-isolated
// path through the public byte facade instead.
#[cfg(all(test, helius_avx512_ifma))]
mod tests {
    use super::*;

    use crate::fp::Fp;

    fn residue(state: &mut u64) -> Fp {
        use crate::limb;
        let mut v = [0u64; 4];
        for limb in &mut v {
            *state ^= *state << 13;
            *state ^= *state >> 7;
            *state ^= *state << 17;
            *limb = *state;
        }
        while limb::gte(&v, &crate::consts::P) {
            v = limb::sub_noborrow(&v, &crate::consts::P);
        }
        Fp(v)
    }

    fn rfp2(s: &mut u64) -> Fp2 {
        Fp2::new(residue(s), residue(s))
    }
    fn rfp6(s: &mut u64) -> Fp6 {
        Fp6::new(rfp2(s), rfp2(s), rfp2(s))
    }
    fn rfp12(s: &mut u64) -> Fp12 {
        Fp12::new(rfp6(s), rfp6(s))
    }

    #[test]
    fn fp2x8_matches_scalar() {
        use crate::{Fr, G2Projective};
        let generator = G2Projective::from(G2Affine::generator());
        let mut points: [G2Affine; 8] = core::array::from_fn(|lane| {
            generator
                .mul_scalar(Fr::from_u64(lane as u64 + 1))
                .to_affine()
        });
        points[2].y.c0 += Fp::ONE;
        points[6] = G2Affine::identity();
        let expected = points.iter().enumerate().fold(0_u8, |mask, (lane, point)| {
            mask | (u8::from(point.is_on_curve()) << lane)
        });
        assert_eq!(g2_on_curve_mask8(&points), expected);

        let twist = {
            let value = twist_b_f2();
            Fp2::new(Fp(value.0), Fp(value.1))
        };
        let non_subgroup = (1_u64..)
            .find_map(|value| {
                let x = Fp2::new(Fp::from_u64(value), Fp::from_u64(value.wrapping_mul(17)));
                (x.square() * x + twist).sqrt().and_then(|y| {
                    let point = G2Affine {
                        x,
                        y,
                        infinity: false,
                    };
                    (!point.is_in_correct_subgroup_assuming_on_curve()).then_some(point)
                })
            })
            .expect("deterministic on-curve non-subgroup point");
        points[2] = non_subgroup;
        let curve_mask = g2_on_curve_mask8(&points);
        let expected_subgroup = points.iter().enumerate().fold(0_u8, |mask, (lane, point)| {
            mask | (u8::from(
                point.is_on_curve() && point.is_in_correct_subgroup_assuming_on_curve(),
            ) << lane)
        });
        assert_eq!(g2_subgroup_mask8(&points, curve_mask), expected_subgroup);
        assert_eq!(expected_subgroup & (1 << 2), 0);

        let mut s = 0x1111_2222_3333_4444u64;
        let inv2 = Fp::from_u64(2).invert().unwrap();
        for _ in 0..4096 {
            let a: [Fp2; 8] = core::array::from_fn(|_| rfp2(&mut s));
            let b: [Fp2; 8] = core::array::from_fn(|_| rfp2(&mut s));
            let (va, vb) = (Fp2x8::load(&a), Fp2x8::load(&b));
            let mul = va.mul(&vb).store();
            let sqr = va.square().store();
            let half = va.half().store();
            let nr = va.mul_by_nonresidue().store();
            let add = va.add(&vb).store();
            let sub = va.sub(&vb).store();
            for i in 0..8 {
                assert_eq!(mul[i], a[i] * b[i], "mul lane {i}");
                assert_eq!(sqr[i], a[i].square(), "sqr lane {i}");
                assert_eq!(
                    half[i],
                    Fp2::new(a[i].c0 * inv2, a[i].c1 * inv2),
                    "half lane {i}"
                );
                assert_eq!(nr[i], a[i].mul_by_nonresidue(), "nonresidue lane {i}");
                assert_eq!(add[i], a[i] + b[i], "add lane {i}");
                assert_eq!(sub[i], a[i] - b[i], "sub lane {i}");
            }
        }
    }

    #[test]
    fn fp6x8_matches_scalar() {
        let mut s = 0x5555_6666_7777_8888u64;
        for _ in 0..4096 {
            let a: [Fp6; 8] = core::array::from_fn(|_| rfp6(&mut s));
            let b: [Fp6; 8] = core::array::from_fn(|_| rfp6(&mut s));
            let (va, vb) = (Fp6x8::load(&a), Fp6x8::load(&b));
            let mul = va.mul(&vb).store();
            let sqr = va.square().store();
            let nr = va.mul_by_nonresidue().store();
            for i in 0..8 {
                assert_eq!(mul[i], a[i] * b[i], "mul lane {i}");
                assert_eq!(sqr[i], a[i].square(), "sqr lane {i}");
                assert_eq!(nr[i], a[i].mul_by_nonresidue(), "nonresidue lane {i}");
            }
        }
    }

    #[test]
    fn fp12x8_matches_scalar() {
        let mut s = 0x9999_aaaa_bbbb_ccccu64;
        for _ in 0..4096 {
            let a: [Fp12; 8] = core::array::from_fn(|_| rfp12(&mut s));
            let b: [Fp12; 8] = core::array::from_fn(|_| rfp12(&mut s));
            let (va, vb) = (Fp12x8::load(&a), Fp12x8::load(&b));
            let mul = va.mul(&vb).store();
            let sqr = va.square().store();
            let conj = va.conjugate().store();
            for i in 0..8 {
                assert_eq!(mul[i], a[i] * b[i], "mul lane {i}");
                assert_eq!(sqr[i], a[i].square(), "sqr lane {i}");
                assert_eq!(conj[i], a[i].conjugate(), "conj lane {i}");
            }
        }
    }

    #[test]
    fn miller8_matches_scalar() {
        use crate::Fr;
        use crate::pairing::miller::miller_loop;
        use crate::{G1Projective, G2Projective};

        let g1 = G1Projective::generator();
        let g2 = G2Projective::from(G2Affine::generator());
        let p: [G1Affine; 8] = core::array::from_fn(|i| {
            g1.mul_scalar(Fr::from_raw([1 + i as u64, 3, 0, 0]))
                .to_affine()
        });
        let q: [G2Affine; 8] = core::array::from_fn(|i| {
            g2.mul_scalar(Fr::from_raw([2 + i as u64, 5, 0, 0]))
                .to_affine()
        });
        let got = miller8(&p, &q).store();
        for i in 0..8 {
            assert_eq!(got[i], miller_loop(&p[i], &q[i]), "lane {i}");
        }
    }

    fn test_pairs() -> ([G1Affine; 8], [G2Affine; 8]) {
        use crate::Fr;
        use crate::{G1Projective, G2Projective};
        let g1 = G1Projective::generator();
        let g2 = G2Projective::from(G2Affine::generator());
        let p = core::array::from_fn(|i| {
            g1.mul_scalar(Fr::from_raw([1 + i as u64, 3, 0, 0]))
                .to_affine()
        });
        let q = core::array::from_fn(|i| {
            g2.mul_scalar(Fr::from_raw([2 + i as u64, 5, 0, 0]))
                .to_affine()
        });
        (p, q)
    }

    #[test]
    fn pairing8_matches_scalar() {
        use crate::pairing::pairing;
        let (p, q) = test_pairs();
        let got = pairing8(&p, &q).store();
        for i in 0..8 {
            assert_eq!(got[i], pairing(&p[i], &q[i]), "lane {i}");
        }
    }

    #[test]
    fn multi_pairing8_matches_scalar() {
        use crate::pairing::multi_pairing;
        let (p, q) = test_pairs();
        let pairs: Vec<(G1Affine, G2Affine)> = (0..8).map(|i| (p[i], q[i])).collect();
        let refs: Vec<(&G1Affine, &G2Affine)> = pairs.iter().map(|(a, b)| (a, b)).collect();
        assert_eq!(multi_pairing8(&pairs), multi_pairing(&refs));
    }

    /// Every chunk-and-mask composition must equal the shared-accumulator
    /// scalar Miller product. The lane engine now backs the production
    /// multi-pairing routes, so the reference must stay engine-independent.
    #[test]
    fn multi_miller_product8_matches_shared_reference() {
        use crate::pairing::miller::multi_miller_loop_shared;
        let g1 = crate::G1Projective::generator();
        let g2 = crate::G2Projective::from(G2Affine::generator());
        let points: Vec<(G1Affine, G2Affine)> = (0..17)
            .map(|i| {
                (
                    g1.mul_scalar(crate::Fr::from_raw([3 + i as u64, 7, 0, 0]))
                        .to_affine(),
                    g2.mul_scalar(crate::Fr::from_raw([5 + i as u64, 11, 0, 0]))
                        .to_affine(),
                )
            })
            .collect();
        for count in 2..=17 {
            let live = &points[..count];
            let Some(product) = multi_miller_product8(live) else {
                assert!(!ifma_available(), "lane product unavailable with IFMA");
                return;
            };
            let refs: Vec<(&G1Affine, &G2Affine)> = live.iter().map(|(a, b)| (a, b)).collect();
            assert_eq!(product, multi_miller_loop_shared(&refs), "count {count}");
        }
    }

    #[test]
    fn multi_pairing8_handles_identities_and_tail() {
        use crate::pairing::multi_pairing;
        use crate::{Fr, G1Projective, G2Projective};

        let g1 = G1Projective::generator();
        let g2 = G2Projective::from(G2Affine::generator());
        let id_p = G1Affine {
            x: Fp::ZERO,
            y: Fp::ZERO,
            infinity: true,
        };
        let id_q = G2Affine {
            x: Fp2::ZERO,
            y: Fp2::ZERO,
            infinity: true,
        };
        // Counts across a group boundary, each with identity pairs mixed in so
        // the scalar tail and the identity skip both exercise.
        for n in [1usize, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 20] {
            let mut pairs: Vec<(G1Affine, G2Affine)> = (0..n)
                .map(|i| {
                    let a = g1
                        .mul_scalar(Fr::from_raw([1 + i as u64, 7, 0, 0]))
                        .to_affine();
                    let b = g2
                        .mul_scalar(Fr::from_raw([3 + i as u64, 2, 0, 0]))
                        .to_affine();
                    (a, b)
                })
                .collect();
            if n > 2 {
                pairs[2] = (id_p, pairs[2].1);
            }
            if n > 9 {
                pairs[9] = (pairs[9].0, id_q);
            }
            let refs: Vec<(&G1Affine, &G2Affine)> = pairs.iter().map(|(a, b)| (a, b)).collect();
            assert_eq!(multi_pairing8(&pairs), multi_pairing(&refs), "n = {n}");
        }
    }

    #[test]
    fn public_facade_matches_scalar_at_ifma_cardinality_boundaries() {
        use crate::batch::{G1Bytes, G2Bytes, InputError, PairBytes, pairing_product_is_one};
        use crate::pairing::multi_pairing;
        use crate::{Fr, G2Projective};

        let p = G1Affine::generator();
        let q = G2Projective::from(G2Affine::generator());
        for n in [5usize, 6, 7, 8, 9, 13, 14, 15, 16, 17] {
            // Every Q is distinct so n >= 8 exercises the IFMA dispatch rather
            // than the public facade's all-equal-Q bilinear reduction.
            let typed: Vec<(G1Affine, G2Affine)> = (0..n)
                .map(|i| {
                    (
                        if i & 1 == 0 { p } else { p.negate() },
                        q.mul_scalar(Fr::from_u64(i as u64 + 1)).to_affine(),
                    )
                })
                .collect();
            let refs: Vec<_> = typed.iter().map(|(g1, g2)| (g1, g2)).collect();
            let encoded: Vec<_> = typed
                .iter()
                .map(|(g1, g2)| PairBytes {
                    g1: G1Bytes::from_affine(g1),
                    g2: G2Bytes::from_affine(g2),
                })
                .collect();
            assert_eq!(
                pairing_product_is_one(&encoded),
                Ok(multi_pairing(&refs) == Fp12::ONE),
                "n = {n}",
            );
        }
        let good = PairBytes {
            g1: G1Bytes::from_affine(&p),
            g2: G2Bytes::from_affine(&G2Affine::generator()),
        };
        for count in [7, 8, 9, 16] {
            let mut pairs = vec![good; count];
            pairs[0].g2.0[127] ^= 1;
            pairs[1].g1.0[63] ^= 1;
            assert_eq!(pairing_product_is_one(&pairs), Err(InputError::NotOnCurve));
            pairs[0] = good;
            pairs[1].g2.0.fill(0xff);
            assert_eq!(pairing_product_is_one(&pairs), Err(InputError::NotOnCurve));
        }
    }

    #[test]
    fn public_facade_ifma_path_accepts_the_256_pair_cap() {
        use crate::batch::{
            G1Bytes, G2Bytes, PAIRING_MAX_PAIRS, PairBytes, pairing_product_is_one,
        };

        let p = G1Affine::generator();
        let q = G2Affine::generator();
        let encoded: Vec<_> = (0..PAIRING_MAX_PAIRS)
            .map(|i| {
                let point = if i & 1 == 0 { p } else { p.negate() };
                PairBytes {
                    g1: G1Bytes::from_affine(&point),
                    g2: G2Bytes::from_affine(&q),
                }
            })
            .collect();
        assert_eq!(pairing_product_is_one(&encoded), Ok(true));
    }
}
