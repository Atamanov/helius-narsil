//! Independent adversarial oracle for the single-pair Miller loop.
//!
//! The reference G2 schedule below intentionally uses the public canonical
//! `Fp2` operations instead of `fp2_fast`, `G2Hom`, or any generated kernel.
//! Future backends only need to keep `prepared_line_coeffs_for_test` wired to
//! their private schedule to inherit this gate.

use ark_bn254::{
    Bn254, Fq as ArkFq, Fr as ArkFr, G1Projective as ArkG1Projective,
    G2Projective as ArkG2Projective,
};
use ark_ec::{CurveGroup, PrimeGroup, pairing::Pairing};
use ark_ff::{BigInt, PrimeField};

use super::{final_exponentiation, miller_loop, miller_loop_prepared, prepare_g2};
use crate::consts::{ATE_LOOP_COUNT, BN_X, R};
use crate::pairing::miller::{
    G2_PREPARED_COEFFS, miller_loop_live_for_test, prepared_line_coeffs_for_test, twist_b_f2,
};
use crate::{Fp, Fp2, Fp6, Fp12, Fr, G1Affine, G1Projective, G2Affine, G2Projective};

type Line = (Fp2, Fp2, Fp2);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Step {
    Double,
    AddPositive,
    AddNegative,
    FrobeniusTail1,
    FrobeniusTail2,
}

#[derive(Clone, Copy)]
struct ReferenceHom {
    x: Fp2,
    y: Fp2,
    z: Fp2,
}

impl ReferenceHom {
    fn from_affine(q: G2Affine) -> Self {
        Self {
            x: q.x,
            y: q.y,
            z: Fp2::ONE,
        }
    }

    fn half(value: Fp2) -> Fp2 {
        value.mul_by_fp(Fp::from_u64(2).invert().expect("two is invertible"))
    }

    /// Canonical-Fp2 transcription of the ePrint 2013/722 D-twist doubling.
    fn double(&mut self, twist_b: Fp2) -> Line {
        let a = Self::half(self.x * self.y);
        let b = self.y.square();
        let c = self.z.square();
        let e = twist_b * (c.double() + c);
        let f = e.double() + e;
        let g = Self::half(b + f);
        let h = (self.y + self.z).square() - b - c;
        let i = e - b;
        let j = self.x.square();

        self.x = a * (b - f);
        self.y = g.square() - e.square().double() - e.square();
        self.z = b * h;

        (-h, j.double() + j, i)
    }

    /// Canonical-Fp2 transcription of the mixed D-twist addition.
    fn add(&mut self, qx: Fp2, qy: Fp2) -> Line {
        let theta = self.y - qy * self.z;
        let lambda = self.x - qx * self.z;
        let c = theta.square();
        let d = lambda.square();
        let e = lambda * d;
        let f = self.z * c;
        let g = self.x * d;
        let h = e + f - g.double();
        self.x = lambda * h;
        self.y = theta * (g - h) - e * self.y;
        self.z *= e;
        let j = theta * qx - lambda * qy;
        (lambda, -theta, j)
    }
}

fn psi_by_scalar(q: G2Affine) -> G2Affine {
    let x = u128::from(BN_X);
    let scalar = 6 * x * x;
    G2Projective::from(q)
        .mul_words(&[scalar as u64, (scalar >> 64) as u64])
        .to_affine()
}

fn reference_twist_b() -> Fp2 {
    let three = Fp2::new(Fp::from_u64(3), Fp::ZERO);
    let xi = Fp2::new(Fp::from_u64(9), Fp::ONE);
    three * xi.invert().expect("9 + u is nonzero")
}

fn reference_schedule(q: G2Affine) -> Vec<(Step, Line)> {
    assert!(!q.is_identity());
    let mut result = Vec::with_capacity(G2_PREPARED_COEFFS);
    let mut r = ReferenceHom::from_affine(q);
    let twist_b = reference_twist_b();
    for i in (1..ATE_LOOP_COUNT.len()).rev() {
        result.push((Step::Double, r.double(twist_b)));
        match ATE_LOOP_COUNT[i - 1] {
            1 => result.push((Step::AddPositive, r.add(q.x, q.y))),
            -1 => result.push((Step::AddNegative, r.add(q.x, -q.y))),
            0 => {}
            digit => panic!("unexpected Ate digit {digit}"),
        }
    }
    let q1 = psi_by_scalar(q);
    let q2 = psi_by_scalar(q1).negate();
    result.push((Step::FrobeniusTail1, r.add(q1.x, q1.y)));
    result.push((Step::FrobeniusTail2, r.add(q2.x, q2.y)));
    assert_eq!(result.len(), G2_PREPARED_COEFFS);
    result
}

fn sub_one(mut value: [u64; 4]) -> [u64; 4] {
    let mut borrow = true;
    for limb in &mut value {
        let (next, next_borrow) = limb.borrowing_sub(0, borrow);
        *limb = next;
        borrow = next_borrow;
    }
    assert!(!borrow);
    value
}

fn scalar_corpus() -> Vec<[u64; 4]> {
    let mut values = vec![
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [2, 0, 0, 0],
        [0xaaaa_aaaa_aaaa_aaaa, 0x5555_5555_5555_5555, 0, 0],
        [0x8000_0000_0000_0001, 0x0000_0000_0000_0001, 0, 0],
        sub_one(R),
    ];
    let mut state = 0x7265_6474_6561_6d21u64;
    for _ in 0..10 {
        let mut words = [0; 4];
        for word in &mut words {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *word = state;
        }
        // `R < 2^254`. Masking to 252 bits gives a canonical scalar without
        // using either field implementation's reduction logic.
        words[3] &= 0x0fff_ffff_ffff_ffff;
        values.push(words);
    }
    values
}

fn ark_scalar(words: [u64; 4]) -> ArkFr {
    ArkFr::from_bigint(BigInt(words)).expect("red-team scalar is canonical")
}

fn fp(value: ArkFq) -> Fp {
    Fp::from_raw(value.into_bigint().0)
}

fn from_ark(value: ark_bn254::Fq12) -> Fp12 {
    let fp2 = |x: ark_bn254::Fq2| Fp2::new(fp(x.c0), fp(x.c1));
    let fp6 = |x: ark_bn254::Fq6| Fp6::new(fp2(x.c0), fp2(x.c1), fp2(x.c2));
    Fp12::new(fp6(value.c0), fp6(value.c1))
}

fn coefficients(value: &Fp12) -> [[u64; 4]; 12] {
    [
        value.c0.c0.c0.to_raw(),
        value.c0.c0.c1.to_raw(),
        value.c0.c1.c0.to_raw(),
        value.c0.c1.c1.to_raw(),
        value.c0.c2.c0.to_raw(),
        value.c0.c2.c1.to_raw(),
        value.c1.c0.c0.to_raw(),
        value.c1.c0.c1.to_raw(),
        value.c1.c1.c0.to_raw(),
        value.c1.c1.c1.to_raw(),
        value.c1.c2.c0.to_raw(),
        value.c1.c2.c1.to_raw(),
    ]
}

#[test]
fn prepared_schedule_has_exact_87_independently_derived_lines() {
    let base = G2Projective::from(G2Affine::generator());
    for scalar in scalar_corpus().into_iter().skip(1).take(10) {
        let q = base.mul_scalar(Fr::from_raw(scalar)).to_affine();
        for (sign, q) in [("positive", q), ("negative", q.negate())] {
            let reference = reference_schedule(q);
            let production = prepared_line_coeffs_for_test(&q);
            assert_eq!(production.len(), 87);
            assert_eq!(reference.len(), production.len());
            assert!(reference.iter().any(|(step, _)| *step == Step::AddPositive));
            assert!(reference.iter().any(|(step, _)| *step == Step::AddNegative));
            assert_eq!(reference[85].0, Step::FrobeniusTail1);
            assert_eq!(reference[86].0, Step::FrobeniusTail2);
            for (index, ((step, want), got)) in reference.into_iter().zip(production).enumerate() {
                assert_eq!(
                    got, want,
                    "line {index} ({step:?}), {sign}, scalar={scalar:x?}"
                );
            }
        }
    }
}

#[test]
fn raw_miller_final_exp_and_prepared_replay_match_ark_coefficientwise() {
    let hg1 = G1Projective::generator();
    let hg2 = G2Projective::from(G2Affine::generator());
    let ag1 = ArkG1Projective::generator();
    let ag2 = ArkG2Projective::generator();

    for (case, scalar) in scalar_corpus().into_iter().enumerate() {
        let other = scalar_corpus()[(case * 7 + 3) % scalar_corpus().len()];
        let hp = hg1.mul_scalar(Fr::from_raw(scalar)).to_affine();
        let hq = hg2.mul_scalar(Fr::from_raw(other)).to_affine();
        let ap = (ag1 * ark_scalar(scalar)).into_affine();
        let aq = (ag2 * ark_scalar(other)).into_affine();

        for (sign, hq, aq) in [("positive", hq, aq), ("negative", hq.negate(), -aq)] {
            let got_raw = miller_loop_live_for_test(&hp, &hq);
            let ark_raw = Bn254::miller_loop(ap, aq).0;
            let want_raw = from_ark(ark_raw);
            assert_eq!(
                coefficients(&got_raw),
                coefficients(&want_raw),
                "raw 12-coefficient Miller mismatch: case {case}, {sign}",
            );

            let got_fe = final_exponentiation(&got_raw);
            let ark_fe = Bn254::final_exponentiation(ark_ec::pairing::MillerLoopOutput(ark_raw))
                .expect("pairing Miller result is nonzero")
                .0;
            assert_eq!(
                coefficients(&got_fe),
                coefficients(&from_ark(ark_fe)),
                "canonical final exponentiation mismatch: case {case}, {sign}",
            );

            let prepared = prepare_g2(&hq).expect("scalar multiple stays in the subgroup");
            assert_eq!(
                coefficients(&miller_loop_prepared(&hp, &prepared)),
                coefficients(&got_raw),
                "prepared/live mismatch: case {case}, {sign}",
            );
            assert_eq!(
                coefficients(&miller_loop(&hp, &hq)),
                coefficients(&got_raw),
                "public cached/live mismatch: case {case}, {sign}",
            );
        }
    }

    let p = G1Affine::generator();
    let q = G2Affine::generator();
    let identity_schedule = prepare_g2(&G2Affine::identity()).expect("identity is valid");
    assert_eq!(miller_loop(&G1Affine::identity(), &q), Fp12::ONE);
    assert_eq!(miller_loop(&p, &G2Affine::identity()), Fp12::ONE);
    assert_eq!(miller_loop_prepared(&p, &identity_schedule), Fp12::ONE);
}

#[test]
fn reference_twist_constant_and_schedule_shape_are_pinned() {
    let twist = twist_b_f2();
    let twist = Fp2::new(Fp(twist.0), Fp(twist.1));
    assert_eq!(twist, reference_twist_b());
    let q = G2Affine::generator();
    assert_eq!(q.y.square(), q.x.square() * q.x + twist);

    let steps: Vec<_> = reference_schedule(q)
        .into_iter()
        .map(|(step, _)| step)
        .collect();
    assert_eq!(
        steps.iter().filter(|&&step| step == Step::Double).count(),
        64
    );
    assert_eq!(
        steps
            .iter()
            .filter(|&&step| matches!(step, Step::AddPositive | Step::AddNegative))
            .count(),
        21,
    );
    assert_eq!(steps.len(), 64 + 21 + 2);
}
