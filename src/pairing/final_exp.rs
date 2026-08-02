//! Final exponentiation for BN254.
//! Easy part + Fuentes-Castaneda hard part (arkworks / eprint 2011/506).

use crate::fp12::Fp12;

/// Arkworks/MCL-compatible BN254 final exponentiation.
///
/// The Fuentes-Castaneda schedule returns the canonical exponent multiplied by
/// `2*x*(6*x^2+3*x+1)`. This factor is coprime to the subgroup order. The
/// result is therefore exact for this API and equivalent for pairing checks.
pub fn final_exponentiation(f: &Fp12) -> Fp12 {
    if f.is_zero() {
        return Fp12::ZERO;
    }

    // Easy: f^{(p^6-1)(p^2+1)}
    let mut f1 = f.conjugate(); // f^{p^6}
    let f2 = f.invert().unwrap();
    let mut r = f1 * f2; // f^{p^6-1}
    f1 = r;
    r = r.frobenius_map_squared(); // f^{(p^6-1)p^2}
    r *= f1; // f^{(p^6-1)(p^2+1)}

    #[cfg(all(
        feature = "std",
        any(helius_avx512_ifma, helius_x86_runtime_ifma),
        not(feature = "force-portable")
    ))]
    if let Some(ifma) = crate::x86_runtime::avx512_ifma() {
        return crate::final_exp_ifma::hard_part(ifma, r);
    }

    hard_part_scalar(r)
}

/// Scalar reference for the Fuentes-Castaneda hard part.
pub(crate) fn hard_part_scalar(r: Fp12) -> Fp12 {
    // X is positive for BN_SNARK1. exp_by_neg_x is conjugate(f^x).
    let y0 = exp_by_neg_x(r);
    let y1 = y0.cyclotomic_square();
    let y2 = y1.cyclotomic_square();
    let mut y3 = y2 * y1;
    let y4 = exp_by_neg_x(y3);
    let y5 = y4.cyclotomic_square();
    let mut y6 = exp_by_neg_x(y5);
    y3 = y3.conjugate();
    y6 = y6.conjugate();
    let y7 = y6 * y4;
    let mut y8 = y7 * y3;
    let y9 = y8 * y1;
    let y10 = y8 * y4;
    let y11 = y10 * r;
    let mut y12 = y9;
    y12 = y12.frobenius_map();
    let y13 = y12 * y11;
    y8 = y8.frobenius_map_squared();
    let y14 = y8 * y13;
    let mut y15 = r.conjugate() * y9;
    y15 = y15.frobenius_map_cubed();
    y15 * y14
}

fn exp_by_neg_x(f: Fp12) -> Fp12 {
    // X_IS_NEGATIVE = false for BN_SNARK1, so this is unitary inverse of f^x
    f.pow_x().conjugate()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consts::{BN_X, P, R};
    use crate::fp::Fp;
    use crate::fp2::Fp2;
    use crate::fp6::Fp6;

    const LIMBS: usize = 20;
    type Big = [u64; LIMBS];

    fn from_limbs<const N: usize>(value: [u64; N]) -> Big {
        let mut out = [0; LIMBS];
        out[..N].copy_from_slice(&value);
        out
    }

    fn cmp(a: &Big, b: &Big) -> core::cmp::Ordering {
        for index in (0..LIMBS).rev() {
            match a[index].cmp(&b[index]) {
                core::cmp::Ordering::Equal => {}
                order => return order,
            }
        }
        core::cmp::Ordering::Equal
    }

    fn add_assign(a: &mut Big, b: &Big) {
        let mut carry = 0u128;
        for index in 0..LIMBS {
            let sum = a[index] as u128 + b[index] as u128 + carry;
            a[index] = sum as u64;
            carry = sum >> 64;
        }
        assert_eq!(carry, 0);
    }

    fn sub_assign(a: &mut Big, b: &Big) {
        let mut borrow = 0u128;
        for index in 0..LIMBS {
            let lhs = a[index] as u128;
            let rhs = b[index] as u128 + borrow;
            a[index] = lhs.wrapping_sub(rhs) as u64;
            borrow = u128::from(lhs < rhs);
        }
        assert_eq!(borrow, 0);
    }

    fn mul(a: &Big, b: &Big) -> Big {
        let mut out = [0u64; LIMBS];
        for i in 0..LIMBS {
            let mut carry = 0u128;
            for j in 0..LIMBS - i {
                let value = a[i] as u128 * b[j] as u128 + out[i + j] as u128 + carry;
                out[i + j] = value as u64;
                carry = value >> 64;
            }
            assert_eq!(carry, 0);
        }
        out
    }

    fn mul_small(a: &Big, b: u64) -> Big {
        mul(a, &from_limbs([b]))
    }

    fn shl1(a: &mut Big) {
        let mut carry = 0;
        for limb in a {
            let next = *limb >> 63;
            *limb = (*limb << 1) | carry;
            carry = next;
        }
        assert_eq!(carry, 0);
    }

    fn bit_len(a: &Big) -> usize {
        for index in (0..LIMBS).rev() {
            if a[index] != 0 {
                return 64 * index + (64 - a[index].leading_zeros() as usize);
            }
        }
        0
    }

    fn div(numerator: &Big, denominator: &Big) -> Big {
        let mut quotient = [0; LIMBS];
        let mut remainder = [0; LIMBS];
        for bit in (0..bit_len(numerator)).rev() {
            shl1(&mut remainder);
            remainder[0] |= (numerator[bit / 64] >> (bit % 64)) & 1;
            if cmp(&remainder, denominator) != core::cmp::Ordering::Less {
                sub_assign(&mut remainder, denominator);
                quotient[bit / 64] |= 1 << (bit % 64);
            }
        }
        assert_eq!(remainder, [0; LIMBS]);
        quotient
    }

    fn hard_exponent() -> Big {
        // d = (p^4 - p^2 + 1) / r.
        let p = from_limbs(P);
        let p2 = mul(&p, &p);
        let mut numerator = mul(&p2, &p2);
        sub_assign(&mut numerator, &p2);
        add_assign(&mut numerator, &from_limbs([1]));
        let d = div(&numerator, &from_limbs(R));

        // The Fuentes-Castaneda schedule uses d times this coprime factor.
        let x = from_limbs([BN_X]);
        let x2 = mul(&x, &x);
        let mut factor = mul_small(&x2, 6);
        add_assign(&mut factor, &mul_small(&x, 3));
        add_assign(&mut factor, &from_limbs([1]));
        mul(&d, &mul_small(&mul(&x, &factor), 2))
    }

    fn pow_big(value: Fp12, exponent: &Big) -> Fp12 {
        let mut result = Fp12::ONE;
        for bit in (0..bit_len(exponent)).rev() {
            result = result.square();
            if (exponent[bit / 64] >> (bit % 64)) & 1 != 0 {
                result *= value;
            }
        }
        result
    }

    fn sample() -> Fp12 {
        let mut next = 1u64;
        let mut fp2 = || {
            next = next.wrapping_mul(6364136223846793005).wrapping_add(1);
            let a = Fp::from_u64(next);
            next = next.wrapping_mul(6364136223846793005).wrapping_add(1);
            Fp2::new(a, Fp::from_u64(next))
        };
        Fp12::new(Fp6::new(fp2(), fp2(), fp2()), Fp6::new(fp2(), fp2(), fp2()))
    }

    #[test]
    fn hard_schedule_matches_independent_exponent_oracle() {
        let value = sample();
        let easy = {
            let t = value.conjugate() * value.invert().unwrap();
            t.frobenius_map_squared() * t
        };
        assert_eq!(hard_part_scalar(easy), pow_big(easy, &hard_exponent()));
    }
}
