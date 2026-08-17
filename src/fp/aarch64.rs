//! AArch64 Montgomery backend generated from `build/a64/schedule.rs`.

use crate::abi::narsil_mont4;
#[cfg(narsil_a64_sosd2)]
use crate::abi::narsil_sosd2;
use crate::fp::Fp;

// `repr(C)` and the assertions bind the Rust constants to the FFI layout.
#[repr(C)]
pub(crate) struct Mont4Constants {
    modulus: [u64; 4],
    negative_inverse: u64,
}

static MONT4_CONSTANTS: Mont4Constants = Mont4Constants {
    modulus: crate::consts::P,
    negative_inverse: crate::consts::P_INV,
};

const _: () = {
    assert!(core::mem::size_of::<Mont4Constants>() == 5 * core::mem::size_of::<u64>());
    assert!(core::mem::align_of::<Mont4Constants>() == core::mem::align_of::<u64>());
    assert!(
        core::mem::offset_of!(Mont4Constants, negative_inverse) == 4 * core::mem::size_of::<u64>()
    );
};

/// Reduced Montgomery multiplication through the shared Apple AArch64 leaf.
///
/// Kernel contract: `a` and `b` are readable, 8-byte-aligned 32-byte arrays.
/// `a` and `b` must both be residues below the BN254 base modulus (the
/// shared backend contract. `Fp::from_raw` reduces first). The A64 schedule
/// additionally tolerates `a < 2^256` (single-chain rows plus CSET give
/// 2^321 headroom, interpreter-verified) but callers must not rely on it.
/// They may alias each other.
/// The wrapper supplies distinct output and immutable constant-table pointers,
/// both suitably aligned and live for the call. The leaf initializes all four
/// output limbs, returns a fully reduced residue, saves every callee-saved
/// register it uses, preserves the 16-byte stack alignment, and neither calls
/// Rust nor unwinds.
#[inline(never)]
pub fn mont_mul(a: &[u64; 4], b: &[u64; 4]) -> Fp {
    debug_assert!(!crate::limb::gte(a, &crate::consts::P));
    debug_assert!(!crate::limb::gte(b, &crate::consts::P));
    let mut z = core::mem::MaybeUninit::<[u64; 4]>::uninit();
    unsafe {
        // SAFETY: fixed-size references and the local output satisfy the
        // complete kernel contract above. The assembly initializes 32 bytes
        // before `assume_init` and cannot retain any pointer.
        narsil_mont4(
            z.as_mut_ptr() as *mut u64,
            a.as_ptr(),
            b.as_ptr(),
            &MONT4_CONSTANTS,
        );
        Fp(z.assume_init())
    }
}

#[inline(always)]
pub fn mont_sqr(a: &[u64; 4]) -> Fp {
    mont_mul(a, a)
}

/// Both Fp halves of `(x0 + x1*u)*(y0 + y1*u)` through the dual-lane leaf.
///
/// Kernel contract: every operand is a readable, 8-byte-aligned 32-byte
/// array holding a residue at most the BN254 base modulus (the SoS operand
/// bound. The leaf forms `p - y1` itself, which needs `y1 <= p`). Operands
/// may alias each other: the Fp2 square passes `x` for both sides. The
/// wrapper supplies a distinct 64-byte output and an immutable constant
/// table, both suitably aligned and live for the call. The leaf initializes
/// all eight output limbs, returns two fully reduced residues, saves every
/// callee-saved register it uses, keeps the 16-byte stack alignment, and
/// neither calls Rust nor unwinds.
#[cfg(narsil_a64_sosd2)]
#[inline(never)]
pub(crate) fn sosd2(
    x0: &[u64; 4],
    x1: &[u64; 4],
    y0: &[u64; 4],
    y1: &[u64; 4],
) -> ([u64; 4], [u64; 4]) {
    for operand in [x0, x1, y0, y1] {
        debug_assert!(!crate::limb::gt(operand, &crate::consts::P));
    }
    let mut z = core::mem::MaybeUninit::<[u64; 8]>::uninit();
    unsafe {
        // SAFETY: fixed-size references and the local output satisfy the
        // complete kernel contract above. The assembly initializes 64 bytes
        // before `assume_init` and cannot retain any pointer.
        narsil_sosd2(
            z.as_mut_ptr() as *mut u64,
            x0.as_ptr(),
            x1.as_ptr(),
            y0.as_ptr(),
            y1.as_ptr(),
            &MONT4_CONSTANTS,
        );
        let z = z.assume_init();
        ([z[0], z[1], z[2], z[3]], [z[4], z[5], z[6], z[7]])
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;
    use crate::consts::{MONT_ONE, MONT_R2, P};
    use crate::fp::Fp;
    use crate::fp::portable;
    use crate::limb;

    #[test]
    fn hot_matches_asm() {
        let a = Fp::from_u64(0x123456789);
        let b = Fp::from_u64(0x987654321);
        assert_eq!(a * b, mont_mul(&a.0, &b.0));
        assert_eq!(a.square(), mont_sqr(&a.0));
    }

    fn next_residue(state: &mut u64) -> [u64; 4] {
        let mut value = [0u64; 4];
        for limb in &mut value {
            *state ^= *state << 13;
            *state ^= *state >> 7;
            *state ^= *state << 17;
            *limb = *state;
        }
        while limb::gte(&value, &P) {
            value = limb::sub_noborrow(&value, &P);
        }
        value
    }

    /// Carry edges, the hot conversion constants, and `random` residues.
    fn residue_corpus(seed: u64, random: usize) -> alloc::vec::Vec<[u64; 4]> {
        let p_minus_one = limb::sub_noborrow(&P, &[1, 0, 0, 0]);
        let mut cases = vec![
            [0; 4],
            [1, 0, 0, 0],
            MONT_ONE,
            MONT_R2,
            p_minus_one,
            [u64::MAX, 0, u64::MAX, 0],
            [0, u64::MAX, 0, 0x1000_0000_0000_0000],
        ];
        for value in cases.iter_mut().skip(5) {
            while limb::gte(value, &P) {
                *value = limb::sub_noborrow(value, &P);
            }
        }
        let mut state = seed;
        cases.extend((0..random).map(|_| next_residue(&mut state)));
        cases
    }

    #[test]
    fn assembly_matches_portable_on_edges_and_carries() {
        let cases = residue_corpus(0x243f_6a88_85a3_08d3, 256);

        for (index, a) in cases.iter().enumerate() {
            assert_eq!(mont_sqr(a), portable::mont_sqr(a), "square case {index}");
            for (other_index, b) in cases.iter().step_by(17).enumerate() {
                assert_eq!(
                    mont_mul(a, b),
                    portable::mont_mul(a, b),
                    "multiply case {index}/{other_index}",
                );
            }
        }
    }

    /// The one check that runs the emitted dual-lane leaf on silicon. The
    /// interpreter proves the schedule's algebra. This proves the assembler,
    /// the ABI, and the callee-saved contract.
    #[cfg(narsil_a64_sosd2)]
    #[test]
    fn sosd2_assembly_matches_portable_on_edges_and_carries() {
        let cases = residue_corpus(0x9e37_79b9_7f4a_7c15, 256);
        for (index, x0) in cases.iter().enumerate() {
            for (step, x1) in cases.iter().step_by(31).enumerate() {
                for (other, y0) in cases.iter().step_by(37).enumerate() {
                    for y1 in cases.iter().step_by(41) {
                        assert_eq!(
                            super::sosd2(x0, x1, y0, y1),
                            crate::fp::sos::sosd2_portable(x0, x1, y0, y1),
                            "case {index}/{step}/{other}",
                        );
                    }
                }
            }
        }
    }

    #[cfg(narsil_a64_sosd2)]
    #[test]
    fn sosd2_assembly_matches_portable_on_random_residues() {
        let mut state = 0xbf58_476d_1ce4_e5b9u64;
        for case in 0..100_000 {
            let x0 = next_residue(&mut state);
            let x1 = next_residue(&mut state);
            let y0 = next_residue(&mut state);
            let y1 = next_residue(&mut state);
            assert_eq!(
                super::sosd2(&x0, &x1, &y0, &y1),
                crate::fp::sos::sosd2_portable(&x0, &x1, &y0, &y1),
                "case {case}",
            );
        }
    }

    #[test]
    #[ignore = "million-case release stress gate; run explicitly before changing field backends"]
    fn million_products_match_assembly_and_arkworks_raw_montgomery() {
        use ark_ff::BigInt;

        let mut state = 0xd1b5_4a32_d192_ed03u64;
        for case in 0..1_000_000 {
            let a = next_residue(&mut state);
            let b = next_residue(&mut state);
            let portable = portable::mont_mul(&a, &b);
            assert_eq!(portable, mont_mul(&a, &b), "assembly case {case}");

            // Arkworks and Narsil use the same four-limb R=2^256 Montgomery
            // representation. `new_unchecked` deliberately consumes those raw
            // limbs, so this oracle compares the kernel result directly: no
            // Narsil conversion path can mask a common error.
            let ark_a = ark_bn254::Fq::new_unchecked(BigInt::new(a));
            let ark_b = ark_bn254::Fq::new_unchecked(BigInt::new(b));
            let ark_product = (ark_a * ark_b).0.0;
            assert_eq!(portable.0, ark_product, "Arkworks case {case}");
        }
    }
}

/// Paired A/B of the A64 leaf against the portable route. Both run in one
/// process, alternating inside each sample, so they see the same core, the
/// same clock and the same background load. The ratio survives noise that
/// would swamp two separate absolute measurements. Run with
/// `cargo test --release --features std --lib ab_sosd2 -- --ignored --nocapture`.
#[cfg(all(test, narsil_a64_sosd2))]
mod ab {
    use super::sosd2;
    use crate::fp::sos::sosd2_portable;
    use core::hint::black_box;
    use std::time::Instant;

    fn residues(seed: u64, n: usize) -> Vec<[u64; 4]> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                let mut r = [0u64; 4];
                for w in &mut r {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    *w = s;
                }
                r[3] &= 0x0fff_ffff_ffff_ffff;
                r
            })
            .collect()
    }

    #[test]
    #[ignore]
    fn ab_sosd2() {
        let xs = residues(1, 256);
        let ys = residues(2, 256);
        let iters = 20_000usize;
        let samples = 41usize;
        let mut leaf = Vec::with_capacity(samples);
        let mut port = Vec::with_capacity(samples);
        // Warm both.
        for i in 0..4096 {
            let k = i & 255;
            black_box(sosd2(&xs[k], &xs[k ^ 1], &ys[k], &ys[k ^ 1]));
            black_box(sosd2_portable(&xs[k], &xs[k ^ 1], &ys[k], &ys[k ^ 1]));
        }
        for s in 0..samples {
            // Alternate which goes first so drift cancels.
            let order = if s % 2 == 0 { [0, 1] } else { [1, 0] };
            for which in order {
                let t = Instant::now();
                if which == 0 {
                    for i in 0..iters {
                        let k = i & 255;
                        black_box(sosd2(
                            black_box(&xs[k]),
                            black_box(&xs[k ^ 1]),
                            black_box(&ys[k]),
                            black_box(&ys[k ^ 1]),
                        ));
                    }
                } else {
                    for i in 0..iters {
                        let k = i & 255;
                        black_box(sosd2_portable(
                            black_box(&xs[k]),
                            black_box(&xs[k ^ 1]),
                            black_box(&ys[k]),
                            black_box(&ys[k ^ 1]),
                        ));
                    }
                }
                let ns = t.elapsed().as_nanos() as f64 / iters as f64;
                if which == 0 {
                    leaf.push(ns)
                } else {
                    port.push(ns)
                }
            }
        }
        leaf.sort_by(|a, b| a.partial_cmp(b).unwrap());
        port.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = |v: &Vec<f64>| v[v.len() / 2];
        let (l, p) = (med(&leaf), med(&port));
        eprintln!("sosd2 leaf     median {:.2} ns  min {:.2}", l, leaf[0]);
        eprintln!("sosd2 portable median {:.2} ns  min {:.2}", p, port[0]);
        eprintln!(
            "ratio leaf/portable  {:.3}   ({:+.1}%)",
            l / p,
            (l / p - 1.0) * 100.0
        );
    }
}
