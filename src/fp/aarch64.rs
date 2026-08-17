//! AArch64 Montgomery backend generated from `build/a64/schedule.rs`.

use crate::abi::narsil_mont4;
#[cfg(narsil_a64_sosd2)]
use crate::abi::narsil_sosd2;
#[cfg(narsil_a64_sosd4)]
use crate::abi::narsil_sosd4;
#[cfg(narsil_a64_sosd6)]
use crate::abi::narsil_sosd6;
#[cfg(all(test, narsil_a64_addsub))]
use crate::abi::{narsil_add_mod, narsil_sub_mod};
use crate::fp::Fp;
#[cfg(any(narsil_a64_sosd4, narsil_a64_sosd6))]
use crate::fp::sos::Fp2Product;

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

/// `a + b mod p` through the modular-add leaf. Production takes the inline
/// rendering of the same schedule in `limb.rs`; this route holds the leaf to
/// the same values on silicon.
///
/// Kernel contract: `a`, `b`, and `p` are readable, 8-byte-aligned four-limb
/// arrays and may alias each other. The leaf is bit-exact with the portable
/// form on all of `[0, 2^256)`; the caller's own bound decides canonicality,
/// and `a, b < p` gives it. The wrapper supplies a distinct output, live for
/// the call. The leaf initializes all four output limbs, saves no register
/// (it stops at x17), moves no stack, and neither calls Rust nor unwinds.
#[cfg(all(test, narsil_a64_addsub))]
#[inline(always)]
pub(crate) fn add_mod(a: &[u64; 4], b: &[u64; 4], p: &[u64; 4]) -> [u64; 4] {
    let mut z = core::mem::MaybeUninit::<[u64; 4]>::uninit();
    unsafe {
        // SAFETY: three fixed-size references and the local output satisfy the
        // complete kernel contract above. The assembly initializes 32 bytes
        // before `assume_init` and cannot retain any pointer.
        narsil_add_mod(
            z.as_mut_ptr() as *mut u64,
            a.as_ptr(),
            b.as_ptr(),
            p.as_ptr(),
        );
        z.assume_init()
    }
}

/// `a - b mod p` through the modular-subtract leaf. Contract as for
/// [`add_mod`].
#[cfg(all(test, narsil_a64_addsub))]
#[inline(always)]
pub(crate) fn sub_mod(a: &[u64; 4], b: &[u64; 4], p: &[u64; 4]) -> [u64; 4] {
    let mut z = core::mem::MaybeUninit::<[u64; 4]>::uninit();
    unsafe {
        // SAFETY: as for `add_mod`.
        narsil_sub_mod(
            z.as_mut_ptr() as *mut u64,
            a.as_ptr(),
            b.as_ptr(),
            p.as_ptr(),
        );
        z.assume_init()
    }
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

/// The operand pointers of one table-ABI call, in kernel table order.
/// Every operand must be a residue at most p, the SoS bound the in-kernel
/// `p - y_i1` image rests on.
#[cfg(any(narsil_a64_sosd4, narsil_a64_sosd6))]
#[inline(always)]
fn table<const N: usize>(operands: [&[u64; 4]; N]) -> [*const u64; N] {
    operands.map(|operand| {
        debug_assert!(!crate::limb::gt(operand, &crate::consts::P));
        operand.as_ptr()
    })
}

/// Both Fp halves of a sum of two Fp2 products through the dual-lane leaf.
///
/// Kernel contract: as for [`sosd2`], for all eight operands. The wrapper
/// builds the pointer table on its own stack, and the leaf reads it during
/// the call and retains nothing, so the table needs no lifetime beyond it.
#[cfg(narsil_a64_sosd4)]
#[inline(never)]
pub(crate) fn sosd4(products: [Fp2Product<'_>; 2]) -> ([u64; 4], [u64; 4]) {
    let [(x00, x01, y00, y01), (x10, x11, y10, y11)] = products;
    let table = table([x00, x01, y00, y01, x10, x11, y10, y11]);
    let mut z = core::mem::MaybeUninit::<[u64; 8]>::uninit();
    unsafe {
        // SAFETY: the local table holds eight live, aligned four-limb operands
        // in the order the kernel indexes them, and the local output satisfies
        // the rest of the contract above. The assembly initializes 64 bytes
        // before `assume_init`.
        narsil_sosd4(z.as_mut_ptr() as *mut u64, table.as_ptr(), &MONT4_CONSTANTS);
        let z = z.assume_init();
        ([z[0], z[1], z[2], z[3]], [z[4], z[5], z[6], z[7]])
    }
}

/// Both Fp halves of a sum of three Fp2 products through the dual-lane leaf.
///
/// Kernel contract: as for [`sosd4`], for all twelve operands.
#[cfg(narsil_a64_sosd6)]
#[inline(never)]
pub(crate) fn sosd6(products: [Fp2Product<'_>; 3]) -> ([u64; 4], [u64; 4]) {
    let [
        (x00, x01, y00, y01),
        (x10, x11, y10, y11),
        (x20, x21, y20, y21),
    ] = products;
    let table = table([x00, x01, y00, y01, x10, x11, y10, y11, x20, x21, y20, y21]);
    let mut z = core::mem::MaybeUninit::<[u64; 8]>::uninit();
    unsafe {
        // SAFETY: as for `sosd4`, for twelve operands.
        narsil_sosd6(z.as_mut_ptr() as *mut u64, table.as_ptr(), &MONT4_CONSTANTS);
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

    /// The eight-operand table ABI on silicon: the assembler, the pointer
    /// table the wrapper builds, and the callee-saved contract. Slots are
    /// swept one at a time over the corpus against saturated backgrounds.
    #[cfg(narsil_a64_sosd4)]
    #[test]
    fn sosd4_assembly_matches_portable_on_edges_and_carries() {
        let cases = residue_corpus(0x3f84_d5b5_b547_0917, 64);
        let p_minus_one = limb::sub_noborrow(&P, &[1, 0, 0, 0]);
        for background in [P, p_minus_one, [0; 4]] {
            for slot in 0..8 {
                for (index, value) in cases.iter().enumerate() {
                    let mut operands = [background; 8];
                    operands[slot] = *value;
                    assert_eq!(
                        sosd4_of(&operands),
                        sosd4_portable_of(&operands),
                        "slot {slot}, case {index}",
                    );
                }
            }
        }
    }

    #[cfg(narsil_a64_sosd4)]
    #[test]
    fn sosd4_assembly_matches_portable_on_random_residues() {
        let mut state = 0x94d0_49bb_1331_11ebu64;
        for case in 0..100_000 {
            let operands: [[u64; 4]; 8] = core::array::from_fn(|_| next_residue(&mut state));
            assert_eq!(
                sosd4_of(&operands),
                sosd4_portable_of(&operands),
                "case {case}",
            );
        }
    }

    /// Eight limb arrays in kernel table order as the two Fp2 products the
    /// dispatch entry points take.
    #[cfg(narsil_a64_sosd4)]
    fn sosd4_products_of(operands: &[[u64; 4]; 8]) -> [crate::fp::sos::Fp2Product<'_>; 2] {
        core::array::from_fn(|i| {
            let [x0, x1, y0, y1] = [0, 1, 2, 3].map(|part| &operands[4 * i + part]);
            (x0, x1, y0, y1)
        })
    }

    #[cfg(narsil_a64_sosd4)]
    fn sosd4_of(operands: &[[u64; 4]; 8]) -> ([u64; 4], [u64; 4]) {
        super::sosd4(sosd4_products_of(operands))
    }

    #[cfg(narsil_a64_sosd4)]
    fn sosd4_portable_of(operands: &[[u64; 4]; 8]) -> ([u64; 4], [u64; 4]) {
        crate::fp::sos::sosd4_portable(sosd4_products_of(operands))
    }

    /// The twelve-operand table ABI on silicon: the assembler, the pointer
    /// table the wrapper builds, and the callee-saved contract. Slots are
    /// swept one at a time over the corpus against saturated backgrounds,
    /// then all twelve are drawn from it at once.
    #[cfg(narsil_a64_sosd6)]
    #[test]
    fn sosd6_assembly_matches_portable_on_edges_and_carries() {
        let cases = residue_corpus(0x452821e6_38d01377, 64);
        let p_minus_one = limb::sub_noborrow(&P, &[1, 0, 0, 0]);
        for background in [P, p_minus_one, [0; 4]] {
            for slot in 0..12 {
                for (index, value) in cases.iter().enumerate() {
                    let mut operands = [background; 12];
                    operands[slot] = *value;
                    assert_eq!(
                        sosd6_of(&operands),
                        sosd6_portable_of(&operands),
                        "slot {slot}, case {index}",
                    );
                }
            }
        }
    }

    #[cfg(narsil_a64_sosd6)]
    #[test]
    fn sosd6_assembly_matches_portable_on_random_residues() {
        let mut state = 0x9216_d5d9_8979_fb1bu64;
        for case in 0..100_000 {
            let operands: [[u64; 4]; 12] = core::array::from_fn(|_| next_residue(&mut state));
            assert_eq!(
                sosd6_of(&operands),
                sosd6_portable_of(&operands),
                "case {case}",
            );
        }
    }

    /// Twelve limb arrays in kernel table order as the three Fp2 products the
    /// dispatch entry points take.
    #[cfg(narsil_a64_sosd6)]
    fn products_of(operands: &[[u64; 4]; 12]) -> [crate::fp::sos::Fp2Product<'_>; 3] {
        core::array::from_fn(|i| {
            let [x0, x1, y0, y1] = [0, 1, 2, 3].map(|part| &operands[4 * i + part]);
            (x0, x1, y0, y1)
        })
    }

    #[cfg(narsil_a64_sosd6)]
    fn sosd6_of(operands: &[[u64; 4]; 12]) -> ([u64; 4], [u64; 4]) {
        super::sosd6(products_of(operands))
    }

    #[cfg(narsil_a64_sosd6)]
    fn sosd6_portable_of(operands: &[[u64; 4]; 12]) -> ([u64; 4], [u64; 4]) {
        crate::fp::sos::sosd6_portable(products_of(operands))
    }

    /// One schedule text renders two ways, so both must answer the portable
    /// form: the leaf symbol through its pointer ABI and the inline template
    /// through registers.
    #[cfg(narsil_a64_addsub)]
    type Fold = fn(&[u64; 4], &[u64; 4], &[u64; 4]) -> [u64; 4];

    #[cfg(narsil_a64_addsub)]
    const ADD_ROUTES: [(&str, Fold); 2] = [
        ("leaf", super::add_mod),
        ("inline", crate::limb::add_mod_inline),
    ];

    #[cfg(narsil_a64_addsub)]
    const SUB_ROUTES: [(&str, Fold); 2] = [
        ("leaf", super::sub_mod),
        ("inline", crate::limb::sub_mod_inline),
    ];

    /// Both modular folds on silicon against the portable forms. The
    /// interpreter proves the schedule algebra; this proves the assembler,
    /// the register-only ABI and the inline operand table. The corpus opens
    /// with zero and p-1, so the sweep covers 0 + 0, a + a, (p-1) + (p-1) and
    /// 0 - (p-1).
    #[cfg(narsil_a64_addsub)]
    #[test]
    fn addsub_assembly_matches_portable_on_edges_and_carries() {
        let cases = residue_corpus(0x0f1e_2d3c_4b5a_6978, 256);
        for (index, a) in cases.iter().enumerate() {
            for (other, b) in cases.iter().enumerate() {
                for (route, add) in ADD_ROUTES {
                    let sum = add(a, b, &P);
                    assert_eq!(
                        sum,
                        limb::add_mod_portable(a, b, &P),
                        "{route} add {index}/{other}",
                    );
                    assert!(
                        !limb::gte(&sum, &P),
                        "{route} unreduced sum {index}/{other}"
                    );
                }
                for (route, sub) in SUB_ROUTES {
                    let difference = sub(a, b, &P);
                    assert_eq!(
                        difference,
                        limb::sub_mod_portable(a, b, &P),
                        "{route} sub {index}/{other}",
                    );
                    assert!(
                        !limb::gte(&difference, &P),
                        "{route} unreduced difference {index}/{other}",
                    );
                }
            }
        }
    }

    #[cfg(narsil_a64_addsub)]
    #[test]
    fn addsub_assembly_matches_portable_on_random_residues() {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        for case in 0..100_000 {
            let a = next_residue(&mut state);
            let b = next_residue(&mut state);
            let sum = limb::add_mod_portable(&a, &b, &P);
            let difference = limb::sub_mod_portable(&a, &b, &P);
            assert!(!limb::gte(&sum, &P), "unreduced sum, case {case}");
            assert!(!limb::gte(&difference, &P), "unreduced difference {case}");
            for (route, add) in ADD_ROUTES {
                assert_eq!(add(&a, &b, &P), sum, "{route} add case {case}");
            }
            for (route, sub) in SUB_ROUTES {
                assert_eq!(sub(&a, &b, &P), difference, "{route} sub case {case}");
            }
        }
    }

    /// The folds also serve Fr, whose modulus differs, and the conditional
    /// subtraction the SoS routes use passes `a` up to `2p` with `b = p`.
    #[cfg(narsil_a64_addsub)]
    #[test]
    fn addsub_assembly_matches_portable_off_the_canonical_bound() {
        let mut state = 0xc2b2_ae3d_27d4_eb4fu64;
        for case in 0..20_000 {
            let a = next_raw(&mut state);
            let b = next_raw(&mut state);
            for modulus in [P, crate::consts::R] {
                for (route, sub) in SUB_ROUTES {
                    assert_eq!(
                        sub(&a, &modulus, &modulus),
                        limb::sub_mod_portable(&a, &modulus, &modulus),
                        "{route} fold case {case}",
                    );
                    assert_eq!(
                        sub(&a, &b, &modulus),
                        limb::sub_mod_portable(&a, &b, &modulus),
                        "{route} sub case {case}",
                    );
                }
                for (route, add) in ADD_ROUTES {
                    assert_eq!(
                        add(&a, &b, &modulus),
                        limb::add_mod_portable(&a, &b, &modulus),
                        "{route} add case {case}",
                    );
                }
            }
        }
    }

    #[cfg(narsil_a64_addsub)]
    fn next_raw(state: &mut u64) -> [u64; 4] {
        core::array::from_fn(|_| {
            *state ^= *state << 13;
            *state ^= *state >> 7;
            *state ^= *state << 17;
            *state
        })
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
/// `cargo test --release --features std --lib ab_sosd -- --ignored --nocapture`.
#[cfg(all(
    test,
    any(
        narsil_a64_sosd2,
        narsil_a64_sosd4,
        narsil_a64_sosd6,
        narsil_a64_addsub
    )
))]
mod ab {
    #[cfg(narsil_a64_sosd2)]
    use super::sosd2;
    #[cfg(narsil_a64_sosd4)]
    use super::sosd4;
    #[cfg(narsil_a64_sosd6)]
    use super::sosd6;
    #[cfg(any(narsil_a64_sosd4, narsil_a64_sosd6))]
    use crate::fp::sos::Fp2Product;
    #[cfg(narsil_a64_sosd2)]
    use crate::fp::sos::sosd2_portable;
    #[cfg(narsil_a64_sosd4)]
    use crate::fp::sos::sosd4_portable;
    #[cfg(narsil_a64_sosd6)]
    use crate::fp::sos::sosd6_portable;
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

    /// Medians of the two sorted sample sets and their ratio.
    fn report(name: &str, leaf: &mut [f64], port: &mut [f64]) {
        leaf.sort_by(|a, b| a.partial_cmp(b).unwrap());
        port.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let (l, p) = (leaf[leaf.len() / 2], port[port.len() / 2]);
        eprintln!("{name} leaf     median {:.2} ns  min {:.2}", l, leaf[0]);
        eprintln!("{name} portable median {:.2} ns  min {:.2}", p, port[0]);
        eprintln!(
            "ratio leaf/portable  {:.3}   ({:+.1}%)",
            l / p,
            (l / p - 1.0) * 100.0
        );
    }

    /// The two folds are single-digit-nanosecond calls, so each sample runs a
    /// dependent chain: the result feeds the next left operand. That measures
    /// the latency the tower actually waits on, not a throughput best case.
    #[cfg(narsil_a64_addsub)]
    #[test]
    #[ignore]
    fn ab_add_mod() {
        ab_fold(
            "add_mod inline",
            crate::limb::add_mod_inline,
            crate::limb::add_mod_portable,
            0x51ed_2701_a4ba_c39f,
        );
        ab_fold(
            "add_mod leaf",
            super::add_mod,
            crate::limb::add_mod_portable,
            0x51ed_2701_a4ba_c39f,
        );
    }

    #[cfg(narsil_a64_addsub)]
    #[test]
    #[ignore]
    fn ab_sub_mod() {
        ab_fold(
            "sub_mod inline",
            crate::limb::sub_mod_inline,
            crate::limb::sub_mod_portable,
            0x1d8e_4e27_c47d_124f,
        );
        ab_fold(
            "sub_mod leaf",
            super::sub_mod,
            crate::limb::sub_mod_portable,
            0x1d8e_4e27_c47d_124f,
        );
    }

    /// Generic over both routes so each one inlines: a fn pointer would add
    /// the same indirect call to both sides and compress the ratio.
    #[cfg(narsil_a64_addsub)]
    fn ab_fold<L, Q>(name: &str, leaf: L, portable: Q, seed: u64)
    where
        L: Fn(&[u64; 4], &[u64; 4], &[u64; 4]) -> [u64; 4],
        Q: Fn(&[u64; 4], &[u64; 4], &[u64; 4]) -> [u64; 4],
    {
        let modulus = crate::consts::P;
        let operands = residues(seed, 256);
        let iters = 200_000usize;
        let samples = 41usize;
        let mut chained = [Vec::with_capacity(samples), Vec::with_capacity(samples)];
        let mut free = [Vec::with_capacity(samples), Vec::with_capacity(samples)];
        for i in 0..8192 {
            let k = i & 255;
            black_box(leaf(&operands[k], &operands[k ^ 1], &modulus));
            black_box(portable(&operands[k], &operands[k ^ 1], &modulus));
        }
        for s in 0..samples {
            // Alternate which goes first so drift cancels.
            let order = if s % 2 == 0 { [0, 1] } else { [1, 0] };
            for which in order {
                let mut acc = operands[0];
                let t = Instant::now();
                if which == 0 {
                    for i in 0..iters {
                        acc = leaf(&acc, black_box(&operands[i & 255]), black_box(&modulus));
                    }
                } else {
                    for i in 0..iters {
                        acc = portable(&acc, black_box(&operands[i & 255]), black_box(&modulus));
                    }
                }
                chained[which].push(t.elapsed().as_nanos() as f64 / iters as f64);
                black_box(acc);

                let t = Instant::now();
                if which == 0 {
                    for i in 0..iters {
                        let k = i & 255;
                        black_box(leaf(&operands[k], &operands[k ^ 1], &modulus));
                    }
                } else {
                    for i in 0..iters {
                        let k = i & 255;
                        black_box(portable(&operands[k], &operands[k ^ 1], &modulus));
                    }
                }
                free[which].push(t.elapsed().as_nanos() as f64 / iters as f64);
            }
        }
        let [mut leaf_chained, mut port_chained] = chained;
        let [mut leaf_free, mut port_free] = free;
        report(
            &format!("{name} chained"),
            &mut leaf_chained,
            &mut port_chained,
        );
        report(
            &format!("{name} independent"),
            &mut leaf_free,
            &mut port_free,
        );
    }

    #[cfg(narsil_a64_sosd2)]
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
        report("sosd2", &mut leaf, &mut port);
    }

    /// Two Fp2 products per call, so one operand set is eight residues.
    #[cfg(narsil_a64_sosd4)]
    fn products4(case: &[[u64; 4]; 8]) -> [Fp2Product<'_>; 2] {
        core::array::from_fn(|i| {
            let [x0, x1, y0, y1] = [0, 1, 2, 3].map(|part| &case[4 * i + part]);
            (x0, x1, y0, y1)
        })
    }

    #[cfg(narsil_a64_sosd4)]
    #[test]
    #[ignore]
    fn ab_sosd4() {
        let pool = residues(5, 256 * 8);
        let cases: Vec<[[u64; 4]; 8]> = (0..256)
            .map(|k| core::array::from_fn(|j| pool[8 * k + j]))
            .collect();
        let iters = 10_000usize;
        let samples = 41usize;
        let mut leaf = Vec::with_capacity(samples);
        let mut port = Vec::with_capacity(samples);
        // Warm both.
        for i in 0..4096 {
            black_box(sosd4(products4(&cases[i & 255])));
            black_box(sosd4_portable(products4(&cases[i & 255])));
        }
        for s in 0..samples {
            // Alternate which goes first so drift cancels.
            let order = if s % 2 == 0 { [0, 1] } else { [1, 0] };
            for which in order {
                let t = Instant::now();
                if which == 0 {
                    for i in 0..iters {
                        black_box(sosd4(products4(black_box(&cases[i & 255]))));
                    }
                } else {
                    for i in 0..iters {
                        black_box(sosd4_portable(products4(black_box(&cases[i & 255]))));
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
        report("sosd4", &mut leaf, &mut port);
    }

    /// Three Fp2 products per call, so one operand set is twelve residues.
    #[cfg(narsil_a64_sosd6)]
    fn products(case: &[[u64; 4]; 12]) -> [Fp2Product<'_>; 3] {
        core::array::from_fn(|i| {
            let [x0, x1, y0, y1] = [0, 1, 2, 3].map(|part| &case[4 * i + part]);
            (x0, x1, y0, y1)
        })
    }

    #[cfg(narsil_a64_sosd6)]
    #[test]
    #[ignore]
    fn ab_sosd6() {
        let pool = residues(3, 256 * 12);
        let cases: Vec<[[u64; 4]; 12]> = (0..256)
            .map(|k| core::array::from_fn(|j| pool[12 * k + j]))
            .collect();
        let iters = 8_000usize;
        let samples = 41usize;
        let mut leaf = Vec::with_capacity(samples);
        let mut port = Vec::with_capacity(samples);
        // Warm both.
        for i in 0..4096 {
            black_box(sosd6(products(&cases[i & 255])));
            black_box(sosd6_portable(products(&cases[i & 255])));
        }
        for s in 0..samples {
            // Alternate which goes first so drift cancels.
            let order = if s % 2 == 0 { [0, 1] } else { [1, 0] };
            for which in order {
                let t = Instant::now();
                if which == 0 {
                    for i in 0..iters {
                        black_box(sosd6(products(black_box(&cases[i & 255]))));
                    }
                } else {
                    for i in 0..iters {
                        black_box(sosd6_portable(products(black_box(&cases[i & 255]))));
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
        report("sosd6", &mut leaf, &mut port);
    }
}
