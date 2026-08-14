//! x86-64 (BMI2+ADX) Montgomery backend using the generated kernel leaves.
//!
//! The assembly is generated at build time from the readable schedule DSL in
//! `build/schedule.rs` and verified by `tests/kernelgen_verify.rs`. See
//! ADR 0001 (build-time amendment). Inspect a copy with
//! `NARSIL_DUMP_ASM=<absolute dir> cargo build`. This module exists only
//! when `build.rs` emitted `narsil_mont4_x86_64_adx`. The target must be
//! x86-64 Linux with `bmi2` and `adx` in the compile-time target features.
//! Anything else uses the portable tier -- never a silent runtime fallback.

use crate::abi::{
    narsil_cyc_sqr_x86, narsil_f2sqr_small_x86, narsil_f2sqr_x86, narsil_fp4_sqr_x86,
    narsil_fp6_mul_x86, narsil_fp12_034_x86, narsil_fp12_034k_x86, narsil_fp12_034l_x86,
    narsil_fp12_mul_x86, narsil_fp12_sqr_mcl_x86, narsil_fp12_sqr_x86, narsil_g2_ysqr_x86,
    narsil_mont4_mul_x86, narsil_mont4_sqr_x86, narsil_sos_x86, narsil_sosd2_small_x86,
    narsil_sosd2_x86, narsil_sosd6_x86,
};
use crate::fp::Fp;

// Arithmetic constants remain typed Rust data. The assembly owns only the
// instruction schedule. `repr(C)` plus the assertions pins the tiny FFI view.
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

/// Extended table of the fp6 kernel: the mont4 shape plus the xi-scaling
/// quotient estimate `mu = floor(2^310/p)`. A separate static so the mont4
/// contract stays untouched.
#[repr(C)]
pub(crate) struct Fp6MulConstants {
    modulus: [u64; 4],
    negative_inverse: u64,
    mu: u64,
}

static FP6_MUL_CONSTANTS: Fp6MulConstants = Fp6MulConstants {
    modulus: crate::consts::P,
    negative_inverse: crate::consts::P_INV,
    mu: crate::consts::P_MU_310,
};

const _: () = {
    assert!(core::mem::size_of::<Fp6MulConstants>() == 6 * core::mem::size_of::<u64>());
    assert!(core::mem::offset_of!(Fp6MulConstants, negative_inverse) == 32);
    assert!(core::mem::offset_of!(Fp6MulConstants, mu) == 40);
};

// Kernel-ABI layout invariant: Fp6 is 24 contiguous limbs in component
// order c0.re, c0.im, c1.re, c1.im, c2.re, c2.im (repr(C) on Fp6/Fp2/Fp).
const _: () = {
    use crate::fp2::Fp2;
    use crate::fp6::Fp6;
    assert!(core::mem::size_of::<Fp6>() == 24 * core::mem::size_of::<u64>());
    assert!(core::mem::align_of::<Fp6>() == core::mem::align_of::<u64>());
    assert!(core::mem::offset_of!(Fp6, c0) == 0);
    assert!(core::mem::offset_of!(Fp6, c1) == 64);
    assert!(core::mem::offset_of!(Fp6, c2) == 128);
    assert!(core::mem::size_of::<Fp2>() == 8 * core::mem::size_of::<u64>());
    assert!(core::mem::offset_of!(Fp2, c0) == 0);
    assert!(core::mem::offset_of!(Fp2, c1) == 32);
};

// Kernel-ABI layout invariant: Fp12 is 48 contiguous limbs, c0 then c1
// (repr(C) on Fp12. The Fp6/Fp2/Fp layers are pinned above).
const _: () = {
    use crate::fp2::Fp2;
    use crate::fp12::Fp12;
    assert!(core::mem::size_of::<Fp12>() == 48 * core::mem::size_of::<u64>());
    assert!(core::mem::align_of::<Fp12>() == core::mem::align_of::<u64>());
    assert!(core::mem::offset_of!(Fp12, c0) == 0);
    assert!(core::mem::offset_of!(Fp12, c1) == 192);
    // The wrapper stages the three sparse coefficients as [Fp2. 3].
    assert!(core::mem::size_of::<[Fp2; 3]>() == 24 * core::mem::size_of::<u64>());
};

#[cfg(any(test, narsil_fp12_mul_asm))]
#[inline(never)]
pub(crate) fn fp12_mul_assign(f: &mut crate::fp12::Fp12, rhs: &crate::fp12::Fp12) {
    #[cfg(debug_assertions)]
    for operand in [&*f, rhs] {
        // SAFETY: repr(C) Fp12 is 48 limbs (asserted above).
        let components = crate::abi::fp12_components(operand);
        for component in components {
            debug_assert!(!crate::limb::gte(component, &crate::consts::P));
        }
    }
    unsafe {
        // SAFETY: the repr(C) Fp12 references satisfy the complete kernel
        // contract above (canonical inputs are the Fp invariant). Z == a is
        // the contract's in-place shape. The assembly initializes all 384
        // output bytes and cannot retain any pointer.
        narsil_fp12_mul_x86(
            f as *mut crate::fp12::Fp12 as *mut u64,
            f as *const crate::fp12::Fp12 as *const u64,
            rhs as *const crate::fp12::Fp12 as *const u64,
            &FP6_MUL_CONSTANTS,
        );
    }
}

#[cfg(any(test, narsil_fp12_sqr_asm))]
#[inline(never)]
pub(crate) fn fp12_sqr_assign(f: &mut crate::fp12::Fp12) {
    #[cfg(debug_assertions)]
    {
        // SAFETY: repr(C) Fp12 is 48 limbs (asserted above).
        let components = crate::abi::fp12_components(f);
        for component in components {
            debug_assert!(!crate::limb::gte(component, &crate::consts::P));
        }
    }
    unsafe {
        // SAFETY: the repr(C) Fp12 reference satisfies the complete kernel
        // contract above (canonical inputs are the Fp invariant). Z == f is
        // the contract's in-place shape. The assembly initializes all 384
        // output bytes and cannot retain any pointer.
        narsil_fp12_sqr_x86(
            f as *mut crate::fp12::Fp12 as *mut u64,
            f as *const crate::fp12::Fp12 as *const u64,
            &FP6_MUL_CONSTANTS,
        );
    }
}

#[cfg(any(test, narsil_fp12_sqr_mcl_asm))]
#[inline(never)]
pub(crate) fn fp12_sqr_mcl_assign(f: &mut crate::fp12::Fp12) {
    #[cfg(debug_assertions)]
    {
        // SAFETY: repr(C) Fp12 is 48 limbs (asserted above).
        let components = crate::abi::fp12_components(f);
        for component in components {
            debug_assert!(!crate::limb::gte(component, &crate::consts::P));
        }
    }
    unsafe {
        // SAFETY: `f` is a live, aligned repr(C) Fp12 of canonical residues.
        // the assembly wrapper initializes all 384 bytes in place, preserves
        // the System V callee-saved set, and retains no pointer.
        narsil_fp12_sqr_mcl_x86(f as *mut crate::fp12::Fp12 as *mut u64, &FP6_MUL_CONSTANTS);
    }
}

#[cfg(any(test, narsil_cyc_sqr_asm))]
#[inline(never)]
pub(crate) fn cyc_sqr_assign(f: &mut crate::fp12::Fp12) {
    #[cfg(debug_assertions)]
    {
        // SAFETY: repr(C) Fp12 is 48 limbs (asserted above).
        let components = crate::abi::fp12_components(f);
        for component in components {
            debug_assert!(!crate::limb::gte(component, &crate::consts::P));
        }
    }
    unsafe {
        // SAFETY: the repr(C) Fp12 reference satisfies the complete kernel
        // contract above (canonical inputs are the Fp invariant). Z == f is
        // the contract's in-place shape. The assembly initializes all 384
        // output bytes and cannot retain any pointer.
        narsil_cyc_sqr_x86(
            f as *mut crate::fp12::Fp12 as *mut u64,
            f as *const crate::fp12::Fp12 as *const u64,
            &FP6_MUL_CONSTANTS,
        );
    }
}

#[cfg(any(test, narsil_fp4_sqr_asm))]
#[inline(always)]
pub(crate) fn fp4_sqr(
    r0: &crate::fp2::Fp2,
    r1: &crate::fp2::Fp2,
) -> (crate::fp2::Fp2, crate::fp2::Fp2) {
    #[cfg(debug_assertions)]
    for operand in [r0, r1] {
        // SAFETY: repr(C) Fp2 is 8 limbs (asserted above).
        let components = crate::abi::fp2_components(operand);
        for component in components {
            debug_assert!(!crate::limb::gte(component, &crate::consts::P));
        }
    }
    let mut z = core::mem::MaybeUninit::<[crate::fp2::Fp2; 2]>::uninit();
    unsafe {
        // SAFETY: repr(C) Fp2 references and the fresh local output satisfy
        // the complete kernel contract above (canonical inputs are the Fp
        // invariant. The local z cannot alias the inputs). The assembly
        // initializes all 128 output bytes and cannot retain any pointer.
        narsil_fp4_sqr_x86(
            z.as_mut_ptr() as *mut u64,
            r0 as *const crate::fp2::Fp2 as *const u64,
            r1 as *const crate::fp2::Fp2 as *const u64,
            &FP6_MUL_CONSTANTS,
        );
        let [t0, t1] = z.assume_init();
        (t0, t1)
    }
}

/// Whole sparse Fp12 product `f *= c0 + c3*w + c4*v*w` (the Miller loop's
/// per-line update, arkworks `mul_by_034`) through the dedicated leaf: six
/// dual-lane T = 6 sums of products with the xi = 9 + u scalings of c3/c4
/// computed in the kernel. `f` and the three coefficients must be canonical.
/// The kernel reads all inputs before it writes the canonical result to `f`.
// Dispatched by default on Intel targets and opt-in elsewhere through
// NARSIL_FP12_034_ASM. Sibling leaves take precedence when enabled (034K >
// 034L > v1). Covered by the leaf differential tests.
#[cfg(any(
    test,
    all(
        narsil_fp12_034_asm,
        not(narsil_fp12_034l_asm),
        not(narsil_fp12_034k_asm)
    )
))]
#[inline(never)]
pub(crate) fn fp12_034_assign(
    f: &mut crate::fp12::Fp12,
    c0: &crate::fp2::Fp2,
    c3: &crate::fp2::Fp2,
    c4: &crate::fp2::Fp2,
) {
    let coefficients = [*c0, *c3, *c4];
    #[cfg(debug_assertions)]
    {
        // SAFETY: repr(C) Fp12 is 48 limbs (asserted above).
        let components = crate::abi::fp12_components(f);
        for component in components {
            debug_assert!(!crate::limb::gte(component, &crate::consts::P));
        }
        // SAFETY: [Fp2. 3] is 24 contiguous limbs (asserted above).
        let staged = crate::abi::fp2_triplet_components(&coefficients);
        for component in staged {
            debug_assert!(!crate::limb::gte(component, &crate::consts::P));
        }
    }
    unsafe {
        // SAFETY: the repr(C) Fp12 reference and the staged coefficients
        // satisfy the complete kernel contract above (canonical inputs are
        // the Fp invariant). Z == f is the contract's in-place shape. The
        // assembly initializes all 384 output bytes and cannot retain any
        // pointer.
        narsil_fp12_034_x86(
            f as *mut crate::fp12::Fp12 as *mut u64,
            f as *const crate::fp12::Fp12 as *const u64,
            coefficients.as_ptr() as *const u64,
            &FP6_MUL_CONSTANTS,
        );
    }
}

#[cfg(any(test, all(narsil_fp12_034l_asm, not(narsil_fp12_034k_asm))))]
#[inline(never)]
pub(crate) fn fp12_034l_assign(
    f: &mut crate::fp12::Fp12,
    c0: &crate::fp2::Fp2,
    c3: &crate::fp2::Fp2,
    c4: &crate::fp2::Fp2,
) {
    let coefficients = [*c0, *c3, *c4];
    #[cfg(debug_assertions)]
    {
        // SAFETY: repr(C) Fp12 is 48 limbs (asserted above).
        let components = crate::abi::fp12_components(f);
        for component in components {
            debug_assert!(!crate::limb::gte(component, &crate::consts::P));
        }
        // SAFETY: [Fp2. 3] is 24 contiguous limbs (asserted above).
        let staged = crate::abi::fp2_triplet_components(&coefficients);
        for component in staged {
            debug_assert!(!crate::limb::gte(component, &crate::consts::P));
        }
    }
    unsafe {
        // SAFETY: the repr(C) Fp12 reference and the staged coefficients
        // satisfy the complete kernel contract above (canonical inputs are
        // the Fp invariant). Z == f is the contract's in-place shape. The
        // assembly initializes all 384 output bytes and cannot retain any
        // pointer.
        narsil_fp12_034l_x86(
            f as *mut crate::fp12::Fp12 as *mut u64,
            f as *const crate::fp12::Fp12 as *const u64,
            coefficients.as_ptr() as *const u64,
            &FP6_MUL_CONSTANTS,
        );
    }
}

#[cfg(any(test, narsil_fp12_034k_asm))]
#[inline(never)]
pub(crate) fn fp12_034k_assign(
    f: &mut crate::fp12::Fp12,
    c0: &crate::fp2::Fp2,
    c3: &crate::fp2::Fp2,
    c4: &crate::fp2::Fp2,
) {
    let coefficients = [*c0, *c3, *c4];
    #[cfg(debug_assertions)]
    {
        // SAFETY: repr(C) Fp12 is 48 limbs (asserted above).
        let components = crate::abi::fp12_components(f);
        for component in components {
            debug_assert!(!crate::limb::gte(component, &crate::consts::P));
        }
        // SAFETY: [Fp2. 3] is 24 contiguous limbs (asserted above).
        let staged = crate::abi::fp2_triplet_components(&coefficients);
        for component in staged {
            debug_assert!(!crate::limb::gte(component, &crate::consts::P));
        }
    }
    unsafe {
        // SAFETY: the repr(C) Fp12 reference and the staged coefficients
        // satisfy the complete kernel contract above (canonical inputs are
        // the Fp invariant). Z == f is the contract's in-place shape. The
        // assembly initializes all 384 output bytes and cannot retain any
        // pointer.
        narsil_fp12_034k_x86(
            f as *mut crate::fp12::Fp12 as *mut u64,
            f as *const crate::fp12::Fp12 as *const u64,
            coefficients.as_ptr() as *const u64,
            &FP6_MUL_CONSTANTS,
        );
    }
}

/// `a` and `b` contain canonical field values. They may alias. `z` contains a
/// canonical `Fp6` value on return.
#[cfg(any(test, narsil_fp6_asm))]
#[inline(never)]
pub(crate) fn fp6_mul(a: &crate::fp6::Fp6, b: &crate::fp6::Fp6) -> crate::fp6::Fp6 {
    #[cfg(debug_assertions)]
    for operand in [a, b] {
        // SAFETY: repr(C) Fp6 is 24 limbs (asserted above).
        let components = crate::abi::fp6_components(operand);
        for component in components {
            debug_assert!(!crate::limb::gte(component, &crate::consts::P));
        }
    }
    let mut z = core::mem::MaybeUninit::<crate::fp6::Fp6>::uninit();
    unsafe {
        // SAFETY: repr(C) Fp6 references and the local output satisfy the
        // complete kernel contract above (canonical inputs are the Fp
        // invariant). The assembly initializes all 192 bytes before
        // `assume_init` and cannot retain any pointer.
        narsil_fp6_mul_x86(
            z.as_mut_ptr() as *mut u64,
            a as *const crate::fp6::Fp6 as *const u64,
            b as *const crate::fp6::Fp6 as *const u64,
            &FP6_MUL_CONSTANTS,
        );
        z.assume_init()
    }
}

/// Reduced Montgomery multiplication through the generated x86-64 leaf.
///
/// Kernel contract: `a` and `b` are readable, 8-byte-aligned 32-byte arrays.
/// `a` and `b` must both be residues below the BN254 base modulus. The
/// dual-chain schedule's 2^320 carry bound needs both operands below p:
/// larger `a` miscomputes (interpreter-caught, hardware-confirmed), so
/// `Fp::from_raw` reduces before calling. They
/// may alias each other. The wrapper supplies distinct output and immutable
/// constant-table
/// pointers, both suitably aligned and live for the call. The leaf
/// initializes all four output limbs, returns a fully reduced residue, saves
/// every callee-saved register it uses, keeps rsp 8-mod-16 aligned as on
/// entry, and neither calls Rust nor unwinds.
#[inline(never)]
pub fn mont_mul(a: &[u64; 4], b: &[u64; 4]) -> Fp {
    debug_assert!(!crate::limb::gte(a, &crate::consts::P));
    debug_assert!(!crate::limb::gte(b, &crate::consts::P));
    let mut z = core::mem::MaybeUninit::<[u64; 4]>::uninit();
    unsafe {
        // SAFETY: fixed-size references and the local output satisfy the
        // complete kernel contract above. The assembly initializes 32 bytes
        // before `assume_init` and cannot retain any pointer.
        narsil_mont4_mul_x86(
            z.as_mut_ptr() as *mut u64,
            a.as_ptr(),
            b.as_ptr(),
            &MONT4_CONSTANTS,
        );
        Fp(z.assume_init())
    }
}

/// Dedicated Montgomery squaring (ten-product schedule). Same contract as
/// [`mont_mul`], with the unused `y` argument passed as `a`.
#[inline(never)]
pub fn mont_sqr(a: &[u64; 4]) -> Fp {
    debug_assert!(!crate::limb::gte(a, &crate::consts::P));
    let mut z = core::mem::MaybeUninit::<[u64; 4]>::uninit();
    unsafe {
        // SAFETY: as in `mont_mul`. The square reads only `x` and the table.
        narsil_mont4_sqr_x86(
            z.as_mut_ptr() as *mut u64,
            a.as_ptr(),
            a.as_ptr(),
            &MONT4_CONSTANTS,
        );
        Fp(z.assume_init())
    }
}

/// `(sum_i a_i * b_i) * R^{-1} mod p` through the rolled SoS leaf: `N/2`
/// operand pairs as a pointer table (`a_0, b_0, a_1, b_1, ...`).
///
/// Kernel contract: an even count of 2..=10 pairs (the inner walk takes two
/// pairs per iteration). Every pointer refers to a readable,
/// 8-byte-aligned 32-byte array holding a value at most p (`negp` output may
/// equal p. The SoS bounds only need operands <= p, unlike the strict < p of
/// the mont4 leaves). The wrapper supplies distinct output and constant-table
/// pointers, live for the call. The leaf initializes all four output limbs,
/// returns the canonical residue, saves every callee-saved register it uses,
/// keeps rsp 8-mod-16 aligned as on entry, and neither calls Rust nor
/// unwinds.
#[inline(always)]
fn sos_leaf<const N: usize>(pairs: &[*const u64; N]) -> [u64; 4] {
    const {
        assert!(N % 4 == 0 && N >= 4 && N <= 20, "even pair count in 2..=10");
    }
    #[cfg(debug_assertions)]
    for operand in pairs {
        // SAFETY: caller passes pointers to live [u64. 4] operands.
        debug_assert!(!crate::limb::gt(
            unsafe { crate::abi::limbs_from_ptr(*operand) },
            &crate::consts::P
        ));
    }
    let mut z = core::mem::MaybeUninit::<[u64; 4]>::uninit();
    unsafe {
        // SAFETY: fixed-size table and the local output satisfy the complete
        // kernel contract above. The assembly initializes 32 bytes before
        // `assume_init` and cannot retain any pointer.
        narsil_sos_x86(
            z.as_mut_ptr() as *mut u64,
            pairs.as_ptr(),
            (N / 2) as u64,
            &MONT4_CONSTANTS,
        );
        z.assume_init()
    }
}

// Leaf-backed SoS entry points, one per portable kernel in `fp/sos.rs`
// (identical semantics. `fp/sos.rs` dispatches here on the ADX tier). The
// dual-lane pairs mirror the portable lane definitions exactly:
// lane0 = sum x_{i0}*y_{i0} - x_{i1}*y_{i1} (subtraction via negp),
// lane1 = sum x_{i0}*y_{i1} + x_{i1}*y_{i0}.

pub(crate) fn sos2(a0: &[u64; 4], b0: &[u64; 4], a1: &[u64; 4], b1: &[u64; 4]) -> [u64; 4] {
    sos_leaf(&[a0.as_ptr(), b0.as_ptr(), a1.as_ptr(), b1.as_ptr()])
}

pub(crate) fn sos4(products: [crate::fp::sos::Product<'_>; 4]) -> [u64; 4] {
    let [(a0, b0), (a1, b1), (a2, b2), (a3, b3)] = products;
    sos_leaf(&[
        a0.as_ptr(),
        b0.as_ptr(),
        a1.as_ptr(),
        b1.as_ptr(),
        a2.as_ptr(),
        b2.as_ptr(),
        a3.as_ptr(),
        b3.as_ptr(),
    ])
}

/// `g^2 - 3*e^2` over Fp2 through the lazy double-width leaf: four raw 4x4
/// products held unreduced, one guarded 512-bit subtraction and one
/// Montgomery reduction per output half.
///
/// Kernel contract: `g`, `e` and `f` are readable, 8-byte-aligned 64-byte
/// arrays of residues below p, `f` is `3*e` in Fp2, and they may alias each
/// other but not the output. The leaf initializes all eight output limbs
/// with canonical residues, saves every callee-saved register it uses, keeps
/// rsp 8-mod-16 aligned as on entry, and neither calls Rust nor unwinds.
#[cfg(any(test, narsil_g2_ysqr_asm))]
pub(crate) fn g2_ysqr(
    g: &[[u64; 4]; 2],
    e: &[[u64; 4]; 2],
    f: &[[u64; 4]; 2],
) -> ([u64; 4], [u64; 4]) {
    #[cfg(debug_assertions)]
    for operand in [g, e, f] {
        debug_assert!(!crate::limb::gte(&operand[0], &crate::consts::P));
        debug_assert!(!crate::limb::gte(&operand[1], &crate::consts::P));
    }
    let mut z = core::mem::MaybeUninit::<[u64; 8]>::uninit();
    unsafe {
        // SAFETY: fixed-size references and the local output satisfy the
        // complete kernel contract above. The assembly initializes 64 bytes
        // before `assume_init` and cannot retain any pointer.
        narsil_g2_ysqr_x86(
            z.as_mut_ptr() as *mut u64,
            g.as_ptr() as *const u64,
            e.as_ptr() as *const u64,
            f.as_ptr() as *const u64,
            &MONT4_CONSTANTS,
        );
        let z = z.assume_init();
        ([z[0], z[1], z[2], z[3]], [z[4], z[5], z[6], z[7]])
    }
}

/// Complex Fp2 square through the fused dual-lane leaf:
/// `((x0+x1)*(x0-x1)/R, x0*(2*x1)/R)`, the two Fp components of
/// `(x0 + x1*u)^2` over `Fp2 = Fp[u]/(u^2 + 1)`.
///
/// Kernel contract: `x0` and `x1` are readable, 8-byte-aligned 32-byte arrays
/// holding residues below p, and may alias. The leaf canonicalizes the three
/// operand images itself, so both Montgomery chains stay inside the mont4
/// carry bound. It initializes all eight output limbs with canonical
/// residues, saves every callee-saved register it uses, keeps rsp 8-mod-16
/// aligned as on entry, and neither calls Rust nor unwinds.
#[cfg(any(test, narsil_f2sqr_asm))]
pub(crate) fn f2sqr(x0: &[u64; 4], x1: &[u64; 4]) -> ([u64; 4], [u64; 4]) {
    debug_assert!(!crate::limb::gte(x0, &crate::consts::P));
    debug_assert!(!crate::limb::gte(x1, &crate::consts::P));
    let leaf = if cfg!(narsil_f2sqr_small) {
        narsil_f2sqr_small_x86
    } else {
        narsil_f2sqr_x86
    };
    let mut z = core::mem::MaybeUninit::<[u64; 8]>::uninit();
    unsafe {
        // SAFETY: fixed-size references and the local output satisfy the
        // complete kernel contract above. The assembly initializes 64 bytes
        // before `assume_init` and cannot retain any pointer.
        leaf(
            z.as_mut_ptr() as *mut u64,
            x0.as_ptr(),
            x1.as_ptr(),
            &MONT4_CONSTANTS,
        );
        let z = z.assume_init();
        ([z[0], z[1], z[2], z[3]], [z[4], z[5], z[6], z[7]])
    }
}

#[cfg(any(test, narsil_sosd2_asm))]
pub(crate) fn sosd2(
    x0: &[u64; 4],
    x1: &[u64; 4],
    y0: &[u64; 4],
    y1: &[u64; 4],
) -> ([u64; 4], [u64; 4]) {
    #[cfg(debug_assertions)]
    for operand in [x0, x1, y0, y1] {
        debug_assert!(!crate::limb::gt(operand, &crate::consts::P));
    }
    let leaf = if cfg!(narsil_sosd2_small) {
        narsil_sosd2_small_x86
    } else {
        narsil_sosd2_x86
    };
    let mut z = core::mem::MaybeUninit::<[u64; 8]>::uninit();
    unsafe {
        // SAFETY: fixed-size references and the local output satisfy the
        // complete kernel contract above. The assembly initializes 64 bytes
        // before `assume_init` and cannot retain any pointer.
        leaf(
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

pub(crate) fn sosd4(products: [crate::fp::sos::Fp2Product<'_>; 2]) -> ([u64; 4], [u64; 4]) {
    let [(x00, x01, y00, y01), (x10, x11, y10, y11)] = products;
    let ny01 = crate::fp::sos::negp(y01);
    let ny11 = crate::fp::sos::negp(y11);
    (
        sos_leaf(&[
            x00.as_ptr(),
            y00.as_ptr(),
            x01.as_ptr(),
            ny01.as_ptr(),
            x10.as_ptr(),
            y10.as_ptr(),
            x11.as_ptr(),
            ny11.as_ptr(),
        ]),
        sos_leaf(&[
            x00.as_ptr(),
            y01.as_ptr(),
            x01.as_ptr(),
            y00.as_ptr(),
            x10.as_ptr(),
            y11.as_ptr(),
            x11.as_ptr(),
            y10.as_ptr(),
        ]),
    )
}

#[cfg(any(test, not(narsil_sosd6_asm)))]
pub(crate) fn sosd6(products: [crate::fp::sos::Fp2Product<'_>; 3]) -> ([u64; 4], [u64; 4]) {
    let [
        (x00, x01, y00, y01),
        (x10, x11, y10, y11),
        (x20, x21, y20, y21),
    ] = products;
    let ny01 = crate::fp::sos::negp(y01);
    let ny11 = crate::fp::sos::negp(y11);
    let ny21 = crate::fp::sos::negp(y21);
    (
        sos_leaf(&[
            x00.as_ptr(),
            y00.as_ptr(),
            x01.as_ptr(),
            ny01.as_ptr(),
            x10.as_ptr(),
            y10.as_ptr(),
            x11.as_ptr(),
            ny11.as_ptr(),
            x20.as_ptr(),
            y20.as_ptr(),
            x21.as_ptr(),
            ny21.as_ptr(),
        ]),
        sos_leaf(&[
            x00.as_ptr(),
            y01.as_ptr(),
            x01.as_ptr(),
            y00.as_ptr(),
            x10.as_ptr(),
            y11.as_ptr(),
            x11.as_ptr(),
            y10.as_ptr(),
            x20.as_ptr(),
            y21.as_ptr(),
            x21.as_ptr(),
            y20.as_ptr(),
        ]),
    )
}

/// Dual-lane sum of three Fp2 products through the dedicated T = 6 leaf:
/// `lane0 = (sum x_i0*y_i0 + x_i1*(p - y_i1))/R`, `lane1 = (sum x_i0*y_i1 +
/// x_i1*y_i0)/R`, both canonical, in one call -- the composed [`sosd6`]
/// route's value with the two lanes' carry chains interleaved in-kernel.
/// Two serial `narsil_sos_x86` walks are each one ~700-instruction
/// single-lane chain, so their independent lanes get no overlap. The leaf
/// alternates lane rows per multiplicand (the sosd2 finding at T = 6) and
/// computes the three `p - y_i1` images itself.
///
/// The leaf's ABI is one staged block built here: the 24 x limbs
/// transposed (so the kernel's multiplicand pointer walks linearly), then
/// five 64-byte y pair blocks `[y00, y01] [y01, y00] [y10, y11] [y11, y10]
/// [y20, y21]` -- pair block i holds pair i's lane0 row source then its
/// lane1 row source, and the kernel overwrites the low vectors of blocks 1
/// and 3 with `p - y01`, `p - y11` in place. Plain value copies here
/// replace the composed route's 24 pointer-table stores and three `negp`
/// temporaries. A 12-pointer-table ABI would instead cost the kernel ~120
/// staging instructions per call (see the schedule's ABI note).
///
/// Kernel contract: every operand is at most p (the portable `sosd6`
/// bound). `stage` is the writable 512-byte block above, `z` receives
/// eight limbs (lane0 then lane1), all initialized and canonical on
/// return. The leaf saves every callee-saved register it uses, keeps rsp
/// 8-mod-16 aligned as on entry, and neither calls Rust nor unwinds.
// Dispatched in production only under NARSIL_SOSD6_ASM=1. Always covered by
// the leaf differential tests.
#[cfg(any(test, narsil_sosd6_asm))]
pub(crate) fn sosd6_leaf(products: [crate::fp::sos::Fp2Product<'_>; 3]) -> ([u64; 4], [u64; 4]) {
    let [
        (x00, x01, y00, y01),
        (x10, x11, y10, y11),
        (x20, x21, y20, y21),
    ] = products;
    #[cfg(debug_assertions)]
    for operand in [x00, x01, y00, y01, x10, x11, y10, y11, x20, x21, y20, y21] {
        debug_assert!(!crate::limb::gt(operand, &crate::consts::P));
    }
    let mut stage = [0u64; 64];
    for (i, x) in [x00, x01, x10, x11, x20, x21].into_iter().enumerate() {
        for (j, limb) in x.iter().enumerate() {
            stage[6 * j + i] = *limb;
        }
    }
    for (block, y) in [y00, y01, y01, y00, y10, y11, y11, y10, y20, y21]
        .into_iter()
        .enumerate()
    {
        stage[24 + 4 * block..28 + 4 * block].copy_from_slice(y);
    }
    let mut z = core::mem::MaybeUninit::<[u64; 8]>::uninit();
    unsafe {
        // SAFETY: the local stage block and fixed-size references satisfy
        // the complete kernel contract above. The assembly initializes 64
        // output bytes before `assume_init`, writes nothing outside z and
        // stage, and cannot retain any pointer.
        narsil_sosd6_x86(
            z.as_mut_ptr() as *mut u64,
            stage.as_mut_ptr(),
            &MONT4_CONSTANTS,
        );
        let z = z.assume_init();
        ([z[0], z[1], z[2], z[3]], [z[4], z[5], z[6], z[7]])
    }
}

#[cfg(test)]
pub(crate) fn sosd8(products: [crate::fp::sos::Fp2Product<'_>; 4]) -> ([u64; 4], [u64; 4]) {
    let [
        (x00, x01, y00, y01),
        (x10, x11, y10, y11),
        (x20, x21, y20, y21),
        (x30, x31, y30, y31),
    ] = products;
    let ny01 = crate::fp::sos::negp(y01);
    let ny11 = crate::fp::sos::negp(y11);
    let ny21 = crate::fp::sos::negp(y21);
    let ny31 = crate::fp::sos::negp(y31);
    (
        sos_leaf(&[
            x00.as_ptr(),
            y00.as_ptr(),
            x01.as_ptr(),
            ny01.as_ptr(),
            x10.as_ptr(),
            y10.as_ptr(),
            x11.as_ptr(),
            ny11.as_ptr(),
            x20.as_ptr(),
            y20.as_ptr(),
            x21.as_ptr(),
            ny21.as_ptr(),
            x30.as_ptr(),
            y30.as_ptr(),
            x31.as_ptr(),
            ny31.as_ptr(),
        ]),
        sos_leaf(&[
            x00.as_ptr(),
            y01.as_ptr(),
            x01.as_ptr(),
            y00.as_ptr(),
            x10.as_ptr(),
            y11.as_ptr(),
            x11.as_ptr(),
            y10.as_ptr(),
            x20.as_ptr(),
            y21.as_ptr(),
            x21.as_ptr(),
            y20.as_ptr(),
            x30.as_ptr(),
            y31.as_ptr(),
            x31.as_ptr(),
            y30.as_ptr(),
        ]),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consts::{MONT_ONE, MONT_R2, P};
    use crate::fp::Fp;
    use crate::fp::portable;
    use crate::limb;

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

    fn edge_and_carry_corpus() -> Vec<[u64; 4]> {
        let p_minus_one = limb::sub_noborrow(&P, &[1, 0, 0, 0]);
        let mut cases = vec![
            [0; 4],
            [1, 0, 0, 0],
            MONT_ONE,
            MONT_R2,
            p_minus_one,
            [u64::MAX, u64::MAX, u64::MAX, P[3] - 1],
            [u64::MAX, 0, u64::MAX, 0],
            [0, u64::MAX, 0, 0x1000_0000_0000_0000],
        ];
        let mut state = 0x243f_6a88_85a3_08d3u64;
        for _ in 0..256 {
            cases.push(next_residue(&mut state));
        }
        cases
    }

    /// Exact division by three of a four-limb multiple of three.
    fn div3(value: [u64; 4]) -> [u64; 4] {
        let mut out = [0u64; 4];
        let mut remainder = 0u128;
        for j in (0..4).rev() {
            let word = (remainder << 64) | value[j] as u128;
            out[j] = (word / 3) as u64;
            remainder = word % 3;
        }
        assert_eq!(remainder, 0, "the value is a multiple of three");
        out
    }

    /// One `(g, e)` operand pair for the lazy `g^2 - 3*e^2` leaf.
    type YsqrCase = ([[u64; 4]; 2], [[u64; 4]; 2]);

    /// `(g, e)` pairs for the lazy `g^2 - 3*e^2` leaf. The crafted entries
    /// are solved for the stage edges the leaf's bound proof rests on, the
    /// corpus entries draw g and e independently.
    fn g2_ysqr_cases() -> Vec<YsqrCase> {
        let zero = [0u64; 4];
        let one = [1u64, 0, 0, 0];
        let p_minus_one = limb::sub_noborrow(&P, &one);
        let p_minus_two = limb::sub_noborrow(&p_minus_one, &one);
        // 3*third = p-1, so e.im = third drives f.im to p-1 and e6 to 2p-2.
        let third = div3(p_minus_one);
        let pow2 = |bit: usize| {
            let mut value = [0u64; 4];
            value[bit / 64] = 1 << (bit % 64);
            value
        };
        let add = |a: [u64; 4], b: [u64; 4]| limb::add_mod(&a, &b, &P);
        let sub = |a: [u64; 4], b: [u64; 4]| limb::sub_mod(&a, &b, &P);
        // gs*gd = 3*2^252 - 1, one unit below es*ed = 3*2^252.
        let g_row_a = add(pow2(252), pow2(251));
        // 2*g.re*g.im = 6*2^200 - 2, two units below 2*e.re*f.im = 6*2^200.
        let g_row_b = sub(add(pow2(201), pow2(200)), one);

        let mut cases = vec![
            // g and e at independent maxima in one call: the raw sums
            // gs = 2p-3 and e6 = 2p-2 both sit at the four-limb ceiling that
            // the uncorrected operand sums rest on.
            ([p_minus_one, p_minus_two], [p_minus_one, third]),
            // The largest raw product the operand bounds admit, the maximal
            // raw sum 2p-3 times the maximal canonical gd = p-1.
            ([p_minus_two, p_minus_one], [p_minus_one, third]),
            // Both e-side maxima at once: ed = p-1 needs f.im = f.re + 1,
            // e6 = 2p-2 needs 3*e.im = p-1 mod p.
            ([p_minus_two, p_minus_one], [add(third, third), third]),
            // Both reductions a dozen units below their p*2^256 precondition.
            ([zero, zero], [[2, 0, 0, 0], one]),
            // Row A one unit below p*2^256, the tightest reduction input.
            ([g_row_a, sub(g_row_a, one)], [pow2(126), zero]),
            // Row B two units below p*2^256. Both of that row's products
            // carry a doubled operand, so no odd gap exists.
            ([one, g_row_b], [pow2(100), pow2(100)]),
        ];
        let corpus = edge_and_carry_corpus();
        for (index, first) in corpus.iter().enumerate() {
            let pick = |stride: usize| corpus[(index * stride + stride) % corpus.len()];
            cases.push(([*first, pick(7)], [pick(11), pick(13)]));
        }
        cases
    }

    /// The assembled lazy leaf against the route it replaces: two Fp2
    /// squares, then `y = g^2 - 3*e^2`. The leaf is assembled wherever this
    /// module exists, so this runs even when the production dispatch keeps
    /// the composed route.
    #[test]
    fn g2_ysqr_leaf_matches_the_composed_route() {
        use crate::fp2_fast::{f2_add, f2_dbl, f2_mul_karatsuba, f2_sub};

        for (case, (g, e)) in g2_ysqr_cases().iter().enumerate() {
            // The leaf's precondition: f = 3e.
            let f = e.map(|component| {
                let value = Fp::from_raw_unchecked(component);
                (value + value + value).mont_limbs()
            });
            let square = |x: &[[u64; 4]; 2]| f2_mul_karatsuba((x[0], x[1]), (x[0], x[1]));
            let e_square = square(e);
            let composed = f2_sub(square(g), f2_add(f2_dbl(e_square), e_square));
            assert_eq!(g2_ysqr(g, e, &f), composed, "case {case}");
        }
    }

    #[test]
    fn hot_matches_asm() {
        let a = Fp::from_u64(0x123456789);
        let b = Fp::from_u64(0x987654321);
        assert_eq!(a * b, mont_mul(&a.0, &b.0));
        assert_eq!(a.square(), mont_sqr(&a.0));
    }

    #[test]
    fn assembly_matches_portable_on_edges_and_carries() {
        let cases = edge_and_carry_corpus();
        for (index, a) in cases.iter().enumerate() {
            assert_eq!(mont_sqr(a), portable::mont_sqr(a), "square case {index}");
            assert_eq!(
                mont_sqr(a),
                mont_mul(a, a),
                "square/multiply divergence, case {index}",
            );
            for (other_index, b) in cases.iter().step_by(17).enumerate() {
                assert_eq!(
                    mont_mul(a, b),
                    portable::mont_mul(a, b),
                    "multiply case {index}/{other_index}",
                );
                assert_eq!(
                    mont_mul(a, b),
                    mont_mul(b, a),
                    "commutativity case {index}/{other_index}",
                );
            }
        }
    }

    #[test]
    fn assembly_matches_portable_on_random_products() {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        for case in 0..65_536 {
            let a = next_residue(&mut state);
            let b = next_residue(&mut state);
            assert_eq!(
                mont_mul(&a, &b),
                portable::mont_mul(&a, &b),
                "multiply case {case}",
            );
            assert_eq!(mont_sqr(&a), portable::mont_sqr(&a), "square case {case}",);
        }
    }

    #[test]
    #[ignore = "million-case release stress gate; run explicitly before changing field backends"]
    fn million_products_match_assembly() {
        let mut state = 0xd1b5_4a32_d192_ed03u64;
        for case in 0..1_000_000 {
            let a = next_residue(&mut state);
            let b = next_residue(&mut state);
            assert_eq!(
                mont_mul(&a, &b),
                portable::mont_mul(&a, &b),
                "multiply case {case}",
            );
            assert_eq!(mont_sqr(&a), portable::mont_sqr(&a), "square case {case}",);
        }
    }
}
