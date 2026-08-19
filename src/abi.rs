//! Private views used by assembly and FFI adapters.

#[cfg(narsil_a64_cyc)]
use crate::Fp12;
#[cfg(narsil_mont4_x86_64_adx)]
use crate::{Fp2, Fp6, Fp12};

#[cfg(narsil_mont4_x86_64_adx)]
const _: () = assert!(core::mem::size_of::<Fp2>() == core::mem::size_of::<[[u64; 4]; 2]>());
#[cfg(narsil_mont4_x86_64_adx)]
const _: () = assert!(core::mem::align_of::<Fp2>() == core::mem::align_of::<[[u64; 4]; 2]>());
#[cfg(narsil_mont4_x86_64_adx)]
const _: () = assert!(core::mem::size_of::<Fp6>() == core::mem::size_of::<[[u64; 4]; 6]>());
#[cfg(narsil_mont4_x86_64_adx)]
const _: () = assert!(core::mem::align_of::<Fp6>() == core::mem::align_of::<[[u64; 4]; 6]>());
#[cfg(any(narsil_mont4_x86_64_adx, narsil_a64_cyc))]
const _: () = assert!(core::mem::size_of::<Fp12>() == core::mem::size_of::<[[u64; 4]; 12]>());
#[cfg(any(narsil_mont4_x86_64_adx, narsil_a64_cyc))]
const _: () = assert!(core::mem::align_of::<Fp12>() == core::mem::align_of::<[[u64; 4]; 12]>());
#[cfg(narsil_mont4_x86_64_adx)]
const _: () = assert!(core::mem::size_of::<[Fp2; 3]>() == core::mem::size_of::<[[u64; 4]; 6]>());

#[cfg(narsil_mont4_x86_64_adx)]
#[inline]
pub(crate) fn fp2_components(value: &Fp2) -> &[[u64; 4]; 2] {
    // SAFETY: Fp and Fp2 are repr(C); the size and alignment match above.
    unsafe { &*(core::ptr::from_ref(value).cast()) }
}

#[cfg(narsil_mont4_x86_64_adx)]
#[inline]
pub(crate) fn fp6_components(value: &Fp6) -> &[[u64; 4]; 6] {
    // SAFETY: the repr(C) Fp2/Fp6 nesting has the asserted size and alignment.
    unsafe { &*(core::ptr::from_ref(value).cast()) }
}

#[cfg(any(narsil_mont4_x86_64_adx, narsil_a64_cyc))]
#[inline]
pub(crate) fn fp12_components(value: &Fp12) -> &[[u64; 4]; 12] {
    // SAFETY: the repr(C) tower has the asserted contiguous size and alignment.
    unsafe { &*(core::ptr::from_ref(value).cast()) }
}

#[cfg(narsil_mont4_x86_64_adx)]
#[inline]
pub(crate) fn fp2_triplet_components(value: &[Fp2; 3]) -> &[[u64; 4]; 6] {
    // SAFETY: arrays are contiguous and each repr(C) Fp2 contains two Fp values.
    unsafe { &*(core::ptr::from_ref(value).cast()) }
}

#[cfg(narsil_mont4_x86_64_adx)]
#[inline]
pub(crate) unsafe fn limbs_from_ptr<'a>(value: *const u64) -> &'a [u64; 4] {
    // SAFETY: the caller guarantees a live, aligned four-limb operand.
    unsafe { &*value.cast() }
}

unsafe extern "C" {
    #[cfg(narsil_a64_kernels)]
    pub(crate) fn narsil_mont4(
        z: *mut u64,
        x: *const u64,
        y: *const u64,
        constants: *const crate::fp::aarch64::Mont4Constants,
    );

    #[cfg(narsil_a64_sosd2)]
    pub(crate) fn narsil_sosd2(
        z: *mut u64,
        x0: *const u64,
        x1: *const u64,
        y0: *const u64,
        y1: *const u64,
        constants: *const crate::fp::aarch64::Mont4Constants,
    );

    #[cfg(narsil_a64_sosd4)]
    pub(crate) fn narsil_sosd4(
        z: *mut u64,
        table: *const *const u64,
        constants: *const crate::fp::aarch64::Mont4Constants,
    );

    #[cfg(narsil_a64_sosd6)]
    pub(crate) fn narsil_sosd6(
        z: *mut u64,
        table: *const *const u64,
        constants: *const crate::fp::aarch64::Mont4Constants,
    );

    #[cfg(narsil_a64_cyc)]
    pub(crate) fn narsil_cyc_fp4(
        z: *mut u64,
        r0: *const u64,
        r1: *const u64,
        constants: *const crate::fp::aarch64::Mont4Constants,
    );

    #[cfg(narsil_a64_cyc)]
    pub(crate) fn narsil_cyc_fold(
        z: *mut u64,
        t: *const u64,
        f: *const u64,
        constants: *const crate::fp::aarch64::Mont4Constants,
    );

    // The production route is the inline rendering in `limb.rs`. The leaf
    // symbols stay for the silicon differential and the paired A/B.
    #[cfg(all(test, narsil_a64_addsub))]
    pub(crate) fn narsil_add_mod(z: *mut u64, a: *const u64, b: *const u64, p: *const u64);

    #[cfg(all(test, narsil_a64_addsub))]
    pub(crate) fn narsil_sub_mod(z: *mut u64, a: *const u64, b: *const u64, p: *const u64);

    #[cfg(narsil_mont4_x86_64_adx)]
    pub(crate) fn narsil_mont4_mul_x86(
        z: *mut u64,
        x: *const u64,
        y: *const u64,
        constants: *const crate::fp::x86_64::Mont4Constants,
    );
    #[cfg(narsil_mont4_x86_64_adx)]
    pub(crate) fn narsil_mont4_sqr_x86(
        z: *mut u64,
        x: *const u64,
        y: *const u64,
        constants: *const crate::fp::x86_64::Mont4Constants,
    );
    #[cfg(narsil_mont4_x86_64_adx)]
    pub(crate) fn narsil_sos_x86(
        z: *mut u64,
        pairs: *const *const u64,
        pair_count: u64,
        constants: *const crate::fp::x86_64::Mont4Constants,
    );
    #[cfg(narsil_mont4_x86_64_adx)]
    pub(crate) fn narsil_sosd2_x86(
        z: *mut u64,
        x0: *const u64,
        x1: *const u64,
        y0: *const u64,
        y1: *const u64,
        constants: *const crate::fp::x86_64::Mont4Constants,
    );
    #[cfg(narsil_mont4_x86_64_adx)]
    pub(crate) fn narsil_sosd2_small_x86(
        z: *mut u64,
        x0: *const u64,
        x1: *const u64,
        y0: *const u64,
        y1: *const u64,
        constants: *const crate::fp::x86_64::Mont4Constants,
    );
    #[cfg(narsil_mont4_x86_64_adx)]
    pub(crate) fn narsil_f2sqr_x86(
        z: *mut u64,
        x0: *const u64,
        x1: *const u64,
        constants: *const crate::fp::x86_64::Mont4Constants,
    );
    #[cfg(narsil_mont4_x86_64_adx)]
    pub(crate) fn narsil_f2sqr_small_x86(
        z: *mut u64,
        x0: *const u64,
        x1: *const u64,
        constants: *const crate::fp::x86_64::Mont4Constants,
    );
    #[cfg(narsil_mont4_x86_64_adx)]
    pub(crate) fn narsil_g2_ysqr_x86(
        z: *mut u64,
        g: *const u64,
        e: *const u64,
        f: *const u64,
        constants: *const crate::fp::x86_64::Mont4Constants,
    );
    #[cfg(narsil_mont4_x86_64_adx)]
    pub(crate) fn narsil_f2mul_x86(
        z: *mut u64,
        x: *const u64,
        y: *const u64,
        constants: *const crate::fp::x86_64::Mont4Constants,
    );
    #[cfg(narsil_mont4_x86_64_adx)]
    pub(crate) fn narsil_sosd6_x86(
        z: *mut u64,
        stage: *mut u64,
        constants: *const crate::fp::x86_64::Mont4Constants,
    );
    #[cfg(narsil_mont4_x86_64_adx)]
    pub(crate) fn narsil_fp6_mul_x86(
        z: *mut u64,
        a: *const u64,
        b: *const u64,
        constants: *const crate::fp::x86_64::Fp6MulConstants,
    );
    #[cfg(narsil_mont4_x86_64_adx)]
    pub(crate) fn narsil_fp12_034_x86(
        z: *mut u64,
        f: *const u64,
        c: *const u64,
        constants: *const crate::fp::x86_64::Fp6MulConstants,
    );
    #[cfg(narsil_mont4_x86_64_adx)]
    pub(crate) fn narsil_fp12_034l_x86(
        z: *mut u64,
        f: *const u64,
        c: *const u64,
        constants: *const crate::fp::x86_64::Fp6MulConstants,
    );
    #[cfg(narsil_mont4_x86_64_adx)]
    pub(crate) fn narsil_fp12_034k_x86(
        z: *mut u64,
        f: *const u64,
        c: *const u64,
        constants: *const crate::fp::x86_64::Fp6MulConstants,
    );
    #[cfg(narsil_mont4_x86_64_adx)]
    pub(crate) fn narsil_fp4_sqr_x86(
        z: *mut u64,
        r0: *const u64,
        r1: *const u64,
        constants: *const crate::fp::x86_64::Fp6MulConstants,
    );
    #[cfg(narsil_mont4_x86_64_adx)]
    pub(crate) fn narsil_fp12_sqr_x86(
        z: *mut u64,
        f: *const u64,
        constants: *const crate::fp::x86_64::Fp6MulConstants,
    );
    #[cfg(narsil_mont4_x86_64_adx)]
    pub(crate) fn narsil_fp12_sqr_mcl_x86(
        f: *mut u64,
        constants: *const crate::fp::x86_64::Fp6MulConstants,
    );
    #[cfg(narsil_mont4_x86_64_adx)]
    pub(crate) fn narsil_cyc_sqr_x86(
        z: *mut u64,
        f: *const u64,
        constants: *const crate::fp::x86_64::Fp6MulConstants,
    );
    #[cfg(narsil_mont4_x86_64_adx)]
    pub(crate) fn narsil_fp12_mul_x86(
        z: *mut u64,
        a: *const u64,
        b: *const u64,
        constants: *const crate::fp::x86_64::Fp6MulConstants,
    );
}
