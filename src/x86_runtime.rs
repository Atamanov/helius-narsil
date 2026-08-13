//! x86 CPUID/XCR0 capability classification and backend authorization.
//!
//! Capability and implementation are deliberately separate. Detecting AVX,
//! AVX2, or AVX-512F does not imply that this crate has a batch kernel for
//! that width. Today only the AVX-512 IFMA batch implementation is authored.

#![cfg_attr(
    not(any(target_arch = "x86", target_arch = "x86_64")),
    allow(dead_code)
)]

use core::sync::atomic::{AtomicU8, Ordering};

/// Highest usable x86 SIMD capability after both CPU and OS-state checks.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CapabilityTier {
    Avx512Ifma,
    Avx512,
    Avx2,
    Avx,
    Simd128,
    Portable,
}

/// Real batch implementation selected for this build and host.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BatchBackend {
    Avx512Ifma,
    ScalarX86Adx,
    PortableRust,
}

/// Proof that this process may enter an AVX-512F+IFMA target-feature boundary.
///
/// The field is private so code outside this module cannot manufacture the
/// proof without first passing the shared CPUID and XCR0 checks.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Avx512Ifma(());

#[derive(Clone, Copy, Debug, Default)]
struct RawCapabilities {
    leaf1_ecx: u32,
    leaf1_edx: u32,
    leaf7_ebx: u32,
    xcr0: u64,
}

const CPUID1_XSAVE: u32 = 1 << 26;
const CPUID1_OSXSAVE: u32 = 1 << 27;
const CPUID1_AVX: u32 = 1 << 28;
const CPUID1_SSE2: u32 = 1 << 26;
const CPUID7_AVX2: u32 = 1 << 5;
const CPUID7_AVX512F: u32 = 1 << 16;
const CPUID7_AVX512IFMA: u32 = 1 << 21;

const XCR0_XMM: u64 = 1 << 1;
const XCR0_YMM: u64 = 1 << 2;
const XCR0_OPMASK: u64 = 1 << 5;
const XCR0_ZMM_HI256: u64 = 1 << 6;
const XCR0_HI16_ZMM: u64 = 1 << 7;
const XCR0_AVX: u64 = XCR0_XMM | XCR0_YMM;
const XCR0_AVX512: u64 = XCR0_AVX | XCR0_OPMASK | XCR0_ZMM_HI256 | XCR0_HI16_ZMM;
const CAPABILITY_UNCACHED: u8 = u8::MAX;

static CACHED_CAPABILITY: AtomicU8 = AtomicU8::new(CAPABILITY_UNCACHED);

impl CapabilityTier {
    fn from_cached(value: u8) -> Option<Self> {
        match value {
            value if value == Self::Avx512Ifma as u8 => Some(Self::Avx512Ifma),
            value if value == Self::Avx512 as u8 => Some(Self::Avx512),
            value if value == Self::Avx2 as u8 => Some(Self::Avx2),
            value if value == Self::Avx as u8 => Some(Self::Avx),
            value if value == Self::Simd128 as u8 => Some(Self::Simd128),
            value if value == Self::Portable as u8 => Some(Self::Portable),
            _ => None,
        }
    }
}

impl RawCapabilities {
    fn classify(self) -> CapabilityTier {
        let avx_cpu = self.leaf1_ecx & (CPUID1_XSAVE | CPUID1_OSXSAVE | CPUID1_AVX)
            == CPUID1_XSAVE | CPUID1_OSXSAVE | CPUID1_AVX;
        let avx_state = avx_cpu && self.xcr0 & XCR0_AVX == XCR0_AVX;
        let avx512_state = avx_state && self.xcr0 & XCR0_AVX512 == XCR0_AVX512;

        if avx512_state
            && self.leaf7_ebx & (CPUID7_AVX512F | CPUID7_AVX512IFMA)
                == CPUID7_AVX512F | CPUID7_AVX512IFMA
        {
            CapabilityTier::Avx512Ifma
        } else if avx512_state && self.leaf7_ebx & CPUID7_AVX512F != 0 {
            CapabilityTier::Avx512
        } else if avx_state && self.leaf7_ebx & CPUID7_AVX2 != 0 {
            CapabilityTier::Avx2
        } else if avx_state {
            CapabilityTier::Avx
        } else if self.leaf1_edx & CPUID1_SSE2 != 0 {
            CapabilityTier::Simd128
        } else {
            CapabilityTier::Portable
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn raw_capabilities() -> RawCapabilities {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::{__cpuid, __cpuid_count, _xgetbv};
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::{__cpuid, __cpuid_count, _xgetbv};

    // SAFETY: CPUID leaf 0 establishes the maximum supported basic leaf.
    // XGETBV is executed only after CPUID reports both XSAVE and OSXSAVE.
    // That is the architectural precondition for reading XCR0.
    unsafe {
        let maximum = __cpuid(0).eax;
        if maximum < 1 {
            return RawCapabilities::default();
        }
        let leaf1 = __cpuid(1);
        let xsave = leaf1.ecx & (CPUID1_XSAVE | CPUID1_OSXSAVE) == CPUID1_XSAVE | CPUID1_OSXSAVE;
        let xcr0 = if xsave { _xgetbv(0) } else { 0 };
        let leaf7_ebx = if maximum >= 7 {
            __cpuid_count(7, 0).ebx
        } else {
            0
        };
        RawCapabilities {
            leaf1_ecx: leaf1.ecx,
            leaf1_edx: leaf1.edx,
            leaf7_ebx,
            xcr0,
        }
    }
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn raw_capabilities() -> RawCapabilities {
    RawCapabilities::default()
}

pub(crate) fn capability_tier() -> CapabilityTier {
    if let Some(cached) = CapabilityTier::from_cached(CACHED_CAPABILITY.load(Ordering::Acquire)) {
        return cached;
    }

    let detected = raw_capabilities().classify();
    match CACHED_CAPABILITY.compare_exchange(
        CAPABILITY_UNCACHED,
        detected as u8,
        Ordering::AcqRel,
        Ordering::Acquire,
    ) {
        Ok(_) => detected,
        Err(cached) => CapabilityTier::from_cached(cached)
            .expect("capability cache is written only with a valid tier"),
    }
}

fn select_batch_backend(
    capability: CapabilityTier,
    ifma_compiled: bool,
    scalar_adx_compiled: bool,
) -> BatchBackend {
    if ifma_compiled && capability == CapabilityTier::Avx512Ifma {
        BatchBackend::Avx512Ifma
    } else if scalar_adx_compiled {
        BatchBackend::ScalarX86Adx
    } else {
        BatchBackend::PortableRust
    }
}

pub(crate) fn batch_backend() -> BatchBackend {
    select_batch_backend(
        capability_tier(),
        cfg!(all(
            any(narsil_avx512_ifma, narsil_x86_runtime_ifma),
            not(feature = "force-portable")
        )),
        cfg!(all(
            narsil_mont4_x86_64_adx,
            not(feature = "force-portable")
        )),
    )
}

/// Authorize an AVX-512F+IFMA entry when both the implementation and the
/// required CPU/OS register state are present.
#[cfg(any(all(narsil_x86_runtime_ifma, not(feature = "force-portable")), test))]
#[inline]
pub(crate) fn avx512_ifma() -> Option<Avx512Ifma> {
    (batch_backend() == BatchBackend::Avx512Ifma).then_some(Avx512Ifma(()))
}

/// Stable diagnostic string for the opt-in benchmark harness.
#[cfg(feature = "backend-report")]
pub(crate) fn backend_report() -> &'static str {
    #[cfg(all(
        target_arch = "aarch64",
        target_vendor = "apple",
        not(feature = "force-portable")
    ))]
    return "non-x86 -> aarch64-apple-scalar";

    #[cfg(all(
        not(any(target_arch = "x86", target_arch = "x86_64")),
        any(
            not(target_arch = "aarch64"),
            not(target_vendor = "apple"),
            feature = "force-portable"
        )
    ))]
    return "non-x86 -> portable-rust";

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    match (capability_tier(), batch_backend()) {
        (CapabilityTier::Avx512Ifma, BatchBackend::Avx512Ifma) => "avx512-ifma",
        (CapabilityTier::Avx512Ifma, BatchBackend::ScalarX86Adx) => {
            "avx512-ifma -> scalar-x86-adx (IFMA batch not compiled)"
        }
        (CapabilityTier::Avx512Ifma, BatchBackend::PortableRust) => {
            "avx512-ifma -> portable-rust (IFMA batch denied)"
        }
        (CapabilityTier::Avx512, BatchBackend::ScalarX86Adx) => {
            "avx512f-no-ifma -> scalar-x86-adx (no authored AVX512F batch kernel)"
        }
        (CapabilityTier::Avx512, BatchBackend::PortableRust) => {
            "avx512f-no-ifma -> portable-rust (no authored AVX512F batch kernel)"
        }
        (CapabilityTier::Avx2, BatchBackend::ScalarX86Adx) => {
            "avx2 -> scalar-x86-adx (no authored AVX2 batch kernel)"
        }
        (CapabilityTier::Avx2, BatchBackend::PortableRust) => {
            "avx2 -> portable-rust (no authored AVX2 batch kernel)"
        }
        (CapabilityTier::Avx, BatchBackend::ScalarX86Adx) => {
            "avx -> scalar-x86-adx (no authored AVX batch kernel)"
        }
        (CapabilityTier::Avx, BatchBackend::PortableRust) => {
            "avx -> portable-rust (no authored AVX batch kernel)"
        }
        (CapabilityTier::Simd128, BatchBackend::ScalarX86Adx) => {
            "simd128 -> scalar-x86-adx (no authored SIMD128 batch kernel)"
        }
        (CapabilityTier::Simd128, BatchBackend::PortableRust) => {
            "simd128 -> portable-rust (no authored SIMD128 batch kernel)"
        }
        (CapabilityTier::Portable, BatchBackend::ScalarX86Adx) => "scalar-x86-adx",
        (CapabilityTier::Portable, BatchBackend::PortableRust) => "portable-rust",
        (_, BatchBackend::Avx512Ifma) => unreachable!("IFMA backend requires IFMA capability"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn avx_cpu() -> u32 {
        CPUID1_XSAVE | CPUID1_OSXSAVE | CPUID1_AVX
    }

    fn classify(leaf1_ecx: u32, leaf1_edx: u32, leaf7_ebx: u32, xcr0: u64) -> CapabilityTier {
        RawCapabilities {
            leaf1_ecx,
            leaf1_edx,
            leaf7_ebx,
            xcr0,
        }
        .classify()
    }

    #[test]
    fn forced_capabilities_follow_the_required_priority() {
        assert_eq!(
            classify(
                avx_cpu(),
                CPUID1_SSE2,
                CPUID7_AVX2 | CPUID7_AVX512F | CPUID7_AVX512IFMA,
                XCR0_AVX512,
            ),
            CapabilityTier::Avx512Ifma
        );
        assert_eq!(
            classify(
                avx_cpu(),
                CPUID1_SSE2,
                CPUID7_AVX2 | CPUID7_AVX512F,
                XCR0_AVX512,
            ),
            CapabilityTier::Avx512
        );
        assert_eq!(
            classify(avx_cpu(), CPUID1_SSE2, CPUID7_AVX2, XCR0_AVX),
            CapabilityTier::Avx2
        );
        assert_eq!(
            classify(avx_cpu(), CPUID1_SSE2, 0, XCR0_AVX),
            CapabilityTier::Avx
        );
        assert_eq!(classify(0, CPUID1_SSE2, 0, 0), CapabilityTier::Simd128);
        assert_eq!(classify(0, 0, 0, 0), CapabilityTier::Portable);
    }

    #[test]
    fn forced_capabilities_require_os_enabled_register_state() {
        // CPUID feature bits alone cannot authorize YMM/ZMM instructions.
        assert_eq!(
            classify(
                avx_cpu(),
                CPUID1_SSE2,
                CPUID7_AVX2 | CPUID7_AVX512F | CPUID7_AVX512IFMA,
                XCR0_XMM,
            ),
            CapabilityTier::Simd128
        );
        // AVX state without opmask/full-ZMM state may use AVX2, never AVX-512.
        assert_eq!(
            classify(
                avx_cpu(),
                CPUID1_SSE2,
                CPUID7_AVX2 | CPUID7_AVX512F | CPUID7_AVX512IFMA,
                XCR0_AVX,
            ),
            CapabilityTier::Avx2
        );
        // AVX512IFMA without AVX512F is not a usable IFMA tier.
        assert_eq!(
            classify(avx_cpu(), CPUID1_SSE2, CPUID7_AVX512IFMA, XCR0_AVX512,),
            CapabilityTier::Avx
        );
        // OSXSAVE without XSAVE (or vice versa) cannot authorize XGETBV use.
        assert_eq!(
            classify(
                CPUID1_OSXSAVE | CPUID1_AVX,
                CPUID1_SSE2,
                CPUID7_AVX2,
                XCR0_AVX,
            ),
            CapabilityTier::Simd128
        );
    }

    #[test]
    fn forced_dispatch_never_relabels_missing_vector_backends() {
        assert_eq!(
            select_batch_backend(CapabilityTier::Avx512Ifma, true, false),
            BatchBackend::Avx512Ifma
        );
        for capability in [
            CapabilityTier::Avx512,
            CapabilityTier::Avx2,
            CapabilityTier::Avx,
            CapabilityTier::Simd128,
            CapabilityTier::Portable,
        ] {
            assert_eq!(
                select_batch_backend(capability, true, false),
                BatchBackend::PortableRust,
                "{capability:?} has no authored vector batch kernel"
            );
            assert_eq!(
                select_batch_backend(capability, true, true),
                BatchBackend::ScalarX86Adx,
                "{capability:?} may use only the real compiled scalar backend"
            );
        }
        assert_eq!(
            select_batch_backend(CapabilityTier::Avx512Ifma, false, true),
            BatchBackend::ScalarX86Adx
        );
        assert_eq!(
            select_batch_backend(CapabilityTier::Avx512Ifma, false, false),
            BatchBackend::PortableRust
        );
    }

    #[test]
    fn capability_proof_matches_selected_backend() {
        assert_eq!(
            avx512_ifma().is_some(),
            batch_backend() == BatchBackend::Avx512Ifma
        );
    }

    #[cfg(all(
        feature = "backend-report",
        feature = "std",
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
    #[test]
    fn harness_report_names_the_real_selected_backend() {
        let report = backend_report();
        match batch_backend() {
            BatchBackend::Avx512Ifma => assert_eq!(report, "avx512-ifma"),
            BatchBackend::ScalarX86Adx => assert!(report.contains("scalar-x86-adx")),
            BatchBackend::PortableRust => assert!(report.contains("portable-rust")),
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn runtime_facade_matches_scalar_boundaries() {
        use crate::batch::{G1Bytes, G2Bytes, PairBytes, pairing_product_is_one};
        use crate::pairing::multi_pairing;
        use crate::{Fp12, Fr, G1Affine, G2Affine, G2Projective};

        let p = G1Affine::generator();
        let q = G2Projective::from(G2Affine::generator());
        for n in [5usize, 6, 7, 8, 9, 15, 16, 17] {
            let typed: alloc::vec::Vec<_> = (0..n)
                .map(|i| {
                    (
                        if i & 1 == 0 { p } else { p.negate() },
                        q.mul_scalar(Fr::from_u64(i as u64 + 1)).to_affine(),
                    )
                })
                .collect();
            let refs: alloc::vec::Vec<_> = typed.iter().map(|(g1, g2)| (g1, g2)).collect();
            let encoded: alloc::vec::Vec<_> = typed
                .iter()
                .map(|(g1, g2)| PairBytes {
                    g1: G1Bytes::from_affine(g1),
                    g2: G2Bytes::from_affine(g2),
                })
                .collect();
            assert_eq!(
                pairing_product_is_one(&encoded),
                Ok(multi_pairing(&refs) == Fp12::ONE),
                "n = {n}, backend = {:?}",
                batch_backend(),
            );
        }
    }
}
