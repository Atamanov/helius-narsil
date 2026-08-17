//! Pure build-policy decisions shared by `build.rs` and host-independent
//! tests. Keep target probing itself in `build.rs`. This module only maps an
//! already classified target plus an optional environment override.

/// Resolve one `NARSIL_*` override. `None` means the value is not a
/// recognized setting and the build must fail rather than guess.
fn override_or(value: Option<&str>, default: bool) -> Option<bool> {
    match value {
        None => Some(default),
        Some("0") => Some(false),
        Some("1") => Some(true),
        Some(_) => None,
    }
}

/// Miller line-path settings, the fused Fp2 square leaf, the dual-lane SoS
/// leaf, and the inlined Fp2 helpers. Intel and Zen 4 both gain on
/// `miller_live`. A target cpu that names neither vendor keeps them off,
/// because these kernels trade text for work and the sign of that trade
/// follows the front end.
pub fn line_path_enabled(value: Option<&str>, target_vendor_is_measured: bool) -> Option<bool> {
    override_or(value, target_vendor_is_measured)
}

pub fn compact_fp12_square_enabled(value: Option<&str>, target_is_amd: bool) -> Option<bool> {
    override_or(value, target_is_amd)
}

/// Runtime CPUID dispatch into the AVX-512 IFMA kernels.
///
/// Two questions must both answer yes and they are not the same question.
/// CPUID answers "may this CPU execute IFMA", and the kernels keep that check
/// at every entry. `base_hosts_zmm` answers "may this build call IFMA code
/// cheaply", which the CPU cannot answer. The kernels carry
/// `#[target_feature(enable = "avx512f,avx512ifma")]`, so without `avx512f` in
/// the base target the caller has no ZMM register class and every 512-bit
/// value crosses the boundary through memory. That traffic costs more than the
/// IFMA kernels save, so a build that cannot host ZMM takes the scalar route.
pub fn runtime_ifma_enabled(value: Option<&str>, base_hosts_zmm: bool) -> Option<bool> {
    override_or(value, base_hosts_zmm)
}

/// Lazy double-width `y = g^2 - 3e^2` leaf of the G2 doubling step. It
/// trades 2 of the step's 20 Montgomery reductions plus three Fp2 helper
/// round trips for text in a line path whose budget is front-end capacity,
/// so the sign is a per-vendor measurement. Intel and Zen 4 both gain.
pub fn g2_ysqr_enabled(value: Option<&str>, target_vendor_is_measured: bool) -> Option<bool> {
    override_or(value, target_vendor_is_measured)
}

/// Lazy double-width Karatsuba Fp2 multiply leaf, mcl's `Fp2Dbl::mulPre`
/// shape. It drops one of the four 4x4 products the fused sums-of-products
/// route pays, but doubles the text and splits one interleaved CIOS chain
/// into a product phase followed by a reduction phase. Line generation is
/// front-end bound, so both trades go the wrong way and the default stays
/// off. The leaf is kept for a host where the multiplier, not the front end,
/// is the limit.
pub fn f2mul_enabled(value: Option<&str>, _target_vendor_is_measured: bool) -> Option<bool> {
    override_or(value, false)
}
