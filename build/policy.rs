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

/// Miller line-path settings whose measurement carries on Intel: the fused
/// Fp2 square leaf, the dual-lane SoS leaf, and the inlined Fp2 helpers.
pub fn intel_line_path_enabled(value: Option<&str>, target_is_intel: bool) -> Option<bool> {
    override_or(value, target_is_intel)
}

pub fn compact_fp12_square_enabled(value: Option<&str>, target_is_amd: bool) -> Option<bool> {
    override_or(value, target_is_amd)
}

/// Lazy double-width `y = g^2 - 3e^2` leaf of the G2 doubling step. It
/// trades 2 of the step's 20 Montgomery reductions plus three Fp2 helper
/// round trips for about 1.9 KiB of text in a line path whose budget is
/// front-end capacity, so the sign is a per-vendor measurement. Measured on
/// Intel; AMD keeps the composed route until someone measures it there.
pub fn g2_ysqr_enabled(value: Option<&str>, target_is_intel: bool) -> Option<bool> {
    override_or(value, target_is_intel)
}
