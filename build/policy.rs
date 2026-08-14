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

/// Enable Intel-default Miller line-path settings unless explicitly
/// overridden: the fused Fp2 square leaf, dual-lane SoS leaf, and inlined
/// Fp2 helpers.
pub fn intel_line_path_enabled(value: Option<&str>, target_is_intel: bool) -> Option<bool> {
    override_or(value, target_is_intel)
}

pub fn compact_fp12_square_enabled(value: Option<&str>, target_is_amd: bool) -> Option<bool> {
    override_or(value, target_is_amd)
}

/// Enable the Intel-default lazy double-width `y = g^2 - 3e^2` leaf of the
/// G2 doubling step unless explicitly overridden. Other targets default to
/// the composed route.
pub fn g2_ysqr_enabled(value: Option<&str>, target_is_intel: bool) -> Option<bool> {
    override_or(value, target_is_intel)
}
