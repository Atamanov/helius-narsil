//! Generates target-specific arithmetic kernels from the Rust schedules.
//! Generated files exist only in Cargo OUT_DIR.

#[path = "build/mod.rs"]
mod kernelgen;

use std::path::{Path, PathBuf};

fn generated_kernels() -> [(&'static str, String); 2] {
    [
        (
            "mont4_aarch64.s",
            kernelgen::a64::render::render_mont4_aarch64(),
        ),
        ("mont4_x86_64.s", kernelgen::render::render_mont4_x86_64()),
    ]
}

fn write_kernel(directory: &Path, name: &str, text: &str) -> PathBuf {
    let path = directory.join(name);
    std::fs::write(&path, text).unwrap_or_else(|error| panic!("write {}: {error}", path.display()));
    path
}

fn build_mcl_oracle() {
    println!("cargo:rerun-if-env-changed=MCL_DIR");
    if std::env::var_os("CARGO_FEATURE_MCL_ORACLE").is_none() {
        return;
    }

    let root = PathBuf::from(
        std::env::var_os("MCL_DIR")
            .expect("mcl-oracle requires MCL_DIR to point at a built MCL tree"),
    );
    let include = root.join("include");
    let archive = root.join("lib256/libmcl.a");
    assert!(
        include.join("mcl/bn.h").is_file() && archive.is_file(),
        "MCL_DIR must contain include/mcl/bn.h and lib256/libmcl.a"
    );

    cc::Build::new()
        .cpp(true)
        .file("tests/mcl_bridge.cpp")
        .include(include)
        .define("MCL_FP_BIT", "256")
        .define("MCL_FR_BIT", "256")
        .define("NDEBUG", None)
        .flag_if_supported("-std=c++17")
        .compile("narsil_mcl_bridge");
    println!(
        "cargo:rustc-link-search=native={}",
        archive.parent().unwrap().display()
    );
    println!("cargo:rustc-link-lib=static=mcl");
}

fn target_cpu_is_intel() -> bool {
    let flags = std::env::var("CARGO_ENCODED_RUSTFLAGS").unwrap_or_default();
    let cpu = flags
        .split('\x1f')
        .filter_map(|f| {
            f.strip_prefix("-Ctarget-cpu=")
                .or_else(|| f.strip_prefix("target-cpu="))
        })
        .next_back()
        .unwrap_or("")
        .to_string();
    if cpu.starts_with("znver") || cpu.is_empty() || cpu.starts_with("x86-64") {
        return false;
    }
    if cpu == "native" {
        return std::fs::read_to_string("/proc/cpuinfo")
            .map(|s| s.contains("GenuineIntel"))
            .unwrap_or(false);
    }
    const INTEL: &[&str] = &[
        "icelake",
        "sapphirerapids",
        "graniterapids",
        "emeraldrapids",
        "skylake",
        "cascadelake",
        "cooperlake",
        "tigerlake",
        "rocketlake",
        "alderlake",
        "raptorlake",
        "meteorlake",
    ];
    INTEL.iter().any(|p| cpu.starts_with(p))
}

fn target_cpu_is_amd() -> bool {
    let flags = std::env::var("CARGO_ENCODED_RUSTFLAGS").unwrap_or_default();
    let cpu = flags
        .split('\x1f')
        .filter_map(|f| {
            f.strip_prefix("-Ctarget-cpu=")
                .or_else(|| f.strip_prefix("target-cpu="))
        })
        .next_back()
        .unwrap_or("");
    if cpu.starts_with("znver") {
        return true;
    }
    if cpu == "native" {
        return std::fs::read_to_string("/proc/cpuinfo")
            .map(|s| s.contains("AuthenticAMD"))
            .unwrap_or(false);
    }
    false
}

fn main() {
    build_mcl_oracle();
    println!("cargo::rustc-check-cfg=cfg(kani)");
    println!("cargo::rustc-check-cfg=cfg(narsil_mont4_x86_64_adx)");
    println!("cargo::rustc-check-cfg=cfg(narsil_x86_intel)");
    println!("cargo::rustc-check-cfg=cfg(narsil_avx512_ifma)");
    println!("cargo::rustc-check-cfg=cfg(narsil_x86_runtime_ifma)");
    println!("cargo::rustc-check-cfg=cfg(narsil_sosd2_small)");
    println!("cargo::rustc-check-cfg=cfg(narsil_f2sqr_asm)");
    println!("cargo::rustc-check-cfg=cfg(narsil_g2_ysqr_asm)");
    println!("cargo::rustc-check-cfg=cfg(narsil_f2sqr_small)");
    println!("cargo::rustc-check-cfg=cfg(narsil_miller_inline)");
    println!("cargo::rustc-check-cfg=cfg(narsil_sosd2_asm)");
    println!("cargo::rustc-check-cfg=cfg(narsil_sosd6_asm)");
    println!("cargo::rustc-check-cfg=cfg(narsil_fp6_asm)");
    println!("cargo::rustc-check-cfg=cfg(narsil_fp12_034_asm)");
    println!("cargo::rustc-check-cfg=cfg(narsil_fp12_034l_asm)");
    println!("cargo::rustc-check-cfg=cfg(narsil_fp12_034k_asm)");
    println!("cargo::rustc-check-cfg=cfg(narsil_fp4_sqr_asm)");
    println!("cargo::rustc-check-cfg=cfg(narsil_fp12_sqr_asm)");
    println!("cargo::rustc-check-cfg=cfg(narsil_fp12_sqr_mcl_asm)");
    println!("cargo::rustc-check-cfg=cfg(narsil_fp12_mul_asm)");
    println!("cargo::rustc-check-cfg=cfg(narsil_cyc_sqr_asm)");
    println!("cargo:rerun-if-env-changed=NARSIL_AVX512_IFMA");
    println!("cargo:rerun-if-env-changed=NARSIL_SOSD2_ASM");
    println!("cargo:rerun-if-env-changed=NARSIL_SOSD2_SMALL");
    println!("cargo:rerun-if-env-changed=NARSIL_F2SQR_ASM");
    println!("cargo:rerun-if-env-changed=NARSIL_G2_YSQR_ASM");
    println!("cargo:rerun-if-env-changed=NARSIL_F2SQR_SMALL");
    println!("cargo:rerun-if-env-changed=NARSIL_MILLER_INLINE");
    println!("cargo:rerun-if-env-changed=NARSIL_SOSD6_ASM");
    println!("cargo:rerun-if-env-changed=NARSIL_FP6_ASM");
    println!("cargo:rerun-if-env-changed=NARSIL_FP12_034_ASM");
    println!("cargo:rerun-if-env-changed=NARSIL_FP12_034L_ASM");
    println!("cargo:rerun-if-env-changed=NARSIL_FP12_034K_ASM");
    println!("cargo:rerun-if-env-changed=NARSIL_FP4_SQR_ASM");
    println!("cargo:rerun-if-env-changed=NARSIL_FP12_SQR_ASM");
    println!("cargo:rerun-if-env-changed=NARSIL_FP12_SQR_MCL_ASM");
    println!("cargo:rerun-if-env-changed=NARSIL_FP12_MUL_ASM");
    println!("cargo:rerun-if-env-changed=NARSIL_CYC_SQR_ASM");
    println!("cargo:rerun-if-env-changed=NARSIL_DUMP_ASM");
    println!("cargo:rerun-if-changed=build");

    // Line-path settings that follow the target CPU. The Rust sites guard
    // themselves on the ADX tier as well, so these are safe to emit here.
    for (variable, cfg) in [
        ("NARSIL_SOSD2_ASM", "narsil_sosd2_asm"),
        ("NARSIL_MILLER_INLINE", "narsil_miller_inline"),
    ] {
        let value = std::env::var(variable).ok();
        match kernelgen::policy::intel_line_path_enabled(value.as_deref(), target_cpu_is_intel()) {
            Some(true) => println!("cargo:rustc-cfg={cfg}"),
            Some(false) => {}
            None => panic!(
                "{variable}={:?} is not recognized; use 0 or 1; unset follows the target CPU",
                value.as_deref().unwrap_or_default()
            ),
        }
    }
    match std::env::var("NARSIL_SOSD6_ASM").ok().as_deref() {
        None => {
            if target_cpu_is_amd() {
                println!("cargo:rustc-cfg=narsil_sosd6_asm");
            }
        }
        Some("1") => println!("cargo:rustc-cfg=narsil_sosd6_asm"),
        Some("0") => {}
        Some(other) => panic!(
            "NARSIL_SOSD6_ASM={other:?} is not recognized; use 0 (composed) or 1 (leaf); unset follows the target CPU"
        ),
    }
    match std::env::var("NARSIL_FP6_ASM").ok().as_deref() {
        None | Some("1") => println!("cargo:rustc-cfg=narsil_fp6_asm"),
        Some("0") => {}
        Some(other) => panic!(
            "NARSIL_FP6_ASM={other:?} is not recognized; use 1 (leaf, default) or 0 (composed)"
        ),
    }
    match std::env::var("NARSIL_FP12_034_ASM").ok().as_deref() {
        None => {
            if target_cpu_is_intel() {
                println!("cargo:rustc-cfg=narsil_fp12_034_asm");
            }
        }
        Some("1") => println!("cargo:rustc-cfg=narsil_fp12_034_asm"),
        Some("0") => {}
        Some(other) => panic!(
            "NARSIL_FP12_034_ASM={other:?} is not recognized; use 0 (composed) or 1 (leaf); unset follows the target CPU"
        ),
    }
    match std::env::var("NARSIL_FP12_034L_ASM").ok().as_deref() {
        None | Some("0") => {}
        Some("1") => println!("cargo:rustc-cfg=narsil_fp12_034l_asm"),
        Some(other) => panic!(
            "NARSIL_FP12_034L_ASM={other:?} is not recognized; use 0 (v1/composed, default) or 1 (lazy leaf)"
        ),
    }
    match std::env::var("NARSIL_FP12_034K_ASM").ok().as_deref() {
        None | Some("0") => {}
        Some("1") => println!("cargo:rustc-cfg=narsil_fp12_034k_asm"),
        Some(other) => panic!(
            "NARSIL_FP12_034K_ASM={other:?} is not recognized; use 0 (v1/composed, default) or 1 (Karatsuba-walk leaf)"
        ),
    }
    match std::env::var("NARSIL_FP4_SQR_ASM").ok().as_deref() {
        None | Some("0") => {}
        Some("1") => println!("cargo:rustc-cfg=narsil_fp4_sqr_asm"),
        Some(other) => panic!(
            "NARSIL_FP4_SQR_ASM={other:?} is not recognized; use 0 (composed, default) or 1 (leaf)"
        ),
    }
    match std::env::var("NARSIL_FP12_SQR_ASM").ok().as_deref() {
        None => {
            if target_cpu_is_intel() {
                println!("cargo:rustc-cfg=narsil_fp12_sqr_asm");
            }
        }
        Some("1") => println!("cargo:rustc-cfg=narsil_fp12_sqr_asm"),
        Some("0") => {}
        Some(other) => panic!(
            "NARSIL_FP12_SQR_ASM={other:?} is not recognized; use 0 (composed) or 1 (leaf); unset follows the target CPU"
        ),
    }
    let compact_square_env = std::env::var("NARSIL_FP12_SQR_MCL_ASM").ok();
    match kernelgen::policy::compact_fp12_square_enabled(
        compact_square_env.as_deref(),
        target_cpu_is_amd(),
    ) {
        Some(true) => println!("cargo:rustc-cfg=narsil_fp12_sqr_mcl_asm"),
        Some(false) => {}
        None => panic!(
            "NARSIL_FP12_SQR_MCL_ASM={:?} is not recognized; use 0 (disable) or 1 (enable); unset follows the target CPU (on for AMD)",
            compact_square_env.as_deref().unwrap_or_default()
        ),
    }
    match std::env::var("NARSIL_FP12_MUL_ASM").ok().as_deref() {
        None => {
            if target_cpu_is_intel() {
                println!("cargo:rustc-cfg=narsil_fp12_mul_asm");
            }
        }
        Some("1") => println!("cargo:rustc-cfg=narsil_fp12_mul_asm"),
        Some("0") => {}
        Some(other) => panic!(
            "NARSIL_FP12_MUL_ASM={other:?} is not recognized; use 0 (composed) or 1 (leaf); unset follows the target CPU"
        ),
    }
    match std::env::var("NARSIL_CYC_SQR_ASM").ok().as_deref() {
        None | Some("1") => println!("cargo:rustc-cfg=narsil_cyc_sqr_asm"),
        Some("0") => {}
        Some(other) => panic!(
            "NARSIL_CYC_SQR_ASM={other:?} is not recognized; use 1 (leaf, default) or 0 (composed)"
        ),
    }
    match std::env::var("NARSIL_SOSD2_SMALL").ok().as_deref() {
        None | Some("1") => println!("cargo:rustc-cfg=narsil_sosd2_small"),
        Some("0") => {}
        Some(other) => panic!(
            "NARSIL_SOSD2_SMALL={other:?} is not recognized; use 1 (rolled, default) or 0 (unrolled)"
        ),
    }
    match std::env::var("NARSIL_F2SQR_SMALL").ok().as_deref() {
        None | Some("1") => println!("cargo:rustc-cfg=narsil_f2sqr_small"),
        Some("0") => {}
        Some(other) => panic!(
            "NARSIL_F2SQR_SMALL={other:?} is not recognized; use 1 (rolled, default) or 0 (unrolled)"
        ),
    }

    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let vendor = std::env::var("CARGO_CFG_TARGET_VENDOR").unwrap_or_default();
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let features = std::env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();
    let has = |feature: &str| features.split(',').any(|entry| entry == feature);

    let out_dir = PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR is set by cargo"));
    let kernels = generated_kernels();

    if let Some(dump) = std::env::var_os("NARSIL_DUMP_ASM") {
        let dump = PathBuf::from(dump);
        assert!(
            dump.is_absolute(),
            "NARSIL_DUMP_ASM must be an absolute directory path"
        );
        std::fs::create_dir_all(&dump)
            .unwrap_or_else(|error| panic!("create {}: {error}", dump.display()));
        for (name, text) in &kernels {
            write_kernel(&dump, name, text);
        }
    }

    let [(aarch64_name, aarch64_text), (x86_name, x86_text)] = &kernels;

    if arch == "aarch64" && vendor == "apple" {
        let path = write_kernel(&out_dir, aarch64_name, aarch64_text);
        cc::Build::new().file(path).compile("narsil_mont4_asm");
    }

    if arch == "x86_64" && os == "linux" && has("bmi2") && has("adx") {
        let path = write_kernel(&out_dir, x86_name, x86_text);
        cc::Build::new().file(path).compile("narsil_mont4_asm");
        println!("cargo:rustc-cfg=narsil_mont4_x86_64_adx");
        if target_cpu_is_intel() {
            println!("cargo:rustc-cfg=narsil_x86_intel");
        }
        let ysqr_env = std::env::var("NARSIL_G2_YSQR_ASM").ok();
        match kernelgen::policy::g2_ysqr_enabled(ysqr_env.as_deref(), target_cpu_is_intel()) {
            Some(true) => println!("cargo:rustc-cfg=narsil_g2_ysqr_asm"),
            Some(false) => {}
            None => panic!(
                "NARSIL_G2_YSQR_ASM={:?} is not recognized; use 0 (composed) or 1 (lazy leaf); unset follows the target CPU",
                ysqr_env.as_deref().unwrap_or_default()
            ),
        }
        let f2sqr_env = std::env::var("NARSIL_F2SQR_ASM").ok();
        match kernelgen::policy::intel_line_path_enabled(
            f2sqr_env.as_deref(),
            target_cpu_is_intel(),
        ) {
            Some(true) => println!("cargo:rustc-cfg=narsil_f2sqr_asm"),
            Some(false) => {}
            None => panic!(
                "NARSIL_F2SQR_ASM={:?} is not recognized; use 0 (two-multiply) or 1 (fused leaf); unset follows the target CPU",
                f2sqr_env.as_deref().unwrap_or_default()
            ),
        }
    }

    let ifma_env = std::env::var("NARSIL_AVX512_IFMA").ok();
    let ifma_baseline = arch == "x86_64" && has("avx512f") && has("avx512ifma");
    match ifma_env.as_deref() {
        Some("1") if arch != "x86_64" => panic!(
            "NARSIL_AVX512_IFMA=1 requests the runtime IFMA implementation, \
             but the target architecture is {arch:?}, not x86_64"
        ),
        Some("0") => {}
        Some(other) if other != "1" => {
            panic!("NARSIL_AVX512_IFMA={other:?} is not recognized; use 1 (force) or 0 (deny)")
        }
        _ => {
            if arch == "x86_64" {
                println!("cargo:rustc-cfg=narsil_x86_runtime_ifma");
            }
            if ifma_baseline {
                println!("cargo:rustc-cfg=narsil_avx512_ifma");
            }
        }
    }
}
