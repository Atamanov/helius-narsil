use std::{env, path::PathBuf, process::Command};

use sha2::{Digest, Sha256};

const MCL_REMEDY: &str = "Build the pinned MCL tree first and point MCL_DIR at it:\n  \
     bench/scripts/provision.sh\n  \
     MCL_DIR=<tree> cargo build --manifest-path bench/Cargo.toml --release --bin four_lane\n\
     The tree must hold include/mcl/bn.hpp, lib256/libmcl.a, and \
     lib256/helius-mcl-native.json. See bench/README.md.";

fn target_cpu_values(rustflags: &str) -> Vec<&str> {
    rustflags
        .split_whitespace()
        .filter_map(|value| {
            value
                .strip_prefix("target-cpu=")
                .or_else(|| value.strip_prefix("-Ctarget-cpu="))
        })
        .collect()
}

fn resolve_executable(path: &std::path::Path) -> PathBuf {
    if path.is_absolute() {
        return path.to_owned();
    }
    env::var_os("PATH")
        .into_iter()
        .flat_map(|value| env::split_paths(&value).collect::<Vec<_>>())
        .map(|directory| directory.join(path))
        .find(|candidate| candidate.is_file())
        .unwrap_or_else(|| panic!("cannot resolve compiler {} through PATH", path.display()))
}

fn main() {
    println!("cargo:rerun-if-changed=src/mcl_bridge.cpp");
    println!("cargo:rerun-if-env-changed=MCL_DIR");
    println!("cargo:rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS");

    let rustflags = env::var("CARGO_ENCODED_RUSTFLAGS")
        .unwrap_or_default()
        .replace('\u{1f}', " ");
    let target_cpus = target_cpu_values(&rustflags);
    assert!(
        target_cpus.len() <= 1,
        "duplicate -C target-cpu settings are forbidden in the comparison harness"
    );
    let target_cpu = target_cpus.first().copied().unwrap_or("unspecified");
    // A pinned non-native CPU would leave the C++ bridge at baseline -O3
    // while the Rust lanes specialize. Only native builds may claim parity.
    assert!(
        target_cpu == "native" || target_cpu == "unspecified",
        "the comparison harness accepts only -C target-cpu=native or no target-cpu, got {target_cpu}"
    );
    println!("cargo:rustc-env=FOUR_LANE_BUILD_RUSTFLAGS={rustflags}");
    println!("cargo:rustc-env=FOUR_LANE_BUILD_TARGET_CPU={target_cpu}");
    let target = env::var("TARGET").unwrap();
    println!(
        "cargo:rustc-env=FOUR_LANE_BUILD_ARK_ASM={}",
        if env::var_os("CARGO_FEATURE_ARK_ASM").is_some() {
            "1"
        } else {
            "0"
        }
    );
    let manifest_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let git = |arguments: &[&str]| {
        Command::new("git")
            .args(arguments)
            .current_dir(&manifest_dir)
            .output()
            .ok()
            .filter(|output| output.status.success())
            .and_then(|output| String::from_utf8(output.stdout).ok())
            .map(|value| value.trim().to_owned())
    };
    // Without these the provenance line can carry the revision of an earlier
    // build that left its artifacts in the target directory. A watched path
    // that does not exist would rerun this script on every build, so a packed
    // branch ref is skipped.
    let head_ref = git(&["symbolic-ref", "--quiet", "HEAD"]);
    for path in ["HEAD"].into_iter().chain(head_ref.as_deref()) {
        let Some(watched) = git(&["rev-parse", "--git-path", path]) else {
            continue;
        };
        if manifest_dir.join(&watched).is_file() {
            println!("cargo:rerun-if-changed={watched}");
        }
    }
    println!(
        "cargo:rustc-env=FOUR_LANE_BUILD_GIT_REV={}",
        git(&["rev-parse", "HEAD"]).unwrap_or_else(|| "UNKNOWN".to_owned())
    );

    let Some(mcl) = env::var_os("MCL_DIR").map(PathBuf::from) else {
        panic!("MCL_DIR is not set, so the mcl lane cannot link.\n{MCL_REMEDY}");
    };
    let include = mcl.join("include");
    let lib = mcl.join("lib256/libmcl.a");
    let native_manifest = mcl.join("lib256/helius-mcl-native.json");

    for required in [include.join("mcl/bn.hpp"), lib.clone()] {
        assert!(
            required.is_file(),
            "MCL_DIR={} has no {}.\n{MCL_REMEDY}",
            mcl.display(),
            required.display()
        );
    }
    println!("cargo:rerun-if-changed={}", lib.display());
    println!("cargo:rerun-if-changed={}", native_manifest.display());
    let archive_sha256 = std::fs::read(&lib)
        .map(|bytes| format!("{:x}", Sha256::digest(bytes)))
        .expect("MCL archive was checked above");
    let native_manifest_sha256 = std::fs::read(&native_manifest)
        .map(|bytes| format!("{:x}", Sha256::digest(bytes)))
        .unwrap_or_else(|_| "MISSING".to_owned());
    println!("cargo:rustc-env=FOUR_LANE_BUILD_MCL_ARCHIVE_SHA256={archive_sha256}");
    println!("cargo:rustc-env=FOUR_LANE_BUILD_MCL_MANIFEST_SHA256={native_manifest_sha256}");
    println!(
        "cargo:rustc-env=FOUR_LANE_BUILD_MCL_MANIFEST_PATH={}",
        std::fs::canonicalize(&native_manifest)
            .unwrap_or(native_manifest.clone())
            .display()
    );

    let native_requested = target_cpu == "native";
    let native_flag = native_requested.then(|| {
        if target.starts_with("aarch64") {
            "-mcpu=native"
        } else {
            "-march=native"
        }
    });
    let mut bridge = cc::Build::new();
    bridge
        .cpp(true)
        .file("src/mcl_bridge.cpp")
        .include(&include)
        .define("MCL_FP_BIT", "256")
        .define("MCL_FR_BIT", "256")
        .define("NDEBUG", None)
        .flag_if_supported("-O3")
        .flag_if_supported("-std=c++17");
    if let Some(flag) = native_flag {
        // A claim-eligible Rust native build must compile its C++ adapter for
        // the same host too. Unlike feature-probing flags, failure is fatal.
        bridge.flag(flag);
    }
    println!(
        "cargo:rustc-env=FOUR_LANE_BUILD_CXX={}",
        resolve_executable(bridge.get_compiler().path()).display()
    );
    println!(
        "cargo:rustc-env=FOUR_LANE_BUILD_CXX_NATIVE_FLAG={}",
        native_flag.unwrap_or("NONE")
    );
    bridge.compile("narsil_mcl_bridge");

    println!(
        "cargo:rustc-link-search=native={}",
        lib.parent().unwrap().display()
    );
    println!("cargo:rustc-link-lib=static=mcl");
}
