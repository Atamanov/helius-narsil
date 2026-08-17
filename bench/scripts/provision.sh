#!/usr/bin/env bash
# Take a fresh Ubuntu x86-64 box to a state where build_lanes.sh can run.
# Installs the build tools and the pinned rustc, then clones MCL, moves it to
# the comparator pin, and builds its 256-bit archive in a scrubbed environment.
# Safe to run again; each stage skips work that is already correct.
set -euo pipefail

SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPTS/../.." && pwd)"
BENCH_WORK="${BENCH_WORK:-$HOME/narsil-bench}"
MCL_DIR="${MCL_DIR:-$BENCH_WORK/mcl}"
MCL_REMOTE="${MCL_REMOTE:-https://github.com/herumi/mcl.git}"
# Upstream base plus the two bundled commits below. Both commits carry a fixed
# author, a fixed committer, and a fixed date, so the pin is the hash of the
# patched tree and of nothing else.
MCL_BASE="e107c70e814aaa3079fbb6fd630a8c48693c4c27"
MCL_PIN="b70288469cd065b28a31764450f85238508c4d1e"
MCL_PATCH="$SCRIPTS/mcl-bench-patches.patch"
PATCH_IDENTITY="helius-bench"
PATCH_EMAIL="bench@helius.local"

APT_PACKAGES=(
    build-essential
    ca-certificates
    clang
    curl
    git
    llvm
    make
    pkg-config
    python3
    util-linux
)

main() {
    require_host
    reject_ambient_environment
    install_packages
    install_rust
    fetch_mcl
    build_mcl
    report
}

fail() {
    echo "provision: $*" >&2
    exit 1
}

note() {
    echo "provision: $*"
}

require_host() {
    [[ "$(uname -s)" == "Linux" ]] || fail "this script provisions Linux only"
    case "$(uname -m)" in
        x86_64 | amd64) ;;
        *) fail "the campaign lanes are x86-64 only, this box is $(uname -m)" ;;
    esac
    [[ -f "$REPO/rust-toolchain.toml" ]] || fail "cannot find the crate root above $0"
    [[ -f "$MCL_PATCH" ]] || fail "missing bundled patch $MCL_PATCH"
    if [[ "$(id -u)" == "0" ]]; then
        SUDO=()
    elif command -v sudo >/dev/null 2>&1; then
        SUDO=(sudo)
    else
        fail "run as root, or install sudo"
    fi
    mkdir -p "$BENCH_WORK"
}

# An ambient compiler variable would enter the MCL archive without entering
# its manifest, so the recorded build would not describe the built archive.
reject_ambient_environment() {
    local variable
    for variable in CC CXX CLANG AR MAKE PYTHON CFLAGS CXXFLAGS CPPFLAGS LDFLAGS \
        CFLAGS_USER CFLAGS_OPT_USER LDFLAGS_USER MARCH MAKEFLAGS; do
        if printenv "$variable" >/dev/null 2>&1; then
            fail "the authoritative MCL build rejects ambient $variable"
        fi
    done
}

install_packages() {
    local missing=()
    local package
    for package in "${APT_PACKAGES[@]}"; do
        if ! dpkg-query --show --showformat='${db:Status-Status}\n' "$package" 2>/dev/null | grep -qx installed; then
            missing+=("$package")
        fi
    done
    if [[ ${#missing[@]} -gt 0 ]]; then
        note "installing ${missing[*]}"
        "${SUDO[@]}" apt-get update
        "${SUDO[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y "${missing[@]}"
    fi
    for tool in cc c++ clang++ ar make python3 git taskset; do
        command -v "$tool" >/dev/null 2>&1 || fail "required tool is still missing after apt: $tool"
    done
    install_perf
}

# Gate B needs a working perf. A container often denies perf_event_open even
# when the tool installs, so a failure here is reported and not fatal.
install_perf() {
    if ! command -v perf >/dev/null 2>&1; then
        note "installing perf for the running kernel"
        "${SUDO[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y \
            "linux-tools-$(uname -r)" linux-tools-common linux-tools-generic ||
            note "no perf package for kernel $(uname -r), gate B will report a skip"
    fi
    if command -v perf >/dev/null 2>&1 && perf stat -x, -e task-clock -- true 2>&1 | grep -q task-clock; then
        note "perf counts task-clock, gate B can run"
    else
        note "perf cannot count task-clock on this host, gate B will report a skip"
    fi
}

install_rust() {
    local channel
    channel="$(grep -E '^[[:space:]]*channel[[:space:]]*=' "$REPO/rust-toolchain.toml" | cut -d'"' -f2)"
    [[ -n "$channel" ]] || fail "cannot read the pinned channel from rust-toolchain.toml"
    # rustup drops its shims here and a shell that has not sourced its env does
    # not see them. Probing before this line reinstalls a rustup that is
    # already present.
    export PATH="$HOME/.cargo/bin:$PATH"
    if ! command -v rustup >/dev/null 2>&1; then
        note "installing rustup with toolchain $channel"
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs |
            sh -s -- -y --default-toolchain "$channel" --profile minimal
    fi
    command -v rustup >/dev/null 2>&1 || fail "rustup is installed but not on PATH"
    rustup toolchain install "$channel" --profile minimal
    rustup component add rustfmt clippy --toolchain "$channel"
    note "pinned toolchain $channel ready"
}

fetch_mcl() {
    if [[ ! -d "$MCL_DIR/.git" ]]; then
        note "cloning MCL into $MCL_DIR"
        git clone "$MCL_REMOTE" "$MCL_DIR"
    fi
    MCL_DIR="$(cd "$MCL_DIR" && pwd)"
    if [[ "$(git -C "$MCL_DIR" rev-parse HEAD)" == "$MCL_PIN" ]]; then
        note "MCL already at the pin $MCL_PIN"
        return
    fi
    git -C "$MCL_DIR" fetch --tags origin
    git -C "$MCL_DIR" checkout --detach "$MCL_BASE"
    git -C "$MCL_DIR" reset --hard "$MCL_BASE"
    git -C "$MCL_DIR" clean -fdx
    # The commit hash binds author, committer, and both dates. Drop any of the
    # four settings below and the tree lands off the pin, which the archive
    # build then rejects.
    env \
        GIT_AUTHOR_NAME="$PATCH_IDENTITY" \
        GIT_AUTHOR_EMAIL="$PATCH_EMAIL" \
        GIT_COMMITTER_NAME="$PATCH_IDENTITY" \
        GIT_COMMITTER_EMAIL="$PATCH_EMAIL" \
        git -C "$MCL_DIR" am --committer-date-is-author-date "$MCL_PATCH"
    local head
    head="$(git -C "$MCL_DIR" rev-parse HEAD)"
    [[ "$head" == "$MCL_PIN" ]] ||
        fail "patched MCL is at $head, expected $MCL_PIN; the git identity or the committer date is wrong"
    note "MCL at the pin $MCL_PIN"
}

# One authoritative MCL archive. env -i below removes every inherited compiler
# variable, so the recorded manifest describes the whole effective build.
build_mcl() {
    local cc cxx clang ar make python arch native_flag hermetic_path status
    cc="$(resolve_system_tool cc)"
    cxx="$(resolve_system_tool c++)"
    clang="$(resolve_system_tool clang++)"
    ar="$(resolve_system_tool ar)"
    make="$(resolve_system_tool make)"
    python="$(resolve_system_tool python3)"
    arch="$(uname -m)"
    native_flag="-march=native"

    local make_variables=(
        "ARCH=$arch"
        "UNAME_S=Linux"
        "CLANG_TARGET="
        "CC=$cc"
        "CXX=$cxx"
        "CLANG=$clang"
        "AR=$ar"
        "ARFLAGS=r"
        "DEBUG=0"
        "SANITIZE_OPT="
        "CFLAGS_USER="
        "CFLAGS_OPT_USER=-fomit-frame-pointer -DNDEBUG -fno-stack-protector -O3 $native_flag"
        "LDFLAGS_USER="
        "MARCH=$native_flag"
        "MCL_FP_BIT=256"
        "MCL_FR_BIT=256"
        "MCL_SIZEOF_UNIT=8"
        "MCL_USE_GMP=0"
        "MCL_USE_GMP_LIB=0"
        "MCL_STATIC_CODE=0"
        "MCL_USE_OMP=0"
        "MCL_USE_PROF=0"
        "MCL_USE_XBYAK=1"
        "MCL_USE_LLVM=1"
        "MCL_BINT_ASM=1"
        "MCL_BINT_ASM_X64=1"
        "MCL_MSM=0"
        "UPDATE_ASM=0"
        "UPDATE_LL=0"
        "PRE="
        "LLVM_VER="
        "LIB_DIR=lib256"
        "OBJ_DIR=obj256"
    )
    hermetic_path="$(dirname "$cc"):$(dirname "$cxx"):$(dirname "$clang"):$(dirname "$ar"):$(dirname "$make"):$(dirname "$python"):/usr/bin:/bin"

    # Both directories are dedicated outputs of this harness.
    rm -rf "$MCL_DIR/obj256" "$MCL_DIR/lib256"
    status="$(git -C "$MCL_DIR" status --porcelain=v1 --untracked-files=all)"
    if [[ -n "$status" ]]; then
        echo "$status" >&2
        fail "the MCL checkout must be clean before the archive build"
    fi
    mkdir -p "$MCL_DIR/obj256" "$MCL_DIR/lib256"
    note "building the 256-bit MCL archive"
    env -i \
        PATH="$hermetic_path" \
        HOME="${HOME:-/tmp}" \
        TMPDIR="${TMPDIR:-/tmp}" \
        PYTHONDONTWRITEBYTECODE=1 \
        LANG=C LC_ALL=C MAKEFLAGS= \
        "$make" --no-print-directory -j1 -C "$MCL_DIR" \
        "${make_variables[@]}" lib256/libmcl.a
    status="$(git -C "$MCL_DIR" status --porcelain=v1 --untracked-files=all |
        grep -Ev '^\?\? (lib256|obj256)/' || true)"
    if [[ -n "$status" ]]; then
        echo "$status" >&2
        fail "the MCL checkout changed during the archive build"
    fi

    "$python" "$SCRIPTS/mcl_manifest.py" \
        --root "$MCL_DIR" \
        --revision "$MCL_PIN" \
        --native-flag="$native_flag" \
        --architecture "$arch" \
        --operating-system Linux \
        --hermetic-path "$hermetic_path" \
        --tool "cc=$cc" --tool "cxx=$cxx" --tool "clang=$clang" \
        --tool "ar=$ar" --tool "make=$make" --tool "python=$python" \
        -- "${make_variables[@]}"
}

resolve_system_tool() {
    if [[ -x "/usr/bin/$1" ]]; then
        printf '%s\n' "/usr/bin/$1"
        return
    fi
    if [[ -x "/bin/$1" ]]; then
        printf '%s\n' "/bin/$1"
        return
    fi
    command -v "$1" || fail "required build tool is unavailable: $1"
}

report() {
    note "work directory  $BENCH_WORK"
    note "MCL            $MCL_DIR"
    note "next           MCL_DIR='$MCL_DIR' bench/scripts/build_lanes.sh"
}

main "$@"
