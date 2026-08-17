#!/usr/bin/env bash
# Build the timing binaries for the four lanes and write lanes.json.
#
# Two binaries carry the four lanes. One is built with ark-ff assembly on and
# answers for helius, arkworks, and mcl. The other is built with ark-ff
# assembly off and answers only for arkworks-portable. The runner refuses to
# start unless the pair keeps that shape and unless both hashes match.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BENCH="$REPO/bench"
BENCH_WORK="${BENCH_WORK:-$HOME/narsil-bench}"
MCL_DIR="${MCL_DIR:-$BENCH_WORK/mcl}"
# A build that names no target CPU selects a configuration that loses to both
# comparators, so the campaign always names one. See bench/README.md.
TARGET_CPU="${NARSIL_TARGET_CPU:-native}"
EXPECTED_BACKEND="${NARSIL_EXPECTED_BACKEND:-avx512-ifma}"
CONTROL=0

main() {
    parse_args "$@"
    reject_ambient_environment
    require_inputs
    build_lanes
    write_manifest
}

fail() {
    echo "build_lanes: $*" >&2
    exit 1
}

note() {
    echo "build_lanes: $*"
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --control) CONTROL=1 ;;
            *) fail "unknown argument $1" ;;
        esac
        shift
    done
}

# An ambient compiler variable would change one lane and not the other, and
# the difference would show up as a performance result.
reject_ambient_environment() {
    local variable
    for variable in CC CXX CLANG AR CFLAGS CXXFLAGS CPPFLAGS LDFLAGS RUSTFLAGS \
        CARGO_ENCODED_RUSTFLAGS CARGO_TARGET_DIR MARCH MAKEFLAGS; do
        if printenv "$variable" >/dev/null 2>&1; then
            fail "the lane build rejects ambient $variable"
        fi
    done
}

require_inputs() {
    [[ -f "$BENCH/Cargo.toml" ]] || fail "no harness crate at $BENCH/Cargo.toml"
    [[ -f "$MCL_DIR/lib256/libmcl.a" && -f "$MCL_DIR/lib256/helius-mcl-native.json" ]] ||
        fail "no MCL archive under $MCL_DIR, run bench/scripts/provision.sh first"
    MCL_DIR="$(cd "$MCL_DIR" && pwd)"
    # provision.sh leaves rustup's shims here. A shell that ran it and has not
    # been restarted since still does not have them on PATH.
    export PATH="$HOME/.cargo/bin:$PATH"
    command -v cargo >/dev/null 2>&1 || fail "cargo is not on PATH, run bench/scripts/provision.sh first"
    # Rebuilding a binary a live campaign is timing would swap the file under
    # the hash the runner already checked.
    if pgrep -x four_lane >/dev/null 2>&1; then
        fail "a four_lane process is live, wait for the campaign to finish"
    fi
    mkdir -p "$BENCH_WORK"
}

build() {
    local target_dir="$1"
    shift
    env RUSTFLAGS="-C target-cpu=$TARGET_CPU" \
        MCL_DIR="$MCL_DIR" \
        CARGO_TARGET_DIR="$target_dir" \
        cargo build --manifest-path "$BENCH/Cargo.toml" \
        --locked --release --bin four_lane "$@"
}

build_lanes() {
    note "target cpu $TARGET_CPU"
    build "$BENCH_WORK/target/lane-asm"
    build "$BENCH_WORK/target/lane-ark-portable" --no-default-features
    if [[ "$CONTROL" == 1 ]]; then
        # Control binary only. It never enters lanes.json, because a portable
        # helius lane answers a different question than the campaign asks.
        build "$BENCH_WORK/target/lane-helius-portable" --features force-portable
        note "control binary $BENCH_WORK/target/lane-helius-portable/release/four_lane"
    fi
}

write_manifest() {
    local asm portable lanes
    asm="$BENCH_WORK/target/lane-asm/release/four_lane"
    portable="$BENCH_WORK/target/lane-ark-portable/release/four_lane"
    lanes="$BENCH_WORK/lanes.json"
    [[ -x "$asm" && -x "$portable" ]] || fail "cargo produced no four_lane binary"
    python3 "$BENCH/scripts/write_lanes.py" \
        --asm-binary "$asm" \
        --portable-binary "$portable" \
        --expected-backend "$EXPECTED_BACKEND" \
        --out "$lanes"
    note "lanes manifest $lanes"
}

main "$@"
