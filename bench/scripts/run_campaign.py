#!/usr/bin/env python3
"""Four-lane BN254 benchmark campaign runner.

Drives one shared native binary (helius, arkworks, mcl) and one portable
binary (arkworks-portable) through the frozen operation matrix. Lane order
rotates per (operation, round) and the ROTATION argument rotates fixtures per
round. Any protocol violation aborts the campaign, in quick mode too.

Each spawn is `four_lane LANE OPERATION ITERATIONS ROTATION`, one process per
cell, pinned to one core with a minimal single-threaded environment.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import pathlib
import platform
import re
import secrets
import statistics
import subprocess
import sys
import time
from typing import Any


PROVENANCE_SCHEMA = "four-lane-provenance-v1"
SUMMARY_SCHEMA = "four-lane-summary-v1"
LANES = ("helius", "arkworks", "arkworks-portable", "mcl")
SHARED_BINARY_LANES = ("helius", "arkworks", "mcl")
PORTABLE_LANE = "arkworks-portable"
PROVENANCE_FIELDS = (
    "schema",
    "rustflags",
    "target_cpu",
    "ark_asm",
    "mcl_archive_sha256",
    "mcl_manifest_sha256",
    "cxx",
    "cxx_native_flag",
    "helius_backend",
    "mcl_runtime",
    "git_rev",
    "timer_overhead_ns_per_iter",
    "tsc_agreement_pct",
    "fixture_seed",
    "fixture_set_sha256",
)
# Line schedules are lane-local layouts. These digests bind each lane's own
# schedule bits and are never compared across lanes, only across repeat
# invocations of the same (operation, lane, rotation).
LANE_LOCAL_DIGEST_OPS = {"g2_prepare_87_lines"}
DIGEST_RE = re.compile(r"[0-9a-f]{16}\Z")
SEED_RE = re.compile(r"\A(?:0[xX][0-9a-fA-F]{1,16}|[0-9]{1,20})\Z")

ITERATIONS = {
    "fp_mul": 16_384,
    "fp_square": 16_384,
    "fp2_mul": 8_192,
    "fp2_square": 8_192,
    "fp12_mul": 1_024,
    "fp12_square": 2_048,
    "fp12_sparse_034": 2_048,
    "fp12_cyclotomic_square": 2_048,
    "g1_line_scaling": 4_096,
    "g2_prepare_87_lines": 16,
    "g1_msm_pub_inputs_1": 512,
    "g1_msm_pub_inputs_2": 512,
    "g1_msm_pub_inputs_3": 512,
    "g1_validate": 4_096,
    "g2_subgroup_check": 1_024,
    "miller_live": 16,
    "miller_prepared": 16,
    "final_exponentiation": 16,
    "full_pairing": 16,
    "full_pairing_prepared": 16,
    "groth16_gnark": 8,
    "groth16_one_input": 8,
    "groth16_committed_rails": 8,
    "groth16_batch8_one_key": 4,
    "groth16_batch8_mixed_keys": 4,
    "groth16_batch3_one_key": 8,
    "groth16_batch3_mixed_keys": 8,
    # Gate E's cross-lane anchor. Fixed work, identical code in every lane.
    "lane_reference": 4_096,
}
# Quick mode still visits every fixture in the pool an operation draws from.
# The pooled Groth16 rows are the deliberate exception. Each holds more
# vectors than one invocation draws, and the rounds walk the rest.
WINDOWED_OPERATIONS = {
    "groth16_gnark",
    "groth16_one_input",
    "groth16_committed_rails",
    "groth16_batch8_one_key",
    "groth16_batch8_mixed_keys",
    "groth16_batch3_one_key",
    "groth16_batch3_mixed_keys",
}
FIXTURE_POOL_MINIMUM = {
    operation: 16
    if operation
    in {
        "g2_prepare_87_lines",
        "miller_live",
        "miller_prepared",
        "final_exponentiation",
        "full_pairing",
        "full_pairing_prepared",
    }
    else 8
    for operation in ITERATIONS
}
QUICK_ITERATIONS = {
    operation: iterations
    if operation in WINDOWED_OPERATIONS
    else max(FIXTURE_POOL_MINIMUM[operation], iterations // 16)
    for operation, iterations in ITERATIONS.items()
}
COMPILER_COMMS = {"rustc", "cargo", "cc1", "clang", "ld", "make", "go"}
# A shared host cannot idle the SMT sibling. Diagnostic runs record the
# measured value up to this bound, claim runs still demand 2 percent.
DIAGNOSTIC_SIBLING_LIMIT = 12.0
MD_ACTIVITY = re.compile(r"\b(resync|recovery|reshape|check|repair)\b\s*=")


def fatal(message: str) -> None:
    raise SystemExit(f"fatal: {message}")


# Gate violations tolerated on a diagnostic host, recorded for the manifest.
HOST_GATE_WARNINGS: list[str] = []


def gate_violation(strict: bool, message: str) -> None:
    if strict:
        fatal(message)
    if message not in HOST_GATE_WARNINGS:
        HOST_GATE_WARNINGS.append(message)
        print(f"diagnostic-host: {message}", file=sys.stderr)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.rounds < 1:
        fatal("rounds must be at least 1")
    if not args.quick and (args.rounds < 4 or args.rounds % 4):
        fatal("claim rounds must be a positive multiple of four")
    args.fixture_seed = normalize_seed(args.fixture_seed)
    os.environ["FOUR_LANE_FIXTURE_SEED"] = args.fixture_seed
    lanes = load_lanes(pathlib.Path(args.lanes))
    gates_enabled = not args.allow_non_linux
    if gates_enabled and platform.system() != "Linux":
        fatal("host gates need Linux; pass --allow-non-linux for local testing")

    lock = None
    if args.lock:
        lock_path = pathlib.Path(args.lock)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock = lock_path.open("a+")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fatal(f"another campaign holds {lock_path}")

    out = pathlib.Path(args.out)
    try:
        out.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        fatal(f"refusing to overwrite existing output directory {out}")

    table = QUICK_ITERATIONS if args.quick else ITERATIONS
    strict_gates = not args.diagnostic_host
    host_state = check_host(args.cpu, strict_gates) if gates_enabled else None
    mitigations = check_mitigations(strict_gates) if gates_enabled else None
    metadata = {
        "schema": "four-lane-metadata-v1",
        "lanes": lanes,
        "cpu": args.cpu,
        "sibling": host_state["sibling"] if host_state else None,
        "host": platform.node(),
        "kernel": platform.release(),
        "boot_id": read_optional("/proc/sys/kernel/random/boot_id"),
        "boost": read_optional("/sys/devices/system/cpu/cpufreq/boost"),
        "host_gates": host_state,
        "mitigations": mitigations,
        "diagnostic_host": args.diagnostic_host,
        "host_gate_warnings": HOST_GATE_WARNINGS,
        "quick": args.quick,
        "rounds": args.rounds,
        "iterations": table,
        "runner_git_rev": runner_git_rev(),
        "started_unix_ns": time.time_ns(),
    }
    def write_metadata() -> None:
        (out / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )

    write_metadata()
    rows, unsupported, provenance = run_campaign(
        lanes, table, args, gates_enabled, out / "raw.jsonl"
    )
    # Every spawn rechecks the host, so a tolerated violation can appear after
    # the first write. The sealed manifest must carry all of them.
    metadata["finished_unix_ns"] = time.time_ns()
    write_metadata()
    summary = summarize(table, rows, unsupported, args, provenance)
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    if lock is not None:
        lock.close()


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lanes", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cpu", default="1")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--allow-non-linux", action="store_true")
    parser.add_argument(
        "--diagnostic-host",
        action="store_true",
        help=(
            "tolerate governor and mitigation gate violations on a host that "
            "cannot satisfy them (rented container); the run records the "
            "violations and can never become claim eligible"
        ),
    )
    parser.add_argument(
        "--fixture-seed",
        help="run scoped u64 fixture seed, decimal or 0x hex, default draws a fresh one",
    )
    parser.add_argument("--lock")
    return parser.parse_args(argv)


def normalize_seed(seed: str | None) -> str:
    """Return the 16-hex form the binary reports back as provenance.

    The runner compares its own seed against every lane's `fixture_seed`
    field, so both sides must spell one value the same way.
    """
    if seed is None:
        return f"0x{secrets.randbits(64):016x}"
    if not SEED_RE.fullmatch(seed):
        fatal(f"fixture seed {seed!r} is not a u64 in decimal or 0x hex")
    value = int(seed, 16) if seed[:2].lower() == "0x" else int(seed)
    if value >> 64:
        fatal(f"fixture seed {seed!r} does not fit in 64 bits")
    return f"0x{value:016x}"


def load_lanes(path: pathlib.Path) -> dict[str, dict[str, Any]]:
    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        fatal(f"cannot read lanes file {path}: {error}")
    if not isinstance(document, dict) or set(document) != set(LANES):
        fatal(f"lanes file must define exactly {sorted(LANES)}")
    for lane, entry in document.items():
        if not isinstance(entry, dict) or not {"binary", "sha256"} <= set(entry):
            fatal(f"lane {lane} needs binary and sha256")
        entry.setdefault("expected_backend", None)
        if lane == "helius" and entry["expected_backend"] is None:
            fatal("lane helius must pin expected_backend in lanes.json")
        binary = pathlib.Path(entry["binary"])
        if not binary.is_file():
            fatal(f"lane {lane} binary {binary} is missing")
        observed = hashlib.sha256(binary.read_bytes()).hexdigest()
        if observed != entry["sha256"]:
            fatal(
                f"lane {lane} binary hash mismatch: expected {entry['sha256']}, "
                f"observed {observed}"
            )
    shared = {
        pathlib.Path(document[lane]["binary"]).resolve()
        for lane in SHARED_BINARY_LANES
    }
    if len(shared) != 1:
        fatal("helius, arkworks, and mcl must share one binary path")
    if pathlib.Path(document[PORTABLE_LANE]["binary"]).resolve() in shared:
        fatal("arkworks-portable must use a different binary than the native lanes")
    return document


def lane_order(operation_index: int, round_index: int) -> list[str]:
    return [
        LANES[(position + operation_index + round_index) % len(LANES)]
        for position in range(len(LANES))
    ]


def run_campaign(
    lanes: dict[str, dict[str, Any]],
    table: dict[str, int],
    args: argparse.Namespace,
    gates_enabled: bool,
    raw_path: pathlib.Path,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], str], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    unsupported: dict[tuple[str, str], str] = {}
    digests: dict[tuple[str, int, str], tuple[str, str]] = {}
    fixture_sets: dict[str, tuple[str, str]] = {}
    mcl_reference: dict[str, str] | None = None
    provenance_by_binary: dict[str, Any] = {}
    strict_gates = not args.diagnostic_host
    with raw_path.open("x") as raw:
        for round_index in range(args.rounds):
            for operation_index, (operation, iterations) in enumerate(table.items()):
                order = lane_order(operation_index, round_index)
                for position, lane in enumerate(order):
                    if gates_enabled:
                        check_host(args.cpu, strict_gates)
                    row = invoke(
                        lanes[lane], lane, operation, iterations, round_index, args
                    )
                    row["order"] = position
                    prov = row["provenance"]
                    binary_kind = "portable" if lane == PORTABLE_LANE else "native"
                    provenance_by_binary.setdefault(binary_kind, prov)
                    mcl_reference = check_mcl_uniform(mcl_reference, prov, lane)
                    check_fixture_set(fixture_sets, operation, lane, prov)
                    check_fixture_seed(args.fixture_seed, prov, lane)
                    if "digest" in row:
                        check_digest(digests, operation, round_index, lane, row)
                        if (operation, lane) in unsupported:
                            fatal(f"{lane} flip-flopped on support for {operation}")
                    else:
                        previous = unsupported.setdefault(
                            (operation, lane), row["unsupported"]
                        )
                        if previous != row["unsupported"] or any(
                            r["operation"] == operation and r["lane"] == lane
                            for r in rows
                            if "digest" in r
                        ):
                            fatal(f"{lane} flip-flopped on support for {operation}")
                    raw.write(json.dumps(row, sort_keys=True) + "\n")
                    raw.flush()
                    rows.append(row)
    return rows, unsupported, provenance_by_binary


def invoke(
    lane_config: dict[str, Any],
    lane: str,
    operation: str,
    iterations: int,
    rotation: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    command = [
        str(lane_config["binary"]),
        lane,
        operation,
        str(iterations),
        str(rotation),
    ]
    if args.cpu != "none":
        command = ["taskset", "-c", args.cpu, *command]
    spawned = time.time_ns()
    completed = subprocess.run(
        command, env=spawn_env(), capture_output=True, text=True
    )
    finished = time.time_ns()
    label = f"{lane}/{operation} rotation {rotation}"
    if completed.returncode == 2:
        fatal(f"usage error (exit 2) from {label}: {completed.stderr.strip()}")
    if completed.returncode == 3:
        fatal(f"build-regime gate failure (exit 3) from {label}")
    if completed.returncode == 4:
        fatal(f"cross-implementation equivalence failure (exit 4) from {label}")
    if completed.returncode == 5:
        fatal(f"timer self-test failure (exit 5) from {label}")
    if completed.returncode not in (0, 6):
        fatal(
            f"unexpected exit code {completed.returncode} from {label}: "
            f"{completed.stderr.strip()}"
        )
    provenance = parse_provenance(completed.stdout, lane_config, lane, label)
    row: dict[str, Any] = {
        "round": rotation,
        "operation": operation,
        "lane": lane,
        "iterations": iterations,
        "exit_code": completed.returncode,
        "provenance": provenance,
        "spawned_unix_ns": spawned,
        "finished_unix_ns": finished,
    }
    if completed.returncode == 6:
        row["unsupported"] = parse_unsupported(
            completed.stdout, lane, operation, label
        )
        return row
    elapsed_ns, digest = parse_result(
        completed.stdout, lane, operation, iterations, label
    )
    row["elapsed_ns"] = elapsed_ns
    row["ns_per_call"] = elapsed_ns / iterations
    row["digest"] = digest
    return row


def spawn_env() -> dict[str, str]:
    keep = {
        "PATH",
        "HOME",
        "USER",
        "LANG",
        "LC_ALL",
        "TMPDIR",
        "FOUR_LANE_FIXTURE_SEED",
    }
    env = {key: value for key, value in os.environ.items() if key in keep}
    env["RAYON_NUM_THREADS"] = "1"
    env["MCL_USE_OMP"] = "0"
    return env


def parse_provenance(
    stdout: str, lane_config: dict[str, Any], lane: str, label: str
) -> dict[str, Any]:
    lines = [line for line in stdout.splitlines() if line.startswith("provenance|")]
    if len(lines) != 1:
        fatal(f"expected exactly one provenance line from {label}, saw {len(lines)}")
    try:
        provenance = json.loads(lines[0].split("|", 1)[1])
    except json.JSONDecodeError as error:
        fatal(f"provenance line from {label} is not JSON: {error}")
    missing = [field for field in PROVENANCE_FIELDS if field not in provenance]
    if missing:
        fatal(f"provenance from {label} is missing {missing}")
    if provenance["schema"] != PROVENANCE_SCHEMA:
        fatal(
            f"provenance schema from {label} is {provenance['schema']!r}, "
            f"expected {PROVENANCE_SCHEMA!r}"
        )
    if not isinstance(provenance["ark_asm"], bool):
        fatal(f"provenance ark_asm from {label} must be a boolean")
    if lane == "arkworks" and not provenance["ark_asm"]:
        fatal(f"arkworks lane ran without ark-ff asm ({label})")
    if lane == PORTABLE_LANE and provenance["ark_asm"]:
        fatal(f"arkworks-portable lane ran with ark-ff asm ({label})")
    expected = lane_config.get("expected_backend")
    if expected is not None and provenance["helius_backend"] != expected:
        fatal(
            f"helius_backend from {label} is {provenance['helius_backend']!r}, "
            f"expected {expected!r}"
        )
    return provenance


def parse_result(
    stdout: str, lane: str, operation: str, iterations: int, label: str
) -> tuple[int, str]:
    lines = [line for line in stdout.splitlines() if line.startswith("result|")]
    if len(lines) != 1:
        fatal(f"expected exactly one result line from {label}, saw {len(lines)}")
    fields = lines[0].split("|")
    if len(fields) != 6:
        fatal(f"malformed result line from {label}: {lines[0]!r}")
    if fields[1:4] != [operation, lane, str(iterations)]:
        fatal(f"result line from {label} echoes a different request: {lines[0]!r}")
    if not fields[3].isdigit() or not fields[4].isdigit():
        fatal(f"non-numeric result fields from {label}: {lines[0]!r}")
    if not DIGEST_RE.fullmatch(fields[5]):
        fatal(f"result digest from {label} is not 16 hex characters: {fields[5]!r}")
    return int(fields[4]), fields[5]


def parse_unsupported(stdout: str, lane: str, operation: str, label: str) -> str:
    lines = [line for line in stdout.splitlines() if line.startswith("unsupported|")]
    if len(lines) != 1:
        fatal(f"exit 6 without exactly one unsupported line from {label}")
    fields = lines[0].split("|", 3)
    if len(fields) != 4 or fields[1] != operation or fields[2] != lane:
        fatal(f"malformed unsupported line from {label}: {lines[0]!r}")
    return fields[3]


def check_mcl_uniform(
    reference: dict[str, str] | None, provenance: dict[str, Any], lane: str
) -> dict[str, str]:
    observed = {
        "mcl_archive_sha256": provenance["mcl_archive_sha256"],
        "mcl_manifest_sha256": provenance["mcl_manifest_sha256"],
    }
    if reference is None:
        return observed
    for key, expected in reference.items():
        if observed[key] != expected:
            fatal(
                f"{key} changed during the campaign ({lane}): expected "
                f"{expected}, observed {observed[key]}"
            )
    return reference


def check_fixture_seed(reference: str, provenance: dict[str, Any], lane: str) -> None:
    """Every lane of one campaign must draw the same run scoped seed."""
    observed = provenance["fixture_seed"]
    if observed != reference:
        fatal(
            f"fixture_seed differs, campaign uses {reference},"
            f" {lane} reported {observed}"
        )


def check_fixture_set(
    fixture_sets: dict[str, tuple[str, str]],
    operation: str,
    lane: str,
    provenance: dict[str, Any],
) -> None:
    observed = provenance["fixture_set_sha256"]
    previous = fixture_sets.setdefault(operation, (lane, observed))
    if previous[1] != observed:
        fatal(
            f"fixture_set_sha256 differs for {operation}: {previous[0]} reported "
            f"{previous[1]}, {lane} reported {observed}"
        )


def check_digest(
    digests: dict[tuple[str, int, str], tuple[str, str]],
    operation: str,
    rotation: int,
    lane: str,
    row: dict[str, Any],
) -> None:
    # Lane-local digests key on the lane too, so cross-lane equality is not
    # required but a repeat of the same cell must still reproduce its digest.
    digest_lane = lane if operation in LANE_LOCAL_DIGEST_OPS else ""
    previous = digests.setdefault(
        (operation, rotation, digest_lane), (lane, row["digest"])
    )
    if previous[1] != row["digest"]:
        fatal(
            f"digest mismatch for {operation} rotation {rotation}: {previous[0]} "
            f"reported {previous[1]}, {lane} reported {row['digest']}"
        )


def summarize(
    table: dict[str, int],
    rows: list[dict[str, Any]],
    unsupported: dict[tuple[str, str], str],
    args: argparse.Namespace,
    provenance_by_binary: dict[str, Any],
) -> dict[str, Any]:
    cells: dict[str, dict[str, Any]] = {}
    for operation in table:
        cells[operation] = {}
        for lane in LANES:
            if (operation, lane) in unsupported:
                cells[operation][lane] = {
                    "unsupported": unsupported[(operation, lane)]
                }
                continue
            samples = [
                row["ns_per_call"]
                for row in rows
                if row["operation"] == operation
                and row["lane"] == lane
                and "digest" in row
            ]
            cells[operation][lane] = {
                "median_ns_per_call": statistics.median(samples),
                "min_ns_per_call": min(samples),
                "iqr_ns_per_call": interquartile_range(samples),
                "samples": len(samples),
            }
    return {
        "schema": SUMMARY_SCHEMA,
        # Only cross_check.py may flip this to true.
        "claim_eligible": False,
        "diagnostic_host": getattr(args, "diagnostic_host", False),
        "quick": args.quick,
        "rounds": args.rounds,
        "fixture_seed": getattr(args, "fixture_seed", None),
        "cells": cells,
        "provenance": provenance_by_binary,
    }


def interquartile_range(samples: list[float]) -> float:
    if len(samples) < 2:
        return 0.0
    quartiles = statistics.quantiles(samples, n=4, method="inclusive")
    return quartiles[2] - quartiles[0]


def check_mitigations(strict: bool = True) -> dict[str, Any]:
    """Abort unless the kernel booted with mitigations=off.

    Default mitigations (the srso family) cripple out-of-order execution on
    this host class and inflate crypto-kernel cycles severalfold, so any
    sample taken under them is invalid. The cmdline and the vulnerability
    states are recorded as provenance. A diagnostic host records the violation
    instead of aborting and can never become claim eligible.
    """
    cmdline = read_sys("/proc/cmdline")
    if "mitigations=off" not in cmdline.split():
        gate_violation(
            strict,
            "kernel booted without mitigations=off; add it to "
            "GRUB_CMDLINE_LINUX in /etc/default/grub, update-grub, and reboot",
        )
    return {
        "cmdline": cmdline,
        "spec_rstack_overflow": read_optional(
            "/sys/devices/system/cpu/vulnerabilities/spec_rstack_overflow"
        ),
        "spectre_v2": read_optional(
            "/sys/devices/system/cpu/vulnerabilities/spectre_v2"
        ),
    }


def check_host(cpu: str, strict: bool = True) -> dict[str, Any]:
    """Abort unless the pinned core can produce a trustworthy sample."""
    # A container may not expose cpufreq at all. That is itself a gate
    # violation, the run cannot show the core is held at a fixed clock.
    governor = read_optional(
        f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor"
    )
    if governor is None:
        gate_violation(strict, f"cpu{cpu} exposes no cpufreq governor")
    elif governor != "performance":
        gate_violation(
            strict, f"cpu{cpu} governor is {governor!r}, need 'performance'"
        )
    sibling = smt_sibling(cpu)
    busy = None
    if sibling is not None:
        # A claim host owns its silicon and must hold the sibling under 2%.
        # A shared diagnostic host never can, another tenant runs on the same
        # physical core, so the run records the measured value instead and
        # stays ineligible for a claim. One transient blip never aborts, a
        # sustained crossing does.
        limit = 2.0 if strict else DIAGNOSTIC_SIBLING_LIMIT
        over = 0
        for _ in range(3):
            busy = cpu_busy_pct(sibling)
            if busy < limit:
                break
            over += 1
            if over >= 2:
                gate_violation(
                    strict,
                    f"SMT sibling cpu{sibling} is {busy:.1f}% busy,"
                    f" need under {limit}%",
                )
                break
    mdstat = read_optional("/proc/mdstat")
    if mdstat and MD_ACTIVITY.search(mdstat):
        fatal("an MD array resync is active")
    compilers = live_compilers()
    if compilers:
        fatal(f"live compiler processes: {sorted(set(compilers))}")
    return {"governor": governor, "sibling": sibling, "sibling_busy_pct": busy}


def smt_sibling(cpu: str) -> int | None:
    siblings = parse_cpu_list(
        read_sys(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
    )
    others = [item for item in siblings if item != int(cpu)]
    return others[0] if others else None


def parse_cpu_list(text: str) -> list[int]:
    cpus: list[int] = []
    for part in text.split(","):
        if not part:
            continue
        bounds = part.split("-", 1)
        cpus.extend(range(int(bounds[0]), int(bounds[-1]) + 1))
    return sorted(set(cpus))


def cpu_busy_pct(cpu: int) -> float:
    first = proc_stat_times(cpu)
    sleep(1.0)
    second = proc_stat_times(cpu)
    total = sum(second) - sum(first)
    if total <= 0:
        return 0.0
    idle = (second[3] + second[4]) - (first[3] + first[4])
    return 100.0 * (total - idle) / total


def proc_stat_times(cpu: int) -> list[int]:
    for line in read_sys("/proc/stat").splitlines():
        fields = line.split()
        if fields and fields[0] == f"cpu{cpu}":
            return [int(value) for value in fields[1:]]
    fatal(f"/proc/stat has no line for cpu{cpu}")
    raise AssertionError("unreachable")


def live_compilers() -> list[str]:
    """Match on the command name, never on a command line.

    A command-line pattern also matches the process that runs the search, and
    the gate would then fire on itself.
    """
    found: list[str] = []
    for comm in pathlib.Path("/proc").glob("[0-9]*/comm"):
        try:
            name = comm.read_text().strip()
        except OSError:
            continue
        if name in COMPILER_COMMS:
            found.append(name)
    return found


def sleep(seconds: float) -> None:
    time.sleep(seconds)


def read_sys(path: str) -> str:
    return pathlib.Path(path).read_text().strip()


def read_optional(path: str) -> str | None:
    try:
        return read_sys(path)
    except OSError:
        return None


def runner_git_rev() -> str | None:
    root = pathlib.Path(__file__).resolve().parents[2]
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


if __name__ == "__main__":
    main()
