#!/usr/bin/env python3
"""Validity gates for a four-lane campaign summary.

Reads summary.json produced by run_campaign.py, applies gates A to D, and
writes validity.json. Only this script may flip claim_eligible to true, and
only when every applicable gate passes.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import statistics
import subprocess
import sys
from typing import Any


VALIDITY_SCHEMA = "four-lane-validity-v1"
LANES = ("helius", "arkworks", "arkworks-portable", "mcl")
# Criterion group -> campaign operation. The bench registers lanes as
# functions named helius, arkworks, and mcl; there is no portable lane.
ANCHOR_GROUPS = {
    "bn254_single_pair_miller_loop": "miller_live",
    "bn254_single_pair_full_pairing": "full_pairing",
    "bn254_final_exponentiation": "final_exponentiation",
    "bn254_groth16_verify_prepared_e2e": "groth16_one_input",
    "bn254_single_pair_miller_loop_prepared_g2": "miller_prepared",
}
CRITERION_LANES = ("helius", "arkworks", "mcl")
CROSS_HARNESS_TOLERANCE = 0.10
PERF_TOLERANCE = 0.05

# Gate C scale windows over every operation, first-principles estimates at
# ~5 GHz Zen5. The pairing anchors keep their original x2 widths, the rest
# are coarse x4-wide windows derived from operation counts in the line
# comments below. One Fp mul is the 4..60 ns unit across the four lanes.
SCALE_WINDOWS_NS = {
    # A 4x4-limb Montgomery multiply is ~35 mul-add plus reduction,
    # roughly 20-150 cycles across the four lanes.
    "fp_mul": (4.0, 60.0),
    # Squaring shares the fp_mul cost class.
    "fp_square": (4.0, 60.0),
    # Karatsuba Fp2 mul is 3 Fp muls plus additions.
    "fp2_mul": (8.0, 250.0),
    # Complex Fp2 squaring is 2 Fp muls plus additions.
    "fp2_square": (6.0, 200.0),
    # Tower Fp12 mul is 18 Fp2 muls, ~54 Fp-mul-equivalents.
    "fp12_mul": (150.0, 3_500.0),
    # Fp12 squaring is ~0.7 of a full Fp12 mul.
    "fp12_square": (100.0, 2_500.0),
    # A sparse 034 line multiply is ~13 Fp2 muls, ~0.7 of a full mul.
    "fp12_sparse_034": (100.0, 2_500.0),
    # Granger-Scott squaring is ~9 Fp2 squares, ~0.4 of a full mul.
    "fp12_cyclotomic_square": (60.0, 1_600.0),
    # Line scaling is 2 Fp muls.
    "g1_line_scaling": (5.0, 130.0),
    # 87 lines, each one G2 double or add of ~30-45 Fp-mul-equivalents,
    # ~3k Fp muls total.
    "g2_prepare_87_lines": (8_000.0, 250_000.0),
    # One 256-bit G1 scalar mul, ~2-3k Fp-mul-equivalents.
    "g1_msm_pub_inputs_1": (8_000.0, 300_000.0),
    # Straus sharing makes two points ~1.2x of one.
    "g1_msm_pub_inputs_2": (9_000.0, 350_000.0),
    # Three points ~1.4x of one.
    "g1_msm_pub_inputs_3": (10_000.0, 400_000.0),
    # Canonical decode of two coordinates plus the curve equation,
    # ~5-10 Fp-mul-equivalents.
    "g1_validate": (15.0, 700.0),
    # helius/arkworks run a ~128-bit endomorphism scalar mul over G2
    # (~5k Fp muls), mcl a full 254-bit order mul (~2x that).
    "g2_subgroup_check": (15_000.0, 800_000.0),
    # The BN254 Miller loop is ~12-13k Fp-mul-equivalents, ~125 us center
    # at 10 ns each.
    "miller_live": (40_000.0, 250_000.0),
    "miller_prepared": (30_000.0, 250_000.0),
    # Miller loop plus final exponentiation.
    "full_pairing": (80_000.0, 500_000.0),
    "full_pairing_prepared": (70_000.0, 500_000.0),
    # The hard part is ~7-12k Fp-mul-equivalents, the same magnitude as
    # the Miller loop.
    "final_exponentiation": (40_000.0, 250_000.0),
    # The plain Groth16 equation over gnark's proofs: three Miller loops,
    # one shared final exponentiation, and a 3-point public-input MSM,
    # ~3-4 pairing-equivalents. The highs span host frequencies from 3.2
    # to 5 GHz.
    "groth16_gnark": (150_000.0, 1_800_000.0),
    # The same equation over a production proof, with a 1-point public-input
    # MSM instead of a 3-point one.
    "groth16_one_input": (150_000.0, 1_600_000.0),
    # groth16_one_input plus a 2-pair proof-of-knowledge product, one extra
    # commitment MSM, and a second final exponentiation.
    "groth16_committed_rails": (200_000.0, 3_600_000.0),
    # 8 proofs under one key: 11 pairs, 8 live members and the key's 3
    # prepared schedules, one final exponentiation, 9 G1 scalar
    # multiplications (one per proof for A, one for the key's weighted alpha)
    # and two multi-scalar multiplications, of 8 points for C and of 2 for the
    # collapsed public inputs. ~5-8 pairing-equivalents.
    "groth16_batch8_one_key": (350_000.0, 6_000_000.0),
    # 8 proofs over 2 keys: 14 pairs, 8 live and 6 prepared, 10 scalar
    # multiplications and four multi-scalar multiplications, of 4 and 2 points
    # per key. A mixed batch aggregates nothing across keys, so it costs more
    # than the same-key one.
    "groth16_batch8_mixed_keys": (400_000.0, 7_000_000.0),
    # 3 proofs under one key: 6 pairs, 3 live and 3 prepared, 4 scalar
    # multiplications, and multi-scalar multiplications of 3 and 2 points.
    "groth16_batch3_one_key": (150_000.0, 3_500_000.0),
    # 3 proofs over 2 keys: 9 pairs, 3 live and 6 prepared, 5 scalar
    # multiplications, and four multi-scalar multiplications of at most 2
    # points each.
    "groth16_batch3_mixed_keys": (200_000.0, 4_500_000.0),
    # Gate E's anchor. 1024 dependent 64-bit multiply-rotate steps, about 5
    # cycles each, so 1 us at 5 GHz and 3.4 us at 1.5 GHz.
    "lane_reference": (700.0, 8_000.0),
}

# Gate E's reference row. Every lane times the same function, so the four
# medians differ only by what the host did between the spawns.
REFERENCE_OPERATION = "lane_reference"
# The four spawns of one row sit minutes apart on a host whose clock the
# harness does not own, so the band has to absorb that drift. Measured spread
# on a shared EPYC 9654 with no governor was under 1 percent, so 5 percent
# leaves room for a worse host and still catches any error large enough to move
# a headline ratio.
REFERENCE_SPREAD_LIMIT = 0.05

# Batch rows, as (batch size, key spread). The size drives the sequential
# comparison, so a new size needs no new identity.
BATCH_ROWS = {
    "groth16_batch8_one_key": (8, "same_vk"),
    "groth16_batch8_mixed_keys": (8, "multi_vk"),
    "groth16_batch3_one_key": (3, "same_vk"),
    "groth16_batch3_mixed_keys": (3, "multi_vk"),
}

# Measured batching crossover per lane, the smallest batch size at which a
# lane's batch route beats that lane's own sequential loop. A lane with a
# faster single verification has a harder baseline to beat, so its crossover
# sits higher. Below the crossover the sequential comparison is reported, not
# gated: a small batch that loses to the loop is a true property of the route
# and the honest answer there is to verify one at a time.
BATCH_CROSSOVER = {
    "same_vk": {"helius": 3, "arkworks": 2, "arkworks-portable": 2, "mcl": 2},
    "multi_vk": {"helius": 4, "arkworks": 4, "arkworks-portable": 4, "mcl": 3},
}

# Comparator trust ceilings. A comparator lane can legitimately run past the
# first-principles high, so its ceiling replaces that high for the named
# (operation, lane). Above the ceiling the harness or the build regime is
# broken, never the library. Components carry half the full-pairing ceiling.
LANE_CEILING_NS = {
    ("full_pairing", "mcl"): 620_000.0,
    ("full_pairing_prepared", "mcl"): 620_000.0,
    ("miller_live", "mcl"): 310_000.0,
    ("miller_prepared", "mcl"): 310_000.0,
    ("final_exponentiation", "mcl"): 310_000.0,
    ("full_pairing", "arkworks"): 1_200_000.0,
    ("full_pairing_prepared", "arkworks"): 1_200_000.0,
    ("miller_live", "arkworks"): 600_000.0,
    ("miller_prepared", "arkworks"): 600_000.0,
    ("final_exponentiation", "arkworks"): 600_000.0,
    ("full_pairing", "arkworks-portable"): 1_200_000.0,
    ("full_pairing_prepared", "arkworks-portable"): 1_200_000.0,
    ("miller_live", "arkworks-portable"): 600_000.0,
    ("miller_prepared", "arkworks-portable"): 600_000.0,
    ("final_exponentiation", "arkworks-portable"): 600_000.0,
}

PAIRING_SPLIT_TOLERANCE = 0.15
SQUARE_MUL_CEILING = 1.1
ARK_ASM_INDISTINCT_BAND = 0.03
ARK_ASM_OPS = ("fp_mul", "fp2_mul", "fp12_mul")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--criterion-dir")
    parser.add_argument("--perf", action="store_true")
    args = parser.parse_args(argv)

    summary_path = pathlib.Path(args.summary)
    summary = json.loads(summary_path.read_text())
    cells = summary["cells"]

    gates = {
        "A_cross_harness": gate_cross_harness(cells, args.criterion_dir),
        "B_cycle_cross_check": gate_perf(cells, summary_path, args.perf),
        "C_scale_windows": gate_scale_windows(cells),
        "D_time_identities": gate_time_identities(cells),
        "E_lane_reference": gate_lane_reference(cells),
    }
    claim_eligible = all(
        gate["status"] in ("pass", "skipped") for gate in gates.values()
    ) and any(gate["status"] == "pass" for gate in gates.values())
    # A diagnostic host tolerated gate violations at run time, so its
    # numbers support no claim regardless of the gates.
    if summary.get("diagnostic_host"):
        claim_eligible = False
    validity = {
        "schema": VALIDITY_SCHEMA,
        "gates": gates,
        "flags": {
            "ark_asm_indistinct": gates["D_time_identities"].get(
                "ark_asm_indistinct", False
            )
        },
        "claim_eligible": claim_eligible,
    }
    pathlib.Path(args.out).write_text(
        json.dumps(validity, indent=2, sort_keys=True) + "\n"
    )
    if claim_eligible:
        summary["claim_eligible"] = True
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    for name, gate in gates.items():
        print(f"{name}: {gate['status']}")
    return 0 if claim_eligible else 1


def cell_median(cells: dict[str, Any], operation: str, lane: str) -> float | None:
    cell = cells.get(operation, {}).get(lane)
    if not cell or "unsupported" in cell:
        return None
    return float(cell["median_ns_per_call"])


def gate_cross_harness(
    cells: dict[str, Any], criterion_dir: str | None
) -> dict[str, Any]:
    """Gate A. One number measured by two independent harnesses.

    A timing bug that lives in the campaign binary alone shows up here as a
    disagreement with Criterion on the same operation and lane.
    """
    if criterion_dir is None:
        return {"status": "skipped", "reason": "no --criterion-dir"}
    root = pathlib.Path(criterion_dir)
    comparisons: list[dict[str, Any]] = []
    failures: list[str] = []
    for group, operation in ANCHOR_GROUPS.items():
        for lane in CRITERION_LANES:
            estimate = criterion_median_ns(root, group, lane)
            wall = cell_median(cells, operation, lane)
            if estimate is None or wall is None:
                continue
            deviation = abs(estimate - wall) / wall
            comparison = {
                "operation": operation,
                "lane": lane,
                "criterion_median_ns": estimate,
                "campaign_median_ns": wall,
                "deviation": deviation,
            }
            comparisons.append(comparison)
            if deviation > CROSS_HARNESS_TOLERANCE:
                failures.append(
                    f"{operation}/{lane} deviates {deviation:.1%} from Criterion"
                )
    if not comparisons:
        return {
            "status": "fail",
            "reason": f"no anchor estimates found below {root}",
            "comparisons": [],
        }
    return {
        "status": "fail" if failures else "pass",
        "tolerance": CROSS_HARNESS_TOLERANCE,
        "comparisons": comparisons,
        "failures": failures,
    }


def criterion_median_ns(root: pathlib.Path, group: str, lane: str) -> float | None:
    for path in sorted((root / group / lane).glob("*/new/estimates.json")):
        try:
            estimates = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        value = (estimates.get("median") or {}).get("point_estimate")
        if value is not None:
            return float(value)
    return None


def gate_perf(
    cells: dict[str, Any], summary_path: pathlib.Path, enabled: bool
) -> dict[str, Any]:
    """Gate B. N/2N differencing under perf.

    One invocation at the campaign iteration count N and one at 2N share every
    setup cost (fixture build, equivalence pass, warm-up), so the task-clock
    difference divided by N is the per-call time of the timed loop alone. A
    timer that reads the wrong clock, or a loop the compiler hoisted, cannot
    agree with the kernel's own accounting.
    """
    if not enabled:
        return {"status": "skipped", "reason": "no --perf"}
    metadata = json.loads((summary_path.parent / "metadata.json").read_text())
    if not perf_available():
        # A rented container can deny perf_event_open outright. A diagnostic
        # run records the gap; a claim host must provide working perf.
        if metadata.get("diagnostic_host"):
            return {
                "status": "skipped",
                "reason": "perf unavailable on diagnostic host",
            }
        return {
            "status": "fail",
            "failures": ["perf_event_open denied on a claim host"],
            "checks": [],
        }
    checks: list[dict[str, Any]] = []
    failures: list[str] = []
    for operation in sorted(set(ANCHOR_GROUPS.values())):
        iterations = metadata["iterations"][operation]
        for lane in LANES:
            wall = cell_median(cells, operation, lane)
            if wall is None:
                continue
            binary = metadata["lanes"][lane]["binary"]
            raw_single, clock_single = perf_stat(
                binary, lane, operation, iterations, metadata["cpu"]
            )
            raw_double, clock_double = perf_stat(
                binary, lane, operation, 2 * iterations, metadata["cpu"]
            )
            check: dict[str, Any] = {
                "operation": operation,
                "lane": lane,
                "iterations": iterations,
                "perf_raw_n": raw_single,
                "perf_raw_2n": raw_double,
            }
            if clock_single is None or clock_double is None:
                failures.append(f"{operation}/{lane} perf stat gave no task-clock")
                checks.append(check)
                continue
            delta_ms = clock_double - clock_single
            if delta_ms <= 0.0:
                failures.append(
                    f"{operation}/{lane} non-positive N/2N task-clock delta "
                    f"({clock_single} ms -> {clock_double} ms), setup dominates "
                    "or perf output was misparsed"
                )
                checks.append(check)
                continue
            derived = delta_ms * 1e6 / iterations
            deviation = abs(derived - wall) / wall
            check.update(
                {
                    "task_clock_ns_per_call": derived,
                    "campaign_median_ns": wall,
                    "deviation": deviation,
                }
            )
            checks.append(check)
            if deviation > PERF_TOLERANCE:
                failures.append(
                    f"{operation}/{lane} perf time deviates {deviation:.1%}"
                )
    return {
        "status": "fail" if failures else "pass",
        "tolerance": PERF_TOLERANCE,
        "checks": checks,
        "failures": failures,
    }


def perf_available() -> bool:
    try:
        probe = subprocess.run(
            ["perf", "stat", "-x,", "-e", "task-clock", "--", "true"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    if probe.returncode != 0:
        return False
    # A working perf writes a CSV row whose first field is a number.
    for line in probe.stderr.splitlines():
        fields = line.split(",")
        if len(fields) > 2 and fields[2] == "task-clock":
            try:
                float(fields[0])
                return True
            except ValueError:
                return False
    return False


def perf_stat(
    binary: str, lane: str, operation: str, iterations: int, cpu: str
) -> tuple[list[str], float | None]:
    command = [
        "perf",
        "stat",
        "-x,",
        "-e",
        "task-clock,cycles",
        "taskset",
        "-c",
        cpu,
        binary,
        lane,
        operation,
        str(iterations),
        "0",
    ]
    completed = subprocess.run(
        command, env=runner().spawn_env(), capture_output=True, text=True
    )
    lines = completed.stderr.strip().splitlines()
    if completed.returncode != 0:
        return lines, None
    for line in lines:
        fields = line.split(",")
        if len(fields) >= 3 and fields[2] == "task-clock":
            try:
                return lines, float(fields[0])
            except ValueError:
                return lines, None
    return lines, None


def runner():
    """Load the runner beside this file, never through sys.path.

    Gate B must spawn the lanes with the same scrubbed environment the
    campaign used, so both scripts have to agree on one definition of it.
    """
    path = pathlib.Path(__file__).with_name("run_campaign.py")
    spec = importlib.util.spec_from_file_location("run_campaign", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load the runner at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gate_scale_windows(cells: dict[str, Any]) -> dict[str, Any]:
    """Gate C. Every median inside a first-principles window.

    A window miss is a regime failure, a wrong build, a wrong operation, or a
    host that is not running the code it reports.
    """
    checks: list[dict[str, Any]] = []
    failures: list[str] = []
    for operation, (low, generic_high) in SCALE_WINDOWS_NS.items():
        for lane in LANES:
            wall = cell_median(cells, operation, lane)
            if wall is None:
                continue
            high = LANE_CEILING_NS.get((operation, lane), generic_high)
            inside = low <= wall <= high
            checks.append(
                {
                    "operation": operation,
                    "lane": lane,
                    "median_ns": wall,
                    "window_ns": [low, high],
                    "inside": inside,
                }
            )
            if not inside:
                failures.append(
                    f"{operation}/{lane} {wall:.1f} ns is outside [{low}, {high}]"
                )
    return {
        "status": "fail" if failures else "pass",
        "checks": checks,
        "failures": failures,
    }


def gate_time_identities(cells: dict[str, Any]) -> dict[str, Any]:
    """Gate D. Relations that hold inside one lane whatever the host clock is.

    Every identity here is a ratio, so a slow or a fast box passes the same
    way. A lane that measured the wrong operation breaks one of them.
    """
    checks: list[dict[str, Any]] = []
    failures: list[str] = []

    def record(
        name: str, lane: str, holds: bool | None, detail: dict[str, Any]
    ) -> None:
        checks.append({"identity": name, "lane": lane, "holds": holds, **detail})
        if holds is False:
            failures.append(f"{name} fails for {lane}")

    # A measured quantity the campaign reports without judging it. It carries
    # its own reason, so a reader sees the row was measured and why no verdict
    # applies, instead of finding no entry at all.
    def report(name: str, lane: str, reason: str, detail: dict[str, Any]) -> None:
        checks.append(
            {
                "identity": name,
                "lane": lane,
                "holds": None,
                "reported_only": True,
                "reason": reason,
                **detail,
            }
        )

    for lane in LANES:
        miller = cell_median(cells, "miller_live", lane)
        final = cell_median(cells, "final_exponentiation", lane)
        pairing = cell_median(cells, "full_pairing", lane)
        if None not in (miller, final, pairing):
            expected = miller + final
            deviation = abs(pairing - expected) / expected
            record(
                "full_pairing ~ miller_live + final_exponentiation",
                lane,
                deviation <= PAIRING_SPLIT_TOLERANCE,
                {"deviation": deviation},
            )
        prepared = cell_median(cells, "miller_prepared", lane)
        if None not in (prepared, miller):
            record(
                "miller_prepared < miller_live",
                lane,
                prepared < miller,
                {"miller_prepared_ns": prepared, "miller_live_ns": miller},
            )
        pairing_prepared = cell_median(cells, "full_pairing_prepared", lane)
        if None not in (pairing_prepared, prepared, final):
            expected = prepared + final
            deviation = abs(pairing_prepared - expected) / expected
            record(
                "full_pairing_prepared ~ miller_prepared + final_exponentiation",
                lane,
                deviation <= PAIRING_SPLIT_TOLERANCE,
                {"deviation": deviation},
            )
        if None not in (pairing_prepared, pairing):
            record(
                "full_pairing_prepared < full_pairing",
                lane,
                pairing_prepared < pairing,
                {"prepared_ns": pairing_prepared, "live_ns": pairing},
            )
        single = cell_median(cells, "groth16_one_input", lane)
        for operation, (size, spread) in BATCH_ROWS.items():
            batch = cell_median(cells, operation, lane)
            if None in (single, batch):
                continue
            crossover = BATCH_CROSSOVER[spread].get(lane)
            name = f"{operation} < {size} * groth16_one_input"
            detail = {
                "batch_ns": batch,
                "single_ns": single,
                "batch_size": size,
                "batch_over_sequential": batch / (size * single),
                "crossover_size": crossover,
            }
            if crossover is not None and size >= crossover:
                record(name, lane, batch < size * single, detail)
            else:
                report(
                    name,
                    lane,
                    f"batch size {size} is below the measured {spread} crossover"
                    f" {crossover} for {lane}, where batching is not expected to win",
                    detail,
                )
        for size, same, multi in (
            (8, "groth16_batch8_one_key", "groth16_batch8_mixed_keys"),
            (3, "groth16_batch3_one_key", "groth16_batch3_mixed_keys"),
        ):
            same_key = cell_median(cells, same, lane)
            multi_key = cell_median(cells, multi, lane)
            if None in (same_key, multi_key):
                continue
            # A mixed batch cannot aggregate gamma, delta, and alpha-beta
            # across keys, so it pays three extra pairs per extra key.
            record(
                f"{multi} > {same}",
                lane,
                multi_key > same_key,
                {"multi_vk_ns": multi_key, "same_vk_ns": same_key, "batch_size": size},
            )
        mul = cell_median(cells, "fp_mul", lane)
        square = cell_median(cells, "fp_square", lane)
        if None not in (mul, square):
            record(
                "fp_square <= 1.1 * fp_mul",
                lane,
                square <= SQUARE_MUL_CEILING * mul,
                {"fp_square_ns": square, "fp_mul_ns": mul},
            )

    indistinct_ops = 0
    compared_ops = 0
    for operation in ARK_ASM_OPS:
        asm = cell_median(cells, operation, "arkworks")
        portable = cell_median(cells, operation, "arkworks-portable")
        if None in (asm, portable):
            continue
        compared_ops += 1
        ratio = portable / asm
        # ark-ff's asm feature is not uniformly faster (measured, fp2_mul
        # runs ~5% quicker portable on Granite Rapids). The identity exists
        # to catch a lane/binary swap, which shows as portable far below
        # asm, so only a large inversion fails.
        record(
            "arkworks-portable not far below arkworks",
            operation,
            ratio >= 0.90,
            {"portable_over_asm": ratio},
        )
        if ratio < 1.0 + ARK_ASM_INDISTINCT_BAND:
            indistinct_ops += 1
    return {
        "status": "fail" if failures else "pass",
        "checks": checks,
        "failures": failures,
        # Collapse the report columns only when every compared field op is
        # inside the 3% band.
        "ark_asm_indistinct": compared_ops > 0 and indistinct_ops == compared_ops,
    }


def gate_lane_reference(cells: dict[str, Any]) -> dict[str, Any]:
    """Gate E. One code path, timed in all four lanes.

    Gates C and D judge a lane against itself. Gate C is a per-cell absolute
    window several times wider than any real result, and every gate D identity
    is a ratio inside one lane, so multiplying a whole lane's column by one
    constant passes both. Nothing else in the file ties one lane's clock to
    another's.

    The `lane_reference` row is the tie. It runs the same Rust function in
    every lane, so its four medians are one quantity measured four times and
    they must agree. A constant applied to one lane's column, whether it came
    from a bad conversion, a spawn that ran at another frequency, or a summary
    edited after the fact, moves that lane's reference median with it and
    breaks the agreement.

    What it cannot catch. An error confined to one operation leaves the
    reference row alone. So does anyone who scales the reference row by the
    same constant; the gate raises the cost of a forgery, it does not make one
    impossible. It says nothing about whether a row computed the right value,
    which is what the cross-lane digests are for. And it holds only for a
    campaign that carries the row, so a summary without it is skipped, not
    passed.
    """
    medians = {
        lane: cell_median(cells, REFERENCE_OPERATION, lane) for lane in LANES
    }
    present = {lane: value for lane, value in medians.items() if value is not None}
    if len(present) < 2:
        return {
            "status": "skipped",
            "reason": f"campaign carries no {REFERENCE_OPERATION} row in two or more lanes",
        }
    # The centre is the median of the lanes, so one moved column is named as
    # the outlier instead of every column that stayed put.
    centre = statistics.median(present.values())
    failures = [
        f"{lane} reference median {value:.1f} ns deviates "
        f"{abs(value - centre) / centre:.1%} from the lane median {centre:.1f} ns"
        for lane, value in sorted(present.items())
        if abs(value - centre) / centre > REFERENCE_SPREAD_LIMIT
    ]
    return {
        "status": "fail" if failures else "pass",
        "tolerance": REFERENCE_SPREAD_LIMIT,
        "medians_ns": present,
        "lane_median_ns": centre,
        "spread": max(present.values()) / min(present.values()) - 1.0,
        "failures": failures,
    }


if __name__ == "__main__":
    sys.exit(main())
