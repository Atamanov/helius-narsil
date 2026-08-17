#!/usr/bin/env python3
"""Tests for the campaign runner and the validity gates.

Everything here runs on a laptop with no lane binary. Stand-in probes speak
the frozen `four_lane LANE OPERATION ITERATIONS ROTATION` protocol, so the
tests exercise the runner's own rules and not the harness under it.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


def load(name: str):
    path = pathlib.Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


RUNNER = load("run_campaign")
CROSS = load("cross_check")

PROBE = """#!/usr/bin/env python3
import hashlib, json, os, sys
lane, operation, iterations, rotation = sys.argv[1:5]
knobs = json.load(open("__KNOBS__"))
def h(text):
    return hashlib.sha256(text.encode()).hexdigest()
prov = {
    "schema": knobs.get("schema", "four-lane-provenance-v1"),
    "rustflags": "-C target-cpu=native",
    "target_cpu": "native",
    "ark_asm": bool(knobs["ark_asm"]),
    "mcl_archive_sha256": h(
        "mcl-archive-" + (operation if knobs.get("mcl_hash_per_op") else "fixed")
    ),
    "mcl_manifest_sha256": h("mcl-manifest"),
    "cxx": "clang 18",
    "cxx_native_flag": "-march=native",
    "helius_backend": knobs["helius_backend"],
    "mcl_runtime": knobs.get("mcl_runtime", "xbyak-jit"),
    "git_rev": "deadbeef",
    "timer_overhead_ns_per_iter": 0.5,
    "tsc_agreement_pct": 99.9,
    "fixture_seed": os.environ.get("FOUR_LANE_FIXTURE_SEED", "0x0000000000000000"),
    "fixture_set_sha256": h(
        "fixtures-" + operation
        + ("-odd" if knobs.get("fixture_odd_lane") == lane else "")
    ),
}
if knobs.get("seed_odd_lane") == lane:
    prov["fixture_seed"] = "0x" + "1" * 16
for field in knobs.get("drop_fields", []):
    prov.pop(field)
print("provenance|" + json.dumps(prov))
if knobs.get("exit3_operation") == operation:
    sys.exit(3)
if [lane, operation] in knobs.get("unsupported", []):
    print(f"unsupported|{operation}|{lane}|no fixture for this lane")
    sys.exit(6)
digest = h(operation + "|" + rotation)[:16]
if knobs.get("lane_local_per_lane") and operation == "g2_prepare_87_lines":
    digest = h(operation + "|" + rotation + "|" + lane)[:16]
if knobs.get("bad_digest_lane") == lane:
    digest = "f" * 16
lanes = ["helius", "arkworks", "arkworks-portable", "mcl"]
per_call = 100 + 10 * lanes.index(lane)
print(f"result|{operation}|{lane}|{iterations}|{per_call * int(iterations)}|{digest}")
"""

NATIVE_DEFAULTS = {"ark_asm": True, "helius_backend": "avx512-ifma"}
PORTABLE_DEFAULTS = {"ark_asm": False, "helius_backend": "portable"}


def sha256_of(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_probe(root: pathlib.Path, name: str, knobs: dict) -> pathlib.Path:
    knobs_path = root / f"{name}.knobs.json"
    knobs_path.write_text(json.dumps(knobs))
    path = root / name
    path.write_text(PROBE.replace("__KNOBS__", str(knobs_path)))
    path.chmod(0o755)
    return path


def make_lanes(
    root: pathlib.Path,
    native_knobs: dict | None = None,
    portable_knobs: dict | None = None,
) -> pathlib.Path:
    native = make_probe(root, "native.py", {**NATIVE_DEFAULTS, **(native_knobs or {})})
    portable = make_probe(
        root, "portable.py", {**PORTABLE_DEFAULTS, **(portable_knobs or {})}
    )
    document = {
        "helius": {
            "binary": str(native),
            "sha256": sha256_of(native),
            "expected_backend": "avx512-ifma",
        },
        "arkworks": {"binary": str(native), "sha256": sha256_of(native)},
        "mcl": {"binary": str(native), "sha256": sha256_of(native)},
        "arkworks-portable": {
            "binary": str(portable),
            "sha256": sha256_of(portable),
            "expected_backend": None,
        },
    }
    path = root / "lanes.json"
    path.write_text(json.dumps(document))
    return path


def runner_argv(root: pathlib.Path, lanes: pathlib.Path, out: str = "run") -> list[str]:
    return [
        "--lanes",
        str(lanes),
        "--out",
        str(root / out),
        "--cpu",
        "none",
        "--rounds",
        "1",
        "--quick",
        "--allow-non-linux",
    ]


class LanesValidationTest(unittest.TestCase):
    def test_hash_mismatch_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            lanes = make_lanes(root)
            document = json.loads(lanes.read_text())
            document["mcl"]["sha256"] = "0" * 64
            lanes.write_text(json.dumps(document))
            with self.assertRaisesRegex(SystemExit, "hash mismatch"):
                RUNNER.main(runner_argv(root, lanes))

    def test_native_lanes_must_share_one_binary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            lanes = make_lanes(root)
            other = make_probe(root, "other.py", {**NATIVE_DEFAULTS, "extra": 1})
            document = json.loads(lanes.read_text())
            document["mcl"] = {"binary": str(other), "sha256": sha256_of(other)}
            lanes.write_text(json.dumps(document))
            with self.assertRaisesRegex(SystemExit, "share one binary"):
                RUNNER.main(runner_argv(root, lanes))

    def test_portable_lane_must_use_its_own_binary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            lanes = make_lanes(root)
            document = json.loads(lanes.read_text())
            document["arkworks-portable"] = dict(document["helius"])
            document["arkworks-portable"]["expected_backend"] = None
            lanes.write_text(json.dumps(document))
            with self.assertRaisesRegex(SystemExit, "different binary"):
                RUNNER.main(runner_argv(root, lanes))

    def test_claim_rounds_must_be_a_multiple_of_four(self) -> None:
        with self.assertRaisesRegex(SystemExit, "multiple of four"):
            RUNNER.main(
                ["--lanes", "x", "--out", "y", "--rounds", "3", "--allow-non-linux"]
            )

    def test_helius_lane_must_pin_expected_backend(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            lanes = make_lanes(root)
            document = json.loads(lanes.read_text())
            del document["helius"]["expected_backend"]
            lanes.write_text(json.dumps(document))
            with self.assertRaisesRegex(SystemExit, "helius must pin expected_backend"):
                RUNNER.main(runner_argv(root, lanes))


class LanesManifestTest(unittest.TestCase):
    """build_lanes.sh emits this document, so it must satisfy the runner."""

    def test_written_manifest_loads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            asm = make_probe(root, "asm.py", NATIVE_DEFAULTS)
            portable = make_probe(root, "portable.py", PORTABLE_DEFAULTS)
            out = root / "lanes.json"
            subprocess.run(
                [
                    sys.executable,
                    str(pathlib.Path(__file__).with_name("write_lanes.py")),
                    "--asm-binary",
                    str(asm),
                    "--portable-binary",
                    str(portable),
                    "--expected-backend",
                    "avx512-ifma",
                    "--out",
                    str(out),
                ],
                check=True,
                capture_output=True,
            )
            document = RUNNER.load_lanes(out)
            self.assertEqual(set(document), set(RUNNER.LANES))
            self.assertEqual(document["helius"]["expected_backend"], "avx512-ifma")
            self.assertIsNone(document["mcl"]["expected_backend"])


class ScheduleTest(unittest.TestCase):
    def test_every_lane_visits_every_position_across_four_rounds(self) -> None:
        for operation_index in (0, 1, 5, 21):
            for position in range(4):
                lanes_seen = {
                    RUNNER.lane_order(operation_index, round_index)[position]
                    for round_index in range(4)
                }
                self.assertEqual(lanes_seen, set(RUNNER.LANES))

    def test_quick_iterations_keep_fixture_pool_coverage(self) -> None:
        self.assertEqual(RUNNER.QUICK_ITERATIONS["miller_live"], 16)
        # Every pooled Groth16 row keeps its full count, because its pool
        # holds more vectors than one invocation draws.
        for operation in RUNNER.WINDOWED_OPERATIONS:
            self.assertEqual(
                RUNNER.QUICK_ITERATIONS[operation], RUNNER.ITERATIONS[operation]
            )
        self.assertEqual(RUNNER.QUICK_ITERATIONS["groth16_one_input"], 8)
        self.assertEqual(RUNNER.QUICK_ITERATIONS["groth16_batch8_one_key"], 4)
        self.assertEqual(RUNNER.QUICK_ITERATIONS["fp_mul"], 1024)
        self.assertEqual(set(RUNNER.QUICK_ITERATIONS), set(RUNNER.ITERATIONS))


class FixtureSeedTest(unittest.TestCase):
    def test_seed_is_normalized_to_the_form_the_binary_reports(self) -> None:
        self.assertEqual(RUNNER.normalize_seed("5"), "0x0000000000000005")
        self.assertEqual(RUNNER.normalize_seed("0xFF"), "0x00000000000000ff")
        self.assertRegex(RUNNER.normalize_seed(None), r"\A0x[0-9a-f]{16}\Z")
        with self.assertRaisesRegex(SystemExit, "not a u64"):
            RUNNER.normalize_seed("cafe")
        with self.assertRaisesRegex(SystemExit, "64 bits"):
            RUNNER.normalize_seed(str(2**64))

    def test_one_seed_reaches_every_lane_and_moves_between_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            lanes = make_lanes(root)
            RUNNER.main(runner_argv(root, lanes, out="run-a"))
            RUNNER.main(runner_argv(root, lanes, out="run-b"))
            first = json.loads((root / "run-a" / "summary.json").read_text())
            second = json.loads((root / "run-b" / "summary.json").read_text())
            self.assertNotEqual(first["fixture_seed"], second["fixture_seed"])
            seeds = {
                json.loads(line)["provenance"]["fixture_seed"]
                for line in (root / "run-a" / "raw.jsonl").read_text().splitlines()
            }
            self.assertEqual(seeds, {first["fixture_seed"]})

    def test_seed_drift_between_lanes_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            lanes = make_lanes(root, native_knobs={"seed_odd_lane": "mcl"})
            with self.assertRaisesRegex(SystemExit, "fixture_seed differs"):
                RUNNER.main(runner_argv(root, lanes))


class CampaignFatalTest(unittest.TestCase):
    def test_missing_provenance_field_is_fatal_in_quick_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            lanes = make_lanes(
                root, native_knobs={"drop_fields": ["tsc_agreement_pct"]}
            )
            with self.assertRaisesRegex(SystemExit, "missing.*tsc_agreement_pct"):
                RUNNER.main(runner_argv(root, lanes))

    def test_provenance_schema_mismatch_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            lanes = make_lanes(root, native_knobs={"schema": "other-v9"})
            with self.assertRaisesRegex(SystemExit, "schema"):
                RUNNER.main(runner_argv(root, lanes))

    def test_mcl_hash_drift_between_invocations_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            lanes = make_lanes(
                root,
                native_knobs={"mcl_hash_per_op": True},
                portable_knobs={"mcl_hash_per_op": True},
            )
            with self.assertRaisesRegex(
                SystemExit, "mcl_archive_sha256 changed during the campaign"
            ):
                RUNNER.main(runner_argv(root, lanes))

    def test_digest_mismatch_across_lanes_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            lanes = make_lanes(root, native_knobs={"bad_digest_lane": "arkworks"})
            with self.assertRaisesRegex(SystemExit, "digest mismatch"):
                RUNNER.main(runner_argv(root, lanes))

    def test_lane_local_digest_ops_tolerate_per_lane_digests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            knobs = {"lane_local_per_lane": True}
            lanes = make_lanes(root, native_knobs=knobs, portable_knobs=knobs)
            RUNNER.main(runner_argv(root, lanes))
            summary = json.loads((root / "run" / "summary.json").read_text())
            self.assertIn("g2_prepare_87_lines", summary["cells"])

    def test_missing_mcl_runtime_field_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            lanes = make_lanes(root, native_knobs={"drop_fields": ["mcl_runtime"]})
            with self.assertRaisesRegex(SystemExit, "missing.*mcl_runtime"):
                RUNNER.main(runner_argv(root, lanes))

    def test_fixture_set_drift_across_lanes_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            lanes = make_lanes(root, native_knobs={"fixture_odd_lane": "mcl"})
            with self.assertRaisesRegex(SystemExit, "fixture_set_sha256 differs"):
                RUNNER.main(runner_argv(root, lanes))

    def test_portable_lane_with_asm_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            lanes = make_lanes(root, portable_knobs={"ark_asm": True})
            with self.assertRaisesRegex(SystemExit, "arkworks-portable.*asm"):
                RUNNER.main(runner_argv(root, lanes))

    def test_wrong_helius_backend_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            lanes = make_lanes(root, native_knobs={"helius_backend": "scalar"})
            with self.assertRaisesRegex(SystemExit, "helius_backend"):
                RUNNER.main(runner_argv(root, lanes))

    def test_exit_three_propagates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            lanes = make_lanes(
                root,
                native_knobs={"exit3_operation": "fp_mul"},
                portable_knobs={"exit3_operation": "fp_mul"},
            )
            with self.assertRaisesRegex(SystemExit, "build-regime gate failure"):
                RUNNER.main(runner_argv(root, lanes))

    def test_output_directory_is_create_new(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            lanes = make_lanes(root)
            (root / "run").mkdir()
            with self.assertRaisesRegex(SystemExit, "refusing to overwrite"):
                RUNNER.main(runner_argv(root, lanes))


class CampaignRunTest(unittest.TestCase):
    def test_quick_run_records_unsupported_as_na_and_summarizes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            unsupported = [["mcl", "groth16_committed_rails"]]
            lanes = make_lanes(root, native_knobs={"unsupported": unsupported})
            RUNNER.main(runner_argv(root, lanes))
            out = root / "run"
            rows = [
                json.loads(line)
                for line in (out / "raw.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(rows), len(RUNNER.ITERATIONS) * 4)
            positions = {
                (row["operation"], row["lane"]): row["order"] for row in rows
            }
            self.assertEqual(len(positions), len(RUNNER.ITERATIONS) * 4)
            summary = json.loads((out / "summary.json").read_text())
            self.assertFalse(summary["claim_eligible"])
            cell = summary["cells"]["groth16_committed_rails"]["mcl"]
            self.assertEqual(cell, {"unsupported": "no fixture for this lane"})
            helius = summary["cells"]["fp_mul"]["helius"]
            self.assertEqual(helius["median_ns_per_call"], 100.0)
            self.assertEqual(helius["samples"], 1)
            self.assertIn("native", summary["provenance"])
            self.assertIn("portable", summary["provenance"])
            metadata = json.loads((out / "metadata.json").read_text())
            self.assertTrue(metadata["quick"])
            self.assertEqual(metadata["iterations"], RUNNER.QUICK_ITERATIONS)

    def test_summary_median_min_and_iqr(self) -> None:
        rows = [
            {
                "operation": "fp_mul",
                "lane": "helius",
                "ns_per_call": value,
                "digest": "0" * 16,
            }
            for value in (10.0, 30.0, 20.0, 40.0)
        ]
        for lane in ("arkworks", "arkworks-portable", "mcl"):
            rows.append(
                {
                    "operation": "fp_mul",
                    "lane": lane,
                    "ns_per_call": 50.0,
                    "digest": "0" * 16,
                }
            )
        args = argparse.Namespace(quick=False, rounds=4)
        summary = RUNNER.summarize({"fp_mul": 16}, rows, {}, args, {})
        cell = summary["cells"]["fp_mul"]["helius"]
        self.assertEqual(cell["median_ns_per_call"], 25.0)
        self.assertEqual(cell["min_ns_per_call"], 10.0)
        self.assertEqual(cell["iqr_ns_per_call"], 15.0)
        self.assertEqual(cell["samples"], 4)
        self.assertEqual(RUNNER.interquartile_range([7.0]), 0.0)


class HostGateTest(unittest.TestCase):
    def patched(self, **overrides):
        values = {
            "governor": "performance",
            "siblings": "1,65",
            "busy": 0.1,
            "mdstat": "md0 : active raid1 sda1[0] sdb1[1]\n",
            "compilers": [],
        }
        values.update(overrides)

        def read_sys(path: str) -> str:
            if path.endswith("scaling_governor"):
                return values["governor"]
            if path.endswith("thread_siblings_list"):
                return values["siblings"]
            raise AssertionError(path)

        def read_optional(path: str):
            if path.endswith("scaling_governor"):
                return values["governor"]
            if path.endswith("mdstat"):
                return values["mdstat"]
            return None

        return (
            mock.patch.object(RUNNER, "read_sys", side_effect=read_sys),
            mock.patch.object(RUNNER, "cpu_busy_pct", return_value=values["busy"]),
            mock.patch.object(RUNNER, "read_optional", side_effect=read_optional),
            mock.patch.object(
                RUNNER, "live_compilers", return_value=values["compilers"]
            ),
        )

    def check(self, strict: bool = True, **overrides):
        patches = self.patched(**overrides)
        with patches[0], patches[1], patches[2], patches[3]:
            return RUNNER.check_host("1", strict)

    def test_clean_host_passes_and_reports_sibling(self) -> None:
        state = self.check()
        self.assertEqual(state["sibling"], 65)
        self.assertEqual(state["governor"], "performance")

    def test_governor_must_be_performance(self) -> None:
        with self.assertRaisesRegex(SystemExit, "governor"):
            self.check(governor="schedutil")

    def test_busy_sibling_is_fatal(self) -> None:
        with self.assertRaisesRegex(SystemExit, "busy"):
            self.check(busy=5.0)

    def test_md_resync_is_fatal(self) -> None:
        with self.assertRaisesRegex(SystemExit, "resync"):
            self.check(mdstat="md0 : active raid1\n [=>..] resync = 8.2%\n")

    def test_live_compiler_is_fatal(self) -> None:
        with self.assertRaisesRegex(SystemExit, "compiler"):
            self.check(compilers=["rustc"])

    def test_diagnostic_host_records_the_violation_instead_of_aborting(self) -> None:
        RUNNER.HOST_GATE_WARNINGS.clear()
        state = self.check(strict=False, governor="powersave", busy=5.0)
        self.assertEqual(state["governor"], "powersave")
        self.assertTrue(
            any("governor" in warning for warning in RUNNER.HOST_GATE_WARNINGS)
        )
        RUNNER.HOST_GATE_WARNINGS.clear()

    def test_cpu_list_parsing(self) -> None:
        self.assertEqual(RUNNER.parse_cpu_list("1,65"), [1, 65])
        self.assertEqual(RUNNER.parse_cpu_list("0-3"), [0, 1, 2, 3])


class MitigationsGateTest(unittest.TestCase):
    def check(self, cmdline: str) -> dict:
        def read_sys(path: str) -> str:
            self.assertEqual(path, "/proc/cmdline")
            return cmdline

        with (
            mock.patch.object(RUNNER, "read_sys", side_effect=read_sys),
            mock.patch.object(RUNNER, "read_optional", return_value="Vulnerable"),
        ):
            return RUNNER.check_mitigations()

    def test_mitigations_off_passes_and_records_provenance(self) -> None:
        state = self.check("quiet mitigations=off nokaslr")
        self.assertIn("mitigations=off", state["cmdline"])
        self.assertEqual(state["spec_rstack_overflow"], "Vulnerable")
        self.assertEqual(state["spectre_v2"], "Vulnerable")

    def test_default_mitigations_are_fatal(self) -> None:
        with self.assertRaisesRegex(SystemExit, "mitigations=off"):
            self.check("quiet nokaslr")
        # A substring of another token must not satisfy the gate.
        with self.assertRaisesRegex(SystemExit, "mitigations=off"):
            self.check("quiet mitigations=offload")


def cell(value: float) -> dict:
    return {
        "median_ns_per_call": value,
        "min_ns_per_call": value,
        "iqr_ns_per_call": 0.0,
        "samples": 4,
    }


def healthy_cells() -> dict:
    cells: dict = {}
    per_lane = {
        "helius": 1.0,
        "arkworks": 1.6,
        "arkworks-portable": 2.4,
        "mcl": 1.7,
    }
    base = {
        "fp_mul": 8.0,
        "fp_square": 7.5,
        "fp2_mul": 30.0,
        "fp12_mul": 500.0,
        "miller_live": 100_000.0,
        "miller_prepared": 50_000.0,
        "final_exponentiation": 80_000.0,
        "full_pairing": 180_000.0,
        "groth16_one_input": 300_000.0,
        "groth16_batch8_one_key": 1_500_000.0,
        "groth16_batch8_mixed_keys": 1_800_000.0,
    }
    for operation, value in base.items():
        cells[operation] = {
            lane: cell(value * scale) for lane, scale in per_lane.items()
        }
    # One code path, so every lane reports the same time for it.
    cells[CROSS.REFERENCE_OPERATION] = {lane: cell(2_000.0) for lane in per_lane}
    return cells


def scale_lane(cells: dict, lane: str, factor: float) -> dict:
    """Multiply every cell of one lane, the shape of a fabricated column."""
    scaled = {
        operation: {name: dict(value) for name, value in lanes.items()}
        for operation, lanes in cells.items()
    }
    for lanes in scaled.values():
        if lane not in lanes:
            continue
        for key in ("median_ns_per_call", "min_ns_per_call"):
            lanes[lane][key] *= factor
    return scaled


class LaneReferenceGateTest(unittest.TestCase):
    """Gate E against the fabrication gates C and D cannot see.

    An audit showed that dividing every helius median by 1.60, a uniform 1.60x
    claim over the whole column, passes gate C and gate D. The gate C windows
    are wide enough to hold it and every gate D identity is a ratio inside one
    lane, which the scaling leaves untouched.
    """

    FABRICATION = 1.0 / 1.60

    def test_uniform_lane_fabrication_passes_gates_c_and_d(self) -> None:
        forged = scale_lane(healthy_cells(), "helius", self.FABRICATION)
        self.assertEqual(CROSS.gate_scale_windows(forged)["status"], "pass")
        self.assertEqual(CROSS.gate_time_identities(forged)["status"], "pass")

    def test_gate_e_catches_the_same_fabrication(self) -> None:
        forged = scale_lane(healthy_cells(), "helius", self.FABRICATION)
        gate = CROSS.gate_lane_reference(forged)
        self.assertEqual(gate["status"], "fail")
        self.assertTrue(gate["failures"])

    def test_gate_e_passes_an_honest_campaign(self) -> None:
        self.assertEqual(
            CROSS.gate_lane_reference(healthy_cells())["status"], "pass"
        )

    def test_gate_e_tolerates_drift_below_the_band(self) -> None:
        drifted = scale_lane(
            healthy_cells(), "mcl", 1.0 + CROSS.REFERENCE_SPREAD_LIMIT / 2
        )
        self.assertEqual(CROSS.gate_lane_reference(drifted)["status"], "pass")

    def test_gate_e_skips_a_campaign_without_the_row(self) -> None:
        cells = healthy_cells()
        del cells[CROSS.REFERENCE_OPERATION]
        self.assertEqual(CROSS.gate_lane_reference(cells)["status"], "skipped")

    def test_the_reference_row_is_in_the_campaign_schedule(self) -> None:
        self.assertIn(CROSS.REFERENCE_OPERATION, RUNNER.ITERATIONS)


class CrossCheckTest(unittest.TestCase):
    def test_anchor_warmups_cancel_in_perf_differencing(self) -> None:
        # Gate B subtracts task-clock at N from 2N. The binary's warmup count
        # is (iters/32).clamp(4,128), so the subtraction cancels warmup only
        # while every anchor keeps the same clamp value at N and 2N.
        def warmups(iterations: int) -> int:
            return min(max(iterations // 32, 4), 128)

        for operation in CROSS.ANCHOR_GROUPS.values():
            n = RUNNER.ITERATIONS[operation]
            self.assertEqual(warmups(n), warmups(2 * n), operation)

    def test_scale_windows_pass_and_fail(self) -> None:
        cells = healthy_cells()
        self.assertEqual(CROSS.gate_scale_windows(cells)["status"], "pass")
        cells["fp_mul"]["helius"] = cell(1.0)
        failing = CROSS.gate_scale_windows(cells)
        self.assertEqual(failing["status"], "fail")
        self.assertIn("fp_mul/helius", failing["failures"][0])

    def test_scale_windows_cover_every_operation(self) -> None:
        self.assertEqual(set(CROSS.SCALE_WINDOWS_NS), set(RUNNER.ITERATIONS))
        for operation, (low, high) in CROSS.SCALE_WINDOWS_NS.items():
            self.assertLess(low, high, operation)
            self.assertGreater(low, 0.0, operation)

    def test_comparator_ceiling_replaces_the_generic_window_high(self) -> None:
        # A comparator lane may legitimately run past the first-principles
        # high, so its ceiling replaces that high and bounds it instead.
        for (operation, lane), ceiling in CROSS.LANE_CEILING_NS.items():
            self.assertNotEqual(lane, "helius", operation)
            self.assertGreater(ceiling, CROSS.SCALE_WINDOWS_NS[operation][1])

        cells = healthy_cells()
        cells["miller_live"]["mcl"] = cell(280_000.0)
        self.assertEqual(CROSS.gate_scale_windows(cells)["status"], "pass")

        cells["miller_live"]["mcl"] = cell(320_000.0)
        gate = CROSS.gate_scale_windows(cells)
        self.assertEqual(gate["status"], "fail")
        self.assertIn("miller_live/mcl", gate["failures"][0])

        # The lane without a ceiling keeps the generic high.
        cells = healthy_cells()
        cells["miller_live"]["helius"] = cell(280_000.0)
        self.assertEqual(CROSS.gate_scale_windows(cells)["status"], "fail")

    def write_perf_run(self, root: pathlib.Path) -> pathlib.Path:
        summary_path = root / "summary.json"
        summary_path.write_text("{}")
        (root / "metadata.json").write_text(
            json.dumps(
                {
                    "iterations": RUNNER.ITERATIONS,
                    "cpu": "1",
                    "lanes": {
                        lane: {"binary": f"/bin/{lane}"} for lane in RUNNER.LANES
                    },
                }
            )
        )
        return summary_path

    def test_gate_perf_differences_n_and_2n_invocations(self) -> None:
        cells = healthy_cells()
        with tempfile.TemporaryDirectory() as directory:
            summary_path = self.write_perf_run(pathlib.Path(directory))

            def fake_perf(binary, lane, operation, iterations, cpu):
                per_call_ms = cells[operation][lane]["median_ns_per_call"] / 1e6
                # 400 ms of setup cancels in the N/2N difference.
                return (["raw"], 400.0 + iterations * per_call_ms)

            with mock.patch.object(CROSS, "perf_available", return_value=True):
                with mock.patch.object(CROSS, "perf_stat", side_effect=fake_perf):
                    gate = CROSS.gate_perf(cells, summary_path, True)
            self.assertEqual(gate["status"], "pass")
            anchors = set(CROSS.ANCHOR_GROUPS.values())
            self.assertEqual({check["operation"] for check in gate["checks"]}, anchors)
            for check in gate["checks"]:
                self.assertAlmostEqual(
                    check["task_clock_ns_per_call"],
                    check["campaign_median_ns"],
                    delta=check["campaign_median_ns"] * 1e-6,
                )
                self.assertIn("perf_raw_n", check)
                self.assertIn("perf_raw_2n", check)

    def test_gate_perf_fails_on_deviation_and_non_positive_delta(self) -> None:
        cells = healthy_cells()
        with tempfile.TemporaryDirectory() as directory:
            summary_path = self.write_perf_run(pathlib.Path(directory))

            def biased_perf(binary, lane, operation, iterations, cpu):
                per_call_ms = cells[operation][lane]["median_ns_per_call"] / 1e6
                return (["raw"], 400.0 + iterations * per_call_ms * 1.2)

            with mock.patch.object(CROSS, "perf_available", return_value=True):
                with mock.patch.object(CROSS, "perf_stat", side_effect=biased_perf):
                    gate = CROSS.gate_perf(cells, summary_path, True)
            self.assertEqual(gate["status"], "fail")
            self.assertIn("deviates", gate["failures"][0])

            def flat_perf(binary, lane, operation, iterations, cpu):
                return (["raw"], 400.0)

            with mock.patch.object(CROSS, "perf_available", return_value=True):
                with mock.patch.object(CROSS, "perf_stat", side_effect=flat_perf):
                    gate = CROSS.gate_perf(cells, summary_path, True)
            self.assertEqual(gate["status"], "fail")
            self.assertIn("non-positive N/2N task-clock delta", gate["failures"][0])

    def test_gate_perf_unavailable_skips_only_diagnostic_hosts(self) -> None:
        cells = healthy_cells()
        with tempfile.TemporaryDirectory() as directory:
            summary_path = self.write_perf_run(pathlib.Path(directory))
            metadata_path = summary_path.parent / "metadata.json"
            metadata = json.loads(metadata_path.read_text())
            with mock.patch.object(CROSS, "perf_available", return_value=False):
                gate = CROSS.gate_perf(cells, summary_path, True)
                self.assertEqual(gate["status"], "fail")
                metadata["diagnostic_host"] = True
                metadata_path.write_text(json.dumps(metadata))
                gate = CROSS.gate_perf(cells, summary_path, True)
                self.assertEqual(gate["status"], "skipped")

    def test_time_identities_pass_and_fail(self) -> None:
        cells = healthy_cells()
        passing = CROSS.gate_time_identities(cells)
        self.assertEqual(passing["status"], "pass")
        self.assertFalse(passing["ark_asm_indistinct"])

        broken = healthy_cells()
        broken["full_pairing"]["helius"] = cell(400_000.0)
        self.assertEqual(CROSS.gate_time_identities(broken)["status"], "fail")

        broken = healthy_cells()
        broken["fp_square"]["mcl"] = cell(
            broken["fp_mul"]["mcl"]["median_ns_per_call"] * 1.5
        )
        self.assertEqual(CROSS.gate_time_identities(broken)["status"], "fail")

        broken = healthy_cells()
        broken["groth16_batch8_one_key"]["mcl"] = cell(
            broken["groth16_one_input"]["mcl"]["median_ns_per_call"] * 8.5
        )
        self.assertEqual(CROSS.gate_time_identities(broken)["status"], "fail")

        broken = healthy_cells()
        broken["groth16_batch8_mixed_keys"]["mcl"] = cell(
            broken["groth16_batch8_one_key"]["mcl"]["median_ns_per_call"] * 0.9
        )
        self.assertEqual(CROSS.gate_time_identities(broken)["status"], "fail")

    def test_portable_faster_than_asm_fails(self) -> None:
        cells = healthy_cells()
        for operation in CROSS.ARK_ASM_OPS:
            cells[operation]["arkworks-portable"] = cell(
                cells[operation]["arkworks"]["median_ns_per_call"] * 0.8
            )
        self.assertEqual(CROSS.gate_time_identities(cells)["status"], "fail")

    def test_ark_asm_indistinct_flag(self) -> None:
        cells = healthy_cells()
        for operation in CROSS.ARK_ASM_OPS:
            cells[operation]["arkworks-portable"] = cell(
                cells[operation]["arkworks"]["median_ns_per_call"] * 1.01
            )
        cells["fp_square"]["arkworks-portable"] = cell(
            cells["fp_mul"]["arkworks-portable"]["median_ns_per_call"] * 0.95
        )
        gate = CROSS.gate_time_identities(cells)
        self.assertEqual(gate["status"], "pass")
        self.assertTrue(gate["ark_asm_indistinct"])

    def test_cross_harness_gate_compares_criterion_medians(self) -> None:
        cells = healthy_cells()
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            estimates = root / "bn254_single_pair_miller_loop/helius/1/new"
            estimates.mkdir(parents=True)
            (estimates / "estimates.json").write_text(
                json.dumps({"median": {"point_estimate": 104_000.0}})
            )
            gate = CROSS.gate_cross_harness(cells, str(root))
            self.assertEqual(gate["status"], "pass")
            (estimates / "estimates.json").write_text(
                json.dumps({"median": {"point_estimate": 150_000.0}})
            )
            self.assertEqual(
                CROSS.gate_cross_harness(cells, str(root))["status"], "fail"
            )
            self.assertEqual(
                CROSS.gate_cross_harness(cells, str(root / "empty"))["status"], "fail"
            )

    def test_main_flips_claim_eligible_only_on_all_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            summary_path = root / "summary.json"
            summary_path.write_text(
                json.dumps({"claim_eligible": False, "cells": healthy_cells()})
            )
            code = CROSS.main(
                ["--summary", str(summary_path), "--out", str(root / "validity.json")]
            )
            self.assertEqual(code, 0)
            validity = json.loads((root / "validity.json").read_text())
            self.assertTrue(validity["claim_eligible"])
            self.assertTrue(json.loads(summary_path.read_text())["claim_eligible"])

            cells = healthy_cells()
            cells["fp_mul"]["helius"] = cell(1.0)
            summary_path.write_text(
                json.dumps({"claim_eligible": False, "cells": cells})
            )
            code = CROSS.main(
                ["--summary", str(summary_path), "--out", str(root / "validity2.json")]
            )
            self.assertEqual(code, 1)
            self.assertFalse(
                json.loads((root / "validity2.json").read_text())["claim_eligible"]
            )
            self.assertFalse(json.loads(summary_path.read_text())["claim_eligible"])

    def test_diagnostic_host_can_never_be_claim_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            summary_path = root / "summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "claim_eligible": False,
                        "diagnostic_host": True,
                        "cells": healthy_cells(),
                    }
                )
            )
            code = CROSS.main(
                ["--summary", str(summary_path), "--out", str(root / "validity.json")]
            )
            self.assertEqual(code, 1)
            validity = json.loads((root / "validity.json").read_text())
            self.assertFalse(validity["claim_eligible"])
            self.assertEqual(validity["gates"]["C_scale_windows"]["status"], "pass")
            self.assertFalse(json.loads(summary_path.read_text())["claim_eligible"])


if __name__ == "__main__":
    unittest.main()
