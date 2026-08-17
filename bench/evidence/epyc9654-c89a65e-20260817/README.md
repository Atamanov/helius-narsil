# Sealed four lane campaign, EPYC 9654, 2026-08-17

The full 28 operation schedule of `bench/scripts/run_campaign.py`, four
rounds, four lanes per operation, 448 timed spawns. It supersedes
`../epyc9654-f73245f-20260817/`.

| Item | Value |
| --- | --- |
| commit | `c89a65e684f10adf2025afde0677a30f110e3645` |
| host | AMD EPYC 9654, Zen 4, container, Ubuntu 24.04, 48 vCPU |
| kernel | 5.15.0-179-generic, pinned to cpu 1, sibling list holds itself |
| build | `-C target-cpu=native`, backend `avx512-ifma` |
| mcl | `b70288469cd065b28a31764450f85238508c4d1e` |
| fixture seed | `0x664c86bb6b618691` |
| `claim_eligible` | false, diagnostic figures from one shared box |

The box carries no git checkout, so `provenance.git_rev` reads `UNKNOWN`.
Every blob hash in `source-manifest.txt` equals `git ls-tree -r c89a65e`.

The campaign ran with `--diagnostic-host`, so `claim_eligible` is false
whatever the gates say. `metadata.json` records the two tolerated violations,
no cpufreq governor on cpu 1 and a kernel booted without `mitigations=off`.

| Gate | Status | Reading |
| --- | --- | --- |
| A, cross harness | pass | 15 anchors, largest deviation 1.13 percent |
| B, cycle cross check | skipped | no `perf` for kernel 5.15.0-179-generic |
| C, scale windows | pass | 112 of 112 medians inside their window |
| D, time identities | pass | 44 hold, 3 reported, size 3 mixed key batch |
| E, lane reference | pass | spread 0.179 percent, tolerance 5 percent |

The three reported identities sit below their own batching crossover, where a
batch is not expected to beat the sequential loop. All four lanes report the
session seed, one `fixture_set_sha256` per operation, and one digest across
all 108 rotations of the 27 shared rows, with `g2_prepare_87_lines` the
lane-local exception. `lane-agreement.txt` replays those checks from
`raw.jsonl` alone and `rotation-gate.txt` is the rotation gate's live negative
control.
