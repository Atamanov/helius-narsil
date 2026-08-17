# Sealed four lane campaign, EPYC 9654, 2026-08-17

A 27 operation schedule of `bench/scripts/run_campaign.py`, four rounds, four
lanes per operation, 432 timed spawns. Superseded by
`../epyc9654-c89a65e-20260817/`, which runs the current harness.

| Item | Value |
| --- | --- |
| commit | `f73245f4d4f66191a39e8e06dea15be37e653619` |
| host | AMD EPYC 9654, Zen 4, container, Ubuntu 24.04, 48 vCPU |
| kernel | 5.15.0-179-generic, pinned to cpu 1, sibling list holds itself |
| build | `-C target-cpu=native`, backend `avx512-ifma` |
| mcl | `b70288469cd065b28a31764450f85238508c4d1e` |
| fixture seed | `0x06a13fbe4590bb3d` |
| `claim_eligible` | false, diagnostic figures from one shared box |

The box carries no git checkout, so `provenance.git_rev` reads `UNKNOWN`, and
every blob hash in `source-manifest.txt` equals `git ls-tree -r f73245f`. The
campaign ran with `--diagnostic-host`, so `claim_eligible` is false whatever
the gates say. `metadata.json` records the two tolerated violations, no
cpufreq governor on cpu 1 and a kernel booted without `mitigations=off`.

| Gate | Status | Reading |
| --- | --- | --- |
| A, cross harness | skipped | no Criterion run, see the addendum |
| B, cycle cross check | skipped | no `perf` for kernel 5.15.0-179-generic |
| C, scale windows | pass | 108 of 108 medians inside their window |
| D, time identities | pass | 44 hold, 3 reported, size 3 mixed key batch |
| E, lane reference | absent | the row postdates this campaign |

The three reported identities sit below their own batching crossover, where a
batch is not expected to beat the sequential loop. All four lanes report the
session seed, one `fixture_set_sha256` per operation, and one digest across
all 104 rotations of the 26 shared rows, with `g2_prepare_87_lines` the
lane-local exception. `lane-agreement.txt` replays those checks.
