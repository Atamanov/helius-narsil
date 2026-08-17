# The four lane BN254 benchmark

Times four lanes of the same BN254 work on one host, proves all four computed
the same values, then decides whether its own evidence supports a claim.
Charts in `../assets/`, sealed runs in `evidence/`.

## Lanes

| Lane | Library and build regime |
| --- | --- |
| `helius` | this crate, explicit target CPU, backend pinned in `lanes.json` |
| `arkworks` | ark-bn254, ark-ec, ark-ff 0.5.0, with `ark-ff/asm` |
| `arkworks-portable` | the same three crates, without `ark-ff/asm` |
| `mcl` | herumi/mcl at the pin below, native C++, 256-bit build |

All but `arkworks-portable` share one binary. The runner hashes both binaries
against `lanes.json` and stops if the two groups collapse onto one file or if
`helius` misses the pinned backend. gnark supplies the proof fixtures and is
not timed. Every lane prepares the key-owned G2 schedules outside the timer
and leaves every proof-owned G2 live. arkworks rebuilds one per iteration
because it cannot lend a schedule, and mcl shares an accumulator over at most
two prepared pairs, so part of any Groth16 gap is API shape.

## Bit-for-bit agreement

All four lanes compute the same equation on the same inputs and must agree bit
for bit. Every lane reports a 16 hex digit result digest per operation and
rotation, and the runner requires one value across the four. Each invocation
first runs an equivalence pass over the pool and a tamper pass with one
corrupted instance per class. A miss exits 4 and stops the run.

| Rows | Digest folds | Strength |
| --- | --- | --- |
| pairing, field, group | the operation's output | full |
| the four batch rows | accumulator, Miller product, verdict | full |
| the three single-proof rows | accumulator, product, verdict | weakest |
| `g2_prepare_87_lines` | its own schedule, replayed | lane-local |

The weakest rows time the prevalidated verifier entry, whose accepting product
is the key's constant `e(alpha, beta)`, so only the public-input point and the
verdict move with the fixture. A schedule layout belongs to the library that
built it, so each lane instead replays its own through its own prepared Miller
entry, and all three replays must equal the live Miller product.

The field-primitive rows run the scalar path, not the AVX-512 IFMA path, and
sit at parity with mcl or behind it. Only the Miller, pairing, and Groth16
rows reach IFMA, and of those only the two batch rows of eight reach the
eight-lane engine, which needs more live terms than the host crossover, 6 on
AMD and 4 on Intel. A round draws `min(iterations, pool_len)` fixtures from
its own start, so the pairing and Groth16 rows get a set no earlier round
drew, the rest rotate an order over a pool of eight, and `lane_reference`
gives gate E one plain 64-bit multiply chain per lane.

## Fixtures

Nothing in this repository proved any Groth16 proof it times. gnark v0.15.0
proved both pools. `fixtures/production-20260816` holds production circuits
compiled for the constraint systems their shipped proving keys were built for,
and a vector enters only when the originating verifier, `helius`, `arkworks`,
and `mcl` all agree on it. `fixtures/gnark-fresh-20260816` holds a MiMC
circuit with three public inputs. Both are hash-pinned, hold more vectors than
one invocation draws, and carry a `README.md` with generator, seed, digests,
and commands. The runner draws one fresh 64-bit `FOUR_LANE_FIXTURE_SEED` per
campaign, gives it to every spawn, records it in `summary.json`, and aborts on
a disagreeing seed or `fixture_set_sha256`.

## Pins

| Component | Pin |
| --- | --- |
| rustc | 1.97.1, from `rust-toolchain.toml` |
| arkworks | ark-bn254, ark-ec, ark-ff, ark-groth16, all 0.5.0 |
| mcl | `b70288469cd065b28a31764450f85238508c4d1e` |
| gnark | v0.15.0, for the proof fixtures |

The mcl pin is upstream `e107c70e814aaa3079fbb6fd630a8c48693c4c27` plus
`scripts/mcl-bench-patches.patch`, which exposes mcl's own `cyclotomicSqr` and
`mulSparseLine` so two rows compare against mcl's code and not a replica. A
commit hash covers both identities and both dates, so `provision.sh` applies
the patch as `helius-bench <bench@helius.local>` under `git am
--committer-date-is-author-date`, then compares HEAD against the pin.

Every lane runs alt_bn128, also called BN_SNARK1, which the harness names
explicitly because mcl's default is a different curve. mcl's own `BN254` has
`z = -0x4080000000000001`, `b = 2`, and `xi = 1 + i` against alt_bn128's
`z = 4965661367192848881`, `b = 3`, and `xi = 9 + i`. mcl publishes a faster
pairing figure for that curve, roughly 232 us against roughly 425 us, and it
verifies none of the proofs this benchmark is about.

## Host

Any x86-64 Linux box with AVX-512 IFMA works, so Zen 4, Zen 5, or Granite
Rapids. `lscpu | grep avx512ifma` must print a match, or the runner stops at
the first spawn against the pinned backend. A build that names no target CPU
resolves to base `x86-64`, which carries no AVX-512, so the crate compiles its
scalar path only and loses to both comparators on every row. `build_lanes.sh`
passes `-C target-cpu=native` and `NARSIL_TARGET_CPU` overrides it.

A shared rented instance has no fixed clock governor, boots with the default
mitigations, which inflate crypto kernels several times over, and usually
locks the performance counters. A run there passes `--diagnostic-host`,
records every tolerated gate in `metadata.json`, and never reaches
`claim_eligible`. It compares the four lanes in one session, so do not carry
an absolute microsecond value to another host or another boot. A claim
eligible run needs every row below, and without `--diagnostic-host` the runner
aborts on the first violation instead of recording it.

| Requirement | Value |
| --- | --- |
| host | bare metal you control |
| `scaling_governor` on the pinned core | `performance` |
| SMT sibling of that core | idle, under 2 percent over one second |
| kernel command line | `mitigations=off` |
| `perf` | working, for gate B |
| Criterion run of the same anchors | present, for gate A |
| rounds | a positive multiple of four |

## Run it

Deploy one commit with `git archive --format=tar HEAD | ssh BOX tar -x`, then
run this on the box, from the crate root.

```sh
W=~/narsil-bench
bench/scripts/provision.sh      # toolchain, build tools, mcl at the pin
bench/scripts/build_lanes.sh    # two lane binaries and lanes.json

python3 bench/scripts/run_campaign.py --lanes $W/lanes.json --out $W/run \
    --cpu 1 --rounds 4 --diagnostic-host --lock $W/campaign.lock

# Gate A. The same anchors under Criterion, same core, same seed.
SEED=$(python3 -c 'import json,sys
print(json.load(open(sys.argv[1]))["fixture_seed"])' $W/run/summary.json)
RUSTFLAGS="-C target-cpu=native" MCL_DIR=$W/mcl CARGO_TARGET_DIR=$W/anchors \
    cargo bench --manifest-path bench/Cargo.toml --bench anchors --no-run
env -i PATH="$PATH" HOME="$HOME" RAYON_NUM_THREADS=1 MCL_USE_OMP=0 \
    CRITERION_HOME=$W/criterion FOUR_LANE_FIXTURE_SEED="$SEED" taskset -c 1 \
    "$(ls -t $W/anchors/release/deps/anchors-* | grep -v '\.d$' | head -1)" \
    --bench --noplot

python3 bench/scripts/cross_check.py --summary $W/run/summary.json \
    --out $W/run/validity.json --criterion-dir $W/criterion

python3 bench/scripts/render_bench_svg.py --primary $W/run \
    --group verification --out $W/bench.svg \
    --host "$(lscpu | grep 'Model name' | cut -d: -f2 | xargs)"
```

Add `--quick` for a shakeout. `--group operations` renders the second chart
and needs `matplotlib`. `cross_check.py` exits 1 when the run is not claim
eligible, the expected outcome on a rented box. On hardware you own drop
`--diagnostic-host` and add `--perf` so gate B runs. Four rounds run for
hours. `test_run_campaign.py` covers the runner and the gates against stand-in
binaries, with no lane build.

A run writes `lanes.json`, `metadata.json` with the host and the gate state,
`raw.jsonl` with one provenance line per spawn, `summary.json` with the median
and interquartile range per cell plus the session seed, and `validity.json`
with every gate and the claim verdict. Keep the directory whole.

## Gates

Any runner check that fails stops the campaign. `cross_check.py` runs after
the campaign and alone may set `claim_eligible`.

| Check | Enforced by | What it catches |
| --- | --- | --- |
| binary sha256, binary shapes | runner | a rebuilt or swapped lane |
| `helius_backend`, `ark_asm` | runner | a wrong ISA or build regime |
| mcl archive and manifest | runner | a comparator rebuilt mid-run |
| `fixture_seed`, `fixture_set_sha256` | runner | a lane on other inputs |
| cross-lane digest equality | runner | skipped work or a wrong value |
| governor, sibling, boot id | runner | a host that cannot hold a clock |
| lane exit 3, 4, 5 | runner | build, equivalence, or timer failure |
| A, cross-harness, 10 percent | Criterion | a bug in this harness alone |
| B, cycle cross-check, 5 percent | `perf` at N and 2N | a denied clock |
| C, scale windows | operation counts | a wrong build or operation |
| D, time identities | ratios inside one lane | a mislabelled row |
| E, lane reference, 5 percent | `lane_reference` | a column times a constant |

Gate C gives mcl and arkworks their own ceilings, and a comparator above its
ceiling means the harness or the host is wrong, never that narsil is fast.
Gate E misses an error confined to one operation.
