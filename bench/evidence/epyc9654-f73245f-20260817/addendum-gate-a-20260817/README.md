# Gate A addendum, EPYC 9654, 2026-08-17

Gate A over the sealed `summary.json` of the parent campaign, plus three
readings the parent could not carry. Every file the campaign wrote is
unchanged. Same box, same boot id, same mcl archive. `claim_eligible` stays
false.

| Reading | Value |
| --- | --- |
| A, cross harness | pass, 15 anchors, largest deviation 1.90 percent |
| B, cycle cross check | skipped, kernel 5.15.0-179-generic has no `perf` |
| E, lane reference | pass, spread 0.88 percent, tolerance 5 percent |
| fixture seed | `0x06a13fbe4590bb3d`, the parent's own session seed |

`bench/benches/anchors.rs` builds the campaign's pools from that seed and
calls the same lane adapters under Criterion's clock and sampling, 50 samples,
2 s warm-up, 10 s measurement, pinned to cpu 1 in the campaign's scrubbed
environment. Gate E reads a `lane_reference` row the parent campaign predates,
so `quick-validity.json` carries its first reading from a quick campaign on
the same host rather than from the sealed one.

`digest_agreement.json` holds 84 cells, the seven Groth16 rows against three
fixture seeds and four rotations, four lanes each. Every cell has one digest
across the four lanes and no row repeats a digest across its seeds or its
rotations.

`mcl-prepared-shape.json`, from `bench/src/bin/mcl_prepared_shape.rs` at 4000
iterations on cpu 1, prices mcl's two-pair prepared ceiling. A one-pair
prepared Miller loop costs 142433 ns and a two-pair one costs 267392 ns, so a
further pair inside an accumulator that exists costs 124959 ns and the
unshared chain plus one Fp12 multiply costs 18906 ns per extra Miller group.
Against the sealed mcl medians that is 1.5 to 3.0 percent of the Groth16 rows.
The figure is an upper bound, because mcl publishes no wider prepared entry
and the marginal cost of a further pair stands in for the counterfactual.
