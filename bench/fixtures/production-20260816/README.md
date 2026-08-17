# Production Groth16 pool

Verification vectors for production circuits and their shipped proving keys,
proved by gnark v0.15.0. Nothing in this repository proved them.

| Item | Value |
| --- | --- |
| generator | `tools/production-fixtures`, held outside this crate |
| gnark | v0.15.0, gnark-crypto v0.20.1, go1.25.7 |
| run seed | `0x005a4f4c414e4131` |
| proving keys | one sha256 per key in `fixture.json` |
| determinism | a seeded reader locks `crypto/rand.Reader` before gnark runs |

```text
bcff5aaf0e1863354242266890db181c2e3bb6e75b05417d42b4de63f16f40e9  pool_sha256
154a91c0178754b22835740c16edcd13e302e6951650ac9325ba67edfd82e38f  fixture.json
```

| Key | Shape | Constraints | Committed | Vectors |
| --- | --- | ---: | --- | ---: |
| `plain_2in_2out` | 2 in, 2 out | 52138 | no | 32 |
| `plain_2in_3out` | 2 in, 3 out | 54136 | no | 32 |
| `committed_2in_3out` | 2 in, 3 out | 245645 | yes | 16 |

Every key declares one public input. `committed_2in_3out` reports
`PublicAndCommitmentCommitted = [[]]`, so its BSB22 preimage is the commitment
point alone, which the harness recomputes and checks.
80 valid vectors pass and 7 tampered ones fail, per key on `proof_a` and on
the public input, plus one on the commitment. No two valid vectors share a
public input, a proof element, or a digest. Each carries `gnark_verdict` from
`groth16.Verify` and `encoded_verdict` from a replay over the alt_bn128 order
hex, and a disagreement stops the generator.

G1 is 64 bytes `X || Y` and G2 is 128 bytes `X.A1 || X.A0 || Y.A1 || Y.A0`,
imaginary limb first, every Fp 32 bytes big-endian, a scalar canonical Fr, and
infinity all zeroes. `fixture.json` also carries gnark `WriteRawTo` and
`WriteTo` output, so gnark alone reads the pool back at the byte level.

```sh
shasum -a 256 -c SHA256SUMS
cd tools/production-fixtures
go run . -seed 0x005a4f4c414e4131 -standard 32 -committed 16 \
  -keys-dir <proving-keys> -out <dir>
go run . -verify <dir>
```
