# gnark fresh Groth16 pool

Plain Groth16 verification vectors made by gnark from its own witnesses. No
code in this repository made the proofs, so this pool checks against gnark.

| Item | Value |
| --- | --- |
| generator | `tools/gnark-fixtures`, held outside this crate |
| gnark | v0.15.0 at `18368a5619e783f349b4aabb8277064495b6a473` |
| module hash | `h1:MwNpcGP2PawnGR3T9AnXDQS67aY22QTNb2Go8p/1gto=` |
| versions | gnark-crypto v0.20.1, go1.25.7 |
| run seed | `0x474e41524b465348` |
| determinism | a seeded reader locks `crypto/rand.Reader` before gnark runs |

```text
c4e003ee9759a58923ef6afb22ccc9ae08437e75cc88b7eeee04ea7716e145ad  pool_sha256
13a4554e32f1ebf56d5b2e66e72f9ad6b1d1e213410be60fa43c37aebc456fef  fixture.json
411388475985bc3b6ab3372f204790619e6ac0544b0f10750674cc3dbc607ae9  verifying key
```

`PoolCircuit` absorbs one salt and eight private preimage limbs into MiMC over
BN254 and pins the digest and the limb sum, over 2972 R1CS constraints. Three
public inputs keep the `gamma_abc` MSM non-trivial, and no `Commit` call keeps
this plain Groth16.

64 valid vectors pass under one key and 2 tampered ones fail, one moving `A`
by a generator step and one moving the `digest` input by one. Both stay on the
curve and in the subgroup, so the rejection comes from the pairing equation
and not from decoding. Every valid vector has a distinct witness, so no two
share public inputs or an A, B, or C element. Each carries `gnark_verdict`
from `groth16.Verify` and `encoded_verdict` from a replay over the alt_bn128
order hex, and a disagreement stops the generator.

G1 is 64 bytes `X || Y` and G2 is 128 bytes `X.A1 || X.A0 || Y.A1 || Y.A0`,
imaginary limb first, every Fp 32 bytes big-endian, a scalar canonical Fr, and
infinity all zeroes. `fixture.json` also carries gnark `WriteRawTo` and
`WriteTo` output, so gnark alone reads the pool back at the byte level.

```sh
shasum -a 256 -c SHA256SUMS
cd tools/gnark-fixtures
go run . -count 64 -seed 0x474e41524b465348 -out <dir>
go run . -verify <dir>
```
