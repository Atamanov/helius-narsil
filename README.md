# helius-narsil

![Narsil](assets/logo.png)

**Same silicon. New edge.**

Narsil is a SIMD hardware optimized Rust library for pairing friendly elliptic
curve operations. Its intended use is **verification**, with the main focus on
pairing check performance.

**DISCLAIMER.** This crate is experimental. **Do not** use it in production.
An audit and a production recommendation come later. Release 1.0 ships with
its own announcement.

BN254 is the only curve in this tree. A BLS family may follow.

## Do not use with secrets

No method in this crate is safe for secret inputs. Constant-time methods will
be named and documented one by one. Until a method is named that way, treat
every function as variable-time.

## Benchmarks

![Narsil Groth16 verification times versus MCL and arkworks](assets/bench.svg)

Verifying one Groth16 proof with one public input takes 298 us against MCL's
741 us and arkworks' 1168 us, so 2.5x MCL and 3.9x arkworks. Committed rails
are 506 us against 1259 us and 2093 us, 2.5x and 4.1x. A batch of eight proofs
under one key is 1174 us against 2275 us and 3880 us, 1.9x and 3.3x, and a
batch of three is 713 us against 1380 us and 2287 us, 1.9x and 3.2x.

![Narsil pairing times versus MCL and arkworks](assets/bench-operations.svg)

A full pairing is 192 us against 428 us and 784 us, 2.2x MCL and 4.1x arkworks.
With a prepared key it is 138 us against 376 us and 692 us, 2.7x and 5.0x. The
prepared Miller loop is 47.6 us against 142 us and 262 us, 3.0x and 5.5x, and
final exponentiation is 89.3 us against 235 us and 428 us, 2.6x and 4.8x.

Medians come from one sealed four round campaign on an AMD EPYC 9654, Zen 4
with AVX-512 IFMA, on the curve alt_bn128, also called BN_SNARK1.

A timed verification computes the public-input MSM, the three pair Miller
product against prepared `gamma` and `delta`, the final exponentiation, and the
comparison against `e(alpha, beta)`. The MSM feeds the pairing, so it cannot be
hoisted. Point decoding and validation sit outside the timer in every lane and
are timed separately by `g1_validate` and `g2_subgroup_check`. Verifying from
untrusted bytes adds two G1 validations and one G2 subgroup check, about 50 us
for narsil and 136 us for MCL, which widens the margin rather than narrowing
it.

Every pairing-level and Groth16-level operation lands at or below 0.55 of MCL's
time and 0.32 of arkworks'. The narrowest is the live Miller loop at 0.54 of
MCL, a 2.6 percent margin.

**The field-primitive rows do not share that result.** They run narsil's scalar
path, not its AVX-512 IFMA path, and they sit at parity with MCL or a little
behind it. Narsil's own G2 line generation costs about 14 percent more than
MCL's. The margin above comes from the fused IFMA pipeline that the compound
operations use, so read the numbers as a claim about pairings and proof
verification, not about every routine in the crate.

The proofs come from production circuits and their shipped proving keys, proved
by gnark v0.15.0. Nothing in this repository proved them. All four lanes verify
every proof before any timing, compute the same equation on the same inputs,
and must agree bit for bit. Proof bytes come from a hash-pinned pool, and a
per-session seed picks which of them a run measures and in what order, so no
two sessions walk the same sequence.

These are diagnostic numbers, not certified ones. The host is a shared rented
container with no fixed clock governor and default kernel mitigations, so the
harness records the run as not claim eligible and says why. A cross-harness
gate against Criterion agrees with the campaign to 1.13 percent, and a
same-work reference row measured in all four lanes spreads 0.18 percent. The
cycle cross-check gate needs `perf`, which that kernel does not carry, so it
did not run.

MCL publishes a faster pairing figure for its own `BN254` parameterization.
That is a different curve. This harness runs MCL on alt_bn128, which is the
curve the proofs use.

`bench/` holds the harness, the protocol, and the sealed evidence. It is
excluded from the published crate.

## Example

Groth16 is `e(alpha, beta) * e(L, gamma) * e(C, delta) * e(-A, B) = 1`.
`alpha` and `beta` stay in the verifying key. `gamma` and `delta` get prepared
G2 schedules. `A`, `B`, `C` and the public-input MSM `L` are online. Needs
`--features std`. See `examples/groth16.rs`.

```rust
# #[cfg(not(feature = "std"))]
# fn main() {}
# #[cfg(feature = "std")]
fn main() -> Result<(), helius_narsil::PreparedVerifierError> {
use helius_narsil::{
    FixedPair, Fr, G1Affine, G1Projective, G2Affine, G2Projective, LivePair,
    PreparedTerm, PreparedVerifier,
};

fn g1(k: u64) -> G1Affine {
    G1Projective::generator().mul_scalar(Fr::from_u64(k)).to_affine()
}
fn g2(k: u64) -> G2Affine {
    G2Projective::from(G2Affine::generator())
        .mul_scalar(Fr::from_u64(k))
        .to_affine()
}

let vk = PreparedVerifier::new(
    &[g2(1), g2(2)],
    &[FixedPair { g1: g1(4), g2: g2(1) }],
)?;
assert!(vk.verify(
    &[
        PreparedTerm { g1: g1(1), prepared_g2: 0 },
        PreparedTerm { g1: g1(1), prepared_g2: 1 },
    ],
    &[LivePair { g1: g1(1).negate(), g2: g2(7) }],
)?);
Ok(())
}
```

Byte-facade product check (`examples/product_check.rs`).

```rust
use helius_narsil::{G1Affine, G1Bytes, G2Affine, G2Bytes, PairBytes};

let p = G1Affine::generator();
let q = G2Affine::generator();
let pairs = [
    PairBytes {
        g1: G1Bytes::from_affine(&p),
        g2: G2Bytes::from_affine(&q),
    },
    PairBytes {
        g1: G1Bytes::from_affine(&p.negate()),
        g2: G2Bytes::from_affine(&q),
    },
];

assert!(helius_narsil::pairing_product_is_one(&pairs)?);
# Ok::<(), helius_narsil::InputError>(())
```

## Tests

[CI](https://github.com/Atamanov/helius-narsil/actions/workflows/ci.yml)
runs `cargo test` on rustc 1.97.1, the MSRV, and on stable. Each toolchain
runs three feature sets, `default` which is `no_std`, `std`, and
`std,force-portable`. The `default` cell also runs fmt, clippy, docs,
cargo-deny, and `cargo publish --dry-run`. CI covers the in-tree arkworks 0.5
native suite and `ark_differential`. MCL is not in CI.

```sh
scripts/install-git-hooks.sh
# pre-commit runs fmt. pre-push runs fmt, clippy, both test profiles, docs,
# the deny check, and the publish dry run.

cargo test --features std
# unit tests, integration tests, and the arkworks 0.5 native suite

cargo test --features std --test ark_differential
# public-API pairing, MSM, Fr, and encoding checks against arkworks 0.5

cargo test --features std --lib arkworks_bn254_0_5_tests
# the 103 default-native ark-bn254 / ark-ec / ark-ff 0.5 cases, adapted in-tree

cargo test --features mcl-oracle --test mcl_differential   # needs MCL_DIR
# pairing, Miller loop, final exp, and a small MSM against a local MCL tree

cargo bench --features std --bench pairing
# Miller loop, prepared Miller loop, final exponentiation, full pairing

cargo bench --features std --bench msm
# G1 variable-time MSM at 1, 8, 32, and 128 points

cargo bench --features std --bench field
# Fp and Fp2 mul / square

cargo doc --no-deps --features std
cargo publish --dry-run
```

`mcl-oracle` links a caller-supplied MCL tree. It is not a production
dependency.

## License

MIT OR Apache-2.0.

Copyright (c) 2026 Helius Blockchain Technologies, Inc.
