<p align="center">
  <img src="assets/logo.png" alt="Narsil" width="640">
</p>

<p align="center">
  <img src="assets/experimental.svg" alt="! EXPERIMENTAL. Do not use in production.">
</p>

# helius-narsil

Same steel. New edge.

Do not use this crate in production. It is experimental. An audit and a
production recommendation come later. Release 1.0 ships with its own
announcement.

BN254 is the only curve in this tree. A BLS family may follow.

## Do not use with secrets

In the current form, no method is safe for secret inputs. Future constant-time
methods will be named and documented one by one. Until a method is named that
way, treat every function as variable-time.

## Benchmarks

<p align="center">
  <img src="assets/bench.svg" alt="Narsil pairing times versus MCL and arkworks">
</p>

Intel Xeon 6952P (Granite Rapids, AVX-512 IFMA). Diagnostic container.
Not claim-eligible. Sealed campaign `four-lane-gnr-diagnostic-20260815e`.

## Example

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

## Commands

```sh
scripts/install-git-hooks.sh
cargo test --features std
cargo test --features std --test ark_differential
cargo test --features std --lib arkworks_bn254_0_5_tests
cargo test --features mcl-oracle --test mcl_differential   # needs MCL_DIR
cargo bench --features std --bench pairing
cargo bench --features std --bench msm
cargo doc --no-deps --features std
cargo publish --dry-run
```

`mcl-oracle` links a caller-supplied MCL tree. It is not a production
dependency.

## License

MIT OR Apache-2.0.

Copyright (c) 2026 Helius Blockchain Technologies, Inc.  
2093 Philadelphia Pike  
Unit 7808  
Claymont, Delaware 19703  
Phone: +1 (917) 933-5224
