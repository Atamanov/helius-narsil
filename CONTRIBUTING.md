# Contributing

```sh
scripts/install-git-hooks.sh
```

That sets `core.hooksPath` to `scripts/git-hooks`. The pre-commit hook runs
`cargo fmt --check`. The pre-push hook runs fmt, clippy, both test profiles,
docs, the deny check when `cargo-deny` is installed, and the crates.io dry
run. The only escape is Git's `--no-verify`. Integrators decide when to use
it.

```sh
cargo test --features std
cargo test --features std --test ark_differential
cargo test --features std --lib arkworks_bn254_0_5_tests
cargo test --features mcl-oracle --test mcl_differential   # needs MCL_DIR
cargo test --no-default-features
cargo clippy --features std --all-targets -- -D warnings
cargo bench --features std --bench pairing
cargo bench --features std --bench msm
cargo doc --no-deps --features std
cargo publish --dry-run
```

Commit subjects are one line. No `Co-Authored-By`. No tool trailers.
Write the body only when a reviewer cannot act without it.
