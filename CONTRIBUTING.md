# Contributing

```sh
scripts/install-git-hooks.sh
```

That sets `core.hooksPath` to `scripts/git-hooks`. pre-commit runs
`cargo fmt --check`. pre-push runs fmt, clippy, both test profiles,
`cargo doc`, and `cargo publish --dry-run`. The only escape is git
`--no-verify`.

```sh
cargo test --features std
cargo test --no-default-features
cargo clippy --features std --all-targets -- -D warnings
```

Commit subjects are one line. No `Co-Authored-By`. No tool trailers.
Write the body only when a reviewer cannot act without it.
