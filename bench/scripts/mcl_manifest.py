#!/usr/bin/env python3
"""Write the manifest that binds the built MCL archive to its inputs.

The harness build reads lib256/helius-mcl-native.json, hashes it, and
carries the hash into every provenance line, so this file is what proves that
all four lanes linked one comparator archive built one way.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess


SCHEMA = "helius-bn254-mcl-native-build-v2"
# Dedicated harness outputs, present in the tree and outside the source hash.
GENERATED_PREFIXES = ("lib256/", "obj256/")


def main() -> None:
    args = parse_args()
    root = pathlib.Path(args.root)
    archive = root / "lib256/libmcl.a"
    tools = dict(item.split("=", 1) for item in args.tool)
    payload = {
        "schema": SCHEMA,
        "revision": args.revision,
        "source_tree": git(root, "rev-parse", "HEAD^{tree}"),
        "source_dirty_content_sha256": source_dirty_digest(root),
        "native_flag": args.native_flag,
        "architecture": args.architecture,
        "operating_system": args.operating_system,
        "hermetic_path": args.hermetic_path,
        "make_jobs": 1,
        "make_variables": dict(
            item.split("=", 1) for item in args.make_variables
        ),
        "archive_sha256": digest_file(archive),
    }
    for name, path in tools.items():
        payload[name] = path
        payload[f"{name}_binary_sha256"] = digest_file(pathlib.Path(path))
        payload[f"{name}_version_sha256"] = version_digest(path)
    manifest = archive.with_name("helius-mcl-native.json")
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"native MCL archive: {archive}")
    print(f"build manifest: {manifest}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--native-flag", required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--operating-system", required=True)
    parser.add_argument("--hermetic-path", required=True)
    parser.add_argument("--tool", action="append", default=[])
    parser.add_argument("make_variables", nargs="*")
    return parser.parse_args()


def digest_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def version_digest(path: str) -> str:
    result = subprocess.run(
        [path, "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    return hashlib.sha256(result.stdout + result.stderr).hexdigest()


def git(root: pathlib.Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def source_dirty_digest(root: pathlib.Path) -> str:
    """Hash every source difference from the pinned commit.

    Length-prefixed fields keep the concatenation unambiguous, so no pair of
    different trees can produce one digest.
    """
    tracked = subprocess.run(
        ["git", "-C", str(root), "diff", "--binary", "--full-index", "HEAD", "--"],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    untracked = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--others", "--exclude-standard", "-z"],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout.split(b"\0")
    hasher = hashlib.sha256()
    field(hasher, b"helius-git-dirty-content-v1")
    field(hasher, tracked)
    for encoded in filter(None, untracked):
        relative = encoded.decode()
        if relative.startswith(GENERATED_PREFIXES):
            continue
        absolute = root / relative
        if absolute.is_symlink():
            kind, contents = b"symlink", absolute.readlink().as_posix().encode()
        elif absolute.is_file():
            kind, contents = b"file", absolute.read_bytes()
        else:
            raise SystemExit(f"unsupported source input: {relative}")
        field(hasher, encoded)
        field(hasher, kind)
        field(hasher, contents)
    return hasher.hexdigest()


def field(hasher: "hashlib._Hash", value: bytes) -> None:
    hasher.update(len(value).to_bytes(8, "big"))
    hasher.update(value)


if __name__ == "__main__":
    main()
