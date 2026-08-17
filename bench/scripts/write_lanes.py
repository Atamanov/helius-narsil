#!/usr/bin/env python3
"""Write lanes.json for run_campaign.py.

Records the sha256 of each lane binary at build time. The runner rehashes
before the first spawn, so a binary replaced between the build and the
campaign cannot reach a timed region.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asm-binary", required=True)
    parser.add_argument("--portable-binary", required=True)
    parser.add_argument("--expected-backend", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    asm = pathlib.Path(args.asm_binary).resolve()
    portable = pathlib.Path(args.portable_binary).resolve()
    if asm == portable:
        raise SystemExit("the two lane binaries must be different files")
    asm_entry = {"binary": str(asm), "sha256": digest(asm)}
    document = {
        # The helius lane pins the backend it must select. A build that falls
        # back to another backend then fails before the first measurement.
        "helius": {**asm_entry, "expected_backend": args.expected_backend},
        "arkworks": {**asm_entry, "expected_backend": None},
        "mcl": {**asm_entry, "expected_backend": None},
        "arkworks-portable": {
            "binary": str(portable),
            "sha256": digest(portable),
            "expected_backend": None,
        },
    }
    pathlib.Path(args.out).write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n"
    )
    print(args.out)


def digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
