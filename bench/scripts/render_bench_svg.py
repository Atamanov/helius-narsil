#!/usr/bin/env python3
"""Render the narsil benchmark charts from sealed campaign evidence.

Reads one campaign directory and writes an SVG bar chart. Every number comes
from summary.json, nothing is typed by hand.

Two charts, because the verification rows and the pairing rows are two decades
apart and one shared axis crushes the faster group. Each chart carries a true
microsecond axis and prints the median on every bar.

Needs matplotlib. Text is emitted as paths, so a chart carries no font
dependency and renders the same everywhere.
"""

from __future__ import annotations

import argparse
import json
import pathlib

# Preferred rows first. A sealed campaign predating a row simply lacks it,
# and the renderer skips what a summary does not carry, so one renderer
# serves both current and archived evidence.
# Two charts. A campaign predating a row simply lacks it, and the renderer
# skips what a summary does not carry, so one renderer serves both current and
# archived evidence.
GROUPS = {
    "verification": {
        "title": "BN254 Groth16 verification",
        "anchor": ["groth16_one_input", "groth16_gnark", "groth16_single"],
        "rows": [
            ("groth16_one_input", "Groth16\n1 public input"),
            ("groth16_committed_rails", "Groth16\ncommitted rails"),
            ("groth16_batch3_one_key", "Batch of 3\none key"),
            ("groth16_batch3_mixed_keys", "Batch of 3\nmixed keys"),
            ("groth16_batch8_one_key", "Batch of 8\none key"),
            ("groth16_batch8_mixed_keys", "Batch of 8\nmixed keys"),
            ("groth16_gnark", "Groth16\n3 public inputs"),
            ("groth16_single", "Groth16\nsingle"),
            ("groth16_committed", "Groth16\ncommitted"),
            ("groth16_multiproof_3", "Batch of 3"),
        ],
    },
    "operations": {
        "title": "BN254 pairing operations",
        "anchor": ["full_pairing"],
        "rows": [
            ("full_pairing", "Full\npairing"),
            ("full_pairing_prepared", "Full pairing\nprepared key"),
            ("miller_live", "Miller\nloop"),
            ("miller_prepared", "Miller loop\nprepared key"),
            ("final_exponentiation", "Final\nexponentiation"),
        ],
    },
}
# Prose must follow the data, never the other way round.
SUBTITLE = {
    "groth16_one_input": "one Groth16 proof, one public input",
    "groth16_gnark": "one Groth16 proof, three public inputs",
    "groth16_single": "one Groth16 proof, one public input",
    "full_pairing": "one full pairing",
}

REFERENCE_WIDTH = 4.2 + 1.35 * 7

BG = "#07112d"
INK = "#eefaff"
DIM = "#9db6de"
FAINT = "#6f8ab5"
ACCENT = "#3ad4ff"
MCL = "#7c4dff"
ARK = "#5075bd"
RULE = "#28477e"

SERIES = [("helius", "Narsil", ACCENT), ("mcl", "MCL", MCL),
          ("arkworks", "arkworks", ARK)]


def load(directory: str) -> dict:
    path = pathlib.Path(directory)
    summary = json.loads((path / "summary.json").read_text())
    metadata = json.loads((path / "metadata.json").read_text())
    validity = json.loads((path / "validity.json").read_text())
    return {"cells": summary["cells"], "seed": summary.get("fixture_seed"),
            "metadata": metadata, "validity": validity}


def us(cells: dict, operation: str, lane: str) -> float | None:
    cell = cells.get(operation, {}).get(lane)
    if not cell or "unsupported" in cell:
        return None
    return float(cell["median_ns_per_call"]) / 1000.0


def fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value >= 100:
        return f"{value:.0f}"
    return f"{value:.1f}"


def pick_headline(cells: dict, anchor: list[str]) -> str:
    for operation in anchor:
        if operation in cells:
            return operation
    raise SystemExit("summary carries no anchor row for this chart")


def install_font(font_dir: str | None) -> str:
    """Register a bundled font family, and fall back to whatever exists."""
    from matplotlib import font_manager

    if font_dir:
        for path in sorted(pathlib.Path(font_dir).glob("*.ttf")):
            font_manager.fontManager.addfont(str(path))
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Inter", "Roboto", "Source Sans 3", "Helvetica Neue",
                 "DejaVu Sans"):
        if name in have:
            return name
    return "sans-serif"


def render(primary: dict, host: str, out: str, font_dir: str | None,
           group: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    family = install_font(font_dir)
    matplotlib.rcParams.update({
        "svg.fonttype": "path",
        "font.family": family,
        "figure.facecolor": BG,
        "axes.facecolor": BG,
    })

    spec = GROUPS[group]
    cells = primary["cells"]
    headline = pick_headline(cells, spec["anchor"])
    subtitle = SUBTITLE[headline]
    present = [(op, label) for op, label in spec["rows"] if op in cells]

    narsil = us(cells, headline, "helius")
    speed_mcl = us(cells, headline, "mcl") / narsil
    speed_ark = us(cells, headline, "arkworks") / narsil

    width = 4.2 + 1.35 * len(present)
    # A reader sees both charts at one column width, so the narrower one
    # is scaled up more. Shrink its type by the same factor to land on
    # equal rendered text.
    scale = width / REFERENCE_WIDTH
    fig, ax = plt.subplots(figsize=(width, 6.6), dpi=100)
    fig.subplots_adjust(left=0.088, right=0.985, top=0.645, bottom=0.125)

    span = 0.27
    tallest = max(v for op, _ in present for v in
                  [us(cells, op, lane) for lane, _, _ in SERIES] if v)

    for slot, (lane, _, colour) in enumerate(SERIES):
        offset = (slot - 1) * span
        for index, (op, _) in enumerate(present):
            value = us(cells, op, lane)
            if not value:
                continue
            ax.bar(index + offset, value, width=span * 0.92, color=colour,
                   zorder=3, linewidth=0)
            ax.text(index + offset, value + tallest * 0.013, fmt(value),
                    ha="center", va="bottom", fontsize=8.0 * scale, color=DIM,
                    zorder=4)

    ax.set_ylim(0, tallest * 1.12)
    ax.set_xlim(-0.62, len(present) - 0.38)
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([label for _, label in present], fontsize=9.8 * scale,
                       color=INK)
    ax.set_ylabel("microseconds, lower is better", fontsize=10 * scale, color=DIM,
                  labelpad=10)
    ax.tick_params(axis="y", colors=FAINT, labelsize=9.5 * scale)
    ax.tick_params(axis="x", length=0, pad=8)
    ax.yaxis.grid(True, color=RULE, linewidth=0.7, alpha=0.75, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(RULE)

    left = 0.088
    fig.text(left, 0.958, spec["title"], fontsize=21 * scale, color=INK,
             weight="bold", va="top")
    fig.text(left, 0.912,
             f"Median time per operation on {host}.",
             fontsize=11.5 * scale, color=DIM, va="top")
    fig.text(left, 0.858,
             f"{speed_ark:.1f}x faster than arkworks      "
             f"{speed_mcl:.1f}x faster than MCL",
             fontsize=24 * scale, color=ACCENT, weight="bold", va="top")
    fig.text(left, 0.788,
             f"On {subtitle}. Bar labels are medians in microseconds.",
             fontsize=10 * scale, color=FAINT, va="top")

    ax.legend(handles=[Patch(facecolor=c, label=n) for _, n, c in SERIES],
              loc="upper left", bbox_to_anchor=(0.0, 1.135), ncol=3,
              frameon=False, fontsize=10.5 * scale, labelcolor=DIM,
              handlelength=1.1, handleheight=1.1, columnspacing=1.6)

    suffix = pathlib.Path(out).suffix.lstrip(".") or "svg"
    fig.savefig(out, format=suffix, facecolor=BG)
    plt.close(fig)

    if suffix == "svg":
        # Glyphs are paths, so a reader with a screen reader gets nothing from
        # the markup unless the figure states its own result.
        describe(out, host, spec["title"], subtitle, narsil, cells, headline,
                 speed_mcl, speed_ark, len(present))


def describe(out: str, host: str, title: str, subtitle: str, narsil: float,
             cells: dict, headline: str, speed_mcl: float, speed_ark: float,
             count: int) -> None:
    path = pathlib.Path(out)
    svg = path.read_text()
    desc = (
        f"Grouped bar chart of {count} operations on {host}, in microseconds, "
        f"where a shorter bar is faster. Each group holds Narsil, MCL and "
        f"arkworks. On {subtitle}, Narsil takes {fmt(narsil)} "
        f"microseconds against {fmt(us(cells, headline, 'mcl'))} for MCL and "
        f"{fmt(us(cells, headline, 'arkworks'))} for arkworks, so "
        f"{speed_mcl:.1f} times faster than MCL and {speed_ark:.1f} times "
        f"faster than arkworks."
    )
    head = svg.index("<svg ")
    close = svg.index(">", head)
    svg = (svg[:close] + ' role="img" aria-labelledby="title desc"'
           + svg[close:close + 1]
           + f"\n <title id=\"title\">{title}, Narsil against MCL and arkworks</title>"
           + f"\n <desc id=\"desc\">{desc}</desc>"
           + svg[close + 1:])
    path.write_text(svg)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--out", required=True,
                        help="output path, or a directory with --all")
    parser.add_argument("--group", default="verification",
                        choices=sorted(GROUPS))
    parser.add_argument("--all", action="store_true",
                        help="render every chart into the --out directory")
    parser.add_argument("--font-dir",
                        help="directory of .ttf files to register")
    args = parser.parse_args()
    primary = load(args.primary)
    if args.all:
        target = pathlib.Path(args.out)
        target.mkdir(parents=True, exist_ok=True)
        for name in sorted(GROUPS):
            stem = "bench" if name == "verification" else f"bench-{name}"
            path = target / f"{stem}.svg"
            render(primary, args.host, str(path), args.font_dir, name)
            print(path)
        return
    render(primary, args.host, args.out, args.font_dir, args.group)
    print(args.out)


if __name__ == "__main__":
    main()
