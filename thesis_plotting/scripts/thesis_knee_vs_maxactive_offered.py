#!/usr/bin/env python3
"""Clean monotonic knee curve for the redesigned knee_vs_maxactive sweep.

X-axis is OFFERED mean RPS (computed from each cell's actual arrival
trace), not achieved throughput. With offered RPS on x, the curve goes
monotonically right as load increases — no back-bending, no fold-overs,
no "oscillations". A configuration that saturates simply shoots
vertically upward in E2E p99 at the saturation point.

Source: cluster_outputs/online_serving_runs/knee_vs_maxactive_<TS>/
Output: thesis_plotting/figures/fig_knee_vs_maxactive_offered.{pdf,png}
"""

import re
import sys
from pathlib import Path

import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO))

from plot_sweep_compare import parse_summary  # noqa: E402

from thesis_plotting.style import (  # noqa: E402
    configure_plotting, save_fig, set_spines, bold_legend,
    FONT_LABEL, FONT_TITLE, FONT_TICK, FONT_ANNOTATE,
)

SWEEP = REPO / "cluster_outputs/online_serving_runs/knee_vs_maxactive_20260608_123805"
OUT_NAME = "fig_knee_vs_maxactive_offered"

CEILING_RPS = 2.13   # architectural ceiling at max_active in [64, 100]
SAFE_RPS    = 1.91   # 0.85× chapter safe operating point

CELL_RE = re.compile(r"^\d+_maxactive_(?P<v>\d+)_s(?P<sx10>\d+)$")


def load_cells(sweep_dir: Path):
    rows = []
    for cell in sorted(sweep_dir.iterdir()):
        if not cell.is_dir():
            continue
        m = CELL_RE.match(cell.name)
        if not m:
            continue
        max_active = int(m.group("v"))
        scale = int(m.group("sx10")) / 10.0
        sumfile = cell / "policy_compare_summary.txt"
        req_csv = cell / "max_util_full" / "requests_out.csv"
        if not sumfile.is_file() or not req_csv.is_file():
            continue
        m_stats = parse_summary(sumfile)
        achieved = m_stats.get("throughput")
        e2e_p99 = m_stats.get("e2e_p99")
        if achieved is None or e2e_p99 is None:
            continue
        n = 0
        max_arrival = 0.0
        with open(req_csv) as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                a = float(row["arrival_ms"])
                if a > max_arrival:
                    max_arrival = a
                n += 1
        if max_arrival <= 0 or n == 0:
            continue
        offered = n / (max_arrival / 1000.0)
        rows.append((max_active, scale, offered, achieved, e2e_p99))
    return rows


def render(rows, out_dir: Path, out_name: str = OUT_NAME, linear: bool = False):
    configure_plotting()
    by_ma = {}
    for ma, scale, offered, achieved, p99 in rows:
        by_ma.setdefault(ma, []).append((offered, achieved, p99, scale))
    ma_values = sorted(by_ma.keys())

    cmap = plt.get_cmap("viridis")
    colors = {ma: cmap(i / max(1, len(ma_values) - 1))
              for i, ma in enumerate(ma_values)}

    fig, ax = plt.subplots(figsize=(11.0, 6.2))

    markers = ["o", "s", "^", "D", "v", "P", "X"]
    for i, ma in enumerate(ma_values):
        pts = sorted(by_ma[ma], key=lambda r: r[0])
        xs = [p[0] for p in pts]
        ys = [p[2] / 1000.0 for p in pts]
        ax.plot(xs, ys, marker=markers[i % len(markers)], ms=9, lw=2.4,
                color=colors[ma], label=f"max_active = {ma}",
                markeredgecolor="black", markeredgewidth=0.6, zorder=4)

    # Reference lines for the safe operating point and the architectural
    # ceiling. Both are *offered* RPS reference lines on this x-axis.
    ax.axvline(SAFE_RPS, color="#1f77b4", linestyle="--", linewidth=1.3,
               zorder=3, alpha=0.7)
    ax.text(SAFE_RPS, ax.get_ylim()[0] * 1.2 if False else 0.02,
            "  safe op  1.91", transform=ax.get_xaxis_transform(),
            ha="left", va="bottom",
            color="#1f77b4", fontsize=FONT_ANNOTATE,
            fontweight="bold")
    ax.axvline(CEILING_RPS, color="#c0392b", linestyle=":", linewidth=1.3,
               zorder=3, alpha=0.7)
    ax.text(CEILING_RPS, 0.02,
            "  ceiling  2.13", transform=ax.get_xaxis_transform(),
            ha="left", va="bottom",
            color="#c0392b", fontsize=FONT_ANNOTATE,
            fontweight="bold")

    if not linear:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel("Offered mean load  (RPS"
                  + ("" if linear else ", log scale") + ")",
                  fontsize=FONT_LABEL + 1, fontweight="bold")
    ax.set_ylabel("E2E p99  (s"
                  + ("" if linear else ", log scale") + ")",
                  fontsize=FONT_LABEL + 1, fontweight="bold")
    ax.set_title("Pi0 @ azure_poisson_wide  —  knee vs max_active",
                 fontsize=FONT_TITLE + 1, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=FONT_TICK + 1)
    ax.grid(True, which="major", alpha=0.45)
    if not linear:
        ax.grid(True, which="minor", alpha=0.18)
    set_spines(ax)
    leg = ax.legend(loc="upper left", fontsize=FONT_TICK + 1,
                    title="active-request cap",
                    title_fontsize=FONT_TICK + 1, framealpha=0.95,
                    ncol=2)
    bold_legend(leg)

    save_fig(fig, out_name, out_dir)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--linear", action="store_true")
    ap.add_argument("--out-name", default=OUT_NAME)
    args = ap.parse_args()

    rows = load_cells(SWEEP)
    if not rows:
        sys.exit(f"[error] no cells found in {SWEEP}")
    print(f"[info] {len(rows)} cells loaded")
    for ma in sorted({r[0] for r in rows}):
        cells = sorted([r for r in rows if r[0] == ma], key=lambda r: r[1])
        print(f"  max_active={ma}:")
        for _, scale, offered, achieved, p99 in cells:
            print(f"    s={scale:>5.2f}  offered={offered:>5.2f}  "
                  f"achieved={achieved:>5.2f}  e2e_p99={p99/1000.0:>8.1f}s")
    name = args.out_name + ("_linear" if args.linear else "")
    render(rows, REPO / "thesis_plotting/figures", name, args.linear)


if __name__ == "__main__":
    main()
