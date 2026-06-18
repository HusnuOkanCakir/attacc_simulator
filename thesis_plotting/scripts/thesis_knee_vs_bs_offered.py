#!/usr/bin/env python3
"""Clean monotonic knee curve for the bs sweep with offered-RPS x-axis.

Mirrors thesis_knee_vs_maxactive_offered.py. With offered RPS on x
instead of achieved throughput, the bs=4 deep-saturation "droop"
disappears — it was an x-axis fold-back artifact, not a real
non-monotonicity in (offered, p99) space.

Source: cluster_outputs/online_serving_runs/knee_vs_bs_20260606_114716/
Output: thesis_plotting/figures/fig_knee_vs_bs_offered.{pdf,png}
"""

import argparse
import csv
import re
import sys
from pathlib import Path

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

SWEEP = REPO / "cluster_outputs/online_serving_runs/knee_vs_bs_20260606_114716"
OUT_NAME = "fig_knee_vs_bs_offered"

CEILING_RPS = 2.10
SAFE_RPS = 1.91

CELL_RE = re.compile(r"^\d+_bs_(?P<v>\d+)_s(?P<sx10>\d+)$")


def load_cells(sweep_dir: Path):
    rows = []
    for cell in sorted(sweep_dir.iterdir()):
        if not cell.is_dir():
            continue
        m = CELL_RE.match(cell.name)
        if not m:
            continue
        bs = int(m.group("v"))
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
        rows.append((bs, scale, offered, achieved, e2e_p99))
    return rows


def render(rows, out_dir: Path, out_name: str = OUT_NAME, linear: bool = False,
           title=None, ref_rps=SAFE_RPS):
    configure_plotting()
    # Local font overrides — defaults (FONT_BASE=7) render ~3 pt after LaTeX
    # scales the figure to \linewidth. Bump for readability.
    FL  = FONT_LABEL    + 8   # 15  axis labels
    FTI = FONT_TITLE    + 8   # 16  title
    FTK = FONT_TICK     + 8   # 14  ticks / legend
    FAN = FONT_ANNOTATE + 9   # 14  safe-op / ceiling annotations
    by_bs = {}
    for bs, scale, offered, achieved, p99 in rows:
        by_bs.setdefault(bs, []).append((offered, achieved, p99, scale))
    bs_values = sorted(by_bs.keys())

    cmap = plt.get_cmap("viridis")
    colors = {bs: cmap(i / max(1, len(bs_values) - 1))
              for i, bs in enumerate(bs_values)}

    fig, ax = plt.subplots(figsize=(11.0, 6.2))

    markers = ["o", "s", "^", "D", "v"]
    for i, bs in enumerate(bs_values):
        pts = sorted(by_bs[bs], key=lambda r: r[0])
        xs = [p[0] for p in pts]
        ys = [p[2] / 1000.0 for p in pts]
        ax.plot(xs, ys, marker=markers[i % len(markers)], ms=9, lw=2.4,
                color=colors[bs], label=f"bs = {bs}",
                markeredgecolor="black", markeredgewidth=0.6, zorder=4)

    # Safe-op marker. The 1.91 RPS chapter operating point is defined for the
    # canonical bs=4 config (the smallest cap that fully reaches the 2.10 RPS
    # architectural ceiling). For bs=1 and bs=2 the configured cap saturates
    # well below 1.91 RPS, so the marker doesn't apply; bs=4, 8, 16 can all
    # operate at 1.91 RPS.
    if ref_rps and ref_rps > 0:
        ax.axvline(ref_rps, color="#1f77b4", linestyle="--", linewidth=1.6,
                   zorder=3, alpha=0.8)
        ax.text(ref_rps, 0.72, f"ref op {ref_rps:.2f}  (bs 4+)  ",
                transform=ax.get_xaxis_transform(),
                ha="right", va="top",
                color="#1f77b4", fontsize=FAN,
                fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="#1f77b4",
                          boxstyle="round,pad=0.3", linewidth=0.9, alpha=0.95))

    if not linear:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel("Offered mean load  (RPS"
                  + ("" if linear else ", log scale") + ")",
                  fontsize=FL, fontweight="bold")
    ax.set_ylabel("E2E p99  (s"
                  + ("" if linear else ", log scale") + ")",
                  fontsize=FL, fontweight="bold")
    ax.set_title(title or "Pi0 @ azure_poisson_wide  —  knee vs decode batch cap",
                 fontsize=FTI, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=FTK)
    ax.grid(True, which="major", alpha=0.45)
    if not linear:
        ax.grid(True, which="minor", alpha=0.18)
    set_spines(ax)
    leg = ax.legend(loc="upper left", fontsize=FTK,
                    title="decode batch cap",
                    title_fontsize=FTK, framealpha=0.95)
    bold_legend(leg)

    save_fig(fig, out_name, out_dir)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--linear", action="store_true")
    ap.add_argument("--out-name", default=OUT_NAME)
    ap.add_argument("--sweep-dirs", type=Path, nargs="+", default=[SWEEP],
                    help="one or more bs knee sweep dirs to merge")
    ap.add_argument("--title", default=None,
                    help="override plot title (e.g. for the bootstrap trace)")
    ap.add_argument("--ref-rps", type=float, default=SAFE_RPS,
                    help="ref-op vertical line; <=0 hides it")
    args = ap.parse_args()

    rows = []
    for d in args.sweep_dirs:
        rows += load_cells(d)
    if not rows:
        sys.exit(f"[error] no cells found in {args.sweep_dirs}")
    print(f"[info] {len(rows)} cells loaded")
    for bs in sorted({r[0] for r in rows}):
        cells = sorted([r for r in rows if r[0] == bs], key=lambda r: r[2])
        print(f"  bs={bs}:")
        for _, scale, offered, achieved, p99 in cells:
            print(f"    s={scale:>5.2f}  offered={offered:>5.2f}  "
                  f"achieved={achieved:>5.2f}  e2e_p99={p99/1000.0:>8.1f}s")
    name = args.out_name + ("_linear" if args.linear else "")
    render(rows, REPO / "thesis_plotting/figures", name, args.linear,
           title=args.title, ref_rps=args.ref_rps)


if __name__ == "__main__":
    main()
