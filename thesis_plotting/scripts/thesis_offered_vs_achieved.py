#!/usr/bin/env python3
"""Offered RPS vs achieved RPS for the n=5k fine-cliff sweep.

Reads each per-cell requests_out.csv and computes:
  offered  = arrivals_count / arrival_span_seconds
  achieved = completed_count / completion_span_seconds

Output:
    thesis_plotting/figures/fig_offered_vs_achieved_pi0_wide_n5k.{pdf,png}

The plot makes the non-monotonic switching region explicit: cells in the
offered range ~1.9-2.1 RPS land in either a drain attractor (achieved ≈
offered) or a collapsed overflow attractor (achieved ≈ 0.8 RPS); below
the switching region cells track the identity line, and above it cells
plateau at the measured 2.13 RPS service ceiling.
"""

import sys
from pathlib import Path
import re

import pandas as pd
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from thesis_plotting.style import (  # noqa: E402
    configure_plotting, save_fig, set_spines, bold_legend,
    COLOR_HEAVY, FONT_LABEL, FONT_TITLE, FONT_ANNOTATE,
)

SWEEP_DIR = REPO / "cluster_outputs/online_serving_runs/rps_knee_pi0_wide_fine_n5k_20260602_183013"
OUT_NAME  = "fig_offered_vs_achieved_pi0_wide_n5k"

CEILING        = 2.13   # measured saturation ceiling (mean throughput)
SAFE_OPERATING = 1.36   # measured safe operating point (= star on the knee plot)


def collect(sweep_dir):
    rows = []
    for cell in sorted(sweep_dir.iterdir()):
        if not cell.is_dir():
            continue
        run_info = cell / "run_info.txt"
        try:
            scale = float(re.search(r"arrival_scale=([\d.]+)",
                                    run_info.read_text()).group(1))
        except Exception:
            continue
        csv = cell / "max_util_full" / "requests_out.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        if df.empty:
            continue
        offered = len(df) / (df["arrival_ms"].max() / 1000.0)
        done = df[df["state"] == "done"]
        if done.empty:
            continue
        achieved = len(done) / (done["completion_ms"].max() / 1000.0)
        rows.append((scale, offered, achieved))
    return sorted(rows)


def classify(offered, achieved):
    """Throughput-deficit classifier mirroring the knee plot's logic."""
    expected = min(offered, CEILING)
    if expected <= 0:
        return "drain"
    deficit = (expected - achieved) / expected
    return "overflow" if deficit > 0.30 else "drain"


def render(pts):
    configure_plotting()
    fig, ax = plt.subplots(figsize=(11.5, 7.0))

    # Sort all cells by offered RPS so the line shows the zig-zag through
    # the non-monotonic switching region.
    pts_sorted = sorted(pts, key=lambda t: t[1])
    xs = [p[1] for p in pts_sorted]
    ys = [p[2] for p in pts_sorted]
    modes = [classify(p[1], p[2]) for p in pts_sorted]

    max_offered = max(xs) * 1.05

    # Identity (perfect drain) reference.
    ax.plot([0, max_offered], [0, max_offered],
            ls="--", color="#888888", lw=1.0, zorder=2,
            label="$y=x$  (achieved $=$ offered)")

    # Saturation ceiling reference.
    ax.axhline(CEILING, color="#c0392b", linestyle=":",
               lw=1.4, zorder=3)
    ax.text(0.985, CEILING, "  ceiling = 2.13 RPS",
            transform=ax.get_yaxis_transform(),
            ha="right", va="bottom",
            fontsize=FONT_ANNOTATE + 1, color="#c0392b", fontweight="bold")

    # Shaded operating-region bands by offered-RPS axis.
    # Use empirical chaos-zone offered-RPS bounds from sweep cell labels:
    # s ∈ [2.65, 2.85] → offered ∈ [≈1.93, ≈2.08].
    ax.axvspan(0, SAFE_OPERATING,    color="#90ee90", alpha=0.10, zorder=1)
    ax.axvspan(SAFE_OPERATING, 1.93, color="#ffe680", alpha=0.10, zorder=1)
    ax.axvspan(1.93, 2.08,           color="#ffa500", alpha=0.15, zorder=1)
    ax.axvspan(2.08, max_offered,    color="#c44e52", alpha=0.10, zorder=1)

    # Single connected polyline through ALL cells in offered-RPS order.
    # This is what makes the non-monotonic jumps visible: the line dives
    # from ≈ 2.0 down to ≈ 0.8 and back as neighboring offered loads flip
    # between drain and overflow attractors.
    ax.plot(xs, ys, "-", color=COLOR_HEAVY[2], lw=1.8, zorder=5,
            label="achieved throughput  (line connects neighboring cells)")

    # Markers colored by attractor.
    drain_x   = [x for x, m in zip(xs, modes) if m == "drain"]
    drain_y   = [y for y, m in zip(ys, modes) if m == "drain"]
    over_x    = [x for x, m in zip(xs, modes) if m == "overflow"]
    over_y    = [y for y, m in zip(ys, modes) if m == "overflow"]
    ax.scatter(drain_x, drain_y, marker="o", s=70, color=COLOR_HEAVY[2],
               edgecolor="black", linewidth=0.5, zorder=6,
               label="drain attractor")
    if over_x:
        ax.scatter(over_x, over_y, marker="o", s=90, color=COLOR_HEAVY[1],
                   edgecolor="black", linewidth=0.5, zorder=7,
                   label="overflow attractor  ($\\approx 0.8$ RPS)")

    # Mark the operating-point regions with text annotations.
    y_anno = 0.06
    ax.text(SAFE_OPERATING / 2, y_anno, "safe zone",
            transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=FONT_ANNOTATE,
            fontweight="bold", color="#1f5b1f",
            bbox=dict(facecolor="white", edgecolor="#4a8a4a",
                      boxstyle="round,pad=0.3", linewidth=0.6, alpha=0.95))
    ax.text((SAFE_OPERATING + 1.93) / 2, y_anno, "cliff approach",
            transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=FONT_ANNOTATE,
            fontweight="bold", color="#a07050",
            bbox=dict(facecolor="white", edgecolor="#c08050",
                      boxstyle="round,pad=0.3", linewidth=0.6, alpha=0.95))
    ax.text(2.005, 0.94, "non-monotonic\nswitching region",
            transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=FONT_ANNOTATE,
            fontweight="bold", color="#a07050",
            bbox=dict(facecolor="white", edgecolor="#c08050",
                      boxstyle="round,pad=0.3", linewidth=0.7, alpha=0.95))
    ax.text((2.08 + max_offered) / 2, y_anno, "stable saturation\n(achieved $\\approx 2.13$ RPS)",
            transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=FONT_ANNOTATE,
            fontweight="bold", color="#a04040",
            bbox=dict(facecolor="white", edgecolor="#c44e52",
                      boxstyle="round,pad=0.3", linewidth=0.6, alpha=0.95))

    ax.set_xlabel("Offered RPS  (arrivals / arrival span)",
                  fontsize=FONT_LABEL + 1, fontweight="bold")
    ax.set_ylabel("Achieved RPS  (completions / completion span)",
                  fontsize=FONT_LABEL + 1, fontweight="bold")
    ax.set_title(
        "Pi0 + azure_poisson_wide v2 — offered vs achieved RPS  "
        "($N=5000$, fine cliff sweep)\n"
        "Identity line = perfect drain; ceiling at 2.13 RPS; "
        "switching-region cells split between DRAIN and OVERFLOW attractors",
        fontsize=FONT_TITLE, fontweight="bold",
    )
    ax.set_xlim(0, max_offered)
    ax.set_ylim(0, max(CEILING, max(p[2] for p in pts)) * 1.10)
    ax.grid(True, alpha=0.4)
    set_spines(ax)
    leg = ax.legend(loc="upper left", fontsize=FONT_LABEL - 1)
    bold_legend(leg)

    fig.tight_layout()
    save_fig(fig, OUT_NAME, REPO / "thesis_plotting/figures")
    plt.close(fig)


def main():
    pts = collect(SWEEP_DIR)
    print(f"[info] {len(pts)} cells from {SWEEP_DIR.name}")
    print()
    print(f"  {'scale':>6}  {'offered':>8}  {'achieved':>9}  mode")
    for s, o, a in pts:
        print(f"  {s:>6.3f}  {o:>8.3f}  {a:>9.3f}  {classify(o, a)}")
    render(pts)


if __name__ == "__main__":
    main()
