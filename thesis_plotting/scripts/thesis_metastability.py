#!/usr/bin/env python3
"""Pi0 + azure_poisson_wide v2 — metastable bistability near the cliff.

The dense knee sweep at scales 2.4-2.7 revealed two stable attractors:
- DRAIN mode: throughput ≈ ceiling (2.13), p99 in seconds
- OVERFLOW mode: throughput collapses to 1.25, p99 in 25-minute range

Between scale=2.65 and 2.50 (a window of ~6% in arrival rate), the
system bistability is sharp and discontinuous. Source sweep:
`rps_knee_pi0_20260601_183211`.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from thesis_plotting.style import (
    configure_plotting, save_fig, set_spines, bold_legend,
    COLOR_HEAVY, FONT_LABEL, FONT_TITLE, FONT_TICK, FONT_ANNOTATE,
)


# Raw data extracted from the 13270702 sweep dir.
DATA = [
    # (scale, rps, ttft_p99_ms, e2e_p99_ms, e2e_p50_ms, span_s, mode)
    (2.70, 2.01,   25387,   465280,   7589, 2486, "drain"),
    (2.65, 1.25, 1512649,  1542527,  27497, 3991, "overflow"),
    (2.60, 1.25, 1554509,  1584637,  45937, 3987, "overflow"),
    (2.55, 1.53,  874879,   905242,  46594, 3262, "partial"),
    (2.50, 2.13,   88924,    91661,  62904, 2349, "drain"),
    (2.45, 2.13,  113088,   116605,  94393, 2347, "drain"),
    (2.40, 2.13,  148402,   155454, 120964, 2346, "drain"),
]


def main():
    configure_plotting()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    scales = [d[0] for d in DATA]
    rps = [d[1] for d in DATA]
    p99 = [d[3] / 1000.0 for d in DATA]
    p50 = [d[4] / 1000.0 for d in DATA]

    drain_color = COLOR_HEAVY[0]    # deep teal
    overflow_color = COLOR_HEAVY[1] # rust
    partial_color = COLOR_HEAVY[3]  # amber

    color_map = {
        "drain": drain_color,
        "overflow": overflow_color,
        "partial": partial_color,
    }

    # ── Panel A: throughput vs arrival_scale ──
    for s, r, _, _, _, _, mode in DATA:
        ax1.scatter([s], [r], s=180, color=color_map[mode],
                    edgecolor="black", linewidth=0.7, zorder=4)
    # Connecting line
    ax1.plot(scales, rps, "-", color="gray", lw=1.5, alpha=0.6, zorder=2)
    # Reference lines
    ax1.axhline(2.13, color=drain_color, ls="--", lw=1.2, alpha=0.6,
                label="drain ceiling (2.13 req/s)")
    ax1.axhline(1.25, color=overflow_color, ls="--", lw=1.2, alpha=0.6,
                label="overflow plateau (1.25 req/s)")
    ax1.set_xlabel("arrival_scale  (smaller = more load)",
                   fontsize=FONT_LABEL, fontweight="bold")
    ax1.set_ylabel("Achieved throughput  (req/s)",
                   fontsize=FONT_LABEL, fontweight="bold")
    ax1.set_title("(a) Throughput bistability",
                  fontsize=FONT_TITLE - 1, fontweight="bold")
    ax1.set_xticks(scales)
    ax1.set_xticklabels([f"{s:.2f}" for s in scales],
                        fontsize=FONT_TICK - 1, rotation=45)
    ax1.set_ylim(0.5, 2.4)
    ax1.invert_xaxis()  # so higher load (lower scale) is to the right
    ax1.grid(True, alpha=0.4)
    bold_legend(ax1.legend(loc="lower right", fontsize=FONT_LABEL - 1))
    set_spines(ax1)

    # ── Panel B: latency on log scale ──
    for s, _, _, p99_ms, p50_ms, _, mode in DATA:
        ax2.scatter([s], [p99_ms / 1000.0], s=180, color=color_map[mode],
                    marker="*", edgecolor="black", linewidth=0.7, zorder=5)
        ax2.scatter([s], [p50_ms / 1000.0], s=80, color=color_map[mode],
                    marker="o", edgecolor="black", linewidth=0.4, zorder=4)
    ax2.plot(scales, p99, "-", color="gray", lw=1.5, alpha=0.5, zorder=2)
    ax2.plot(scales, p50, "--", color="gray", lw=1.0, alpha=0.5, zorder=2)
    ax2.set_yscale("log")
    ax2.set_xlabel("arrival_scale",
                   fontsize=FONT_LABEL, fontweight="bold")
    ax2.set_ylabel("E2E latency  (s, log scale)",
                   fontsize=FONT_LABEL, fontweight="bold")
    ax2.set_title("(b) Latency — stars=p99, circles=p50",
                  fontsize=FONT_TITLE - 1, fontweight="bold")
    ax2.set_xticks(scales)
    ax2.set_xticklabels([f"{s:.2f}" for s in scales],
                        fontsize=FONT_TICK - 1, rotation=45)
    ax2.invert_xaxis()
    ax2.grid(True, alpha=0.4)
    set_spines(ax2)

    # Build a manual legend for the modes.
    from matplotlib.lines import Line2D
    mode_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=drain_color,
               markersize=11, markeredgecolor="black", markeredgewidth=0.7,
               label="DRAIN mode  (throughput = 2.13 / 1.53)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=overflow_color,
               markersize=11, markeredgecolor="black", markeredgewidth=0.7,
               label="OVERFLOW mode  (throughput = 1.25)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=partial_color,
               markersize=11, markeredgecolor="black", markeredgewidth=0.7,
               label="partial recovery (rps=1.53)"),
    ]
    bold_legend(ax2.legend(handles=mode_handles, loc="lower right",
                            fontsize=FONT_LABEL - 1))

    fig.suptitle(
        "Pi0 @ azure_poisson_wide v2 — metastable bistability at the cliff edge\n"
        "between scale=2.65 and scale=2.50 (rps 1.25–2.13)",
        fontsize=FONT_TITLE, fontweight="bold", y=1.02)

    fig.tight_layout()
    out_dir = REPO / "thesis_plotting/figures"
    save_fig(fig, "fig_metastability_pi0_wide", out_dir)
    plt.close(fig)


if __name__ == "__main__":
    main()
