#!/usr/bin/env python3
"""6-cell cross-regime tail anatomy on the fine-cliff n=5k sweep.

For each of 6 representative cells (deep sub-cliff → safe a → last
clean drain → non-monotonic drain → non-monotonic overflow → stable saturation),
reproduce the 3-panel tail-anatomy view and stack them into a single
6×3 figure.

Output: thesis_plotting/figures/fig_tail_anatomy_compare_pi0_wide.{pdf,png}
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

from analyze_request_tail import (  # noqa: E402
    load_cell,
)

SWEEP = REPO / "cluster_outputs/online_serving_runs/rps_knee_pi0_wide_fine_n5k_20260602_183013"

# (cell-dir-basename, regime-label)
CELLS = [
    ("01_azure_poisson_wide_s1000", "(1) deep sub-cliff"),
    ("04_azure_poisson_wide_s0400", "(2) safe point"),
    ("05_azure_poisson_wide_s0350", "(3) last clean drain"),
    ("10_azure_poisson_wide_s0269", "(4) non-monotonic drain"),
    ("09_azure_poisson_wide_s0270", "(5) non-monotonic overflow"),
    ("21_azure_poisson_wide_s0250", "(6) stable saturation"),
]
TOP_N = 50
POLICY = "max_util_full"


def main() -> None:
    fig, axes = plt.subplots(len(CELLS), 3,
                             figsize=(20, 4.0 * len(CELLS)),
                             squeeze=False)

    for row_idx, (cell_name, regime_label) in enumerate(CELLS):
        run_dir = SWEEP / cell_name
        if not run_dir.is_dir():
            print(f"[skip] {cell_name}: not found")
            continue
        d, summary, _meta = load_cell(run_dir, POLICY)
        top = d.nlargest(TOP_N, "e2e_ms")
        rps = summary.get("throughput", "?")
        e2e_p99_s = summary.get("e2e_p99", float("nan")) / 1000

        # ── Panel A: arrival vs e2e, color = context_tokens
        ax = axes[row_idx][0]
        sc = ax.scatter(
            d.arrival_ms / 1000, d.e2e_ms / 1000,
            c=d.context_tokens, s=10, alpha=0.55,
            cmap="viridis",
            norm=mcolors.LogNorm(
                vmin=max(1, d.context_tokens.min()),
                vmax=max(2, d.context_tokens.max()),
            ),
            edgecolors="none", zorder=2,
        )
        ax.scatter(top.arrival_ms / 1000, top.e2e_ms / 1000,
                   marker="o", facecolor="none", edgecolor="#c44e52",
                   s=50, linewidth=0.9, zorder=3)
        ax.set_ylabel("e2e (s)", fontweight="bold")
        if row_idx == 0:
            ax.set_title("A. (arrival, e2e) — color = ctx", fontweight="bold")
        ax.grid(True, alpha=0.4)
        cbar = plt.colorbar(sc, ax=ax, pad=0.01)
        cbar.set_label("ctx", fontsize=8)

        # ── Panel B: context vs e2e, color = route
        ax = axes[row_idx][1]
        pim = d[d.route == "lpddr5_pim_bank"]
        gpu = d[d.route == "gpu_only"]
        ax.scatter(pim.context_tokens, pim.e2e_ms / 1000,
                   c="#1f77b4", s=10, alpha=0.45,
                   edgecolors="none", zorder=2,
                   label=f"PIM (n={len(pim)})")
        ax.scatter(gpu.context_tokens, gpu.e2e_ms / 1000,
                   c="#d62728", s=10, alpha=0.55,
                   edgecolors="none", zorder=2,
                   label=f"GPU (n={len(gpu)})")
        ax.scatter(top.context_tokens, top.e2e_ms / 1000,
                   marker="o", facecolor="none", edgecolor="black",
                   s=44, linewidth=0.7, zorder=3,
                   label=f"top-{TOP_N}")
        if row_idx == 0:
            ax.set_title("B. (context, e2e) — color = route",
                         fontweight="bold")
        ax.grid(True, alpha=0.4)
        ax.legend(loc="upper left", fontsize=8)
        # GPU share annotation
        gpu_overall = (d.route == "gpu_only").mean() * 100
        gpu_in_top = (top.route == "gpu_only").mean() * 100
        ax.text(0.02, 0.96,
                f"GPU share:\noverall {gpu_overall:.1f}%\nin top-{TOP_N}: {gpu_in_top:.0f}%",
                transform=ax.transAxes, fontsize=8, fontweight="bold",
                ha="left", va="top",
                bbox=dict(facecolor="white", edgecolor="#888",
                          boxstyle="round,pad=0.25", linewidth=0.5,
                          alpha=0.95))

        # ── Panel C: boxplot e2e by arrival decile
        ax = axes[row_idx][2]
        data_per_decile = [d[d.arr_decile == i].e2e_ms / 1000
                            for i in range(10)]
        bp = ax.boxplot(data_per_decile, positions=range(10),
                        widths=0.7, showfliers=True, patch_artist=True)
        for box in bp["boxes"]:
            box.set_facecolor("#90c0e0")
            box.set_alpha(0.7)
        # Highlight top-N
        for i in range(10):
            in_decile = top[top.arr_decile == i]
            if len(in_decile):
                ax.scatter(np.full(len(in_decile), i),
                           in_decile.e2e_ms / 1000,
                           color="#c44e52", marker="o", s=24,
                           zorder=3, alpha=0.85)
        ax.set_yscale("log")
        ax.set_xticks(range(10))
        if row_idx == 0:
            ax.set_title("C. e2e by arrival decile (red = top-N)",
                         fontweight="bold")
        ax.grid(True, alpha=0.4, which="both")

        # ── Row label on the leftmost panel
        row_label = (f"{regime_label}\n"
                     f"RPS={rps}   "
                     f"e2e p99={e2e_p99_s:.0f}s")
        axes[row_idx][0].text(
            -0.32, 0.5, row_label,
            transform=axes[row_idx][0].transAxes,
            ha="center", va="center",
            fontsize=11, fontweight="bold", rotation=90,
            color="#333",
        )

    fig.suptitle(
        "Pi0 + azure_poisson_wide v2 (n=5,000)  —  per-cell tail anatomy "
        "across 6 regimes\n"
        "Rows: deep sub-cliff → safe point → clean drain → non-monotonic drain → non-monotonic overflow → stable saturation",
        fontsize=14, fontweight="bold", y=0.998,
    )
    fig.tight_layout(rect=(0.025, 0, 1, 0.98))

    out_dir = REPO / "thesis_plotting/figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / "fig_tail_anatomy_compare_pi0_wide.pdf"
    png = out_dir / "fig_tail_anatomy_compare_pi0_wide.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=130)
    plt.close(fig)
    print(f"[pdf] {pdf}")
    print(f"[png] {png}")


if __name__ == "__main__":
    main()
