#!/usr/bin/env python3
"""Pi0 @ azure_poisson_wide v2 — hybrid vs gpu-only at s=5 (≈ s=4 proxy).

The original G figure (`fig_g_pi0_knee_azure_poisson_wide.png`) was
rendered at arrival_scale=6.5 (peak ≈ a=2.13). The system has metastable
bistability in scale 2.55–2.65 and an unstable cliff at scale 2.7, so
"operating at a" is unsafe in practice.

This re-render uses arrival_scale=5.0 (peak ≈ a=1.09), which sits at
the closest available G-comparison cell below the cliff. The original
6.5 figure is preserved untouched.

Source cells (same dataset n=5000, same canonical config):
- hybrid:   cluster_outputs/online_serving_runs/knee_vs_route_20260601_151039/03_route_hybrid_s050
- gpu-only: cluster_outputs/online_serving_runs/knee_vs_route_20260601_151039/09_route_gpu_only_s050

Output: thesis_plotting/figures/fig_g_pi0_knee_azure_poisson_wide_s5.{pdf,png}
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO))

from plot_sweep_compare import parse_summary  # noqa: E402

from thesis_plotting.style import (  # noqa: E402
    configure_plotting, save_fig, set_spines,
    COLOR_ROUTE, FONT_LABEL, FONT_TITLE, FONT_TICK,
)


HYBRID_CELL = (REPO / "cluster_outputs/online_serving_runs"
               / "knee_vs_route_20260601_151039"
               / "03_route_hybrid_s050")
GPUONLY_CELL = (REPO / "cluster_outputs/online_serving_runs"
                / "knee_vs_route_20260601_151039"
                / "09_route_gpu_only_s050")

ARRIVAL_SCALE = 5.0
DATASET_LABEL = "azure_poisson_wide  (Poisson, mixture v2)"
MODEL_DISPLAY = "Pi0"

MODE_LABEL = {"hybrid": "Hybrid", "gpuonly": "GPU-only"}
MODE_COLOR = {"hybrid": COLOR_ROUTE["lpddr5_pim_bank"],
              "gpuonly": COLOR_ROUTE["gpu_only"]}


def load_cell(cell_dir: Path) -> dict | None:
    sumfile = cell_dir / "policy_compare_summary.txt"
    if not sumfile.is_file():
        inner = next((c / "policy_compare_summary.txt"
                      for c in cell_dir.iterdir() if c.is_dir()), None)
        if inner and inner.is_file():
            sumfile = inner
    if not sumfile.is_file():
        return None
    return parse_summary(sumfile)


def render(cells: dict, out_dir: Path, out_name: str):
    configure_plotting()
    if not cells.get("hybrid") or not cells.get("gpuonly"):
        sys.exit("[error] missing hybrid or gpuonly cell")

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))

    metrics = [
        ("throughput", "Throughput  (req/s)",  lambda v: v,
         False, "higher"),
        ("ttft_p99",   "TTFT p99  (ms)",       lambda v: v,
         True,  "lower"),
        ("e2e_p99",    "E2E p99  (s)",         lambda v: v / 1000.0,
         True,  "lower"),
    ]
    modes = ["gpuonly", "hybrid"]
    xs = np.array([0.35, 0.65])

    for col, (key, ylabel, transform, log, winning) in enumerate(metrics):
        ax = axes[col]
        vals = []
        for mode in modes:
            v = cells[mode].get(key)
            vals.append(transform(v) if v is not None else 0.0)

        colors = [MODE_COLOR[m] for m in modes]
        bars = ax.bar(xs, vals, width=0.28, color=colors,
                      edgecolor="black", linewidth=0.5, zorder=3)
        if log:
            ax.set_yscale("log")

        gpu_v, hyb_v = vals
        if gpu_v > 0:
            delta_signed = (hyb_v - gpu_v) / gpu_v * 100.0
            if abs(delta_signed) < 0.5:
                label_str = "= 0%"
                color = "#5a5a5a"
            else:
                arrow = "↑" if delta_signed > 0 else "↓"
                label_str = f"{arrow}{abs(delta_signed):.0f}%"
                hybrid_wins = ((winning == "lower" and delta_signed < 0)
                               or (winning == "higher" and delta_signed > 0))
                color = "#2ca02c" if hybrid_wins else "#c44e52"
            top_y = max(vals)
            ax.text(0.5, top_y * (2.0 if log else 1.18),
                    label_str,
                    ha="center", va="bottom",
                    fontsize=FONT_LABEL + 4, fontweight="bold",
                    color=color)

        for bar, v in zip(bars, vals):
            label = f"{v:.2f}" if v < 100 else f"{v:.0f}"
            if log:
                y_lab = bar.get_height() * 1.07
            else:
                y_lab = bar.get_height() + max(vals) * 0.015
            ax.text(bar.get_x() + bar.get_width() / 2, y_lab, label,
                    ha="center", va="bottom",
                    fontsize=FONT_LABEL, fontweight="bold")

        ax.set_xticks(xs)
        ax.set_xticklabels([MODE_LABEL[m] for m in modes],
                           fontsize=FONT_TICK, fontweight="bold")
        ax.set_xlim(0.10, 0.90)
        ax.set_ylabel(ylabel, fontsize=FONT_LABEL, fontweight="bold")
        if log:
            ax.set_ylim(top=max(vals) * 4.5)
        else:
            ax.set_ylim(0, max(vals) * 1.32)
        set_spines(ax)

    # Title encodes the s=4 derivation while honestly noting s=5 source.
    sup = (f"{MODEL_DISPLAY} @ safe operating point   "
           f"({DATASET_LABEL}, arrival_scale = {ARRIVAL_SCALE:g} ≈ s=4 proxy)")
    fig.suptitle(sup, fontsize=FONT_TITLE, fontweight="bold", y=1.00)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    save_fig(fig, out_name, out_dir)
    plt.close(fig)


def main():
    cells = {
        "hybrid":  load_cell(HYBRID_CELL),
        "gpuonly": load_cell(GPUONLY_CELL),
    }
    for mode, c in cells.items():
        if c is None:
            sys.exit(f"[error] could not parse {mode} cell")
        print(f"[info] {mode}: thru={c.get('throughput'):.2f} "
              f"ttft_p99={c.get('ttft_p99'):.0f}ms "
              f"e2e_p99={c.get('e2e_p99')/1000.0:.1f}s")
    render(cells, REPO / "thesis_plotting/figures",
           "fig_g_pi0_knee_azure_poisson_wide_s5")


if __name__ == "__main__":
    main()
