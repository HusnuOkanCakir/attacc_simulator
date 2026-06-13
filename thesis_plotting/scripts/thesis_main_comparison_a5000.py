#!/usr/bin/env python3
"""Pi0 @ azure_poisson_wide v2 — hybrid vs gpu-only at the new
max_active=5000 default and the new ref operating point.

Source cells (same dataset n=5000, max_active=5000, arrival_scale=2.85,
mean offered ≈ 1.91 RPS, kv_pool=64 GiB):
- hybrid:   cluster_outputs/online_serving_runs/main_comparison_a5000_20260605_235416/A_hybrid_a5000
- gpu-only: cluster_outputs/online_serving_runs/main_comparison_a5000_20260605_235416/B_gpu_only_a5000

Output: thesis_plotting/figures/fig_main_comparison_a5000_pi0_wide.{pdf,png}
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
               / "main_comparison_a5000_20260605_235416"
               / "A_hybrid_a5000")
GPUONLY_CELL = (REPO / "cluster_outputs/online_serving_runs"
                / "main_comparison_a5000_20260605_235416"
                / "B_gpu_only_a5000")

DATASET_LABEL = "azure_poisson_wide  (Poisson, mixture v2)"
MODEL_DISPLAY = "Pi0"
MEAN_OFFERED_RPS = 1.91

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

    # Local font overrides — defaults (FONT_BASE=7) render ~3pt after LaTeX
    # scales the figure to \linewidth. Bump for readability.
    FL  = FONT_LABEL + 8    # axis label / value label / tick text     -> 15
    FTI = FONT_TITLE + 8    # suptitle                                  -> 16
    FTK = FONT_TICK  + 8    # x-tick labels (GPU-only / Hybrid)         -> 14
    FDL = FONT_LABEL + 10   # delta label (↑21%, ↓7×)                   -> 17

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 5.0))

    metrics = [
        ("throughput", "Throughput  (RPS)",  lambda v: v,
         False, "higher"),
        # TTFT p99: linear y-axis capped at 3000 ms — the two values
        # (~1.6 s each) are within 7% of each other and don't benefit
        # from log scaling.
        ("ttft_p99",   "TTFT p99  (ms)",       lambda v: v,
         False, "lower"),
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
                    fontsize=FDL, fontweight="bold",
                    color=color)

        for bar, v in zip(bars, vals):
            label = f"{v:.2f}" if v < 100 else f"{v:.0f}"
            if log:
                y_lab = bar.get_height() * 1.07
            else:
                y_lab = bar.get_height() + max(vals) * 0.015
            ax.text(bar.get_x() + bar.get_width() / 2, y_lab, label,
                    ha="center", va="bottom",
                    fontsize=FL, fontweight="bold")

        ax.set_xticks(xs)
        ax.set_xticklabels([MODE_LABEL[m] for m in modes],
                           fontsize=FTK, fontweight="bold")
        ax.set_xlim(0.10, 0.90)
        ax.set_ylabel(ylabel, fontsize=FL, fontweight="bold")
        ax.tick_params(axis="y", which="major", labelsize=FTK - 2)
        if key == "ttft_p99":
            # Hand-tuned linear cap so the two close-together bars
            # remain readable with a bit of headroom for the value
            # labels and the delta callout.
            ax.set_ylim(0, 3000)
        elif log:
            ax.set_ylim(top=max(vals) * 4.5)
        else:
            ax.set_ylim(0, max(vals) * 1.32)
        set_spines(ax)

    sup = (f"{MODEL_DISPLAY} @ ref operating point   "
           f"({DATASET_LABEL}, mean offered load ≈ {MEAN_OFFERED_RPS:.2f} RPS)")
    fig.suptitle(sup, fontsize=FTI, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
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
           "fig_main_comparison_a5000_pi0_wide")


if __name__ == "__main__":
    main()
