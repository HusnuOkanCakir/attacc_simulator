#!/usr/bin/env python3
"""Architecture-level GPU vs PIM comparison from the calibrated cost models.

This figure isolates the per-shape hardware time and energy from the
scheduler: for each (Lin, Lout) request shape at bs=1, the cost tables
predict prefill time, per-token decode time, prefill energy, and
per-token decode energy on the GPU-only route and on the GPU+PIM
(`lpddr5_pim_bank`) route. The thesis chapter uses this to motivate
why the `min_finish` scheduler routes long-context requests to PIM
even though prefill is identical on both routes.

Output:
    thesis_plotting/figures/fig_route_arch_compare_pi0_a6000.{pdf,png}

Layout (2×2):
    (a) decode time speedup    — GPU / PIM, log color, > 1 means PIM faster
    (b) decode energy ratio    — PIM / GPU, log color, > 1 means PIM hotter
    (c) total-request time speedup at bs=1, with three representative
        shape markers overlaid
    (d) total-request energy ratio, same markers

Total request cost is composed from the cost tables as
    total_time   = prefill_ms + Lout * decode_ms_per_token
    total_energy = prefill_nJ + Lout * decode_nJ_per_token
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from thesis_plotting.style import (  # noqa: E402
    configure_plotting, save_fig, set_spines,
    FONT_LABEL, FONT_TITLE, FONT_ANNOTATE, FONT_TICK,
)

# Energy-fixed, A6000-target cost tables (single-module LPDDR5-PIM
# energy accounting + GDDR6 GPU memory energy). Generated on the
# cluster by tools/run_cost_table_sweep.sbatch with
# OUT_DIR=cost_tables_energyfix_pi0_a6000.
GPU_CSV = REPO / "cluster_outputs/cost_tables_energyfix_pi0_a6000/gpu_only.csv"
PIM_CSV = REPO / "cluster_outputs/cost_tables_energyfix_pi0_a6000/lpddr5_pim_bank.csv"
BS      = 1

# Three representative shapes chosen to span the azure_poisson_wide
# workload regimes. Lout=208 is the maximum sampled in the cost tables.
SHAPES = [
    ("Short",  512,  64),
    ("Medium", 2048, 192),
    ("Long",   5120, 208),
]


def load(csv_path):
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"g_time (ms)": "g_time_ms",
                            "s_energy (nJ)": "s_energy_nJ",
                            "g_energy (nJ)": "g_energy_nJ"})
    df = df[df["bs"] == BS].copy()
    return df[["Lin", "Lout", "s_time", "g_time_ms",
               "s_energy_nJ", "g_energy_nJ"]]


def grid(df, value_col):
    p = df.pivot_table(index="Lout", columns="Lin", values=value_col,
                       aggfunc="mean").sort_index().sort_index(axis=1)
    return p


def diverging_log_norm(data, max_dev=None):
    """Symmetric-log color norm around 1.0 for ratio heatmaps."""
    a = np.log2(data)
    if max_dev is None:
        max_dev = float(np.nanmax(np.abs(a)))
        max_dev = max(max_dev, 0.5)
    return mcolors.Normalize(vmin=-max_dev, vmax=max_dev)


# Local font overrides — defaults (FONT_BASE=7) render ~3pt after LaTeX
# scales the figure to \linewidth. Bump for readability.
FL  = FONT_LABEL    + 8   # 15  axis label / colorbar label
FTI = FONT_TITLE    + 8   # 16  panel title
FTK = FONT_TICK     + 8   # 14  ticks
FAN = FONT_ANNOTATE + 10  # 15  shape marker label


def imshow_log_ratio(ax, ratio_df, title, cbar_label, max_dev=None):
    log_ratio = np.log2(ratio_df.values)
    if max_dev is None:
        max_dev = float(np.nanmax(np.abs(log_ratio)))
        max_dev = max(max_dev, 0.5)
    im = ax.imshow(log_ratio, origin="lower", aspect="auto",
                   cmap="RdBu_r", vmin=-max_dev, vmax=max_dev,
                   extent=[ratio_df.columns.min(), ratio_df.columns.max(),
                           ratio_df.index.min(), ratio_df.index.max()])
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(cbar_label, fontsize=FL, fontweight="bold")

    # Convert tick labels back to linear ratio.
    ticks = cbar.get_ticks()
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{2**t:.2f}×" for t in ticks])
    cbar.ax.tick_params(labelsize=FTK)

    ax.set_xlabel("Input context  $L_{in}$  (tokens)",
                  fontsize=FL, fontweight="bold")
    ax.set_ylabel("Output length  $L_{out}$  (tokens)",
                  fontsize=FL, fontweight="bold")
    ax.set_title(title, fontsize=FTI, fontweight="bold", loc="left")
    ax.tick_params(labelsize=FTK)
    set_spines(ax)
    return im


def mark_shapes(ax, shapes, color="black"):
    for label, lin, lout in shapes:
        ax.scatter([lin], [lout], marker="o", s=240, facecolor="white",
                   edgecolor=color, linewidth=2.2, zorder=5)
        ax.text(lin, lout, f"  {label}", fontsize=FAN,
                fontweight="bold", color=color, va="center", ha="left",
                bbox=dict(facecolor="white", edgecolor="none",
                          pad=1.5, alpha=0.85))


def main():
    configure_plotting()

    gpu = load(GPU_CSV)
    pim = load(PIM_CSV)
    merged = gpu.merge(pim, on=["Lin", "Lout"], suffixes=("_gpu", "_pim"))

    # Per-token decode metrics.
    merged["decode_speedup_pim_over_gpu"] = (
        merged["g_time_ms_gpu"] / merged["g_time_ms_pim"]
    )
    merged["decode_energy_ratio_pim_over_gpu"] = (
        merged["g_energy_nJ_pim"] / merged["g_energy_nJ_gpu"]
    )

    # Total request cost at bs=1, composed from the cost tables.
    def total_time(row, prefix):
        return row[f"s_time_{prefix}"] + row["Lout"] * row[f"g_time_ms_{prefix}"]

    def total_energy(row, prefix):
        return (row[f"s_energy_nJ_{prefix}"]
                + row["Lout"] * row[f"g_energy_nJ_{prefix}"])

    merged["total_time_gpu"] = merged.apply(lambda r: total_time(r, "gpu"), axis=1)
    merged["total_time_pim"] = merged.apply(lambda r: total_time(r, "pim"), axis=1)
    merged["total_speedup_pim_over_gpu"] = (
        merged["total_time_gpu"] / merged["total_time_pim"]
    )
    merged["total_energy_gpu"] = merged.apply(lambda r: total_energy(r, "gpu"), axis=1)
    merged["total_energy_pim"] = merged.apply(lambda r: total_energy(r, "pim"), axis=1)
    merged["total_energy_ratio_pim_over_gpu"] = (
        merged["total_energy_pim"] / merged["total_energy_gpu"]
    )

    g_decode_time   = grid(merged, "decode_speedup_pim_over_gpu")
    g_decode_energy = grid(merged, "decode_energy_ratio_pim_over_gpu")
    g_total_time    = grid(merged, "total_speedup_pim_over_gpu")
    g_total_energy  = grid(merged, "total_energy_ratio_pim_over_gpu")

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 11.0),
                             constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.3, h_pad=0.6,
                                    hspace=0.08, wspace=0.08)
    (ax_a, ax_b), (ax_c, ax_d) = axes

    imshow_log_ratio(
        ax_a, g_decode_time,
        title="(a) Per-token decode time speedup\n(PIM faster $\\rightarrow$ red)",
        cbar_label="GPU / PIM  (×)",
        max_dev=1.0,
    )

    imshow_log_ratio(
        ax_b, g_decode_energy,
        title="(b) Per-token decode energy ratio\n(red $>1$: PIM hotter; blue $<1$: PIM cooler)",
        cbar_label="PIM / GPU  (×)",
        max_dev=1.0,
    )

    imshow_log_ratio(
        ax_c, g_total_time,
        title="(c) Total request time speedup\n(prefill + $L_{out}\\cdot$ decode)",
        cbar_label="GPU / PIM  (×)",
        max_dev=1.0,
    )
    mark_shapes(ax_c, SHAPES)

    imshow_log_ratio(
        ax_d, g_total_energy,
        title="(d) Total request energy ratio\n(red $>1$: PIM hotter; blue $<1$: PIM cooler)",
        cbar_label="PIM / GPU  (×)",
        max_dev=1.0,
    )
    mark_shapes(ax_d, SHAPES)

    fig.suptitle(
        "Architectural per-shape comparison: Pi0 on A6000 cost models, bs=1",
        fontsize=FTI + 4, fontweight="bold",
    )

    out_dir = REPO / "thesis_plotting/figures"
    save_fig(fig, "fig_route_arch_compare_pi0_a6000", out_dir)
    plt.close(fig)

    # ── Sanity print: representative-shape table ──────────────────────────
    print()
    print(f"{'shape':<8} {'Lin':>5} {'Lout':>5} "
          f"{'g_GPU':>10} {'g_PIM':>10} {'speedup':>9} "
          f"{'e_GPU':>11} {'e_PIM':>11} {'e_ratio':>9}")
    for label, lin, lout in SHAPES:
        r = merged[(merged["Lin"] == lin) & (merged["Lout"] == lout)]
        if r.empty:
            print(f"{label:<8} {lin:>5} {lout:>5}   (no row in cost table)")
            continue
        r = r.iloc[0]
        print(f"{label:<8} {lin:>5} {lout:>5} "
              f"{r['g_time_ms_gpu']:>9.3f}m {r['g_time_ms_pim']:>9.3f}m "
              f"{r['decode_speedup_pim_over_gpu']:>8.2f}× "
              f"{r['g_energy_nJ_gpu']:>10.2e} {r['g_energy_nJ_pim']:>10.2e} "
              f"{r['decode_energy_ratio_pim_over_gpu']:>8.2f}×")
    print()
    print(f"grid coverage: Lin ∈ [{merged.Lin.min()}, {merged.Lin.max()}], "
          f"Lout ∈ [{merged.Lout.min()}, {merged.Lout.max()}], "
          f"{len(merged)} cells")


if __name__ == "__main__":
    main()
