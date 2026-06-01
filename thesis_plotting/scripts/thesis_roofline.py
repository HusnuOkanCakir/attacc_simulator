#!/usr/bin/env python3
"""Thesis roofline figure: Pi0 attention kernel on A100 GPU vs LPDDR5-PIM.

Reuses the operating-point extraction from tools/plot_roofline.py but
applies the thesis style on top — cleaner palette, fewer series, focus
on the attention-AI=1 story (PIM gets 3.4× more BW for memory-bound
attention).

Output: thesis_plotting/figures/fig_roofline.{pdf,png}
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO))

from plot_roofline import (  # noqa: E402
    load_tables, operating_points,
    GPU_COMPUTE_TFLOPS, GPU_BW_TBS, GPU_RIDGE,
    PIM_ATTN_BW_TBS, PIM_COMPUTE_TFLOPS, PIM_RIDGE,
    AI_ATTN,
)

from thesis_plotting.style import (  # noqa: E402
    configure_plotting, save_fig, set_spines, bold_legend,
    COLOR_ROUTE, FONT_ANNOTATE, FONT_LABEL, FONT_TITLE,
)


DEFAULT_GPU = REPO / "cluster_outputs/cost_tables_full_energy/gpu_only.csv"
DEFAULT_PIM = REPO / "cluster_outputs/cost_tables_full_energy/lpddr5_pim_bank.csv"


def plot_roofline_thesis(pts: dict, batch: int, out_dir: Path, out_name: str):
    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    ai_range = np.logspace(-2, 4, 800)

    GPU_COLOR = COLOR_ROUTE["gpu_only"]
    GPU_DARK  = "#1f3a4e"      # darker for emphasis on the roof line
    PIM_COLOR = "#c44e52"
    PIM_DARK  = "#8b1a1a"

    gpu_roof = np.minimum(GPU_COMPUTE_TFLOPS, GPU_BW_TBS * ai_range)
    pim_roof = np.minimum(PIM_COMPUTE_TFLOPS, PIM_ATTN_BW_TBS * ai_range)

    ax.plot(ai_range, gpu_roof, color=GPU_DARK, lw=2.0, zorder=4,
            label=f"GPU roof  (A100: {GPU_BW_TBS} TB/s, "
                  f"{GPU_COMPUTE_TFLOPS:.0f} TFLOPS)")
    ax.plot(ai_range, pim_roof, color=PIM_DARK, lw=2.0, linestyle="--",
            zorder=4,
            label=f"PIM-attn roof  ({PIM_ATTN_BW_TBS} TB/s, "
                  f"{PIM_COMPUTE_TFLOPS:.0f} TFLOPS)")

    ax.axvline(GPU_RIDGE, color=GPU_DARK, lw=0.6, alpha=0.4,
               linestyle=":", zorder=2)
    ax.axvline(PIM_RIDGE, color=PIM_DARK, lw=0.6, alpha=0.4,
               linestyle=":", zorder=2)
    ax.text(GPU_RIDGE * 1.05, GPU_COMPUTE_TFLOPS * 0.5,
            f"GPU ridge\n{GPU_RIDGE:.0f} FLOPs/B",
            color=GPU_DARK, fontsize=FONT_ANNOTATE,
            fontweight="bold", va="center")
    ax.text(PIM_RIDGE * 1.05, PIM_COMPUTE_TFLOPS * 0.4,
            f"PIM ridge\n{PIM_RIDGE:.0f} FLOPs/B",
            color=PIM_DARK, fontsize=FONT_ANNOTATE,
            fontweight="bold", va="center")

    label_lins = {512, 2048, 4096}

    # ── Decode-attention operating points (the AI=1 region) ────────────
    if pts["dec_gpu_attn"]:
        xs, ys, ls = zip(*pts["dec_gpu_attn"])
        ax.scatter(xs, ys, marker="o", s=46,
                   color=GPU_COLOR, edgecolor=GPU_DARK, linewidth=0.7,
                   zorder=6,
                   label="Decode attention (GPU, varies L)")
        for ai, perf, lin in pts["dec_gpu_attn"]:
            if lin in label_lins:
                ax.annotate(f"L={lin}", (ai, perf),
                            xytext=(6, -10), textcoords="offset points",
                            fontsize=FONT_ANNOTATE, color=GPU_DARK,
                            fontweight="bold")

    if pts["dec_pim_attn"]:
        xs, ys, ls = zip(*pts["dec_pim_attn"])
        ax.scatter(xs, ys, marker="^", s=60,
                   color=PIM_COLOR, edgecolor=PIM_DARK, linewidth=0.7,
                   zorder=6,
                   label="Decode attention (PIM, varies L)")
        for ai, perf, lin in pts["dec_pim_attn"]:
            if lin in label_lins:
                ax.annotate(f"L={lin}", (ai, perf),
                            xytext=(6, 4), textcoords="offset points",
                            fontsize=FONT_ANNOTATE, color=PIM_DARK,
                            fontweight="bold")

    # FC decode (single point at the FC AI=1 location, GPU only)
    ax.scatter([pts["ai_fc"]], [pts["fc_achieved_gpu"]],
               marker="D", s=70, color="#2c3e50",
               edgecolor="black", linewidth=0.5,
               zorder=7, label="FC/FFN decode (GPU)")

    # Memory-bound / compute-bound region annotations
    ax.text(0.025, GPU_COMPUTE_TFLOPS * 0.7, "Memory-bound",
            fontsize=FONT_ANNOTATE + 1, color="#555",
            fontstyle="italic", ha="left")
    ax.text(800, 12, "Compute-bound",
            fontsize=FONT_ANNOTATE + 1, color="#555",
            fontstyle="italic", ha="center")

    # Key insight: at AI=1 GPU never saturates HBM (small KV reads are
    # weight-traffic-bound); PIM's 1.62 TB/s sustained beats GPU's
    # ~0.47 TB/s effective on this kernel — ~3.4× advantage. The scatter
    # points confirm: red PIM triangles sit above blue GPU dots at AI=1.
    insight = (
        "Decode attention sits at AI ≈ 1\n"
        "→ both roofs are BW-limited;\n"
        "PIM ≈ 3.4× GPU effective BW\n"
        "on this kernel"
    )
    ax.annotate(insight,
                xy=(AI_ATTN, PIM_ATTN_BW_TBS * AI_ATTN),
                xytext=(0.05, 0.04),
                textcoords="axes fraction",
                ha="left", va="bottom",
                fontsize=FONT_ANNOTATE + 1,
                color=PIM_DARK, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor=PIM_DARK,
                          boxstyle="round,pad=0.3", linewidth=0.5),
                arrowprops=dict(arrowstyle="->", color=PIM_DARK,
                                lw=0.6))

    ax.set_xscale("log")
    ax.set_yscale("log")
    fmt = mticker.LogFormatterSciNotation(base=10)
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    ax.set_xlim(0.02, 5000)
    ax.set_ylim(0.05, GPU_COMPUTE_TFLOPS * 2.5)

    ax.set_xlabel("Arithmetic intensity (FLOPs/byte)",
                  fontsize=FONT_LABEL, fontweight="bold")
    ax.set_ylabel("Achieved performance (TFLOPS)",
                  fontsize=FONT_LABEL, fontweight="bold")
    ax.set_title(f"Roofline — Pi0 attention on A100 GPU vs LPDDR5-PIM  "
                 f"(bs={batch})",
                 fontsize=FONT_TITLE, fontweight="bold")

    ax.grid(True, which="both", alpha=0.35)
    set_spines(ax)
    leg = ax.legend(loc="lower right", fontsize=FONT_ANNOTATE + 1,
                    framealpha=0.95)
    bold_legend(leg)

    save_fig(fig, out_name, out_dir)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu-csv", type=Path, default=DEFAULT_GPU)
    ap.add_argument("--pim-csv", type=Path, default=DEFAULT_PIM)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--out-dir", type=Path,
                    default=REPO / "thesis_plotting" / "figures")
    ap.add_argument("--out-name", type=str, default="fig_roofline")
    args = ap.parse_args()

    for p in (args.gpu_csv, args.pim_csv):
        if not p.is_file():
            sys.exit(f"[error] cost table missing: {p}")

    configure_plotting()
    gpu_df, pim_df = load_tables(str(args.gpu_csv), str(args.pim_csv))
    pts = operating_points(gpu_df, pim_df, bs=args.batch)
    print(f"[info] {len(pts['dec_pim_attn'])} PIM attention points, "
          f"{len(pts['dec_gpu_attn'])} GPU attention points")
    plot_roofline_thesis(pts, args.batch, args.out_dir, args.out_name)


if __name__ == "__main__":
    main()
