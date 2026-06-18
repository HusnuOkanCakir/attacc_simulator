#!/usr/bin/env python3
"""AttAcc-Figure-15-style normalized energy-per-token breakdown.

For each representative (Lin, Lout) shape at bs=1, draws two stacked
bars — GPU-only and GPU+PIM — normalized to the GPU-only bar, with
the absolute GPU-only energy per output token in parentheses under
the group label. Components mirror AttAcc Fig. 15:

  FC layer (off-chip memory access)        = g_fc_mem_energy
  Attention layer (off-chip memory access) = g_attn_mem_energy
  FC layer (on-chip memory & compute)      = g_fc_comp_energy
  Attention layer (on-chip mem & compute)  = g_attn_comp_energy
  Communication                            = g_comm_energy
  Etc                                      = g_etc_mem + g_etc_comp

(Component identity verified: the six buckets sum to g_energy.)

Source: cluster_outputs/cost_tables_energyfix_pi0_a6000/{gpu_only,
        lpddr5_pim_bank}.csv
Output: thesis_plotting/figures/fig_energy_breakdown_pi0_a6000.{pdf,png}
"""

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from thesis_plotting.style import (  # noqa: E402
    configure_plotting, save_fig, set_spines,
    FONT_ANNOTATE, FONT_LABEL, FONT_TICK, FONT_TITLE,
)

GPU_CSV = REPO / "cluster_outputs/cost_tables_energyfix_pi0_a6000/gpu_only.csv"
PIM_CSV = REPO / "cluster_outputs/cost_tables_energyfix_pi0_a6000/lpddr5_pim_bank.csv"
BS = 1

SHAPES = [
    ("Short",  512,  64),
    ("Medium", 2048, 192),
    ("Long",   5120, 208),
]

# (label, column(s), facecolor, hatch)
COMPONENTS = [
    ("FC layer\n(off-chip mem access)",        ["g_fc_mem_energy"],
     "#2c4d75", None),
    ("Attention layer\n(off-chip mem access)", ["g_attn_mem_energy"],
     "#e8702a", None),
    ("FC layer\n(on-chip mem & compute)",      ["g_fc_comp_energy"],
     "#9db8d6", "////"),
    ("Attention layer\n(on-chip mem & compute)", ["g_attn_comp_energy"],
     "#f5b78f", "////"),
    ("Communication",                          ["g_comm_energy"],
     "#b8b8b8", None),
    ("Etc",                                    ["g_etc_mem_energy",
                                                "g_etc_comp_energy"],
     "#ffffff", None),
]


def cell(df, lin, lout):
    r = df[(df.Lin == lin) & (df.Lout == lout) & (df.bs == BS)]
    if r.empty:
        sys.exit(f"[error] missing cell ({lin},{lout}) bs={BS}")
    return r.iloc[0]


def main():
    configure_plotting()
    FL  = FONT_LABEL    + 8   # 15
    FTI = FONT_TITLE    + 8   # 16
    FTK = FONT_TICK     + 8   # 14
    FAN = FONT_ANNOTATE + 8   # 13

    gpu = pd.read_csv(GPU_CSV)
    pim = pd.read_csv(PIM_CSV)

    fig, ax = plt.subplots(figsize=(13.5, 6.8))

    bar_w = 0.36
    group_gap = 1.5
    xticks, xlabels = [], []
    ATTN_COMP = "Attention layer\n(on-chip mem & compute)"

    print(f"{'shape':<8} {'route':<5} " +
          " ".join(f"{c[0].splitlines()[0][:12]:>14}" for c in COMPONENTS) +
          f" {'total mJ/tok':>13}")

    for gi, (label, lin, lout) in enumerate(SHAPES):
        x0 = gi * group_gap
        g_row, p_row = cell(gpu, lin, lout), cell(pim, lin, lout)
        g_total = g_row["g_energy (nJ)"]

        for bi, (route, row) in enumerate([("GPU", g_row), ("PIM", p_row)]):
            x = x0 + (bi - 0.5) * (bar_w + 0.04)
            bottom = 0.0
            vals = []
            attn_comp = None
            for _name, cols, color, hatch in COMPONENTS:
                v = sum(row[c] for c in cols) / g_total
                vals.append(v)
                ax.bar(x, v, bar_w, bottom=bottom,
                       facecolor=color, hatch=hatch,
                       edgecolor="black", linewidth=0.7, zorder=3)
                if _name == ATTN_COMP:
                    attn_comp = (bottom + v / 2.0, v)  # (segment center y, value)
                bottom += v
            ax.text(x, bottom + 0.015, route,
                    ha="center", va="bottom",
                    fontsize=FAN, fontweight="bold")
            # Explicit numeric callout for the faint on-chip attention sliver:
            # GPU label to the left, PIM label to the right, with a thin leader.
            if attn_comp is not None:
                yc, v = attn_comp
                side = -1 if route == "GPU" else 1
                ax.annotate(
                    f"{v*100:.2f}%",
                    xy=(x + side * bar_w / 2.0, yc),
                    xytext=(x + side * (bar_w / 2.0 + 0.28), yc),
                    ha=("right" if side < 0 else "left"), va="center",
                    fontsize=FAN - 4, color="#a8541f",
                    arrowprops=dict(arrowstyle="-", color="#a8541f", lw=0.6),
                    zorder=6)
            print(f"{label:<8} {route:<5} " +
                  " ".join(f"{v:>14.4f}" for v in vals) +
                  f" {sum(row[c] for cc in COMPONENTS for c in cc[1])/1e6:>13.1f}")

        xticks.append(x0)
        xlabels.append(f"{label}\n$L_{{in}}$={lin}, $L_{{out}}$={lout}\n"
                       f"({g_total/1e6:.0f} mJ)")

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=FTK)
    ax.set_xlim(xticks[0] - 0.95, xticks[-1] + 0.95)
    ax.set_ylabel("Normalized energy per output token",
                  fontsize=FL, fontweight="bold")
    ax.set_ylim(0, 1.30)
    ax.set_title("Per-token decode energy breakdown: GPU-only vs GPU+PIM "
                 "(Pi0, A6000 target, bs=1)\n"
                 "normalized to GPU-only per shape; absolute GPU-only "
                 "energy in parentheses",
                 fontsize=FTI, fontweight="bold")
    ax.tick_params(axis="y", which="major", labelsize=FTK)
    ax.grid(axis="y", alpha=0.4)
    set_spines(ax)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c[2], hatch=c[3],
                             edgecolor="black", linewidth=0.7)
               for c in COMPONENTS]
    leg = ax.legend(handles, [c[0] for c in COMPONENTS],
                    loc="upper center", bbox_to_anchor=(0.5, -0.22),
                    ncol=3, fontsize=FAN, framealpha=0.95)
    for t in leg.get_texts():
        t.set_fontweight("bold")

    fig.tight_layout()
    save_fig(fig, "fig_energy_breakdown_pi0_a6000",
             REPO / "thesis_plotting/figures")
    plt.close(fig)


if __name__ == "__main__":
    main()
