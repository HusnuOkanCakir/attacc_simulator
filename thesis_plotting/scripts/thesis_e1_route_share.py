#!/usr/bin/env python3
"""Thesis E1 figure: route divergence — PIM share vs arrival_scale.

Highlights the architecture-driven routing finding: across the same
arrival_scale range, Pi0 routes >80% to PIM while OpenVLA routes <50%.

Output: thesis_plotting/figures/fig_e1_route_share.{pdf,png}
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO))

from plot_e1_route_share_vs_scale import (  # noqa: E402
    group_by_model, pim_share,
)
from plot_sweep_compare import discover_runs  # noqa: E402

from thesis_plotting.style import (  # noqa: E402
    configure_plotting, save_fig, set_spines, bold_legend,
    COLOR_MODEL, COLOR_CATEGORY, STD_FIGSIZE, FONT_ANNOTATE,
    FONT_LABEL, FONT_TITLE,
)


DEFAULT_SWEEP = REPO / "cluster_outputs/online_serving_runs/" \
                       "e1_route_mix_20260519_214353"

MODEL_DISPLAY = {"pi0": "Pi0", "openvla": "OpenVLA"}


def plot_route_share(grouped, out_dir, out_name):
    fig, ax = plt.subplots(figsize=(STD_FIGSIZE[0], 3.2))

    # GPU-dominant / PIM-dominant region backgrounds.
    ax.axhspan(0, 50,  color=COLOR_CATEGORY["low"],  alpha=0.18, zorder=0)
    ax.axhspan(50, 100, color=COLOR_CATEGORY["high"], alpha=0.18, zorder=0)

    for model, points in grouped.items():
        scales = [p[0] for p in points]
        shares = [pim_share(p[1]) for p in points]
        ax.plot(scales, shares,
                marker="o", linewidth=1.6, markersize=5,
                color=COLOR_MODEL[model],
                label=MODEL_DISPLAY[model],
                zorder=3)
        for s, v in zip(scales, shares):
            ax.text(s, v + 3, f"{v:.0f}%",
                    ha="center", va="bottom",
                    fontsize=FONT_ANNOTATE,
                    color=COLOR_MODEL[model], fontweight="bold")

    ax.axhline(50, color="#444", linewidth=0.7, linestyle=":", zorder=2)
    ax.text(ax.get_xlim()[1] * 0.92, 50, "50%",
            ha="right", va="bottom", fontsize=FONT_ANNOTATE,
            color="#444")

    # Side-panel labels for the two regions
    ax.text(0.02, 0.05, "GPU-dominant",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=FONT_ANNOTATE + 1, fontweight="bold",
            color="#2d6b2d")
    ax.text(0.02, 0.95, "PIM-dominant",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=FONT_ANNOTATE + 1, fontweight="bold",
            color="#8b1a1a")

    ax.set_xscale("log")
    fmt = mticker.FuncFormatter(lambda x, _p: f"{x:g}")
    ax.xaxis.set_major_formatter(fmt)
    # Force ticks at every actual sweep point.
    all_scales = sorted({s for points in grouped.values()
                         for s, _ in points})
    ax.set_xticks(all_scales)
    ax.xaxis.set_minor_locator(mticker.NullLocator())

    ax.set_xlabel("arrival_scale  (smaller = denser arrivals)",
                  fontsize=FONT_LABEL, fontweight="bold")
    ax.set_ylabel("PIM route share (%)",
                  fontsize=FONT_LABEL, fontweight="bold")
    ax.set_title("E1 — Architecture-driven routing divergence",
                 fontsize=FONT_TITLE, fontweight="bold")

    ax.set_ylim(-3, 108)
    ax.grid(True, alpha=0.5, which="both")
    set_spines(ax)
    leg = ax.legend(loc="center right", title="Model")
    bold_legend(leg)
    if leg.get_title() is not None:
        leg.get_title().set_fontweight("bold")

    save_fig(fig, out_name, out_dir)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-dir", type=Path, default=DEFAULT_SWEEP)
    ap.add_argument("--out-dir", type=Path,
                    default=REPO / "thesis_plotting" / "figures")
    ap.add_argument("--out-name", type=str, default="fig_e1_route_share")
    args = ap.parse_args()

    if not args.sweep_dir.is_dir():
        sys.exit(f"[error] sweep dir not found: {args.sweep_dir}")

    configure_plotting()
    rows = discover_runs(args.sweep_dir)
    if not rows:
        sys.exit("[error] no sub-runs found")
    grouped = group_by_model(rows)
    print(f"[info] {len(rows)} cells from {args.sweep_dir.name}")
    for model, points in grouped.items():
        print(f"  {MODEL_DISPLAY[model]:<8}  "
              f"scales={[p[0] for p in points]}  "
              f"PIM%={[round(pim_share(p[1]), 1) for p in points]}")

    plot_route_share(grouped, args.out_dir, args.out_name)


if __name__ == "__main__":
    main()
