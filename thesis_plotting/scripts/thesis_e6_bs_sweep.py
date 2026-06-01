#!/usr/bin/env python3
"""Thesis E6 batching figures: 1/B disproved (bs=1→8 scales) +
bs=16 starvation cliff.

Produces three sub-figures, written into thesis_plotting/figures/:

  fig_e6_bs_throughput.{pdf,png}     — throughput vs configured bs
                                       (log2-x), one line per model,
                                       bs=8 star + bs=16 cliff X mark.
  fig_e6_bs_actual_vs_cfg.{pdf,png}  — grouped bars: configured vs
                                       achieved mean_bs per cell, two
                                       subplots (one per model).
  fig_e6_bs_histogram.{pdf,png}      — 2×5 small multiples of actual
                                       batch-size distribution per cell.
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

from plot_decode_batching import (  # noqa: E402
    parse_label, load_histogram, group_by_model,
)
from plot_sweep_compare import discover_runs  # noqa: E402

from thesis_plotting.style import (  # noqa: E402
    configure_plotting, save_fig, set_spines, bold, bold_legend,
    COLOR_MODEL, STD_FIGSIZE, WIDE_FIGSIZE, FONT_ANNOTATE, FONT_LABEL,
    FONT_TITLE,
)


DEFAULT_SWEEP = REPO / "cluster_outputs/online_serving_runs/" \
                       "e6_bs_sweep_azure_20260528_201000"

MODEL_DISPLAY = {"pi0": "Pi0", "openvla": "OpenVLA"}


# ── figure 1: throughput vs configured bs ───────────────────────────────

def plot_throughput(grouped, out_dir, out_name):
    fig, ax = plt.subplots(figsize=(STD_FIGSIZE[0], 3.0))
    peak_xy = {}  # model → (x_peak, y_peak)
    cliff_xy = {}

    for model, points in grouped.items():
        cfgs   = [p[0] for p in points]
        thrus  = [p[1].get("throughput", float("nan")) for p in points]
        color = COLOR_MODEL[model]
        ax.plot(cfgs, thrus,
                marker="o", linewidth=1.5, markersize=4,
                color=color, label=MODEL_DISPLAY[model], zorder=3)

        # Find best non-collapse cell (mean_bs / cfg ≥ 0.5).
        best_idx = -1
        best_val = -1.0
        for i, (cfg, summary, _label) in enumerate(points):
            mean_bs = summary.get("decode_mean_bs", 0.0)
            if cfg > 0 and mean_bs / cfg >= 0.5 and thrus[i] > best_val:
                best_val = thrus[i]
                best_idx = i
        if best_idx >= 0:
            peak_xy[model] = (cfgs[best_idx], thrus[best_idx])

        # Find starvation cliff (mean_bs / cfg < 0.2 AND throughput drop).
        for i, (cfg, summary, _label) in enumerate(points):
            mean_bs = summary.get("decode_mean_bs", 0.0)
            if cfg > 1 and mean_bs / cfg < 0.2:
                cliff_xy[model] = (cfgs[i], thrus[i])
                break

    # Star markers on peaks
    for model, (x, y) in peak_xy.items():
        ax.scatter([x], [y], marker="*", s=160, zorder=5,
                   color=COLOR_MODEL[model],
                   edgecolor="black", linewidth=0.5)
        ax.annotate(f"bs={x} peak",
                    xy=(x, y), xytext=(0, 12),
                    textcoords="offset points",
                    ha="center", fontsize=FONT_ANNOTATE + 1,
                    fontweight="bold",
                    color=COLOR_MODEL[model])

    # Red X on cliffs
    for model, (x, y) in cliff_xy.items():
        ax.scatter([x], [y], marker="X", s=80, zorder=5,
                   color="#c0392b",
                   edgecolor="black", linewidth=0.5)

    if cliff_xy:
        any_model = next(iter(cliff_xy))
        x, y = cliff_xy[any_model]
        ax.annotate("starvation cliff\n(mean_bs $\\ll$ cfg)",
                    xy=(x, y), xytext=(-25, 18),
                    textcoords="offset points",
                    ha="right", fontsize=FONT_ANNOTATE + 1,
                    fontweight="bold", color="#c0392b",
                    arrowprops=dict(arrowstyle="->", color="#c0392b",
                                    lw=0.6))

    ax.set_xscale("log", base=2)
    fmt = mticker.FuncFormatter(lambda x, _p: f"{int(x)}")
    ax.xaxis.set_major_formatter(fmt)
    ax.set_xlabel("Configured max_decode_batch_size",
                  fontsize=FONT_LABEL, fontweight="bold")
    ax.set_ylabel("Throughput (req/s)",
                  fontsize=FONT_LABEL, fontweight="bold")
    # Extra headroom so the title doesn't collide with the peak star.
    ax.set_ylim(top=max(max([p[1].get("throughput", 0)
                             for p in pts]) for pts in grouped.values()) * 1.45)
    ax.set_title("E6 — Throughput vs configured batch size",
                 fontsize=FONT_TITLE, fontweight="bold")
    ax.grid(True, alpha=0.5, which="both")
    set_spines(ax)
    leg = ax.legend(loc="upper left", title="Model")
    bold_legend(leg)
    if leg.get_title() is not None:
        leg.get_title().set_fontweight("bold")
    save_fig(fig, out_name, out_dir)
    plt.close(fig)


# ── figure 2: actual vs configured ─────────────────────────────────────

def plot_actual_vs_configured(grouped, out_dir, out_name):
    models = list(grouped.keys())
    fig, axes = plt.subplots(1, len(models),
                             figsize=(WIDE_FIGSIZE[0], 3.0),
                             squeeze=False)
    for ax, model in zip(axes[0], models):
        points = grouped[model]
        cfgs   = [p[0] for p in points]
        means  = [p[1].get("decode_mean_bs", 0.0) for p in points]
        x = np.arange(len(cfgs))
        width = 0.38
        ax.bar(x - width / 2, cfgs, width,
               label="configured",
               color="#cfd8dc", edgecolor="black", linewidth=0.4,
               zorder=3)
        ax.bar(x + width / 2, means, width,
               label="achieved (mean)",
               color=COLOR_MODEL[model], edgecolor="black", linewidth=0.4,
               zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in cfgs],
                           fontsize=FONT_LABEL, fontweight="bold")
        ax.set_xlabel("Configured max_decode_batch_size",
                      fontsize=FONT_LABEL, fontweight="bold")
        ax.set_ylabel("Batch size",
                      fontsize=FONT_LABEL, fontweight="bold")
        ax.set_title(MODEL_DISPLAY[model],
                     fontsize=FONT_TITLE, fontweight="bold")
        ax.set_ylim(0, max(cfgs + [1]) * 1.22)
        ax.grid(axis="y", alpha=0.5)
        ax.grid(axis="x", visible=False)
        set_spines(ax)
        # Fill ratio annotation per cell.
        for i, (cfg, mean) in enumerate(zip(cfgs, means)):
            ratio = (mean / cfg * 100.0) if cfg > 0 else 0.0
            color = "#c0392b" if ratio < 20 else "#444"
            ax.text(x[i], max(cfg, mean) * 1.03,
                    f"{ratio:.0f}%",
                    ha="center", va="bottom",
                    fontsize=FONT_ANNOTATE, color=color,
                    fontweight="bold")
        leg = ax.legend(loc="upper left", fontsize=FONT_ANNOTATE + 1)
        bold_legend(leg)
    fig.suptitle("E6 — Configured vs achieved decode batch size",
                 fontsize=FONT_TITLE, fontweight="bold")
    save_fig(fig, out_name, out_dir)
    plt.close(fig)


# ── figure 3: histogram per cell ────────────────────────────────────────

def plot_histogram(sweep_dir, grouped, out_dir, out_name):
    cells = []
    for model, points in grouped.items():
        for bs, _summary, label in points:
            hist = load_histogram(sweep_dir, label)
            if not hist:
                continue
            cells.append((model, bs, hist))
    if not cells:
        return
    n = len(cells)
    cols = min(5, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                             figsize=(WIDE_FIGSIZE[0], 1.7 * rows),
                             squeeze=False)
    axes_flat = axes.flatten()
    for ax in axes_flat[n:]:
        ax.axis("off")
    for ax, (model, bs, hist) in zip(axes_flat, cells):
        total = float(sum(hist))
        pct = [100.0 * h / total if total > 0 else 0.0 for h in hist]
        bs_x = list(range(1, len(hist) + 1))
        ax.bar(bs_x, pct,
               color=COLOR_MODEL[model],
               edgecolor="black", linewidth=0.3, zorder=3)
        ax.set_xticks(bs_x[::max(1, len(bs_x) // 4)])
        ax.set_xlabel("actual bs", fontsize=FONT_ANNOTATE + 1)
        ax.set_ylabel("% steps", fontsize=FONT_ANNOTATE + 1)
        ax.set_title(f"{MODEL_DISPLAY[model]} cfg={bs}",
                     fontsize=FONT_ANNOTATE + 2, fontweight="bold")
        ax.set_ylim(0, max(pct + [1]) * 1.15)
        ax.grid(axis="y", alpha=0.4)
        ax.grid(axis="x", visible=False)
        set_spines(ax)
    fig.suptitle("E6 — Per-cell decode-batch distribution",
                 fontsize=FONT_TITLE, fontweight="bold")
    save_fig(fig, out_name, out_dir)
    plt.close(fig)


# ── main ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-dir", type=Path, default=DEFAULT_SWEEP)
    ap.add_argument("--out-dir", type=Path,
                    default=REPO / "thesis_plotting" / "figures")
    ap.add_argument("--out-suffix", default="",
                    help="appended to figure base names (e.g. '_knee' → fig_e6_bs_throughput_knee)")
    args = ap.parse_args()

    if not args.sweep_dir.is_dir():
        sys.exit(f"[error] sweep dir not found: {args.sweep_dir}")

    configure_plotting()
    rows = discover_runs(args.sweep_dir)
    if not rows:
        sys.exit("[error] no sub-runs found")
    print(f"[info] {len(rows)} cells from {args.sweep_dir.name}")
    grouped = group_by_model(rows)
    for model, points in grouped.items():
        print(f"  {MODEL_DISPLAY[model]:<8}  "
              f"cfgs={[p[0] for p in points]}  "
              f"thru={[round(p[1].get('throughput', 0), 2) for p in points]}  "
              f"mean_bs={[round(p[1].get('decode_mean_bs', 0), 2) for p in points]}")

    sfx = args.out_suffix
    plot_throughput(grouped, args.out_dir, f"fig_e6_bs_throughput{sfx}")
    plot_actual_vs_configured(grouped, args.out_dir, f"fig_e6_bs_actual_vs_cfg{sfx}")
    plot_histogram(args.sweep_dir, grouped, args.out_dir, f"fig_e6_bs_histogram{sfx}")


if __name__ == "__main__":
    main()
