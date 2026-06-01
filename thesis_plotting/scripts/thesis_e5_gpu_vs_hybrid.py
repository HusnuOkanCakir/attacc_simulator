#!/usr/bin/env python3
"""Thesis main-claim figure: GPU-only vs hybrid (Pi0 + OpenVLA, Azure conv).

Replaces the auto-generated sweep_compare.png for the thesis §6.5.0a
headline. 2×2 grid: rows = metrics (throughput, E2E p99), cols = models.
Within each panel: grouped bars (gpu_only vs hybrid). Hybrid gets the
deep-teal emphasis color, gpu_only gets the lighter baseline tone.
Annotates the +%/−% delta on the hybrid bar.

Usage:
    python thesis_plotting/scripts/thesis_e5_gpu_vs_hybrid.py \\
        [--sweep-dir <PATH>] [--out-name <FIG_NAME>]

Defaults to the E5 v2 sweep (`e5_gpu_vs_hybrid_azure_20260528_152128`).
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO))

from plot_sweep_compare import parse_summary  # noqa: E402

from thesis_plotting.style import (  # noqa: E402
    configure_plotting, save_fig, set_spines, bold, bold_legend,
    COLOR_ROUTE, COLOR_MODEL, WIDE_FIGSIZE, FONT_ANNOTATE, FONT_LABEL,
    FONT_TITLE,
)


# ── data ────────────────────────────────────────────────────────────────

DEFAULT_SWEEP = REPO / "cluster_outputs/online_serving_runs/" \
                       "e5_gpu_vs_hybrid_azure_20260528_152128"


def discover_cells(sweep_dir: Path) -> dict:
    """Returns {(model, mode): metrics_dict} for one E5-style sweep.

    Recognises labels like '01_pi0_hybrid', '02_pi0_gpuonly', etc.
    """
    out = {}
    label_re = re.compile(r"^(?:\d+_)?(?P<model>pi0|openvla)_(?P<mode>hybrid|gpuonly)$")
    for sub in sorted(sweep_dir.iterdir()):
        if not sub.is_dir():
            continue
        m = label_re.match(sub.name)
        if not m:
            continue
        sumfile = sub / "policy_compare_summary.txt"
        if not sumfile.is_file():
            for child in sub.iterdir():
                if child.is_dir():
                    alt = child / "policy_compare_summary.txt"
                    if alt.is_file():
                        sumfile = alt
                        break
        metrics = parse_summary(sumfile)
        if not metrics:
            print(f"[warn] no metrics in {sub.name}", file=sys.stderr)
            continue
        out[(m.group("model"), m.group("mode"))] = metrics
    return out


# ── plot ────────────────────────────────────────────────────────────────

MODELS = ["pi0", "openvla"]
MODES  = ["gpuonly", "hybrid"]
MODE_LABEL = {"gpuonly": "GPU-only", "hybrid": "Hybrid"}
MODE_COLOR = {"gpuonly": COLOR_ROUTE["gpu_only"],
              "hybrid":  COLOR_ROUTE["lpddr5_pim_bank"]}


def make_figure(cells: dict, out_dir: Path, out_name: str):
    fig, axes = plt.subplots(2, 2, figsize=(WIDE_FIGSIZE[0], 4.4),
                             sharex=False)

    metrics = [
        ("throughput",
         lambda v: v,                    # identity
         "Throughput (req/s)",
         "higher",                       # winning direction
         False),                         # log y
        ("e2e_p99",
         lambda v: v / 1000.0,           # ms -> s
         "E2E p99 (s)",
         "lower",
         True),
    ]

    for row, (metric_key, transform, ylabel, winning, log) in enumerate(metrics):
        for col, model in enumerate(MODELS):
            ax = axes[row, col]
            vals = []
            for mode in MODES:
                m = cells.get((model, mode), {})
                v = m.get(metric_key)
                vals.append(transform(v) if v is not None else 0.0)

            # Tight pair: bars sit at x=0.35 and x=0.65 instead of 0, 1.
            xs = np.array([0.35, 0.65])
            colors = [MODE_COLOR[mode] for mode in MODES]
            bars = ax.bar(xs, vals, width=0.28, color=colors,
                          edgecolor="black", linewidth=0.5, zorder=3)

            gpu_v, hyb_v = vals[0], vals[1]
            if gpu_v > 0:
                delta_signed = (hyb_v - gpu_v) / gpu_v * 100.0
                arrow = "↑" if delta_signed > 0 else "↓"
                delta_str = f"{arrow}{abs(delta_signed):.0f}%"
            else:
                delta_signed = 0.0
                delta_str = ""

            # Value labels just above each bar (works for log + linear).
            for bar, v in zip(bars, vals):
                label = f"{v:.2f}" if v < 100 else f"{v:.0f}"
                if log:
                    # In log space, "just above" means multiply by ~1.06.
                    y_lab = bar.get_height() * 1.06
                else:
                    y_lab = bar.get_height() + max(vals) * 0.015
                ax.text(bar.get_x() + bar.get_width() / 2,
                        y_lab,
                        label,
                        ha="center", va="bottom",
                        fontsize=FONT_LABEL + 2,
                        color="black", fontweight="bold",
                        zorder=4)

            if delta_str:
                # Centered delta annotation, well above the value labels.
                top_y = max(vals)
                ax.text(0.5, top_y * (1.18 if not log else 2.1),
                        delta_str,
                        ha="center", va="bottom",
                        fontsize=FONT_LABEL + 5,
                        color=COLOR_MODEL[model], fontweight="bold")

            ax.set_xticks(xs)
            ax.set_xticklabels([MODE_LABEL[mode] for mode in MODES],
                               fontsize=FONT_LABEL, fontweight="bold")
            ax.set_xlim(0.10, 0.90)

            if col == 0:
                ax.set_ylabel(ylabel, fontsize=FONT_LABEL + 1,
                              fontweight="bold")

            if row == 0:
                title = "Pi0" if model == "pi0" else "OpenVLA"
                ax.set_title(title, fontsize=FONT_TITLE + 2,
                             fontweight="bold")

            if log:
                ax.set_yscale("log")
                ax.set_ylim(top=max(vals) * 3.0,
                            bottom=min(vals) * 0.7)
            else:
                ax.set_ylim(0, max(vals) * 1.30)

            ax.grid(axis="y", alpha=0.5)
            ax.grid(axis="x", visible=False)
            set_spines(ax)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=MODE_COLOR[m],
                             edgecolor="black", linewidth=0.4)
               for m in MODES]
    labels = [MODE_LABEL[m] for m in MODES]
    leg = fig.legend(handles, labels, loc="upper center",
                     ncol=2, bbox_to_anchor=(0.5, 1.02))
    bold_legend(leg)

    save_fig(fig, out_name, out_dir)
    plt.close(fig)


# ── main ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-dir", type=Path, default=DEFAULT_SWEEP,
                    help="E5 sweep directory (default: e5_gpu_vs_hybrid_azure_20260528_152128)")
    ap.add_argument("--out-dir", type=Path,
                    default=REPO / "thesis_plotting" / "figures")
    ap.add_argument("--out-name", type=str, default="fig_e5_gpu_vs_hybrid")
    args = ap.parse_args()

    if not args.sweep_dir.is_dir():
        sys.exit(f"[error] sweep dir not found: {args.sweep_dir}")

    configure_plotting()
    cells = discover_cells(args.sweep_dir)
    if not cells:
        sys.exit("[error] no (model, mode) cells found")

    print(f"[info] {len(cells)} cells from {args.sweep_dir.name}")
    for (model, mode), m in cells.items():
        print(f"  {model:>8} {mode:>8}  thru={m.get('throughput'):.2f}  "
              f"e2e_p99={m.get('e2e_p99'):.0f}ms")

    make_figure(cells, args.out_dir, args.out_name)


if __name__ == "__main__":
    main()
