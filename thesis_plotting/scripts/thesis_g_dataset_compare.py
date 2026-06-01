#!/usr/bin/env python3
"""Cross-dataset Pi0 comparison at the knee (G azure_full19k vs azure_poisson).

Visualises the central finding of the G sweeps: hybrid wins on the
real Azure heavy-tail trace but the synthetic Poisson trace inverts the
E2E story. One 2x2 figure with dataset on rows, metric on cols.

Usage:
    python thesis_plotting/scripts/thesis_g_dataset_compare.py
"""

import sys
from pathlib import Path
import re

import numpy as np
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO))

from plot_sweep_compare import parse_summary  # noqa: E402

from thesis_plotting.style import (  # noqa: E402
    configure_plotting, save_fig, set_spines,
    COLOR_ROUTE, FONT_LABEL, FONT_TITLE, FONT_TICK,
)


DEFAULTS = {
    "azure_full19k":  REPO / "cluster_outputs/online_serving_runs/"
                             "g_knee_main_claim_20260530_164544",
    "azure_poisson":  REPO / "cluster_outputs/online_serving_runs/"
                             "g_knee_azure_poisson_20260530_174416",
}

DATASET_LABEL = {
    "azure_full19k": "Azure full-19k  (real, heavy-tail)",
    "azure_poisson": "Azure-shaped Poisson  (synthetic, smoothed)",
}

MODE_LABEL  = {"hybrid": "Hybrid", "gpuonly": "GPU-only"}
MODE_COLOR  = {"hybrid": COLOR_ROUTE["lpddr5_pim_bank"],
               "gpuonly": COLOR_ROUTE["gpu_only"]}


def discover_pi0(sweep_dir: Path) -> dict:
    """Returns {mode: metrics_dict} for the Pi0 hybrid + gpuonly cells."""
    out = {}
    label_re = re.compile(r"^\d+_pi0_(?P<mode>hybrid|gpuonly)$")
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
                    cand = child / "policy_compare_summary.txt"
                    if cand.is_file():
                        sumfile = cand
                        break
        if not sumfile.is_file():
            continue
        metrics = parse_summary(sumfile)
        out[m["mode"]] = metrics
    return out


def render(data: dict, out_dir: Path, out_name: str):
    """data = {dataset_name: {mode: metrics}}"""
    configure_plotting()
    datasets = list(data.keys())
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 5.4), sharex="col")

    metrics = [
        ("ttft_p99", "TTFT p99 (ms)", lambda v: v),
        ("e2e_p99",  "E2E p99 (s)",   lambda v: v / 1000.0),
    ]
    modes = ["gpuonly", "hybrid"]
    xs = np.array([0.35, 0.65])

    for row, dataset in enumerate(datasets):
        cells = data[dataset]
        for col, (metric_key, ylabel, transform) in enumerate(metrics):
            ax = axes[row, col]
            vals = []
            for mode in modes:
                m = cells.get(mode, {})
                v = m.get(metric_key)
                vals.append(transform(v) if v is not None else 0.0)
            colors = [MODE_COLOR[m] for m in modes]
            bars = ax.bar(xs, vals, width=0.28, color=colors,
                          edgecolor="black", linewidth=0.5, zorder=3)
            ax.set_yscale("log")

            gpu_v, hyb_v = vals[0], vals[1]
            if gpu_v > 0:
                delta_signed = (hyb_v - gpu_v) / gpu_v * 100.0
                arrow = "↑" if delta_signed > 0 else "↓"
                outcome_color = "#c44e52" if delta_signed > 0 else "#2ca02c"
                ax.text(0.5, max(vals) * 2.4,
                        f"{arrow}{abs(delta_signed):.0f}%",
                        ha="center", va="bottom",
                        fontsize=FONT_LABEL + 4, fontweight="bold",
                        color=outcome_color, transform=ax.transData)

            for bar, v in zip(bars, vals):
                label = f"{v:.2f}" if v < 100 else f"{v:.0f}"
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() * 1.08,
                        label, ha="center", va="bottom",
                        fontsize=FONT_LABEL + 1, fontweight="bold")

            ax.set_xticks(xs)
            ax.set_xticklabels([MODE_LABEL[m] for m in modes],
                               fontsize=FONT_TICK, fontweight="bold")
            ax.set_xlim(0.10, 0.90)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=FONT_LABEL,
                              fontweight="bold")
            else:
                ax.set_ylabel(ylabel, fontsize=FONT_LABEL,
                              fontweight="bold")
            ax.set_ylim(top=max(vals) * 5.5)
            set_spines(ax)

            if row == 0:
                ax.set_title(["TTFT p99", "E2E p99"][col],
                             fontsize=FONT_TITLE - 1, fontweight="bold")

        # Row label as left-side text on first column.
        axes[row, 0].text(-0.36, 0.5, DATASET_LABEL[datasets[row]],
                          transform=axes[row, 0].transAxes,
                          rotation=90, va="center", ha="center",
                          fontsize=FONT_LABEL + 1, fontweight="bold")

    fig.suptitle("Pi0 at the knee — real Azure (top) vs synthetic Poisson (bottom)",
                 fontsize=FONT_TITLE, fontweight="bold", y=1.00)

    # Single legend at the top.
    handles = [plt.Rectangle((0, 0), 1, 1,
                             facecolor=MODE_COLOR[m],
                             edgecolor="black", linewidth=0.5,
                             label=MODE_LABEL[m]) for m in modes]
    fig.legend(handles=handles, loc="upper right",
               bbox_to_anchor=(0.99, 1.00),
               ncol=2, frameon=True, fontsize=FONT_LABEL,
               prop={"weight": "bold"})

    fig.tight_layout(rect=(0.04, 0.00, 1.0, 0.95))
    save_fig(fig, out_name, out_dir)
    plt.close(fig)


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full19k-dir", type=Path,
                    default=DEFAULTS["azure_full19k"])
    ap.add_argument("--poisson-dir", type=Path,
                    default=DEFAULTS["azure_poisson"])
    ap.add_argument("--out-dir", type=Path,
                    default=REPO / "thesis_plotting/figures")
    ap.add_argument("--out-name", default="fig_g_pi0_dataset_compare")
    args = ap.parse_args()

    data = {
        "azure_full19k": discover_pi0(args.full19k_dir),
        "azure_poisson": discover_pi0(args.poisson_dir),
    }
    for k, v in data.items():
        print(f"[{k}] {sorted(v.keys())}")
        for mode, metrics in v.items():
            print(f"  {mode:8s}  ttft_p99={metrics.get('ttft_p99')}ms"
                  f"  e2e_p99={metrics.get('e2e_p99')}ms"
                  f"  thru={metrics.get('throughput')}")
    render(data, args.out_dir, args.out_name)


if __name__ == "__main__":
    main()
