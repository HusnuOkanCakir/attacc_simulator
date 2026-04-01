#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


def _plot_one(route_df: pd.DataFrame,
              out_path: Path,
              guard_ms: float) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {
        "gpu_only": "tab:blue",
        "lpddr5_pim_bank": "tab:orange",
    }
    for route, grp in route_df.groupby("chosen_route"):
        ax.scatter(grp["finish_delta_hybrid_minus_gpu_ms"],
                   grp["scheduler_energy_delta_hybrid_minus_gpu_nj"],
                   s=30,
                   alpha=0.8,
                   color=colors.get(str(route), "tab:gray"),
                   label=str(route))

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:blue", markersize=8, label="chosen gpu_only"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:orange", markersize=8, label="chosen lpddr5_pim_bank"),
    ]

    if guard_ms > 0.0:
        ax.axvline(-guard_ms, color="tab:green", linestyle=":", linewidth=1.5)
        ax.axvline(+guard_ms, color="tab:green", linestyle=":", linewidth=1.5)
        ax.axvspan(-guard_ms, +guard_ms, color="tab:green", alpha=0.08)
        legend_handles.append(Line2D([0], [0],
                                     color="tab:green",
                                     linestyle=":",
                                     linewidth=1.5,
                                     label=f"latency guard ±{guard_ms:g} ms"))

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_pad = 0.04 * (xlim[1] - xlim[0])
    y_pad = 0.05 * (ylim[1] - ylim[0])
    label_box = dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="none")
    ax.text(xlim[0] + x_pad, ylim[1] - y_pad,
            "Hybrid faster\nGPU lower energy",
            ha="left", va="top", fontsize=8.5, color="tab:purple", bbox=label_box)
    ax.text(xlim[1] - x_pad, ylim[1] - y_pad,
            "GPU wins both",
            ha="right", va="top", fontsize=8.5, color="tab:blue", bbox=label_box)
    ax.text(xlim[0] + x_pad, ylim[0] + y_pad,
            "Hybrid wins both",
            ha="left", va="bottom", fontsize=8.5, color="tab:orange", bbox=label_box)
    ax.text(xlim[1] - x_pad, ylim[0] + y_pad,
            "GPU faster\nHybrid lower energy",
            ha="right", va="bottom", fontsize=8.5, color="tab:green", bbox=label_box)

    ax.set_xlabel("Finish delta: hybrid - gpu (ms)")
    ax.set_ylabel("Scheduler energy delta: hybrid - gpu (nJ)")
    ax.set_title("Frozen-State Route Delta Scatter")
    ax.grid(True, alpha=0.3)
    ax.legend(handles=legend_handles, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot frozen-state route-delta boundaries from one route-delta CSV.")
    ap.add_argument("--route-delta-csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--guard-values", nargs="+", type=float, required=True)
    args = ap.parse_args()

    route_df = pd.read_csv(args.route_delta_csv)
    route_df = route_df[["finish_delta_hybrid_minus_gpu_ms",
                         "scheduler_energy_delta_hybrid_minus_gpu_nj",
                         "chosen_route"]].dropna()
    if route_df.empty:
        raise SystemExit("No usable route-delta rows found")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for guard_ms in args.guard_values:
        tag = str(guard_ms).replace(".", "p")
        out_path = args.out_dir / f"{args.prefix}_frozen_guard{tag}_route_selection_delta_scatter.png"
        _plot_one(route_df,
                  out_path=out_path,
                  guard_ms=float(guard_ms))
        print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
