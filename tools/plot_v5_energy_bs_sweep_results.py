#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

BS_RE = re.compile(r"bs(\d+)")


def _load_rows(summary_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for path in sorted(summary_dir.glob("replay_summary_v5_energy_bs*.json")):
        match = BS_RE.search(path.stem)
        if not match:
            continue
        bs = int(match.group(1))
        with path.open() as f:
            data = json.load(f)
        decode_batching = data.get("decode_batching", {})
        route_counts = data.get("route_counts", {})
        total_energy = data.get("energy_nj", {})
        decode_energy = data.get("decode_energy_nj", {})
        total_routes = float(sum(route_counts.values())) if route_counts else 0.0
        rows.append({
            "max_decode_batch_size": bs,
            "throughput_tokps": data.get("throughput_tokps"),
            "throughput_rps": data.get("throughput_rps"),
            "ttft_p95": data.get("ttft_ms", {}).get("p95"),
            "e2e_p95": data.get("e2e_ms", {}).get("p95"),
            "tbt_p95": data.get("tbt_ms", {}).get("p95"),
            "gpu_util": data.get("gpu_util"),
            "pim_util": data.get("pim_util"),
            "decode_mean_batch_size": decode_batching.get("mean_batch_size"),
            "num_decode_steps": decode_batching.get("num_decode_steps"),
            "gpu_route_count": route_counts.get("gpu_only", 0.0),
            "hybrid_route_count": route_counts.get("lpddr5_pim_bank", 0.0),
            "gpu_route_frac": (route_counts.get("gpu_only", 0.0) / total_routes) if total_routes > 0 else 0.0,
            "energy_total_nj": total_energy.get("total", decode_energy.get("total")),
            "decode_energy_total_nj": decode_energy.get("total"),
            "decode_energy_gpu_only_nj": decode_energy.get("by_route", {}).get("gpu_only", 0.0),
            "decode_energy_hybrid_nj": decode_energy.get("by_route", {}).get("lpddr5_pim_bank", 0.0),
            "decode_energy_per_decode_token": data.get("decode_energy_nj_per_decode_token"),
        })
    if not rows:
        raise FileNotFoundError(f"No replay_summary_v5_energy_bs*.json files found in {summary_dir}")
    return pd.DataFrame(rows).sort_values("max_decode_batch_size").reset_index(drop=True)


def _save_line_plot(df: pd.DataFrame,
                    x_col: str,
                    y_cols: List[str],
                    labels: List[str],
                    title: str,
                    ylabel: str,
                    out_path: Path,
                    y_lim: tuple[float, float] | None = None) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for col, label in zip(y_cols, labels):
        ax.plot(df[x_col], df[col], marker="o", linewidth=2, label=label)
    if y_lim is not None:
        ax.set_ylim(*y_lim)
    ax.set_xlabel("Max decode batch size")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _save_dual_axis_plot(df: pd.DataFrame,
                         x_col: str,
                         left_col: str,
                         left_label: str,
                         right_col: str,
                         right_label: str,
                         title: str,
                         out_path: Path) -> None:
    fig, ax_left = plt.subplots(figsize=(8, 5))
    ax_right = ax_left.twinx()

    left_color = "#4C72B0"
    right_color = "#C44E52"

    left_line = ax_left.plot(df[x_col],
                             df[left_col],
                             marker="o",
                             linewidth=2,
                             color=left_color,
                             label=left_label)[0]
    right_line = ax_right.plot(df[x_col],
                               df[right_col],
                               marker="s",
                               linewidth=2,
                               color=right_color,
                               label=right_label)[0]

    ax_left.set_xlabel("Max decode batch size")
    ax_left.set_ylabel(left_label, color=left_color)
    ax_right.set_ylabel(right_label, color=right_color)
    ax_left.tick_params(axis="y", colors=left_color)
    ax_right.tick_params(axis="y", colors=right_color)
    ax_left.set_title(title)
    ax_left.grid(True, alpha=0.3)
    ax_left.legend([left_line, right_line], [left_label, right_label], loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _save_triple_axis_latency_plot(df: pd.DataFrame,
                                   x_col: str,
                                   out_path: Path) -> None:
    fig, ax_left = plt.subplots(figsize=(8, 5))
    ax_right = ax_left.twinx()
    ax_right2 = ax_left.twinx()
    ax_right2.spines["right"].set_position(("outward", 55))

    ttft_color = "#4C72B0"
    e2e_color = "#55A868"
    tbt_color = "#C44E52"

    line1 = ax_left.plot(df[x_col],
                         df["ttft_p95"],
                         marker="o",
                         linewidth=2,
                         color=ttft_color,
                         label="TTFT p95")[0]
    line2 = ax_right.plot(df[x_col],
                          df["e2e_p95"],
                          marker="s",
                          linewidth=2,
                          color=e2e_color,
                          label="E2E p95")[0]
    line3 = ax_right2.plot(df[x_col],
                           df["tbt_p95"],
                           marker="^",
                           linewidth=2,
                           color=tbt_color,
                           label="TBT p95")[0]

    ax_left.set_xlabel("Max decode batch size")
    ax_left.set_ylabel("TTFT p95 (ms)", color=ttft_color)
    ax_right.set_ylabel("E2E p95 (ms)", color=e2e_color)
    ax_right2.set_ylabel("TBT p95 (ms)", color=tbt_color)
    ax_left.tick_params(axis="y", colors=ttft_color)
    ax_right.tick_params(axis="y", colors=e2e_color)
    ax_right2.tick_params(axis="y", colors=tbt_color)
    ax_left.set_title("Max Decode Batch Sweep: p95 Latencies")
    ax_left.grid(True, alpha=0.3)
    ax_left.legend([line1, line2, line3], ["TTFT p95", "E2E p95", "TBT p95"], loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description="Plot v5 energy-aware max decode batch size sweep results.")
    p.add_argument("--summary-dir",
                   type=Path,
                   default=Path("cluster_outputs/v5_energy_bs_sweep"),
                   help="Directory containing replay_summary_v5_energy_bs*.json files")
    p.add_argument("--out-dir",
                   type=Path,
                   default=Path("cluster_outputs/v5_energy_bs_sweep_plots"),
                   help="Directory to write plots and merged CSV")
    p.add_argument("--prefix",
                   type=str,
                   default="v5_energy_bs_sweep",
                   help="Output file prefix")
    args = p.parse_args()

    df = _load_rows(args.summary_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    merged_csv = args.out_dir / f"{args.prefix}_summary.csv"
    df.to_csv(merged_csv, index=False)

    _save_dual_axis_plot(df,
                         x_col="max_decode_batch_size",
                         left_col="throughput_tokps",
                         left_label="Throughput (tok/s)",
                         right_col="throughput_rps",
                         right_label="Throughput (req/s)",
                         title="Max Decode Batch Sweep: Throughput",
                         out_path=args.out_dir / f"{args.prefix}_throughput.png")

    _save_line_plot(df,
                    x_col="max_decode_batch_size",
                    y_cols=["throughput_tokps"],
                    labels=["Throughput (tok/s)"],
                    title="Max Decode Batch Sweep: Token Throughput",
                    ylabel="Throughput (tok/s)",
                    out_path=args.out_dir / f"{args.prefix}_throughput_tokps.png")

    _save_line_plot(df,
                    x_col="max_decode_batch_size",
                    y_cols=["throughput_rps"],
                    labels=["Throughput (req/s)"],
                    title="Max Decode Batch Sweep: Request Throughput",
                    ylabel="Throughput (req/s)",
                    out_path=args.out_dir / f"{args.prefix}_throughput_rps.png")

    _save_triple_axis_latency_plot(df,
                                   x_col="max_decode_batch_size",
                                   out_path=args.out_dir / f"{args.prefix}_latency_p95.png")

    _save_line_plot(df,
                    x_col="max_decode_batch_size",
                    y_cols=["ttft_p95"],
                    labels=["TTFT p95"],
                    title="Max Decode Batch Sweep: TTFT p95",
                    ylabel="Latency (ms)",
                    out_path=args.out_dir / f"{args.prefix}_ttft_p95_only.png")

    _save_line_plot(df,
                    x_col="max_decode_batch_size",
                    y_cols=["e2e_p95"],
                    labels=["E2E p95"],
                    title="Max Decode Batch Sweep: E2E p95",
                    ylabel="Latency (ms)",
                    out_path=args.out_dir / f"{args.prefix}_e2e_p95_only.png")

    _save_line_plot(df,
                    x_col="max_decode_batch_size",
                    y_cols=["tbt_p95"],
                    labels=["TBT p95"],
                    title="Max Decode Batch Sweep: TBT p95",
                    ylabel="Latency (ms)",
                    out_path=args.out_dir / f"{args.prefix}_tbt_p95_only.png")

    _save_line_plot(df,
                    x_col="max_decode_batch_size",
                    y_cols=["energy_total_nj"],
                    labels=["Total request energy (nJ)"],
                    title="Max Decode Batch Sweep: Total Request Energy",
                    ylabel="Energy (nJ)",
                    out_path=args.out_dir / f"{args.prefix}_energy_total.png")

    _save_line_plot(df,
                    x_col="max_decode_batch_size",
                    y_cols=["decode_energy_per_decode_token"],
                    labels=["Decode energy per token (nJ)"],
                    title="Max Decode Batch Sweep: Decode Energy Per Token",
                    ylabel="Energy per decode token (nJ)",
                    out_path=args.out_dir / f"{args.prefix}_decode_energy_per_token.png")

    _save_dual_axis_plot(df,
                         x_col="max_decode_batch_size",
                         left_col="decode_energy_gpu_only_nj",
                         left_label="GPU-only decode energy (nJ)",
                         right_col="decode_energy_hybrid_nj",
                         right_label="Hybrid decode energy (nJ)",
                         title="Max Decode Batch Sweep: Decode Energy by Route",
                         out_path=args.out_dir / f"{args.prefix}_decode_energy_by_route.png")

    _save_line_plot(df,
                    x_col="max_decode_batch_size",
                    y_cols=["decode_energy_gpu_only_nj"],
                    labels=["GPU-only decode energy (nJ)"],
                    title="Max Decode Batch Sweep: GPU-Only Decode Energy",
                    ylabel="Energy (nJ)",
                    out_path=args.out_dir / f"{args.prefix}_decode_energy_gpu_only.png")

    _save_line_plot(df,
                    x_col="max_decode_batch_size",
                    y_cols=["decode_energy_hybrid_nj"],
                    labels=["Hybrid decode energy (nJ)"],
                    title="Max Decode Batch Sweep: Hybrid Decode Energy",
                    ylabel="Energy (nJ)",
                    out_path=args.out_dir / f"{args.prefix}_decode_energy_hybrid.png")

    _save_line_plot(df,
                    x_col="max_decode_batch_size",
                    y_cols=["gpu_route_frac"],
                    labels=["GPU-only route fraction"],
                    title="Max Decode Batch Sweep: GPU Route Fraction",
                    ylabel="Fraction",
                    out_path=args.out_dir / f"{args.prefix}_gpu_route_fraction.png",
                    y_lim=(0.0, 1.0))

    _save_dual_axis_plot(df,
                         x_col="max_decode_batch_size",
                         left_col="decode_mean_batch_size",
                         left_label="Mean decode batch size",
                         right_col="num_decode_steps",
                         right_label="Decode steps",
                         title="Max Decode Batch Sweep: Decode Batching",
                         out_path=args.out_dir / f"{args.prefix}_decode_batching.png")

    _save_line_plot(df,
                    x_col="max_decode_batch_size",
                    y_cols=["decode_mean_batch_size"],
                    labels=["Mean decode batch size"],
                    title="Max Decode Batch Sweep: Mean Decode Batch Size",
                    ylabel="Mean decode batch size",
                    out_path=args.out_dir / f"{args.prefix}_decode_mean_batch_size_only.png")

    _save_line_plot(df,
                    x_col="max_decode_batch_size",
                    y_cols=["num_decode_steps"],
                    labels=["Decode steps"],
                    title="Max Decode Batch Sweep: Decode Steps",
                    ylabel="Decode steps",
                    out_path=args.out_dir / f"{args.prefix}_decode_steps_only.png")

    _save_dual_axis_plot(df,
                         x_col="max_decode_batch_size",
                         left_col="gpu_util",
                         left_label="GPU util",
                         right_col="pim_util",
                         right_label="PIM util",
                         title="Max Decode Batch Sweep: Resource Utilization",
                         out_path=args.out_dir / f"{args.prefix}_utilization.png")

    _save_line_plot(df,
                    x_col="max_decode_batch_size",
                    y_cols=["gpu_util"],
                    labels=["GPU util"],
                    title="Max Decode Batch Sweep: GPU Utilization",
                    ylabel="Utilization",
                    out_path=args.out_dir / f"{args.prefix}_gpu_util_only.png",
                    y_lim=(0.0, 1.0))

    _save_line_plot(df,
                    x_col="max_decode_batch_size",
                    y_cols=["pim_util"],
                    labels=["PIM util"],
                    title="Max Decode Batch Sweep: PIM Utilization",
                    ylabel="Utilization",
                    out_path=args.out_dir / f"{args.prefix}_pim_util_only.png",
                    y_lim=(0.0, 1.0))

    print(f"Wrote merged CSV: {merged_csv}")
    print(f"Wrote plots to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
