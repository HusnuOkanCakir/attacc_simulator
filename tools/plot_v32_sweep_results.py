#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd


ORDER = ["v31_best", "v32_prefill2", "v32_prefill4"]


def _format_bar_value(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:.0f}"
    if abs(value) >= 100:
        return f"{value:.1f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.3f}"


def _load_rows(summary_dir: Path) -> pd.DataFrame:
    rows: List[dict] = []
    for label in ORDER:
        path = summary_dir / f"replay_summary_{label}.json"
        if not path.exists():
            continue
        with path.open() as f:
            d = json.load(f)
        decode_batching = d.get("decode_batching", {})
        prefill_batching = d.get("prefill_batching", {})
        route_counts = d.get("route_counts", {})
        total_routed = sum(route_counts.values()) or 1
        rows.append({
            "case": label,
            "throughput_tokps": d["throughput_tokps"],
            "throughput_rps": d["throughput_rps"],
            "ttft_p95": d["ttft_ms"]["p95"],
            "prefill_wait_p95": d["prefill_wait_ms"]["p95"],
            "e2e_p95": d["e2e_ms"]["p95"],
            "tbt_p95": d["tbt_ms"]["p95"],
            "gpu_util": d["gpu_util"],
            "pim_util": d["pim_util"],
            "gpu_route_frac": route_counts.get("gpu_only", 0) / total_routed,
            "decode_mean_batch_size": decode_batching.get("mean_batch_size", 0.0),
            "prefill_mean_batch_size": prefill_batching.get("mean_batch_size", 0.0),
            "decode_steps": decode_batching.get("num_decode_steps", 0.0),
            "prefill_steps": prefill_batching.get("num_prefill_steps", 0.0),
        })
    if not rows:
        raise FileNotFoundError(f"No v3.2 summary JSONs found in {summary_dir}")
    df = pd.DataFrame(rows)
    df["case"] = pd.Categorical(df["case"], categories=ORDER, ordered=True)
    return df.sort_values("case").reset_index(drop=True)


def _bar_plot(df: pd.DataFrame, y: str, ylabel: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    bars = ax.bar(df["case"], df[y], color=colors[:len(df)])
    ax.set_xlabel("Case")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=35)
    ymax = float(df[y].max()) if len(df) else 0.0
    if ymax > 0:
        ax.set_ylim(top=ymax * 1.15)
    ax.bar_label(bars, labels=[_format_bar_value(v) for v in df[y]], padding=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description="Plot v3.2 prefill batching sweep results.")
    p.add_argument("--summary-dir",
                   type=Path,
                   default=Path("cluster_outputs/v32_sweep"),
                   help="Directory containing replay_summary_*.json for v3.2 cases")
    p.add_argument("--out-dir",
                   type=Path,
                   default=Path("cluster_outputs/v32_sweep_plots"),
                   help="Output directory for plots")
    p.add_argument("--prefix",
                   type=str,
                   default="v32",
                   help="Output filename prefix")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = _load_rows(args.summary_dir)
    summary_csv = args.out_dir / f"{args.prefix}_summary.csv"
    df.to_csv(summary_csv, index=False)
    print(f"Wrote merged CSV: {summary_csv}")

    _bar_plot(df, "throughput_tokps", "Throughput (tok/s)", args.out_dir / f"{args.prefix}_throughput.png")
    _bar_plot(df, "ttft_p95", "TTFT p95 (ms)", args.out_dir / f"{args.prefix}_ttft_p95.png")
    _bar_plot(df, "prefill_wait_p95", "Prefill wait p95 (ms)", args.out_dir / f"{args.prefix}_prefill_wait_p95.png")
    _bar_plot(df, "e2e_p95", "E2E p95 (ms)", args.out_dir / f"{args.prefix}_e2e_p95.png")
    _bar_plot(df, "tbt_p95", "TBT p95 (ms)", args.out_dir / f"{args.prefix}_tbt_p95.png")
    _bar_plot(df, "gpu_route_frac", "GPU-only route fraction", args.out_dir / f"{args.prefix}_gpu_route_frac.png")
    _bar_plot(df, "decode_mean_batch_size", "Mean decode batch size", args.out_dir / f"{args.prefix}_decode_mean_batch_size.png")
    _bar_plot(df, "prefill_mean_batch_size", "Mean prefill batch size", args.out_dir / f"{args.prefix}_prefill_mean_batch_size.png")
    _bar_plot(df, "decode_steps", "Decode steps", args.out_dir / f"{args.prefix}_decode_steps.png")
    _bar_plot(df, "prefill_steps", "Prefill steps", args.out_dir / f"{args.prefix}_prefill_steps.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["case"], df["gpu_util"], marker="o", label="GPU util")
    ax.plot(df["case"], df["pim_util"], marker="o", label="PIM util")
    ax.set_xlabel("Case")
    ax.set_ylabel("Utilization")
    ax.set_title("Resource Utilization")
    ax.grid(alpha=0.25)
    ax.legend()
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(args.out_dir / f"{args.prefix}_utilization.png", dpi=180)
    plt.close(fig)

    print(f"Wrote plots to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
