#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd


ORDER = ["nobatch", "batch2", "batch4", "batch4_guard", "batch4_guard_cap"]


def _load_rows(summary_dir: Path) -> pd.DataFrame:
    rows: List[dict] = []
    for label in ORDER:
        path = summary_dir / f"replay_summary_{label}.json"
        if not path.exists():
            continue
        with path.open() as f:
            d = json.load(f)
        batching = d.get("decode_batching", {})
        local = d.get("local_scheduling", {})
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
            "mean_batch_size": batching.get("mean_batch_size", 0.0),
            "max_batch_size": batching.get("max_batch_size", 0.0),
            "num_decode_steps": batching.get("num_decode_steps", 0.0),
            "prefill_guard_trigger_count": local.get("prefill_guard_trigger_count", 0.0),
            "decode_limit_trigger_count": local.get("decode_limit_trigger_count", 0.0),
        })
    if not rows:
        raise FileNotFoundError(f"No v3.1 summary JSONs found in {summary_dir}")
    df = pd.DataFrame(rows)
    df["case"] = pd.Categorical(df["case"], categories=ORDER, ordered=True)
    return df.sort_values("case").reset_index(drop=True)


def _bar_plot(df: pd.DataFrame, y: str, ylabel: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(df["case"], df[y], color=["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"][:len(df)])
    ax.set_xlabel("Case")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description="Plot v3.1 prompt-aware batching sweep results.")
    p.add_argument("--summary-dir",
                   type=Path,
                   default=Path("cluster_outputs/v31_sweep"),
                   help="Directory containing replay_summary_*.json for v3.1 cases")
    p.add_argument("--out-dir",
                   type=Path,
                   default=Path("cluster_outputs/v31_sweep_plots"),
                   help="Output directory for plots")
    p.add_argument("--prefix",
                   type=str,
                   default="v31",
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
    _bar_plot(df, "mean_batch_size", "Mean decode batch size", args.out_dir / f"{args.prefix}_mean_batch_size.png")
    _bar_plot(df, "num_decode_steps", "Decode steps", args.out_dir / f"{args.prefix}_num_decode_steps.png")
    _bar_plot(df, "gpu_route_frac", "GPU-only route fraction", args.out_dir / f"{args.prefix}_gpu_route_frac.png")
    _bar_plot(df, "prefill_guard_trigger_count", "Prefill guard trigger count",
              args.out_dir / f"{args.prefix}_prefill_guard_triggers.png")
    _bar_plot(df, "decode_limit_trigger_count", "Decode limit trigger count",
              args.out_dir / f"{args.prefix}_decode_limit_triggers.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["case"], df["gpu_util"], marker="o", label="GPU util")
    ax.plot(df["case"], df["pim_util"], marker="o", label="PIM util")
    ax.set_xlabel("Case")
    ax.set_ylabel("Utilization")
    ax.set_title("Resource Utilization")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out_dir / f"{args.prefix}_utilization.png", dpi=180)
    plt.close(fig)

    print(f"Wrote plots to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
