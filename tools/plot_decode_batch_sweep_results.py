#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd


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
    for label in ("nobatch", "batch2", "batch4"):
        path = summary_dir / f"replay_summary_{label}.json"
        if not path.exists():
            continue
        with path.open() as f:
            d = json.load(f)
        batching = d.get("decode_batching", {})
        rows.append({
            "case": label,
            "throughput_tokps": d["throughput_tokps"],
            "throughput_rps": d["throughput_rps"],
            "ttft_p95": d["ttft_ms"]["p95"],
            "e2e_p95": d["e2e_ms"]["p95"],
            "tbt_p95": d["tbt_ms"]["p95"],
            "gpu_util": d["gpu_util"],
            "pim_util": d["pim_util"],
            "mean_batch_size": batching.get("mean_batch_size", 0.0),
            "max_batch_size": batching.get("max_batch_size", 0.0),
            "num_decode_steps": batching.get("num_decode_steps", 0.0),
        })
    if not rows:
        raise FileNotFoundError(f"No decode-batch summary JSONs found in {summary_dir}")
    df = pd.DataFrame(rows)
    order = ["nobatch", "batch2", "batch4"]
    df["case"] = pd.Categorical(df["case"], categories=order, ordered=True)
    return df.sort_values("case").reset_index(drop=True)


def _bar_plot(df: pd.DataFrame, x: str, y: str, ylabel: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(df[x], df[y], color=["#4C72B0", "#55A868", "#C44E52"][:len(df)])
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
    p = argparse.ArgumentParser(description="Plot decode-batching sweep results.")
    p.add_argument("--summary-dir",
                   type=Path,
                   default=Path("cluster_outputs/decode_batch_sweep"),
                   help="Directory containing replay_summary_{nobatch,batch2,batch4}.json")
    p.add_argument("--out-dir",
                   type=Path,
                   default=Path("cluster_outputs/decode_batch_sweep_plots"),
                   help="Output directory for plots")
    p.add_argument("--prefix",
                   type=str,
                   default="decode_batch",
                   help="Output filename prefix")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = _load_rows(args.summary_dir)
    summary_csv = args.out_dir / f"{args.prefix}_summary.csv"
    df.to_csv(summary_csv, index=False)
    print(f"Wrote merged CSV: {summary_csv}")

    _bar_plot(df, "case", "throughput_tokps", "Throughput (tok/s)",
              args.out_dir / f"{args.prefix}_throughput.png")
    _bar_plot(df, "case", "ttft_p95", "TTFT p95 (ms)",
              args.out_dir / f"{args.prefix}_ttft_p95.png")
    _bar_plot(df, "case", "e2e_p95", "E2E p95 (ms)",
              args.out_dir / f"{args.prefix}_e2e_p95.png")
    _bar_plot(df, "case", "tbt_p95", "TBT p95 (ms)",
              args.out_dir / f"{args.prefix}_tbt_p95.png")
    _bar_plot(df, "case", "mean_batch_size", "Mean decode batch size",
              args.out_dir / f"{args.prefix}_mean_batch_size.png")
    _bar_plot(df, "case", "num_decode_steps", "Decode steps",
              args.out_dir / f"{args.prefix}_num_decode_steps.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["case"], df["gpu_util"], marker="o", label="GPU util")
    ax.plot(df["case"], df["pim_util"], marker="o", label="PIM util")
    ax.set_xlabel("Case")
    ax.set_ylabel("Utilization")
    ax.set_title("Resource Utilization")
    ax.grid(alpha=0.25)
    ax.legend()
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    util_path = args.out_dir / f"{args.prefix}_utilization.png"
    fig.savefig(util_path, dpi=180)
    plt.close(fig)

    print(f"Wrote plots to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
