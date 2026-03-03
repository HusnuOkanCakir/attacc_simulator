#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


SLO_RE = re.compile(r"slo(\d+)")


def _load_rows(summary_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for path in sorted(summary_dir.glob("replay_summary_ml_slack_slo*.json")):
        match = SLO_RE.search(path.stem)
        if not match:
            continue
        slo = int(match.group(1))
        with path.open() as f:
            data = json.load(f)
        rows.append({
            "slo_e2e_ms": slo,
            "throughput_tokps": data.get("throughput_tokps"),
            "throughput_rps": data.get("throughput_rps"),
            "slo_e2e_miss_rate": data.get("slo_e2e_miss_rate"),
            "gpu_util": data.get("gpu_util"),
            "pim_util": data.get("pim_util"),
            "route_fallback_count": data.get("route_fallback_count"),
            "ttft_p95": data.get("ttft_ms", {}).get("p95"),
            "ttft_p99": data.get("ttft_ms", {}).get("p99"),
            "e2e_p95": data.get("e2e_ms", {}).get("p95"),
            "e2e_p99": data.get("e2e_ms", {}).get("p99"),
            "tbt_p95": data.get("tbt_ms", {}).get("p95"),
            "tbt_p99": data.get("tbt_ms", {}).get("p99"),
        })
    if not rows:
        raise FileNotFoundError(f"No replay_summary_ml_slack_slo*.json files found in {summary_dir}")
    df = pd.DataFrame(rows).sort_values("slo_e2e_ms").reset_index(drop=True)
    return df


def _save_line_plot(df: pd.DataFrame,
                    x_col: str,
                    y_cols: List[str],
                    labels: List[str],
                    title: str,
                    ylabel: str,
                    out_path: Path,
                    x_log: bool = True,
                    y_lim: tuple[float, float] | None = None) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for col, label in zip(y_cols, labels):
        ax.plot(df[x_col], df[col], marker="o", linewidth=2, label=label)
    if x_log:
        ax.set_xscale("log")
    if y_lim is not None:
        ax.set_ylim(*y_lim)
    ax.set_xlabel("SLO E2E deadline (ms)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description="Plot replay SLO sweep summary results.")
    p.add_argument("--summary-dir",
                   type=Path,
                   default=Path("cluster_outputs/slo_sweep"),
                   help="Directory containing replay_summary_ml_slack_slo*.json files")
    p.add_argument("--out-dir",
                   type=Path,
                   default=Path("cluster_outputs/slo_sweep_plots"),
                   help="Directory to write plots and merged CSV")
    p.add_argument("--prefix",
                   type=str,
                   default="slo_sweep",
                   help="Output file prefix")
    args = p.parse_args()

    df = _load_rows(args.summary_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    merged_csv = args.out_dir / f"{args.prefix}_summary.csv"
    df.to_csv(merged_csv, index=False)

    _save_line_plot(df,
                    x_col="slo_e2e_ms",
                    y_cols=["slo_e2e_miss_rate"],
                    labels=["E2E miss rate"],
                    title="SLO Sweep: E2E Miss Rate",
                    ylabel="Miss rate",
                    out_path=args.out_dir / f"{args.prefix}_miss_rate.png",
                    y_lim=(0.0, 1.0))

    _save_line_plot(df,
                    x_col="slo_e2e_ms",
                    y_cols=["throughput_tokps", "throughput_rps"],
                    labels=["Throughput (tok/s)", "Throughput (req/s)"],
                    title="SLO Sweep: Throughput",
                    ylabel="Throughput",
                    out_path=args.out_dir / f"{args.prefix}_throughput.png")

    _save_line_plot(df,
                    x_col="slo_e2e_ms",
                    y_cols=["ttft_p95", "e2e_p95", "tbt_p95"],
                    labels=["TTFT p95", "E2E p95", "TBT p95"],
                    title="SLO Sweep: p95 Latencies",
                    ylabel="Latency (ms)",
                    out_path=args.out_dir / f"{args.prefix}_latency_p95.png")

    _save_line_plot(df,
                    x_col="slo_e2e_ms",
                    y_cols=["gpu_util", "pim_util"],
                    labels=["GPU util", "PIM util"],
                    title="SLO Sweep: Resource Utilization",
                    ylabel="Utilization",
                    out_path=args.out_dir / f"{args.prefix}_utilization.png",
                    y_lim=(0.0, 1.0))

    print(f"Wrote merged CSV: {merged_csv}")
    print(f"Wrote plots to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
