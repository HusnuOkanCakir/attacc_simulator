#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


FILENAME_RE = re.compile(r"replay_summary_wait(?P<tag>[0-9p]+)\.json$")


def tag_to_value(tag: str) -> float:
    return float(tag.replace("p", "."))


def load_rows(summary_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(summary_dir.glob("replay_summary_wait*.json")):
        m = FILENAME_RE.match(path.name)
        if not m:
            continue
        wait_ms = tag_to_value(m.group("tag"))
        with path.open() as f:
            data = json.load(f)
        route_counts = data.get("route_counts", {})
        admission = data.get("admission", {})
        queue = data.get("admission_queue", {})
        total_reqs = max(1.0, float(data.get("total_requests", 0.0)))
        rows.append({
            "wait_ms": wait_ms,
            "gpu_only_count": float(route_counts.get("gpu_only", 0.0)),
            "hybrid_count": float(route_counts.get("lpddr5_pim_bank", 0.0)),
            "gpu_only_frac": float(route_counts.get("gpu_only", 0.0)) / total_reqs,
            "hybrid_frac": float(route_counts.get("lpddr5_pim_bank", 0.0)) / total_reqs,
            "held_count": float(admission.get("held_count", 0.0)),
            "blocked_count": float(admission.get("blocked_count", 0.0)),
            "bypassed_count": float(admission.get("bypassed_count", 0.0)),
            "best_effort_count": float(admission.get("best_effort_count", 0.0)),
            "blocked_due_to_tbt_count": float(admission.get("blocked_due_to_tbt_count", 0.0)),
            "queue_max_len": float(queue.get("max_len", 0.0)),
            "queue_mean_wait_ms": float(queue.get("mean_wait_ms", 0.0)),
            "throughput_tokps": float(data.get("throughput_tokps", 0.0)),
            "ttft_p95_ms": float(data.get("ttft_ms", {}).get("p95", 0.0)),
            "e2e_p95_ms": float(data.get("e2e_ms", {}).get("p95", 0.0)),
            "tbt_p95_ms": float(data.get("tbt_ms", {}).get("p95", 0.0)),
            "slo_e2e_miss_rate": float(data.get("slo_e2e_miss_rate", 0.0) or 0.0),
        })
    if not rows:
        raise FileNotFoundError(f"No replay_summary_wait*.json files found in {summary_dir}")
    return pd.DataFrame(rows).sort_values("wait_ms").reset_index(drop=True)


def plot_lines(df: pd.DataFrame,
               x_col: str,
               cols: list[str],
               labels: list[str],
               title: str,
               xlabel: str,
               ylabel: str,
               out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for col, label in zip(cols, labels):
        ax.plot(df[x_col], df[col], marker="o", linewidth=2, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if len(cols) > 1:
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot admission_max_wait_ms sweep results.")
    ap.add_argument("--summary-dir", type=Path,
                    default=Path("cluster_outputs/azure_admission_wait_sweep"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("cluster_outputs/azure_admission_wait_sweep_plots"))
    ap.add_argument("--prefix", default="azure_admission_wait_sweep")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = load_rows(args.summary_dir)
    df.to_csv(args.out_dir / f"{args.prefix}_summary_table.csv", index=False)

    plot_lines(df, "wait_ms",
               ["held_count", "bypassed_count", "best_effort_count", "blocked_count"],
               ["held", "bypassed", "best-effort", "blocked"],
               "Admission Outcomes vs Max Wait",
               "admission_max_wait_ms",
               "Count",
               args.out_dir / f"{args.prefix}_admission_counts.png")

    plot_lines(df, "wait_ms",
               ["queue_max_len", "queue_mean_wait_ms"],
               ["queue max len", "queue mean wait ms"],
               "Queue Pressure vs Max Wait",
               "admission_max_wait_ms",
               "Value",
               args.out_dir / f"{args.prefix}_queue_pressure.png")

    plot_lines(df, "wait_ms",
               ["ttft_p95_ms"],
               ["TTFT p95"],
               "TTFT p95 vs Max Wait",
               "admission_max_wait_ms",
               "TTFT p95 (ms)",
               args.out_dir / f"{args.prefix}_ttft_p95.png")

    plot_lines(df, "wait_ms",
               ["e2e_p95_ms"],
               ["E2E p95"],
               "E2E p95 vs Max Wait",
               "admission_max_wait_ms",
               "E2E p95 (ms)",
               args.out_dir / f"{args.prefix}_e2e_p95.png")

    plot_lines(df, "wait_ms",
               ["tbt_p95_ms"],
               ["TBT p95"],
               "TBT p95 vs Max Wait",
               "admission_max_wait_ms",
               "TBT p95 (ms)",
               args.out_dir / f"{args.prefix}_tbt_p95.png")

    plot_lines(df, "wait_ms",
               ["throughput_tokps"],
               ["Throughput"],
               "Throughput vs Max Wait",
               "admission_max_wait_ms",
               "Throughput (tok/s)",
               args.out_dir / f"{args.prefix}_throughput_tokps.png")

    plot_lines(df, "wait_ms",
               ["slo_e2e_miss_rate"],
               ["E2E miss rate"],
               "E2E Miss Rate vs Max Wait",
               "admission_max_wait_ms",
               "Miss rate",
               args.out_dir / f"{args.prefix}_slo_e2e_miss_rate.png")

    plot_lines(df, "wait_ms",
               ["gpu_only_count", "hybrid_count"],
               ["gpu_only route count", "lpddr5_pim_bank route count"],
               "Route Counts vs Max Wait",
               "admission_max_wait_ms",
               "Requests",
               args.out_dir / f"{args.prefix}_route_counts.png")

    plot_lines(df, "wait_ms",
               ["gpu_only_frac", "hybrid_frac"],
               ["gpu_only fraction", "lpddr5_pim_bank fraction"],
               "Route Fractions vs Max Wait",
               "admission_max_wait_ms",
               "Fraction of requests",
               args.out_dir / f"{args.prefix}_route_fractions.png")

    print(f"Wrote sweep summary table: {args.out_dir / f'{args.prefix}_summary_table.csv'}")
    print(f"Wrote sweep plots to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
