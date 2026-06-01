#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


FILENAME_RE = re.compile(r"replay_summary_factor(?P<tag>[0-9p]+)\.json$")


def tag_to_value(tag: str) -> float:
    return float(tag.replace("p", "."))


def load_rows(summary_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(summary_dir.glob("replay_summary_factor*.json")):
        m = FILENAME_RE.match(path.name)
        if not m:
            continue
        factor = tag_to_value(m.group("tag"))
        with path.open() as f:
            data = json.load(f)
        route_counts = data.get("route_counts", {})
        admission = data.get("admission", {})
        queue = data.get("admission_queue", {})
        total_reqs = max(1.0, float(data.get("total_requests", 0.0)))
        rows.append({
            "energy_efficiency_factor_ms_per_joule": factor,
            "gpu_only_count": float(route_counts.get("gpu_only", 0.0)),
            "hybrid_count": float(route_counts.get("lpddr5_pim_bank", 0.0)),
            "gpu_only_frac": float(route_counts.get("gpu_only", 0.0)) / total_reqs,
            "hybrid_frac": float(route_counts.get("lpddr5_pim_bank", 0.0)) / total_reqs,
            "energy_total_nj": float(data.get("energy_nj", {}).get("total", 0.0)),
            "decode_energy_per_token_nj": float(data.get("decode_energy_nj_per_decode_token", 0.0)),
            "ttft_p95_ms": float(data.get("ttft_ms", {}).get("p95", 0.0)),
            "e2e_p95_ms": float(data.get("e2e_ms", {}).get("p95", 0.0)),
            "tbt_p95_ms": float(data.get("tbt_ms", {}).get("p95", 0.0)),
            "held_count": float(admission.get("held_count", 0.0)),
            "bypassed_count": float(admission.get("bypassed_count", 0.0)),
            "best_effort_count": float(admission.get("best_effort_count", 0.0)),
            "queue_max_len": float(queue.get("max_len", 0.0)),
            "queue_mean_wait_ms": float(queue.get("mean_wait_ms", 0.0)),
            "throughput_tokps": float(data.get("throughput_tokps", 0.0)),
        })
    if not rows:
        raise FileNotFoundError(f"No replay_summary_factor*.json files found in {summary_dir}")
    return pd.DataFrame(rows).sort_values("energy_efficiency_factor_ms_per_joule").reset_index(drop=True)


def plot_lines(df: pd.DataFrame,
               cols: list[str],
               labels: list[str],
               title: str,
               ylabel: str,
               out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for col, label in zip(cols, labels):
        ax.plot(df["energy_efficiency_factor_ms_per_joule"], df[col], marker="o", linewidth=2, label=label)
    ax.set_xlabel("Energy efficiency factor (ms/J)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if len(cols) > 1:
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot linear-guard energy-efficiency-factor sweep results.")
    ap.add_argument("--summary-dir", type=Path,
                    default=Path("cluster_outputs/azure_energy_efficiency_factor_sweep"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("cluster_outputs/azure_energy_efficiency_factor_sweep_plots"))
    ap.add_argument("--prefix", default="azure_energy_efficiency_factor_sweep")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = load_rows(args.summary_dir)
    df.to_csv(args.out_dir / f"{args.prefix}_summary_table.csv", index=False)

    plot_lines(df,
               ["gpu_only_count", "hybrid_count"],
               ["gpu_only route count", "lpddr5_pim_bank route count"],
               "Route Counts vs Energy Efficiency Factor",
               "Requests",
               args.out_dir / f"{args.prefix}_route_counts.png")

    plot_lines(df,
               ["gpu_only_frac", "hybrid_frac"],
               ["gpu_only fraction", "lpddr5_pim_bank fraction"],
               "Route Fractions vs Energy Efficiency Factor",
               "Fraction of requests",
               args.out_dir / f"{args.prefix}_route_fractions.png")

    plot_lines(df,
               ["energy_total_nj"],
               ["Total request energy"],
               "Total Energy vs Energy Efficiency Factor",
               "Energy (nJ)",
               args.out_dir / f"{args.prefix}_energy_total.png")

    plot_lines(df,
               ["decode_energy_per_token_nj"],
               ["Decode energy per token"],
               "Decode Energy Per Token vs Energy Efficiency Factor",
               "Energy per decode token (nJ)",
               args.out_dir / f"{args.prefix}_decode_energy_per_token.png")

    plot_lines(df,
               ["throughput_tokps"],
               ["Throughput"],
               "Throughput vs Energy Efficiency Factor",
               "Throughput (tok/s)",
               args.out_dir / f"{args.prefix}_throughput_tokps.png")

    plot_lines(df,
               ["ttft_p95_ms"],
               ["TTFT p95"],
               "TTFT p95 vs Energy Efficiency Factor",
               "TTFT p95 (ms)",
               args.out_dir / f"{args.prefix}_ttft_p95.png")

    plot_lines(df,
               ["e2e_p95_ms"],
               ["E2E p95"],
               "E2E p95 vs Energy Efficiency Factor",
               "E2E p95 (ms)",
               args.out_dir / f"{args.prefix}_e2e_p95.png")

    plot_lines(df,
               ["tbt_p95_ms"],
               ["TBT p95"],
               "TBT p95 vs Energy Efficiency Factor",
               "TBT p95 (ms)",
               args.out_dir / f"{args.prefix}_tbt_p95.png")

    plot_lines(df,
               ["held_count", "bypassed_count", "best_effort_count"],
               ["held", "bypassed", "best-effort"],
               "Admission Outcomes vs Energy Efficiency Factor",
               "Count",
               args.out_dir / f"{args.prefix}_admission_counts.png")

    plot_lines(df,
               ["queue_max_len", "queue_mean_wait_ms"],
               ["queue max len", "queue mean wait ms"],
               "Queue Pressure vs Energy Efficiency Factor",
               "Value",
               args.out_dir / f"{args.prefix}_queue_pressure.png")

    print(f"Wrote sweep summary table: {args.out_dir / f'{args.prefix}_summary_table.csv'}")
    print(f"Wrote sweep plots to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
