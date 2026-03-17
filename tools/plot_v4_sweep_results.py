#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_CASES = [
    "v31_best=cluster_outputs/v32_sweep/replay_summary_v31_best.json",
    "v40_pred=cluster_outputs/replay_summary_v40_pred.json",
    "v41_fcfs_nopred_l1000=cluster_outputs/replay_summary_v41_fcfs_nopred_l1000.json",
    "v41_fcfs_pred_l1000=cluster_outputs/replay_summary_v41_fcfs_pred_l1000.json",
]


def _format_bar_value(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:.0f}"
    if abs(value) >= 100:
        return f"{value:.1f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.3f}"


def _parse_cases(items: List[str]) -> List[tuple[str, Path]]:
    pairs: List[tuple[str, Path]] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --case '{item}', expected label=path")
        label, path = item.split("=", 1)
        label = label.strip()
        p = Path(path.strip())
        if not label or not path:
            raise ValueError(f"Invalid --case '{item}', empty label/path")
        pairs.append((label, p))
    return pairs


def _load_rows(case_pairs: List[tuple[str, Path]]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for label, path in case_pairs:
        if not path.exists():
            raise FileNotFoundError(f"Summary file not found: {path}")
        with path.open() as f:
            d = json.load(f)
        route_counts = d.get("route_counts", {})
        total_routed = sum(route_counts.values()) or 1
        batching = d.get("decode_batching", {})
        local = d.get("local_scheduling", {})
        admission = d.get("admission", {})
        admission_q = d.get("admission_queue", {})
        total_requests = float(d.get("total_requests", 0))
        lost_count = float(admission.get("lost_count", 0.0))
        rows.append({
            "case": label,
            "total_requests": total_requests,
            "completed_requests": float(d.get("completed_requests", 0)),
            "throughput_tokps": float(d["throughput_tokps"]),
            "throughput_rps": float(d["throughput_rps"]),
            "ttft_p95": float(d["ttft_ms"]["p95"]),
            "e2e_p95": float(d["e2e_ms"]["p95"]),
            "tbt_p95": float(d["tbt_ms"]["p95"]),
            "gpu_util": float(d["gpu_util"]),
            "pim_util": float(d["pim_util"]),
            "gpu_route_frac": float(route_counts.get("gpu_only", 0) / total_routed),
            "decode_mean_batch_size": float(batching.get("mean_batch_size", 0.0)),
            "decode_num_steps": float(batching.get("num_decode_steps", 0.0)),
            "admission_reject_count": float(admission.get("reject_count", 0.0)),
            "admission_lost_count": lost_count,
            "admission_shadow_timeout_count": float(admission.get("shadow_timeout_count", 0.0)),
            "admission_queue_max_len": float(admission_q.get("max_len", 0.0)),
            "admission_queue_mean_wait_ms": float(admission_q.get("mean_wait_ms", 0.0)),
            "slo_e2e_miss_rate": float(d["slo_e2e_miss_rate"]) if d.get("slo_e2e_miss_rate") is not None else float("nan"),
            "lost_frac": (lost_count / total_requests) if total_requests > 0 else 0.0,
            "local_policy": str(local.get("local_scheduling_policy", "priority")),
        })
    df = pd.DataFrame(rows)
    order = [label for label, _ in case_pairs]
    df["case"] = pd.Categorical(df["case"], categories=order, ordered=True)
    return df.sort_values("case").reset_index(drop=True)


def _bar_plot(df: pd.DataFrame, y: str, ylabel: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"]
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
    p = argparse.ArgumentParser(description="Plot v4 replay comparisons in the same style as v31 sweep plots.")
    p.add_argument("--case",
                   action="append",
                   default=[],
                   help="Case as label=summary_json. Can be repeated.")
    p.add_argument("--out-dir",
                   type=Path,
                   default=Path("cluster_outputs/v4_sweep_plots"),
                   help="Output directory")
    p.add_argument("--prefix",
                   type=str,
                   default="v4",
                   help="Output filename prefix")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    case_items = args.case if args.case else DEFAULT_CASES
    df = _load_rows(_parse_cases(case_items))

    summary_csv = args.out_dir / f"{args.prefix}_summary.csv"
    df.to_csv(summary_csv, index=False)
    print(f"Wrote merged CSV: {summary_csv}")

    _bar_plot(df, "throughput_tokps", "Throughput (tok/s)", args.out_dir / f"{args.prefix}_throughput.png")
    _bar_plot(df, "ttft_p95", "TTFT p95 (ms)", args.out_dir / f"{args.prefix}_ttft_p95.png")
    _bar_plot(df, "e2e_p95", "E2E p95 (ms)", args.out_dir / f"{args.prefix}_e2e_p95.png")
    _bar_plot(df, "tbt_p95", "TBT p95 (ms)", args.out_dir / f"{args.prefix}_tbt_p95.png")
    _bar_plot(df, "gpu_route_frac", "GPU-only route fraction", args.out_dir / f"{args.prefix}_gpu_route_frac.png")
    _bar_plot(df, "decode_mean_batch_size", "Mean decode batch size",
              args.out_dir / f"{args.prefix}_decode_mean_batch_size.png")
    _bar_plot(df, "decode_num_steps", "Decode steps", args.out_dir / f"{args.prefix}_decode_steps.png")
    _bar_plot(df, "admission_reject_count", "Admission reject count", args.out_dir / f"{args.prefix}_admission_rejects.png")
    _bar_plot(df, "admission_lost_count", "Admission lost count", args.out_dir / f"{args.prefix}_admission_lost.png")
    _bar_plot(df, "admission_shadow_timeout_count", "Admission shadow timeout count",
              args.out_dir / f"{args.prefix}_admission_shadow_timeouts.png")
    _bar_plot(df, "admission_queue_max_len", "Admission queue max length",
              args.out_dir / f"{args.prefix}_admission_queue_max_len.png")
    _bar_plot(df, "admission_queue_mean_wait_ms", "Admission queue mean wait (ms)",
              args.out_dir / f"{args.prefix}_admission_queue_mean_wait_ms.png")
    _bar_plot(df, "slo_e2e_miss_rate", "SLO E2E miss rate", args.out_dir / f"{args.prefix}_slo_e2e_miss_rate.png")
    _bar_plot(df, "lost_frac", "Lost request fraction", args.out_dir / f"{args.prefix}_lost_frac.png")
    _bar_plot(df, "total_requests", "Total requests (run)", args.out_dir / f"{args.prefix}_total_requests.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["case"], df["gpu_util"], marker="o", label="GPU util")
    ax.plot(df["case"], df["pim_util"], marker="o", label="PIM util")
    ax.set_xlabel("Case")
    ax.set_ylabel("Utilization")
    ax.set_title("Resource utilization")
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
