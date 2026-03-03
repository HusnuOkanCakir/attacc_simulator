#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ecdf(values: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.array([]), np.array([])
    x = np.sort(arr)
    y = np.arange(1, x.size + 1) / x.size
    return x, y


def _safe_prefix(path: Path, prefix: Optional[str]) -> str:
    if prefix:
        return prefix
    return path.stem


def _plot_latency_ecdf(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False
    for col, label in [
        ("ttft_ms", "TTFT"),
        ("e2e_ms", "E2E"),
        ("mean_tbt_ms", "Mean TBT"),
    ]:
        if col not in df:
            continue
        x, y = _ecdf(df[col].dropna())
        if x.size == 0:
            continue
        ax.plot(x, y, label=label, linewidth=2)
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xscale("log")
    ax.set_xlabel("Latency (ms, log scale)")
    ax.set_ylabel("ECDF")
    ax.set_title("Replay Latency ECDF")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_arrival_scatter(df: pd.DataFrame, out_path: Path) -> None:
    cols = [c for c in ["ttft_ms", "e2e_ms"] if c in df]
    if not cols or "arrival_ms" not in df:
        return
    fig, axes = plt.subplots(1, len(cols), figsize=(7 * len(cols), 5), sharex=True)
    if len(cols) == 1:
        axes = [axes]
    clipped = (df.get("clipped_lin", False).astype(str).str.lower() == "true") | (
        df.get("clipped_lout", False).astype(str).str.lower() == "true"
    )
    for ax, col in zip(axes, cols):
        ax.scatter(df.loc[~clipped, "arrival_ms"], df.loc[~clipped, col], s=8, alpha=0.35, label="in-range")
        if clipped.any():
            ax.scatter(df.loc[clipped, "arrival_ms"], df.loc[clipped, col], s=10, alpha=0.6, label="clipped")
        ax.set_xlabel("Arrival time (ms)")
        ax.set_ylabel(f"{col} (ms)")
        ax.set_title(f"Arrival vs {col}")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_predicted_vs_actual(df: pd.DataFrame, out_path: Path) -> None:
    if "chosen_predicted_finish_ms" not in df or "e2e_ms" not in df:
        return
    tmp = df[["chosen_predicted_finish_ms", "e2e_ms"]].dropna()
    if tmp.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(tmp["chosen_predicted_finish_ms"], tmp["e2e_ms"], s=10, alpha=0.35)
    lo = min(tmp["chosen_predicted_finish_ms"].min(), tmp["e2e_ms"].min())
    hi = max(tmp["chosen_predicted_finish_ms"].max(), tmp["e2e_ms"].max())
    ax.plot([lo, hi], [lo, hi], "--", color="black", linewidth=1.5, label="ideal")
    ax.set_xlabel("Predicted finish (ms)")
    ax.set_ylabel("Actual E2E (ms)")
    ax.set_title("Predicted Finish vs Actual E2E")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_route_mix(df: pd.DataFrame, summary: Optional[dict], out_path: Path) -> None:
    route_counts = df["route"].fillna("None").value_counts().sort_index()
    reason_counts = df["route_decision_reason"].fillna("None").value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(route_counts.index.astype(str), route_counts.values)
    axes[0].set_title("Route Counts")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(reason_counts.index.astype(str), reason_counts.values)
    axes[1].set_title("Decision Reasons")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(True, axis="y", alpha=0.3)

    if summary:
        fig.suptitle(
            f"Replay Summary: tok/s={summary.get('throughput_tokps', float('nan')):.2f}, "
            f"GPU util={summary.get('gpu_util', float('nan')):.3f}, "
            f"PIM util={summary.get('pim_util', float('nan')):.3f}",
            fontsize=11,
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_slack_hist(df: pd.DataFrame, out_path: Path) -> None:
    if "chosen_slack_ms" not in df:
        return
    vals = pd.to_numeric(df["chosen_slack_ms"], errors="coerce").dropna()
    if vals.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(vals, bins=50, alpha=0.8)
    ax.axvline(0.0, color="red", linestyle="--", linewidth=2, label="deadline boundary")
    ax.set_xlabel("Chosen slack (ms)")
    ax.set_ylabel("Request count")
    ax.set_title("Slack Distribution")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_queue_pressure_hist(df: pd.DataFrame, out_path: Path) -> None:
    if "chosen_queue_pressure_ms" not in df:
        return
    vals = pd.to_numeric(df["chosen_queue_pressure_ms"], errors="coerce").dropna()
    if vals.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(vals, bins=50, alpha=0.8)
    ax.set_xlabel("Chosen queue pressure (ms)")
    ax.set_ylabel("Request count")
    ax.set_title("Queue Pressure Distribution")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_queue_pressure_vs_e2e(df: pd.DataFrame, out_path: Path) -> None:
    if "chosen_queue_pressure_ms" not in df or "e2e_ms" not in df:
        return
    tmp = df[["chosen_queue_pressure_ms", "e2e_ms"]].dropna()
    if tmp.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(tmp["chosen_queue_pressure_ms"], tmp["e2e_ms"], s=10, alpha=0.35)
    ax.set_xlabel("Chosen queue pressure (ms)")
    ax.set_ylabel("Actual E2E (ms)")
    ax.set_title("Queue Pressure vs Actual E2E")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description="Plot replay result diagnostics from per-request CSV and summary JSON.")
    p.add_argument("--requests-csv", type=Path, required=True, help="Per-request replay CSV")
    p.add_argument("--summary-json", type=Path, default=None, help="Optional replay summary JSON")
    p.add_argument("--out-dir", type=Path, default=Path("cluster_outputs/replay_plots"), help="Output directory")
    p.add_argument("--prefix", type=str, default=None, help="Output file prefix")
    args = p.parse_args()

    df = pd.read_csv(args.requests_csv)
    summary = None
    if args.summary_json and args.summary_json.exists():
        with args.summary_json.open() as f:
            summary = json.load(f)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    prefix = _safe_prefix(args.requests_csv, args.prefix)

    _plot_latency_ecdf(df, args.out_dir / f"{prefix}_latency_ecdf.png")
    _plot_arrival_scatter(df, args.out_dir / f"{prefix}_arrival_scatter.png")
    _plot_predicted_vs_actual(df, args.out_dir / f"{prefix}_predicted_vs_actual.png")
    _plot_route_mix(df, summary, args.out_dir / f"{prefix}_route_mix.png")
    _plot_slack_hist(df, args.out_dir / f"{prefix}_slack_hist.png")
    _plot_queue_pressure_hist(df, args.out_dir / f"{prefix}_queue_pressure_hist.png")
    _plot_queue_pressure_vs_e2e(df, args.out_dir / f"{prefix}_queue_pressure_vs_e2e.png")

    print(f"Wrote plots to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
