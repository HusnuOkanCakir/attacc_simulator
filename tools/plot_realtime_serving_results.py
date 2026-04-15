#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TIMELINE_IN_PER_REQUEST = 0.28
MAX_FIGURE_HEIGHT_IN = 30

STATE_COLORS = {
    "prefill_window": "#aec7e8",
    "decode_window": "#ffbb78",
}


def _safe_prefix(path: Path, prefix: Optional[str]) -> str:
    if prefix:
        return prefix
    return path.stem


def _ecdf(values: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.array([]), np.array([])
    x = np.sort(arr)
    y = np.arange(1, x.size + 1) / x.size
    return x, y


def _plot_latency_ecdf(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False
    for col, label in [
        ("ttft_ms", "TTFT"),
        ("e2e_ms", "E2E"),
        ("tbt_ms", "Mean TBT"),
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
    ax.set_title("Realtime Serving Latency ECDF")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_arrival_scatter(df: pd.DataFrame, out_path: Path) -> None:
    cols = [c for c in ["ttft_ms", "e2e_ms", "tbt_ms"] if c in df.columns]
    if not cols or "arrival_ms" not in df.columns:
        return
    fig, axes = plt.subplots(1, len(cols), figsize=(6 * len(cols), 5), sharex=True)
    if len(cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        for route, sub_df in df.groupby("route"):
            ax.scatter(sub_df["arrival_ms"], sub_df[col], s=18, alpha=0.65, label=route)
        ax.set_xlabel("Arrival time (ms)")
        ax.set_ylabel(f"{col} (ms)")
        ax.set_title(f"Arrival vs {col}")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_route_mix(df: pd.DataFrame, summary: Optional[dict], out_path: Path) -> None:
    counts = None
    if summary:
        counts = (summary.get("frontend", {}) or {}).get("route_counts")
    if not counts:
        counts = df["route"].value_counts(dropna=False).to_dict()
    if not counts:
        return

    labels = list(counts.keys())
    values = [counts[k] for k in labels]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=["#4c78a8", "#f58518", "#54a24b", "#b279a2"][: len(labels)])
    ax.set_ylabel("Requests")
    ax.set_title("Realtime Route Mix")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(value), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_request_execution_timeline(df: pd.DataFrame, out_path: Path) -> None:
    required = {"request_id", "arrival_ms", "ttft_ms", "e2e_ms", "route"}
    if not required.issubset(df.columns):
        return
    df = df.copy().sort_values(["arrival_ms", "request_id"])
    fig_h = min(MAX_FIGURE_HEIGHT_IN, max(4, TIMELINE_IN_PER_REQUEST * len(df) + 1.5))
    fig, ax = plt.subplots(figsize=(12, fig_h))
    annotate_rows = len(df) <= 300

    yticks = []
    ylabels = []
    for row_idx, (_, row) in enumerate(df.iterrows()):
        request_id = int(row["request_id"])
        arrival = float(row["arrival_ms"])
        ttft = float(row["ttft_ms"])
        e2e = float(row["e2e_ms"])
        first_token = arrival + ttft
        finish = arrival + e2e
        y = row_idx
        yticks.append(y + 0.4)
        ylabels.append(f"r{request_id}")

        if first_token > arrival:
            ax.broken_barh([(arrival, first_token - arrival)], (y + 0.05, 0.7),
                           facecolors=STATE_COLORS["prefill_window"], edgecolors="black", linewidth=0.4)
        if finish > first_token:
            ax.broken_barh([(first_token, finish - first_token)], (y + 0.05, 0.7),
                           facecolors=STATE_COLORS["decode_window"], edgecolors="black", linewidth=0.4)

        if annotate_rows:
            ax.text(finish + max(0.01, finish * 0.005), y + 0.4, str(row.get("route", "-")), va="center", fontsize=8)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=STATE_COLORS["prefill_window"], edgecolor="black", linewidth=0.4),
        plt.Rectangle((0, 0), 1, 1, facecolor=STATE_COLORS["decode_window"], edgecolor="black", linewidth=0.4),
    ]
    ax.legend(handles, ["prefill window", "decode window"], loc="lower right")
    ax.set_xlabel("Simulation time (ms)")
    ax.set_ylabel("Request")
    ax.set_title("Realtime Request Execution Timeline")
    tick_step = max(1, int(math.ceil(len(yticks) / 80.0)))
    ax.set_yticks(yticks[::tick_step])
    ax.set_yticklabels(ylabels[::tick_step], fontsize=8 if len(yticks) <= 200 else 6)
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description="Plot diagnostics for realtime serving per-request CSV output.")
    p.add_argument("--requests-csv", type=Path, required=True, help="Realtime per-request CSV")
    p.add_argument("--summary-json", type=Path, default=None, help="Optional realtime summary JSON")
    p.add_argument("--out-dir", type=Path, default=Path("cluster_outputs/realtime_plots"), help="Output directory")
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
    _plot_route_mix(df, summary, args.out_dir / f"{prefix}_route_mix.png")
    _plot_request_execution_timeline(df, args.out_dir / f"{prefix}_request_execution_timeline.png")

    print(f"Wrote plots to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
