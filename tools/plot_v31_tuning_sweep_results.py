#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd


PATTERN = re.compile(r"replay_summary_cap(\d+)_guard(\d+)_cons(\d+)\.json")


def _load_rows(summary_dir: Path) -> pd.DataFrame:
    rows: List[dict] = []
    for path in sorted(summary_dir.glob("replay_summary_*.json")):
        match = PATTERN.match(path.name)
        if not match:
            continue
        with path.open() as f:
            data = json.load(f)
        batching = data.get("decode_batching", {})
        local = data.get("local_scheduling", {})
        route_counts = data.get("route_counts", {})
        total_routed = sum(route_counts.values()) or 1
        rows.append({
            "decode_cap": int(match.group(1)),
            "prefill_guard_ms": int(match.group(2)),
            "max_consecutive_decode_batches": int(match.group(3)),
            "throughput_tokps": data["throughput_tokps"],
            "throughput_rps": data["throughput_rps"],
            "ttft_p95": data["ttft_ms"]["p95"],
            "prefill_wait_p95": data["prefill_wait_ms"]["p95"],
            "e2e_p95": data["e2e_ms"]["p95"],
            "tbt_p95": data["tbt_ms"]["p95"],
            "gpu_util": data["gpu_util"],
            "pim_util": data["pim_util"],
            "gpu_route_frac": route_counts.get("gpu_only", 0) / total_routed,
            "mean_batch_size": batching.get("mean_batch_size", 0.0),
            "max_batch_size": batching.get("max_batch_size", 0.0),
            "num_decode_steps": batching.get("num_decode_steps", 0.0),
            "prefill_guard_trigger_count": local.get("prefill_guard_trigger_count", 0.0),
            "decode_limit_trigger_count": local.get("decode_limit_trigger_count", 0.0),
        })
    if not rows:
        raise FileNotFoundError(f"No tuning sweep summary JSONs found in {summary_dir}")
    df = pd.DataFrame(rows)
    return df.sort_values(
        ["max_consecutive_decode_batches", "prefill_guard_ms", "decode_cap"]
    ).reset_index(drop=True)


def _heatmap_grid(
    df: pd.DataFrame,
    metric: str,
    title: str,
    out_path: Path,
    cmap: str = "viridis",
):
    consec_values = sorted(df["max_consecutive_decode_batches"].unique())
    guard_values = sorted(df["prefill_guard_ms"].unique())
    cap_values = sorted(df["decode_cap"].unique())

    fig, axes = plt.subplots(1, len(consec_values), figsize=(5 * len(consec_values), 4.5), squeeze=False)
    vmin = df[metric].min()
    vmax = df[metric].max()
    image = None

    for ax, consec in zip(axes[0], consec_values):
        subset = df[df["max_consecutive_decode_batches"] == consec]
        pivot = (
            subset.pivot(index="prefill_guard_ms", columns="decode_cap", values=metric)
            .reindex(index=guard_values, columns=cap_values)
        )
        image = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"max_consec={consec}")
        ax.set_xlabel("decode_batch_cap_with_prefill")
        ax.set_ylabel("prefill_guard_ms")
        ax.set_xticks(range(len(cap_values)))
        ax.set_xticklabels(cap_values)
        ax.set_yticks(range(len(guard_values)))
        ax.set_yticklabels(guard_values)

        for row_idx, guard in enumerate(guard_values):
            for col_idx, cap in enumerate(cap_values):
                value = pivot.loc[guard, cap]
                if pd.isna(value):
                    continue
                text = f"{value:.1f}" if abs(value) < 1000 else f"{value:.0f}"
                ax.text(col_idx, row_idx, text, ha="center", va="center", color="white", fontsize=8)

    fig.suptitle(title)
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _line_plot(df: pd.DataFrame, metric: str, ylabel: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    for consec in sorted(df["max_consecutive_decode_batches"].unique()):
        subset = df[df["max_consecutive_decode_batches"] == consec].copy()
        subset["label"] = subset.apply(
            lambda row: f"cap{int(row['decode_cap'])}-g{int(row['prefill_guard_ms'])}", axis=1
        )
        top = subset.sort_values(metric, ascending=(metric not in {"throughput_tokps", "gpu_route_frac"})).head(3)
        ax.plot(
            top["label"],
            top[metric],
            marker="o",
            label=f"max_consec={consec}",
        )
    ax.set_xlabel("Top configs per max_consec")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    ax.grid(alpha=0.25)
    ax.legend()
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot v3.1 tuning sweep results.")
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=Path("cluster_outputs/v31_tuning_sweep"),
        help="Directory containing replay_summary_cap*_guard*_cons*.json files",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("cluster_outputs/v31_tuning_sweep_plots"),
        help="Output directory for plots",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="v31_tuning",
        help="Output filename prefix",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = _load_rows(args.summary_dir)

    summary_csv = args.out_dir / f"{args.prefix}_summary.csv"
    df.to_csv(summary_csv, index=False)
    print(f"Wrote merged CSV: {summary_csv}")

    _heatmap_grid(
        df,
        "throughput_tokps",
        "Throughput (tok/s)",
        args.out_dir / f"{args.prefix}_throughput_heatmap.png",
    )
    _heatmap_grid(
        df,
        "ttft_p95",
        "TTFT p95 (ms)",
        args.out_dir / f"{args.prefix}_ttft_p95_heatmap.png",
        cmap="magma_r",
    )
    _heatmap_grid(
        df,
        "prefill_wait_p95",
        "Prefill wait p95 (ms)",
        args.out_dir / f"{args.prefix}_prefill_wait_p95_heatmap.png",
        cmap="magma_r",
    )
    _heatmap_grid(
        df,
        "e2e_p95",
        "E2E p95 (ms)",
        args.out_dir / f"{args.prefix}_e2e_p95_heatmap.png",
        cmap="magma_r",
    )
    _heatmap_grid(
        df,
        "tbt_p95",
        "TBT p95 (ms)",
        args.out_dir / f"{args.prefix}_tbt_p95_heatmap.png",
        cmap="magma_r",
    )
    _heatmap_grid(
        df,
        "gpu_route_frac",
        "GPU-only route fraction",
        args.out_dir / f"{args.prefix}_gpu_route_frac_heatmap.png",
    )

    _line_plot(
        df,
        "throughput_tokps",
        "Top throughput configs per max_consec",
        args.out_dir / f"{args.prefix}_throughput_top3.png",
    )
    _line_plot(
        df,
        "e2e_p95",
        "Top E2E p95 configs per max_consec",
        args.out_dir / f"{args.prefix}_e2e_p95_top3.png",
    )

    print(f"Wrote plots to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
