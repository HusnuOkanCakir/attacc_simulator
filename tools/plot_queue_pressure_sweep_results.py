#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


TAG_RE = re.compile(r"replay_summary_g([0-9p]+)_p([0-9p]+)_d([0-9p]+)\.json")


def _dec(token: str) -> float:
    return float(token.replace("p", "."))


def _load_rows(summary_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for path in sorted(summary_dir.glob("replay_summary_*.json")):
        match = TAG_RE.match(path.name)
        if not match:
            continue
        with path.open() as f:
            data = json.load(f)
        rows.append({
            "gpu_queue_alpha": _dec(match.group(1)),
            "pim_queue_alpha": _dec(match.group(2)),
            "decode_token_alpha": _dec(match.group(3)),
            "throughput_tokps": data.get("throughput_tokps"),
            "throughput_rps": data.get("throughput_rps"),
            "slo_e2e_miss_rate": data.get("slo_e2e_miss_rate"),
            "gpu_util": data.get("gpu_util"),
            "pim_util": data.get("pim_util"),
            "route_fallback_count": data.get("route_fallback_count"),
            "ttft_p95": data.get("ttft_ms", {}).get("p95"),
            "e2e_p95": data.get("e2e_ms", {}).get("p95"),
            "tbt_p95": data.get("tbt_ms", {}).get("p95"),
        })
    if not rows:
        raise FileNotFoundError(f"No replay_summary_*.json files found in {summary_dir}")
    return pd.DataFrame(rows).sort_values(
        ["gpu_queue_alpha", "pim_queue_alpha", "decode_token_alpha"]
    ).reset_index(drop=True)


def _pivot_plot(df: pd.DataFrame,
                value_col: str,
                title: str,
                out_path: Path) -> None:
    decode_vals = sorted(df["decode_token_alpha"].unique())
    piv = (df.pivot_table(index="gpu_queue_alpha",
                          columns="pim_queue_alpha",
                          values=value_col,
                          aggfunc="mean")
             .sort_index()
             .sort_index(axis=1))
    fig, axes = plt.subplots(1, len(decode_vals), figsize=(5 * len(decode_vals), 4), sharey=True)
    if len(decode_vals) == 1:
        axes = [axes]
    vmin = df[value_col].min()
    vmax = df[value_col].max()
    for ax, decode_alpha in zip(axes, decode_vals):
        sub = df[df["decode_token_alpha"] == decode_alpha]
        mat = (sub.pivot_table(index="gpu_queue_alpha",
                               columns="pim_queue_alpha",
                               values=value_col,
                               aggfunc="mean")
                 .sort_index()
                 .sort_index(axis=1))
        im = ax.imshow(mat.values, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(mat.columns)))
        ax.set_xticklabels([str(v) for v in mat.columns])
        ax.set_yticks(range(len(mat.index)))
        ax.set_yticklabels([str(v) for v in mat.index])
        ax.set_xlabel("pim_queue_alpha")
        ax.set_ylabel("gpu_queue_alpha")
        ax.set_title(f"decode_token_alpha={decode_alpha}")
    fig.suptitle(title)
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _line_plot(df: pd.DataFrame,
               x_col: str,
               y_col: str,
               group_col: str,
               title: str,
               ylabel: str,
               out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    grouped = df.groupby(group_col)
    for group_val, sub in grouped:
        sub = sub.sort_values(x_col)
        ax.plot(sub[x_col], sub[y_col], marker="o", linewidth=2, label=f"{group_col}={group_val}")
    ax.set_xlabel(x_col)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description="Plot queue-pressure sweep summaries.")
    p.add_argument("--summary-dir",
                   type=Path,
                   default=Path("cluster_outputs/queue_pressure_sweep"),
                   help="Directory containing replay_summary_*.json files")
    p.add_argument("--out-dir",
                   type=Path,
                   default=Path("cluster_outputs/queue_pressure_sweep_plots"),
                   help="Output directory")
    p.add_argument("--prefix",
                   type=str,
                   default="queue_pressure",
                   help="Output filename prefix")
    args = p.parse_args()

    df = _load_rows(args.summary_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    merged_csv = args.out_dir / f"{args.prefix}_summary.csv"
    df.to_csv(merged_csv, index=False)

    _pivot_plot(df,
                value_col="slo_e2e_miss_rate",
                title="Queue-Pressure Sweep: E2E Miss Rate",
                out_path=args.out_dir / f"{args.prefix}_miss_rate_heatmap.png")

    _pivot_plot(df,
                value_col="throughput_tokps",
                title="Queue-Pressure Sweep: Throughput (tok/s)",
                out_path=args.out_dir / f"{args.prefix}_throughput_heatmap.png")

    _line_plot(df,
               x_col="gpu_queue_alpha",
               y_col="slo_e2e_miss_rate",
               group_col="decode_token_alpha",
               title="Miss Rate vs GPU Queue Alpha",
               ylabel="E2E miss rate",
               out_path=args.out_dir / f"{args.prefix}_miss_rate_vs_gpu_alpha.png")

    _line_plot(df,
               x_col="gpu_queue_alpha",
               y_col="throughput_tokps",
               group_col="decode_token_alpha",
               title="Throughput vs GPU Queue Alpha",
               ylabel="Throughput (tok/s)",
               out_path=args.out_dir / f"{args.prefix}_throughput_vs_gpu_alpha.png")

    print(f"Wrote merged CSV: {merged_csv}")
    print(f"Wrote plots to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
