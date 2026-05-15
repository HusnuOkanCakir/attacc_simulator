#!/usr/bin/env python3
"""
Overlay simulator cost-table predictions on NCU/roofline ground truth.

Reads:
  - NCU points CSV from /home/okan/lerobot/prof/runs/<RUN>/  (label, group, stage,
    block_type, attn_subtype, ai, perf, time_s, mem_bw_pct). `time_s` despite
    its label is the kernel-group time in microseconds.
  - Simulator cost-table CSV (e.g. cluster_outputs/cost_tables_full_energy_openvla/
    gpu_only.csv). `s_time` (prefill ms) and `g_time (ms)` (decode-step ms) at
    the (Lin, Lout, bs) row.

Produces a 2-panel figure:
  (a) Grouped bar chart — measured (NCU) vs predicted (sim) ms per stage.
  (b) Per-kernel scatter — NCU kernels color-coded by block_type, with the
      simulator's predicted (prefill, decode) marker overlaid.

Usage:
    python tools/plot_predicted_vs_ncu.py \\
        --ncu-csv /home/okan/lerobot/prof/runs/<RUN>/<...>_points_bs<N>_bf16.csv \\
        --cost-table cluster_outputs/cost_tables_full_energy_openvla/gpu_only.csv \\
        --bs 16 --lin 300 --lout 32 --model openvla \\
        [--out-png plots/predicted_vs_ncu_<run>_bs<N>.png]

NCU runs with batched bs in a single CSV use the file pattern
`*_stage_split_points_bs<N>_bf16.csv`. Use the per-bs file matching --bs.
"""

import argparse
import math
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Per-stage NCU `group` values map to one of two simulator stages.
STAGE_GROUP_MAP = {
    "sum_attn_gemm":  "prefill",
    "sum_attn_other": "prefill",
    "sum_fc":         "prefill",
    "gen_attn_gemm":  "decode",
    "gen_attn_other": "decode",
    "gen_fc":         "decode",
}


def load_ncu(ncu_csv: Path) -> pd.DataFrame:
    """Load NCU points CSV; convert `time_s` (μs) to ms; classify stage."""
    if not ncu_csv.exists():
        sys.exit(f"[error] NCU CSV not found: {ncu_csv}")
    df = pd.read_csv(ncu_csv)
    needed = {"group", "stage", "ai", "time_s", "mem_bw_pct"}
    missing = needed - set(df.columns)
    if missing:
        sys.exit(f"[error] NCU CSV missing required cols: {missing}")
    # The CSV label `time_s` is actually microseconds. Convert to ms.
    df["time_ms"] = df["time_s"].astype(float) / 1000.0
    df["sim_stage"] = df["group"].map(STAGE_GROUP_MAP).fillna("other")
    return df


def aggregate_ncu_by_stage(df: pd.DataFrame) -> dict[str, float]:
    """Sum NCU kernel time (ms) per simulator stage."""
    return df.groupby("sim_stage")["time_ms"].sum().to_dict()


def load_sim_row(cost_csv: Path, lin: int, lout: int, bs: int) -> pd.Series:
    """Look up one row of the cost table for the (Lin, Lout, bs) triple."""
    if not cost_csv.exists():
        sys.exit(f"[error] cost-table CSV not found: {cost_csv}")
    df = pd.read_csv(cost_csv)
    needed = {"Lin", "Lout", "bs"}
    missing = needed - set(df.columns)
    if missing:
        sys.exit(f"[error] cost-table missing cols: {missing} "
                 f"(found {sorted(df.columns)[:10]}...)")

    # Exact match preferred; otherwise nearest neighbor on (Lin, Lout) at this bs.
    exact = df[(df.Lin == lin) & (df.Lout == lout) & (df.bs == bs)]
    if not exact.empty:
        return exact.iloc[0]

    bs_match = df[df.bs == bs]
    if bs_match.empty:
        sys.exit(f"[error] no rows in cost-table at bs={bs}")
    # Nearest by Euclidean distance in (Lin, Lout) space.
    dist = (bs_match["Lin"] - lin)**2 + (bs_match["Lout"] - lout)**2
    nearest = bs_match.loc[dist.idxmin()]
    print(f"[sim] no exact match for (Lin={lin}, Lout={lout}, bs={bs}); "
          f"using nearest (Lin={int(nearest.Lin)}, Lout={int(nearest.Lout)}, bs={bs})")
    return nearest


def extract_sim_times(row: pd.Series) -> dict[str, float]:
    """Extract per-stage time (ms) from a cost-table row."""
    # Column names: `s_time` (prefill ms, no parentheses) and `g_time (ms)`
    # (decode-step ms). Total decode = g_time × Lout.
    prefill_ms = float(row.get("s_time", math.nan))
    decode_step_ms = float(row.get("g_time (ms)", math.nan))
    lout = int(row.get("Lout", 1))
    total_decode_ms = decode_step_ms * max(lout - 1, 0)  # Lout-1 decode steps after prefill emits token 1
    return {
        "prefill": prefill_ms,
        "decode":  total_decode_ms,
        "decode_step": decode_step_ms,
    }


def make_plot(ncu_df: pd.DataFrame,
              ncu_times: dict[str, float],
              sim_times: dict[str, float],
              sim_row: pd.Series,
              model: str,
              ncu_csv: Path,
              cost_csv: Path,
              bs: int, lin: int, lout: int,
              out_png: Path) -> None:
    fig, (ax_bar, ax_scatter) = plt.subplots(1, 2, figsize=(12, 4.8))

    # ── (a) Grouped bar chart: measured vs predicted per stage ─────────────────
    stages = ["prefill", "decode"]
    measured = [ncu_times.get(s, 0.0) for s in stages]
    predicted = [sim_times[s] for s in stages]
    x = np.arange(len(stages))
    width = 0.35
    ax_bar.bar(x - width/2, measured, width, label="Measured (NCU)",
               color="#1f77b4", edgecolor="black")
    ax_bar.bar(x + width/2, predicted, width, label="Predicted (sim)",
               color="#ff7f0e", edgecolor="black")
    for i, (m, p) in enumerate(zip(measured, predicted)):
        ax_bar.text(i - width/2, m, f"{m:.2f}", ha="center", va="bottom", fontsize=8)
        ax_bar.text(i + width/2, p, f"{p:.2f}", ha="center", va="bottom", fontsize=8)
        if m > 0:
            ratio = p / m
            ax_bar.text(i, max(m, p) * 1.15, f"{ratio:.2f}×",
                        ha="center", va="bottom", fontsize=9,
                        color="darkred" if ratio > 1.5 or ratio < 0.5 else "black")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([s.title() for s in stages])
    ax_bar.set_ylabel("Total time (ms)")
    ax_bar.set_title(f"NCU measured vs simulator predicted\n"
                     f"{model} bs={bs} Lin={lin} Lout={lout}", fontsize=10)
    ax_bar.legend(fontsize=9)
    ax_bar.grid(True, axis="y", alpha=0.3)

    # ── (b) Per-kernel scatter on (AI, time_ms) axes ──────────────────────────
    palette = {
        "sum_attn_gemm":  "#1f77b4",
        "sum_attn_other": "#aec7e8",
        "sum_fc":         "#9467bd",
        "gen_attn_gemm":  "#d62728",
        "gen_attn_other": "#ff9896",
        "gen_fc":         "#ff7f0e",
    }
    for grp, sub in ncu_df.groupby("group"):
        ax_scatter.scatter(sub["ai"], sub["time_ms"],
                           s=60, c=palette.get(grp, "gray"),
                           edgecolor="black", linewidth=0.5,
                           label=grp, alpha=0.85)
    # Add simulator points at conventional AI ~ 30 (compute-bound side) for
    # visual reference; user reads bars for quantitative compare.
    sim_ai = sim_row.get("sys_opb", 30.0) if not pd.isna(sim_row.get("sys_opb", math.nan)) else 30.0
    ax_scatter.scatter([sim_ai], [sim_times["prefill"]], marker="*", s=240,
                       c="orange", edgecolor="black", linewidth=1.0,
                       label=f"Sim prefill (AI≈{sim_ai:.0f})", zorder=5)
    ax_scatter.scatter([sim_ai], [sim_times["decode_step"]], marker="*", s=240,
                       c="red", edgecolor="black", linewidth=1.0,
                       label="Sim decode-step", zorder=5)
    ax_scatter.set_xscale("log")
    ax_scatter.set_yscale("log")
    ax_scatter.set_xlabel("Arithmetic intensity (FLOP/byte)")
    ax_scatter.set_ylabel("Kernel time (ms)")
    ax_scatter.set_title("Per-kernel measured + sim overlay", fontsize=10)
    ax_scatter.legend(fontsize=7, loc="best")
    ax_scatter.grid(True, which="both", alpha=0.3)

    fig.suptitle(f"Predicted vs NCU — {model.upper()}  ({ncu_csv.name})",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out_png}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ncu-csv", required=True, type=Path,
                    help="NCU points CSV (e.g. <run>/openvla_stage_split_points_bs16_bf16.csv).")
    ap.add_argument("--cost-table", required=True, type=Path,
                    help="Simulator cost-table CSV "
                         "(e.g. cluster_outputs/cost_tables_full_energy_openvla/gpu_only.csv).")
    ap.add_argument("--model", required=True,
                    help="Model name for plot title (openvla, pi0, ...).")
    ap.add_argument("--bs", type=int, required=True,
                    help="Batch size of the NCU run.")
    ap.add_argument("--lin", type=int, required=True,
                    help="Context length to look up in the cost table (VLA: ≈300 for image+text).")
    ap.add_argument("--lout", type=int, required=True,
                    help="Output tokens to look up (VLA: 7 OpenVLA, 50 Pi0).")
    ap.add_argument("--out-png", type=Path, default=None,
                    help="Output PNG path. Default: <ncu-csv-dir>/predicted_vs_ncu_<stem>.png")
    args = ap.parse_args()

    ncu_df = load_ncu(args.ncu_csv)
    ncu_times = aggregate_ncu_by_stage(ncu_df)
    sim_row = load_sim_row(args.cost_table, args.lin, args.lout, args.bs)
    sim_times = extract_sim_times(sim_row)

    print(f"[ncu]  prefill={ncu_times.get('prefill', 0):.3f} ms  "
          f"decode={ncu_times.get('decode', 0):.3f} ms  "
          f"(from {args.ncu_csv.name})")
    print(f"[sim]  prefill={sim_times['prefill']:.3f} ms  "
          f"decode_step={sim_times['decode_step']:.3f} ms × ({args.lout}-1) = "
          f"{sim_times['decode']:.3f} ms")
    if ncu_times.get("prefill", 0) > 0:
        print(f"[ratio] prefill sim/ncu = {sim_times['prefill'] / ncu_times['prefill']:.2f}×")
    if ncu_times.get("decode", 0) > 0:
        print(f"[ratio] decode  sim/ncu = {sim_times['decode'] / ncu_times['decode']:.2f}×")

    if args.out_png is None:
        stem = args.ncu_csv.stem
        out_png = args.ncu_csv.parent / f"predicted_vs_ncu_{stem}.png"
    else:
        out_png = args.out_png

    make_plot(ncu_df, ncu_times, sim_times, sim_row,
              args.model, args.ncu_csv, args.cost_table,
              args.bs, args.lin, args.lout, out_png)


if __name__ == "__main__":
    main()
