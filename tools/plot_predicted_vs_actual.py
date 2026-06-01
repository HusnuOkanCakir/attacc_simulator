#!/usr/bin/env python3
"""
Post-hoc predicted-vs-actual scatter for serving_online runs.

Reads requests_out.csv from one or more policy run directories, queries the
HistGBR cost-model bundles (.pkl) at the (Lin, Lout, bs=1) the simulator used,
and plots predicted vs actual prefill / decode-total / E2E latency on log-log
axes. The diagonal indicates perfect agreement.

For new runs that include scheduler_predicted_* columns, this also emits a
second plot comparing the scheduler's admission-time predicted finish/E2E
against the actual completion/E2E observed under online load.

Usage:
  python tools/plot_predicted_vs_actual.py \
      --run-dir cluster_outputs/online_serving_runs/<RUN> \
      --gpu-pkl    cluster_outputs/cost_models_full_energy_openvla/gpu_only.pkl \
      --hybrid-pkl cluster_outputs/cost_models_full_energy_openvla/lpddr5_pim_bank.pkl
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROUTE_COLORS = {
    "gpu_only":        "#ff7f0e",
    "lpddr5_pim_bank": "#1f77b4",
}


def _predict(bundle: dict, lin: int, lout: int, bs: int = 1) -> tuple[float, float]:
    """Return (prefill_e2e_ms, decode_e2e_ms) for one (Lin, Lout, bs) tuple."""
    feat = bundle.get("feature_names", ["Lin", "Lout", "bs"])
    x = pd.DataFrame([[lin, lout, bs]], columns=feat)
    pre = float(bundle["estimators"]["prefill_e2e_ms"].predict(x)[0])
    dec = float(bundle["estimators"]["decode_e2e_ms"].predict(x)[0])
    return pre, dec


def _augment_with_predictions(df: pd.DataFrame, bundles: dict[str, dict],
                               vision_prefix_tokens: int) -> pd.DataFrame:
    df = df.copy()
    pred_pre = np.full(len(df), np.nan)
    pred_dec = np.full(len(df), np.nan)
    for i, row in enumerate(df.itertuples(index=False)):
        b = bundles.get(row.route)
        if b is None:
            continue
        # The simulator's cost-table key uses inflated context (vision prefix
        # is added at scheduler load-time). For Pi0 vision_prefix_tokens=0,
        # so this is a no-op.
        lin = int(row.context_tokens)  # already inflated in requests_out
        lout = int(row.generated_tokens)
        ppre, pdec = _predict(b, lin, lout, bs=1)
        pred_pre[i] = ppre
        pred_dec[i] = pdec
    df["pred_prefill_ms"]      = pred_pre
    df["pred_decode_step_ms"]  = pred_dec
    df["pred_decode_total_ms"] = pred_dec * (df["generated_tokens"] - 1).clip(lower=0)
    df["pred_e2e_ms"]          = df["pred_prefill_ms"] + df["pred_decode_total_ms"]

    df["actual_prefill_ms"]      = df["prefill_end_ms"] - df["prefill_start_ms"]
    df["actual_decode_total_ms"] = df["completion_ms"]  - df["first_token_ms"]
    df["actual_e2e_ms"]          = df["e2e_ms"]
    return df


def _plot_one(ax, df, x_col, y_col, title, label):
    for r, c in ROUTE_COLORS.items():
        m = (df["route"] == r) & df[x_col].notna() & df[y_col].notna() \
            & (df[x_col] > 0) & (df[y_col] > 0)
        if m.sum() == 0: continue
        ax.loglog(df[x_col][m], df[y_col][m], "o", color=c, alpha=0.7,
                  markersize=6, label=f"{r} (n={m.sum()})")
    finite = pd.concat([df[x_col].dropna(), df[y_col].dropna()])
    finite = finite[finite > 0]
    if len(finite):
        lo, hi = float(finite.min()), float(finite.max())
        ax.loglog([lo, hi], [lo, hi], "k--", alpha=0.5, label="y = x")
    ax.set_xlabel(f"actual {label} (ms)")
    ax.set_ylabel(f"predicted {label} (ms)")
    ax.set_title(title)
    ax.grid(which="both", alpha=0.3)
    ax.legend(fontsize=8)


def _plot_scheduler_predictions(run_dir: Path) -> None:
    """Plot admission-time scheduler finish predictions when CSV columns exist."""
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    required = {
        "scheduler_predicted_finish_ms",
        "scheduler_predicted_e2e_ms",
        "completion_ms",
        "e2e_ms",
    }

    for csv_path in sorted(run_dir.glob("*/requests_out.csv")):
        policy = csv_path.parent.name
        df = pd.read_csv(csv_path)
        missing = required - set(df.columns)
        if missing:
            print(f"[scheduler plot skip] {policy}: missing {sorted(missing)}")
            continue
        df = df[df["state"] == "done"].copy()
        if df.empty:
            print(f"[scheduler plot skip] {policy}: no completed requests")
            continue

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        _plot_one(axes[0], df, "completion_ms", "scheduler_predicted_finish_ms",
                  "Absolute Finish Time", "finish time")
        _plot_one(axes[1], df, "e2e_ms", "scheduler_predicted_e2e_ms",
                  "E2E Including Admission-Time Queue Estimate", "E2E")
        fig.suptitle(
            f"{policy} — scheduler predicted finish vs actual completion",
            fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        out = plots_dir / f"{policy}_scheduler_predicted_vs_actual_loglog.png"
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"[plot] {out.relative_to(run_dir.parent.parent)}")


def plot_run_dir(run_dir: Path, gpu_pkl: Path, hybrid_pkl: Path,
                 vision_prefix_tokens: int = 0) -> None:
    _plot_scheduler_predictions(run_dir)

    bundles: dict[str, dict] = {}
    if gpu_pkl.exists():
        try:
            bundles["gpu_only"] = pickle.load(open(gpu_pkl, "rb"))
        except Exception as exc:
            print(f"[service plot skip] failed to load {gpu_pkl}: {exc}")
    if hybrid_pkl.exists():
        try:
            bundles["lpddr5_pim_bank"] = pickle.load(open(hybrid_pkl, "rb"))
        except Exception as exc:
            print(f"[service plot skip] failed to load {hybrid_pkl}: {exc}")
    if not bundles:
        print("[service plot skip] no cost-model bundles loaded")
        return

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Each policy subdir has its own requests_out.csv.
    csvs = sorted(run_dir.glob("*/requests_out.csv"))
    for csv_path in csvs:
        policy = csv_path.parent.name
        df = pd.read_csv(csv_path)
        df = df[df["state"] == "done"]
        if df.empty:
            print(f"[skip] {policy}: no completed requests")
            continue
        df = _augment_with_predictions(df, bundles, vision_prefix_tokens)

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        _plot_one(axes[0], df, "actual_prefill_ms",      "pred_prefill_ms",      "Prefill",       "prefill")
        _plot_one(axes[1], df, "actual_decode_total_ms", "pred_decode_total_ms", "Decode (total)", "decode total")
        _plot_one(axes[2], df, "actual_e2e_ms",          "pred_e2e_ms",
                  "E2E: Service Prediction vs Loaded Actual", "E2E")
        fig.suptitle(f"{policy} — predicted (cost model) vs actual (simulator)", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        out = plots_dir / f"{policy}_predicted_vs_actual_loglog.png"
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"[plot] {out.relative_to(run_dir.parent.parent)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True,
                    help="Policy-compare run directory (parent of guaranteed_no_evict/, etc.)")
    ap.add_argument("--gpu-pkl", type=Path, required=True)
    ap.add_argument("--hybrid-pkl", type=Path, required=True)
    ap.add_argument("--vision-prefix-tokens", type=int, default=0,
                    help="Informational only — simulator already inflated context_tokens "
                         "in requests_out.csv before writing it.")
    args = ap.parse_args()
    plot_run_dir(args.run_dir, args.gpu_pkl, args.hybrid_pkl, args.vision_prefix_tokens)


if __name__ == "__main__":
    main()
