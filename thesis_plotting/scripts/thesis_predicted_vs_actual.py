#!/usr/bin/env python3
"""Thesis cost-model validation figure: ML predicted vs cost-table actual.

Three-panel log-log scatter comparing the HistGBR cost model's prediction
against the dense source-of-truth cost table across the (Lin, Lout, bs)
sweep grid. One column per metric (prefill_e2e_ms, decode_e2e_ms,
g_time_ms), per-route scatter (gpu_only + lpddr5_pim_bank), R² annotated.

This validates that the ML model is a faithful surrogate of the cost
table itself — a cleaner story than a saturated end-to-end sim
comparison (which would conflate queueing with cost-model error).

Output: thesis_plotting/figures/fig_predicted_vs_actual.{pdf,png}
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from thesis_plotting.style import (  # noqa: E402
    configure_plotting, save_fig, set_spines, bold_legend,
    COLOR_ROUTE, WIDE_FIGSIZE, FONT_ANNOTATE, FONT_LABEL, FONT_TITLE,
)


DEFAULT_MODEL_DIR = REPO / "cluster_outputs/cost_models_full_energy_openvla"
DEFAULT_TABLE_DIR = REPO / "cluster_outputs/cost_tables_full_energy_openvla"

ROUTE_COLOR = {
    "gpu_only":        COLOR_ROUTE["gpu_only"],
    "lpddr5_pim_bank": COLOR_ROUTE["lpddr5_pim_bank"],
}


def _r_squared(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = np.isfinite(actual) & np.isfinite(predicted) \
         & (actual > 0) & (predicted > 0)
    if mask.sum() < 2:
        return float("nan")
    a = np.log(actual[mask])
    p = np.log(predicted[mask])
    ss_res = np.sum((a - p) ** 2)
    ss_tot = np.sum((a - a.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def predict_route(bundle: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Apply the ML cost-model bundle on the cost table's (Lin, Lout, bs)
    grid, returning a frame with pred + actual columns side by side."""
    feat = bundle.get("feature_names", ["Lin", "Lout", "bs"])
    x = df[feat].copy()

    out = pd.DataFrame({"Lin": df["Lin"], "Lout": df["Lout"], "bs": df["bs"]})

    target_cols = {
        "prefill_e2e_ms": "s_time",          # serial prefill in ms
        "decode_e2e_ms":  "g_time (ms)",     # per-step decode in ms
    }
    for pred_key, actual_col in target_cols.items():
        if pred_key in bundle["estimators"] and actual_col in df.columns:
            out[f"pred_{pred_key}"]   = bundle["estimators"][pred_key].predict(x)
            out[f"actual_{pred_key}"] = df[actual_col].values
    return out


def plot_panel(ax, frames: dict, pred_col: str, actual_col: str, title: str):
    finite_a, finite_p = [], []
    for route, color in ROUTE_COLOR.items():
        if route not in frames:
            continue
        sub = frames[route]
        if pred_col not in sub.columns or actual_col not in sub.columns:
            continue
        m = sub[pred_col].notna() & sub[actual_col].notna() \
          & (sub[pred_col] > 0) & (sub[actual_col] > 0)
        if m.sum() == 0:
            continue
        ax.loglog(sub[actual_col][m], sub[pred_col][m],
                  "o", color=color, alpha=0.45, markersize=3.5,
                  markeredgecolor="none",
                  label=f"{route.replace('_', '-')}  (n={m.sum()})",
                  zorder=3)
        finite_a.append(sub[actual_col][m].values)
        finite_p.append(sub[pred_col][m].values)

    if finite_a:
        all_a = np.concatenate(finite_a)
        all_p = np.concatenate(finite_p)
        lo = max(min(all_a.min(), all_p.min()), 1e-4)
        hi = max(all_a.max(), all_p.max()) * 1.1
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.6,
                zorder=2, label="y = x")
        r2 = _r_squared(all_a, all_p)
        ax.text(0.04, 0.93, f"$R^2$ = {r2:.4f}",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=FONT_ANNOTATE + 2, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="#888",
                          boxstyle="round,pad=0.25", linewidth=0.4))

    ax.set_xlabel("actual cost-table (ms)",
                  fontsize=FONT_LABEL, fontweight="bold")
    ax.set_ylabel("ML-predicted (ms)",
                  fontsize=FONT_LABEL, fontweight="bold")
    ax.set_title(title, fontsize=FONT_TITLE, fontweight="bold")
    ax.grid(which="both", alpha=0.4)
    set_spines(ax)
    leg = ax.legend(loc="lower right", fontsize=FONT_ANNOTATE + 1)
    bold_legend(leg)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR,
                    help="Directory with gpu_only.pkl + lpddr5_pim_bank.pkl")
    ap.add_argument("--table-dir", type=Path, default=DEFAULT_TABLE_DIR,
                    help="Directory with gpu_only.csv + lpddr5_pim_bank.csv")
    ap.add_argument("--out-dir", type=Path,
                    default=REPO / "thesis_plotting" / "figures")
    ap.add_argument("--out-name", type=str, default="fig_predicted_vs_actual")
    ap.add_argument("--label", type=str, default="OpenVLA / A100",
                    help="Title suffix, e.g. model + GPU.")
    args = ap.parse_args()

    configure_plotting()
    frames: dict[str, pd.DataFrame] = {}
    for route in ("gpu_only", "lpddr5_pim_bank"):
        pkl = args.model_dir / f"{route}.pkl"
        csv = args.table_dir / f"{route}.csv"
        if not pkl.is_file() or not csv.is_file():
            print(f"[skip] {route}: missing {pkl.name} or {csv.name}")
            continue
        bundle = pickle.load(open(pkl, "rb"))
        df = pd.read_csv(csv)
        frames[route] = predict_route(bundle, df)
        print(f"[info] {route}: {len(df)} cost-table rows")

    if not frames:
        sys.exit("[error] no route data loaded")

    fig, axes = plt.subplots(1, 2, figsize=(WIDE_FIGSIZE[0] * 0.7, 3.4))
    plot_panel(axes[0], frames,
               "pred_prefill_e2e_ms", "actual_prefill_e2e_ms",
               "Prefill (per-request, ms)")
    plot_panel(axes[1], frames,
               "pred_decode_e2e_ms",  "actual_decode_e2e_ms",
               "Decode (per-step, ms)")
    fig.suptitle(
        f"Cost-model validation — HistGBR vs source cost table  ({args.label})",
        fontsize=FONT_TITLE, fontweight="bold",
    )
    save_fig(fig, args.out_name, args.out_dir)
    plt.close(fig)


if __name__ == "__main__":
    main()
