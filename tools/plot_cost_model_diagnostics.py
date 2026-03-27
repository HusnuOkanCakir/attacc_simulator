#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import sys
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error, r2_score)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.trace_replay import OutputCsvCostModel


FEATURE_NAMES = ["Lin", "Lout", "bs"]
TARGET_NAMES = [
    "prefill_e2e_ms",
    "prefill_gpu_ms",
    "prefill_pim_ms",
    "prefill_energy_nj",
    "decode_e2e_ms",
    "decode_gpu_ms",
    "decode_pim_ms",
    "decode_energy_nj",
]


def _parse_route_map(items: Sequence[str], arg_name: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid {arg_name} '{item}'. Expected route=path.")
        route, path = item.split("=", 1)
        route = route.strip()
        path = path.strip()
        if not route or not path:
            raise ValueError(f"Invalid {arg_name} '{item}'. Empty route/path.")
        out[route] = path
    return out


def _build_training_frame(route: str,
                          csv_path: str,
                          sum_offload_to_pim: bool) -> pd.DataFrame:
    raw_df = pd.read_csv(csv_path)
    rows = []
    for _, row in raw_df.iterrows():
        point = OutputCsvCostModel._row_to_point(route, row, csv_path, sum_offload_to_pim)
        rows.append({
            "Lin": int(row["Lin"]),
            "Lout": int(row["Lout"]),
            "bs": int(row["bs"]),
            "prefill_e2e_ms": point.prefill_e2e_ms,
            "prefill_gpu_ms": point.prefill_gpu_ms,
            "prefill_pim_ms": point.prefill_pim_ms,
            "prefill_energy_nj": point.prefill_energy_nj,
            "decode_e2e_ms": point.decode_e2e_ms,
            "decode_gpu_ms": point.decode_gpu_ms,
            "decode_pim_ms": point.decode_pim_ms,
            "decode_energy_nj": point.decode_energy_nj,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"{csv_path}: no rows found")
    return df


def _load_bundle(path: str) -> Dict[str, object]:
    with open(path, "rb") as f:
        return pickle.load(f)


def _metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
    }


def _plot_pred_vs_actual(route: str,
                         df: pd.DataFrame,
                         preds: pd.DataFrame,
                         targets: Sequence[str],
                         out_path: Path,
                         metric_map: Dict[str, Dict[str, float]]) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()
    for ax, target in zip(axes, targets):
        x = df[target]
        y = preds[target]
        lo = min(float(x.min()), float(y.min()))
        hi = max(float(x.max()), float(y.max()))
        points = ax.scatter(x, y, s=36, alpha=0.8, label="Calibration sample")
        ideal_line, = ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1, label="Ideal prediction (y=x)")
        m = metric_map[target]
        ax.set_title(f"{target}\nR2={m['r2']:.3f} MAE={m['mae']:.4f}")
        ax.set_xlabel("Actual value")
        ax.set_ylabel("Predicted value")
        ax.legend(handles=[points, ideal_line], loc="best", fontsize=8)
    for ax in axes[len(targets):]:
        ax.axis("off")
    fig.suptitle(f"{route}: Predicted vs Actual")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_error_by_shape(route: str,
                         df: pd.DataFrame,
                         preds: pd.DataFrame,
                         out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    size_map = {int(bs): 45 + 12 * int(bs) for bs in df["bs"].dropna().unique()}
    for ax, target in zip(axes, ["prefill_e2e_ms", "decode_e2e_ms"]):
        denom = df[target].abs().clip(lower=1e-9)
        ape = ((preds[target] - df[target]).abs() / denom) * 100.0
        sc = ax.scatter(df["Lin"],
                        df["Lout"],
                        c=ape,
                        s=[size_map[int(bs)] for bs in df["bs"]],
                        cmap="viridis")
        ax.set_title(f"{target} APE (%)")
        ax.set_xlabel("Lin")
        ax.set_ylabel("Lout")
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("APE (%)")
    fig.suptitle(f"{route}: Error by Shape")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Plot interpolation diagnostics for learned route cost models.")
    p.add_argument("--profile",
                   action="append",
                   required=True,
                   help="Training CSV in form route=path/to/cost_table.csv")
    p.add_argument("--ml-profile",
                   action="append",
                   required=True,
                   help="Learned model bundle in form route=path/to/model.pkl")
    p.add_argument("--out-dir",
                   type=Path,
                   default=Path("cluster_outputs/cost_model_plots"),
                   help="Directory to write plot PNGs and metrics JSON")
    p.add_argument("--sum-offload-to-pim",
                   action="store_true",
                   help="Interpret CSV targets as if sum/prefill is offloaded to PIM")
    p.add_argument("--summary-json",
                   type=Path,
                   default=None,
                   help="Optional combined metrics JSON path")
    args = p.parse_args()

    route_to_csv = _parse_route_map(args.profile, "--profile")
    route_to_model = _parse_route_map(args.ml_profile, "--ml-profile")
    routes = sorted(set(route_to_csv.keys()) & set(route_to_model.keys()))
    if not routes:
        raise ValueError("No overlapping routes between --profile and --ml-profile")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, object] = {}

    for route in routes:
        df = _build_training_frame(route=route,
                                   csv_path=route_to_csv[route],
                                   sum_offload_to_pim=args.sum_offload_to_pim)
        bundle = _load_bundle(route_to_model[route])
        features = df[FEATURE_NAMES]
        available_targets = [target for target in TARGET_NAMES if target in bundle["estimators"]]

        preds = {}
        metrics = {}
        for target in available_targets:
            model = bundle["estimators"][target]
            pred = pd.Series(model.predict(features), index=df.index)
            preds[target] = pred
            metrics[target] = _metrics(df[target], pred)
        pred_df = pd.DataFrame(preds)

        _plot_pred_vs_actual(route=route,
                             df=df,
                             preds=pred_df,
                             targets=available_targets,
                             out_path=args.out_dir / f"{route}_pred_vs_actual.png",
                             metric_map=metrics)
        _plot_error_by_shape(route=route,
                             df=df,
                             preds=pred_df,
                             out_path=args.out_dir / f"{route}_error_by_shape.png")

        metrics_json = {
            "csv_path": route_to_csv[route],
            "model_path": route_to_model[route],
            "num_rows": int(len(df)),
            "metrics": metrics,
        }
        (args.out_dir / f"{route}_metrics.json").write_text(
            json.dumps(metrics_json, indent=2, sort_keys=True) + "\n")
        summary[route] = metrics_json
        print(f"[plotted] {route}")
        print(f"  pred-vs-actual: {args.out_dir / f'{route}_pred_vs_actual.png'}")
        print(f"  error-by-shape: {args.out_dir / f'{route}_error_by_shape.png'}")

    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(f"Wrote summary: {args.summary_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
