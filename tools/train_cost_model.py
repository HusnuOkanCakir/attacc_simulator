#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import sys
from typing import Dict, List, Sequence, Tuple

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error, r2_score)
from sklearn.model_selection import train_test_split

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


def _default_model_path(out_dir: Path, route: str) -> Path:
    return out_dir / f"{route}.pkl"


def _build_training_frame(route: str,
                          csv_path: str,
                          sum_offload_to_pim: bool) -> pd.DataFrame:
    raw_df = pd.read_csv(csv_path)
    rows: List[Dict[str, float]] = []
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
        raise ValueError(f"{csv_path}: no training rows")
    return df


def _fit_route_models(df: pd.DataFrame,
                      test_fraction: float,
                      random_state: int,
                      max_depth: int | None,
                      max_iter: int,
                      learning_rate: float) -> Tuple[Dict[str, object], Dict[str, Dict[str, float]]]:
    X = df[FEATURE_NAMES]
    metrics: Dict[str, Dict[str, float]] = {}
    estimators: Dict[str, object] = {}

    if len(df) >= 5 and 0.0 < test_fraction < 1.0:
        X_train, X_test = train_test_split(X,
                                           test_size=test_fraction,
                                           random_state=random_state)
        split_by_index = True
    else:
        X_train, X_test = X, X
        split_by_index = False

    for target in TARGET_NAMES:
        y = df[target]
        if split_by_index:
            y_train = y.loc[X_train.index]
            y_test = y.loc[X_test.index]
        else:
            y_train = y
            y_test = y

        model = HistGradientBoostingRegressor(max_depth=max_depth,
                                              max_iter=max_iter,
                                              learning_rate=learning_rate,
                                              random_state=random_state)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics[target] = {
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "mape": float(mean_absolute_percentage_error(y_test, y_pred)),
            "r2": float(r2_score(y_test, y_pred)) if len(X_test) > 1 else float("nan"),
        }
        estimators[target] = model

    return estimators, metrics


def main() -> int:
    p = argparse.ArgumentParser(
        description=("Train lightweight ML cost models from output.csv-style "
                     "route calibration tables."))
    p.add_argument("--profile",
                   action="append",
                   required=True,
                   help="Training CSV in form route=path/to/cost_table.csv")
    p.add_argument("--model-out",
                   action="append",
                   default=[],
                   help="Optional output path override in form route=path/to/model.pkl")
    p.add_argument("--out-dir",
                   type=Path,
                   default=Path("cluster_outputs/cost_models"),
                   help="Default directory for saved model bundles")
    p.add_argument("--sum-offload-to-pim",
                   action="store_true",
                   help="Train targets assuming sum/prefill offload to PIM")
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--max-depth", type=int, default=3)
    p.add_argument("--max-iter", type=int, default=300)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--summary-json",
                   type=Path,
                   default=None,
                   help="Optional path to write training metrics summary")
    args = p.parse_args()

    route_to_csv = _parse_route_map(args.profile, "--profile")
    route_to_out = _parse_route_map(args.model_out, "--model-out") if args.model_out else {}

    summary: Dict[str, object] = {}
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for route, csv_path in route_to_csv.items():
        out_path = Path(route_to_out.get(route, str(_default_model_path(args.out_dir, route))))
        df = _build_training_frame(route=route,
                                   csv_path=csv_path,
                                   sum_offload_to_pim=args.sum_offload_to_pim)
        estimators, metrics = _fit_route_models(df=df,
                                                test_fraction=args.test_fraction,
                                                random_state=args.random_state,
                                                max_depth=args.max_depth,
                                                max_iter=args.max_iter,
                                                learning_rate=args.learning_rate)

        bounds = {
            feat: (float(df[feat].min()), float(df[feat].max()))
            for feat in FEATURE_NAMES
        }
        bundle = {
            "route": route,
            "feature_names": FEATURE_NAMES,
            "target_names": TARGET_NAMES,
            "bounds": bounds,
            "estimators": estimators,
            "metadata": {
                "csv_path": csv_path,
                "num_rows": int(len(df)),
                "sum_offload_to_pim": bool(args.sum_offload_to_pim),
                "test_fraction": float(args.test_fraction),
                "random_state": int(args.random_state),
                "max_depth": None if args.max_depth is None else int(args.max_depth),
                "max_iter": int(args.max_iter),
                "learning_rate": float(args.learning_rate),
                "metrics": metrics,
            },
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("wb") as f:
            pickle.dump(bundle, f)

        summary[route] = {
            "csv_path": csv_path,
            "model_path": str(out_path),
            "num_rows": int(len(df)),
            "bounds": bounds,
            "metrics": metrics,
        }

        print(f"[trained] {route}")
        print(f"  rows: {len(df)}")
        print(f"  model: {out_path}")
        for target in TARGET_NAMES:
            m = metrics[target]
            print(f"  {target}: mae={m['mae']:.6f} mape={m['mape']:.6f} r2={m['r2']:.6f}")

    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(f"Wrote summary: {args.summary_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
