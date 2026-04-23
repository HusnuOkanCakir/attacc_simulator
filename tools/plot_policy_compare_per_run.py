#!/usr/bin/env python3
"""
Generate per-run plots (execution timeline, route mix, predicted vs actual
finish) for each subdirectory under a policy-compare run directory.

Reuses the three plot_* functions from tools/run_serving_online_kv_oom_abc.py
so the look matches the A/B/C kv_oom_abc plots exactly.

Typical use
-----------
  # Latest run under cluster_outputs/online_serving_runs/*_policy_compare/
  python tools/plot_policy_compare_per_run.py

  # Or a specific run dir
  python tools/plot_policy_compare_per_run.py \\
      --run-dir cluster_outputs/online_serving_runs/20260421_202815_policy_compare
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

# Reuse the three single-run plot helpers from the kv_oom_abc runner.
from run_serving_online_kv_oom_abc import (  # noqa: E402
    plot_execution_timeline,
    plot_route_mix,
    plot_predicted_vs_actual,
    _load_cost_table,
)

RUN_ROOT = REPO / "cluster_outputs/online_serving_runs"

LABEL_MAP = {
    "guaranteed_no_evict": "guaranteed_no_evict",
    "max_util_full":       "max_util_full",
    "max_util_tail":       "max_util_tail",
    "max_util_tail_pred":  "max_util_tail_pred",
}


def latest_policy_compare_dir() -> Path:
    dirs = sorted(RUN_ROOT.glob("*_policy_compare"))
    if not dirs:
        raise FileNotFoundError(
            f"No *_policy_compare directory under {RUN_ROOT}. "
            "Run tools/run_serving_online_policy_compare.py first.")
    return dirs[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="Policy-compare run dir (default: latest under "
                         "cluster_outputs/online_serving_runs/).")
    args = ap.parse_args()

    run_dir = (args.run_dir or latest_policy_compare_dir()).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {run_dir}")

    print(f"[run_dir] {run_dir}")
    cost_tables = _load_cost_table()

    for subdir, label in LABEL_MAP.items():
        sub = run_dir / subdir
        csv_path = sub / "requests_out.csv"
        if not csv_path.exists():
            print(f"[skip] {subdir}: missing requests_out.csv")
            continue

        df = pd.read_csv(csv_path)
        plots_dir = sub / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        slug = label

        print(f"\n[{label}]")
        plot_execution_timeline(df, label,
            plots_dir / f"{slug}_request_execution_timeline.png")
        plot_route_mix(df, label,
            plots_dir / f"{slug}_route_mix.png")
        plot_predicted_vs_actual(df, label, cost_tables,
            plots_dir / f"{slug}_predicted_finish_vs_actual_finish.png")

    print(f"\n[done] plots under {run_dir}/*/plots/")


if __name__ == "__main__":
    main()
