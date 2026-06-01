#!/usr/bin/env python3
"""Generate a dense cost table CSV from trained HistGBR pkl models.

The existing sparse cost tables have ~436 rows with Lin steps of ~512 tokens.
This script evaluates the trained models on a fine grid (default step=32 tokens)
and writes a CSV in the explicit serving format that cost_table.cpp (Path 1) reads
directly — no C++ changes needed.

Usage:
  python tools/gen_dense_cost_table.py \\
    --models cluster_outputs/cost_models_full_energy/gpu_only.pkl \\
             cluster_outputs/cost_models_full_energy/lpddr5_pim_bank.pkl \\
    --out-dir cluster_outputs/cost_tables_full_energy \\
    --lin-step 32
"""
import argparse
import pickle
from pathlib import Path

import pandas as pd

TARGET_NAMES = [
    "prefill_e2e_ms", "prefill_gpu_ms", "prefill_pim_ms", "prefill_energy_nj",
    "decode_e2e_ms",  "decode_gpu_ms",  "decode_pim_ms",  "decode_energy_nj",
]
OUT_COLS = ["route", "lin", "lout", "bs"] + TARGET_NAMES

LOUT_VALUES = [2, 16, 32, 48, 64, 96, 128, 160, 192, 208]
BS_VALUES   = [1, 8, 32]


def load_bundle(pkl_path: str) -> dict:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def gen_grid(bundle: dict, lin_step: int) -> pd.DataFrame:
    lin_min = int(bundle["bounds"]["Lin"][0])
    lin_max = int(bundle["bounds"]["Lin"][1])
    lout_max = int(bundle["bounds"]["Lout"][1])
    bs_max   = int(bundle["bounds"]["bs"][1])

    lin_vals  = list(range(lin_min, lin_max + 1, lin_step))
    lout_vals = [v for v in LOUT_VALUES if v <= lout_max]
    # Keep requested bs values but cap at model's bs_max (e.g. pim model tops at 8)
    bs_vals = sorted({min(v, bs_max) for v in BS_VALUES})

    rows = []
    for bs in bs_vals:
        for lout in lout_vals:
            feats = pd.DataFrame({
                "Lin":  [float(lin) for lin in lin_vals],
                "Lout": [float(lout)] * len(lin_vals),
                "bs":   [float(bs)]   * len(lin_vals),
            })
            preds = {t: bundle["estimators"][t].predict(feats) for t in TARGET_NAMES}
            for i, lin in enumerate(lin_vals):
                row = {"route": bundle["route"], "lin": lin, "lout": lout, "bs": bs}
                for t in TARGET_NAMES:
                    row[t] = max(0.0, float(preds[t][i]))
                rows.append(row)

    return pd.DataFrame(rows, columns=OUT_COLS)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True,
                    help="Paths to .pkl model files (one per route).")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Directory to write *_dense.csv files into.")
    ap.add_argument("--lin-step", type=int, default=32,
                    help="Lin grid step in tokens (default: 32).")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for pkl_path in args.models:
        bundle = load_bundle(pkl_path)
        route  = bundle["route"]
        df     = gen_grid(bundle, args.lin_step)
        out    = args.out_dir / f"{route}_dense.csv"
        df.to_csv(out, index=False)
        print(f"[gen] {route}: {len(df)} rows  lin_step={args.lin_step}  -> {out}")


if __name__ == "__main__":
    main()
