#!/usr/bin/env python3
"""Post-hoc serving-level energy accounting.

The serving simulator does not report energy; this script reconstructs
it per request from requests_out.csv and the (energy-fixed) cost
tables:

    E(req) = s_energy(Lin)                      # prefill, bs=1 row
           + gen_tokens * g_energy(Lin, BS)     # decode, bs=BS rows

plus a recompute-waste term from the cell's summary statistics
(preempted work is re-executed):

    E_waste = tokens_recomputed * g_energy_route(mean Lin, BS)
            + full_evictions   * s_energy_route(mean Lin)

Nearest-neighbor lookup on (Lin, Lout); decode rows at the configured
decode-batch cap (default 4) so batched weight-read amortization is
reflected; the same convention is applied to every config so the
comparison is internally consistent.

Usage:
    python tools/compute_serving_energy.py \
        --cost-dir cluster_outputs/cost_tables_energyfix_pi0_a6000 \
        --run-dir <cell_dir> [<cell_dir> ...]

Each <cell_dir> is a runner cell (containing <policy>/requests_out.csv
and policy_compare_summary.txt).
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PIM_ROUTE = "lpddr5_pim_bank"
GPU_ROUTE = "gpu_only"


class EnergyTable:
    """Nearest-neighbor (Lin, Lout) energy lookup for one route."""

    def __init__(self, csv_path: Path, bs_decode: int):
        df = pd.read_csv(csv_path)
        df = df.rename(columns=lambda c: c.strip())
        scol = "s_energy (nJ)"
        gcol = "g_energy (nJ)"
        if scol not in df.columns or gcol not in df.columns:
            sys.exit(f"[error] {csv_path} missing energy columns")
        self.prefill = df[df.bs == 1][["Lin", "Lout", scol]].to_numpy()
        dec = df[df.bs == bs_decode]
        if dec.empty:
            # fall back to closest available bs
            avail = sorted(df.bs.unique())
            pick = min(avail, key=lambda b: abs(b - bs_decode))
            print(f"[warn] {csv_path.name}: no bs={bs_decode} rows, "
                  f"using bs={pick}", file=sys.stderr)
            dec = df[df.bs == pick]
        self.decode = dec[["Lin", "Lout", gcol]].to_numpy()

    @staticmethod
    def _nearest(table, lin, lout):
        d = (np.abs(table[:, 0] - lin) / 6144.0
             + np.abs(table[:, 1] - lout) / 800.0)
        return table[int(np.argmin(d)), 2]

    def prefill_nj(self, lin, lout):
        return self._nearest(self.prefill, lin, lout)

    def decode_nj_per_tok(self, lin, lout):
        return self._nearest(self.decode, lin, lout)


def parse_summary(sumfile: Path):
    """Extract recompute-related stats from policy_compare_summary.txt."""
    stats = {"tokens_recomputed": 0, "preempt_full": 0}
    if not sumfile.is_file():
        return stats
    for line in sumfile.read_text().splitlines():
        if line.startswith("Tokens recomputed"):
            try:
                stats["tokens_recomputed"] = int(line.split()[-1])
            except ValueError:
                pass
        elif line.startswith("Preempt count"):
            # "Preempt count (tail/full)   337 (0/337)"
            m = re.search(r"\((\d+)/(\d+)\)\s*$", line)
            if m:
                stats["preempt_full"] = int(m.group(2))
    return stats


def analyse_cell(cell_dir: Path, tables: dict, bs_decode: int):
    req_csv = next(iter(cell_dir.glob("*/requests_out.csv")), None)
    if req_csv is None:
        print(f"[skip] {cell_dir.name}: no requests_out.csv")
        return None
    d = pd.read_csv(req_csv)
    done = d[d.state == "done"]

    total_nj = 0.0
    total_tokens = 0
    route_counts = {}
    for _, r in done.iterrows():
        route = r["route"]
        tab = tables.get(route)
        if tab is None:
            continue
        lin = float(r["context_tokens"])
        gen = float(r["generated_tokens"])
        total_nj += tab.prefill_nj(lin, gen)
        total_nj += gen * tab.decode_nj_per_tok(lin, gen)
        total_tokens += gen
        route_counts[route] = route_counts.get(route, 0) + 1

    # Recompute waste: preempted work re-executed on the PIM route.
    stats = parse_summary(cell_dir / "policy_compare_summary.txt")
    pim_tab = tables.get(PIM_ROUTE)
    waste_nj = 0.0
    if pim_tab is not None and len(done):
        mean_lin = float(done.context_tokens.mean())
        mean_gen = float(done.generated_tokens.mean())
        waste_nj += (stats["tokens_recomputed"]
                     * pim_tab.decode_nj_per_tok(mean_lin, mean_gen))
        waste_nj += stats["preempt_full"] * pim_tab.prefill_nj(mean_lin, mean_gen)

    grand_nj = total_nj + waste_nj
    return {
        "cell": cell_dir.name,
        "n_done": len(done),
        "routes": route_counts,
        "useful_j": total_nj / 1e9,
        "waste_j": waste_nj / 1e9,
        "total_j": grand_nj / 1e9,
        "j_per_tok": grand_nj / 1e9 / max(total_tokens, 1),
        "mj_per_tok": grand_nj / 1e6 / max(total_tokens, 1),
        "j_per_req": grand_nj / 1e9 / max(len(done), 1),
        "waste_share": waste_nj / max(grand_nj, 1e-9),
        "tokens_recomputed": stats["tokens_recomputed"],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cost-dir", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, nargs="+", required=True)
    ap.add_argument("--bs", type=int, default=4,
                    help="Decode batch cap used by the runs (default 4).")
    args = ap.parse_args()

    tables = {
        GPU_ROUTE: EnergyTable(args.cost_dir / "gpu_only.csv", args.bs),
        PIM_ROUTE: EnergyTable(args.cost_dir / "lpddr5_pim_bank.csv", args.bs),
    }

    print(f"{'cell':<32} {'n':>5} {'PIM/GPU':>11} {'total J':>9} "
          f"{'waste %':>8} {'mJ/tok':>8} {'J/req':>7}")
    for run_dir in args.run_dir:
        res = analyse_cell(run_dir, tables, args.bs)
        if res is None:
            continue
        pim = res["routes"].get(PIM_ROUTE, 0)
        gpu = res["routes"].get(GPU_ROUTE, 0)
        print(f"{res['cell']:<32} {res['n_done']:>5} "
              f"{pim:>5}/{gpu:<5} {res['total_j']:>9.1f} "
              f"{100*res['waste_share']:>7.1f}% "
              f"{res['mj_per_tok']:>8.2f} {res['j_per_req']:>7.2f}")


if __name__ == "__main__":
    main()
