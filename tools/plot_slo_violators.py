#!/usr/bin/env python3
"""Retroactively compute SLO violator rates from a sweep_dir's requests_out.csv
files and the per-cell config.yaml. Produces a violators-vs-SLO curve per model.

Use case: any sweep that varied --slo-e2e-ms across cells (e.g. E3 SLO sweep).
For each sub-run `<NN>_<model>_slo<NNNN>`, reads the policy's requests_out.csv,
counts how many completed requests had `e2e_ms > slo_e2e_ms`, and plots the
violator percentage on a log-x SLO axis with one line per model.

Output:
  <sweep_dir>/plots/slo_violators.png
  <sweep_dir>/plots/slo_violators_table.txt

Usage:
  python tools/plot_slo_violators.py \\
      --sweep-dir cluster_outputs/online_serving_runs/e3_slo_sweep_<TS>
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


LABEL_RE = re.compile(r"^(?:\d+_)?(?P<model>[a-z0-9_]+?)_slo(?P<slo>\d+)$")


def discover_cells(sweep_dir: Path):
    """Find sub-runs and parse their (model, slo_ms) labels."""
    cells = []
    for sub in sorted(sweep_dir.iterdir()):
        if not sub.is_dir():
            continue
        m = LABEL_RE.match(sub.name)
        if not m:
            continue
        slo = int(m.group("slo"))
        model = m.group("model")
        # The runner emits requests_out.csv under <sub>/<policy>/requests_out.csv
        # or <sub>/requests_out.csv depending on layout. Try both.
        csv_candidates = [
            sub / "requests_out.csv",
        ]
        for child in sub.iterdir():
            if child.is_dir():
                csv_candidates.append(child / "requests_out.csv")
        csv = next((c for c in csv_candidates if c.exists()), None)
        if csv is None:
            print(f"[skip] {sub.name}: no requests_out.csv")
            continue
        cells.append((model, slo, csv))
    return cells


def compute_violators(csv: Path, slo_ms: int) -> tuple[int, int, float, float]:
    """Returns (violators, total_done, e2e_p99, e2e_p50)."""
    df = pd.read_csv(csv)
    d = df[df["state"] == "done"]
    if len(d) == 0:
        return 0, 0, 0.0, 0.0
    violators = int((d["e2e_ms"] > slo_ms).sum())
    p99 = float(d["e2e_ms"].quantile(0.99))
    p50 = float(d["e2e_ms"].quantile(0.50))
    return violators, len(d), p99, p50


def plot_violators(rows: dict[str, list[tuple[int, int, int, float, float]]],
                   out_path: Path):
    """rows[model] = list of (slo_ms, violators, total, e2e_p99, e2e_p50)"""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for model, points in rows.items():
        slos = [p[0] for p in points]
        pct = [100.0 * p[1] / max(1, p[2]) for p in points]
        ax.plot(slos, pct, marker="o", linewidth=2, label=model)
    ax.set_xscale("log")
    ax.set_xlabel("SLO E2E target (ms)")
    ax.set_ylabel("% requests violating SLO")
    ax.set_title("SLO violator rate vs SLO target (synthetic VLA)")
    ax.set_ylim(-2, 102)
    ax.grid(True, alpha=0.3, which="both")
    ax.axhline(50, color="gray", linewidth=0.7, linestyle=":")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[plot] {out_path}")


def write_table(rows, out_path: Path):
    with out_path.open("w") as f:
        f.write(f"{'model':<10}{'slo_ms':>8}{'done':>8}"
                f"{'violators':>12}{'pct%':>8}{'e2e_p50':>10}{'e2e_p99':>10}\n")
        f.write("-" * 66 + "\n")
        for model, points in rows.items():
            for slo, v, total, p99, p50 in points:
                pct = 100.0 * v / max(1, total)
                f.write(f"{model:<10}{slo:>8}{total:>8}{v:>12}"
                        f"{pct:>7.1f}%{p50:>10.0f}{p99:>10.0f}\n")
    print(f"[table] {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-dir", type=Path, required=True)
    args = ap.parse_args()
    if not args.sweep_dir.is_dir():
        sys.exit(f"[error] sweep-dir not a directory: {args.sweep_dir}")

    cells = discover_cells(args.sweep_dir)
    if not cells:
        sys.exit("[error] no sub-runs matched <NN>_<model>_slo<NNNN>")

    rows = {}
    for model, slo, csv in cells:
        v, total, p99, p50 = compute_violators(csv, slo)
        rows.setdefault(model, []).append((slo, v, total, p99, p50))
    for model in rows:
        rows[model].sort(key=lambda x: x[0])

    plots_dir = args.sweep_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_violators(rows, plots_dir / "slo_violators.png")
    write_table(rows, plots_dir / "slo_violators_table.txt")

    print()
    for model, points in rows.items():
        print(f"=== {model} ===")
        for slo, v, total, p99, p50 in points:
            pct = 100.0 * v / max(1, total)
            print(f"  slo={slo:>5}ms  violators={v}/{total} ({pct:.1f}%)  "
                  f"e2e_p50={p50:.0f}  e2e_p99={p99:.0f}")


if __name__ == "__main__":
    main()
