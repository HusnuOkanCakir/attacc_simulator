#!/usr/bin/env python3
"""RPS-knee plot: E2E p99 latency (y) vs OFFERED load (x), serial vs pipeline.

The knee is where E2E p99 turns up sharply as offered load approaches the
sustainable throughput. Comparing the serial and pipeline executors shows
whether the pipeline pushes the knee to a higher offered RPS / lower tail.

One or more datasets can be overlaid; pass matched --dirs and --labels.

Usage:
    # single dataset
    python thesis_plotting/scripts/thesis_pipeline_knee.py \
        --dirs cluster_outputs/online_serving_runs/pipe_knee_bootstrap_pi0_<TS> \
        --labels bootstrap

    # both datasets overlaid
    python thesis_plotting/scripts/thesis_pipeline_knee.py \
        --dirs <bootstrap_dir> <wide_dir> \
        --labels bootstrap wide \
        --out-name fig_pipeline_knee
"""
import argparse
import csv
import glob
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
FIGDIR = REPO / "thesis_plotting" / "figures"

DS_COLOR = {0: "#406ea3", 1: "#c0504d", 2: "#5a9b5a", 3: "#8064a2"}
EXEC_STYLE = {"serial":   dict(ls="--", marker="o", mfc="white"),
              "pipeline": dict(ls="-",  marker="s")}


def parse_e2e_p99(cell):
    fs = glob.glob(str(Path(cell) / "policy_compare_summary.txt"))
    if not fs:
        return None
    m = re.search(r"E2E p99 \(ms\)\s+([0-9.]+)", open(fs[0]).read())
    return float(m.group(1)) if m else None


def offered_rps(cell):
    cs = glob.glob(str(Path(cell) / "*" / "requests_out.csv"))
    if not cs:
        return None
    n, mx = 0, 0.0
    for r in csv.DictReader(open(cs[0])):
        a = float(r["arrival_ms"]); mx = max(mx, a); n += 1
    return n / (mx / 1000.0) if mx > 0 else None


def load_curves(sweep_dir):
    """dict[exec] -> sorted [(offered_rps, e2e_p99_ms)]."""
    out = {"serial": [], "pipeline": []}
    for d in sorted(glob.glob(str(Path(sweep_dir) / "*_s*"))):
        name = Path(d).name
        exec_ = ("serial" if name.startswith("serial")
                 else "pipeline" if name.startswith("pipeline") else None)
        if exec_ is None:
            continue
        off = offered_rps(d); p99 = parse_e2e_p99(d)
        if off is None or p99 is None:
            continue
        out[exec_].append((off, p99))
    for k in out:
        out[k].sort()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True,
                    help="one sweep dir per dataset")
    ap.add_argument("--labels", nargs="+", required=True,
                    help="dataset labels, matched to --dirs")
    ap.add_argument("--out-name", default="fig_pipeline_knee")
    ap.add_argument("--out-dir", type=Path, default=FIGDIR)
    ap.add_argument("--logy", action="store_true", help="log-scale E2E p99 axis")
    args = ap.parse_args()
    if len(args.dirs) != len(args.labels):
        ap.error("--dirs and --labels must have equal length")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    print(f"{'dataset':<12}{'exec':<10}{'offered':>9}{'E2Ep99ms':>10}")
    for di, (sweep, label) in enumerate(zip(args.dirs, args.labels)):
        curves = load_curves(sweep)
        color = DS_COLOR[di % len(DS_COLOR)]
        for exec_ in ("serial", "pipeline"):
            pts = curves[exec_]
            if not pts:
                continue
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            lbl = f"{label} — {exec_}"
            ax.plot(xs, ys, color=color, lw=2.4, ms=8,
                    markeredgecolor="black", markeredgewidth=0.6,
                    label=lbl, **EXEC_STYLE[exec_])
            for off, p99 in pts:
                print(f"{label:<12}{exec_:<10}{off:>9.2f}{p99:>10.0f}")

    ax.set_xlabel("Offered load  (req/s)", fontsize=13, fontweight="bold")
    ax.set_ylabel("E2E p99 latency  (ms)", fontsize=13, fontweight="bold")
    if args.logy:
        ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.35)
    ax.tick_params(labelsize=11)
    ax.set_title("RPS knee: E2E p99 vs offered load — serial vs pipeline executor",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11, framealpha=0.95)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"{args.out_name}.{ext}", dpi=130)
    plt.close(fig)
    print(f"\nwrote {args.out_name}.{{png,pdf}} to {args.out_dir}")


if __name__ == "__main__":
    main()
