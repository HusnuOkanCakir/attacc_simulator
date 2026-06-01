#!/usr/bin/env python3
"""E3-specific plot: route mix (PIM share %) vs SLO value, one line per model.

Reads each sub-run's `policy_compare_summary.txt`, expects sub-run labels of
the form `<NN>_<model>_slo<NNNN>` (e.g. `01_pi0_slo0100`, `08_openvla_slo3200`).
Produces:
  - <sweep_dir>/plots/slo_route_mix.png    (PIM share % vs SLO, per model)
  - <sweep_dir>/plots/slo_metrics.png       (4-panel: throughput, TTFT p99,
                                              E2E p99, KV-OOM holds vs SLO,
                                              one line per model)

Usage:
  python tools/plot_slo_route_mix.py --sweep-dir cluster_outputs/online_serving_runs/e3_slo_sweep_<TS>
"""

import argparse
import re
import sys
from pathlib import Path

# Reuse parse_summary + discover_runs from plot_sweep_compare.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from plot_sweep_compare import parse_summary, discover_runs  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Accept both `01_pi0_slo0100` and `pi0_slo0100` (plot_sweep_compare strips `<NN>_`).
LABEL_RE = re.compile(r"^(?:\d+_)?(?P<model>[a-z0-9_]+?)_slo(?P<slo>\d+)$")


def parse_label(label: str):
    m = LABEL_RE.match(label)
    if not m:
        return None
    return m.group("model"), int(m.group("slo"))


def group_by_model(rows):
    grouped: dict[str, list[tuple[int, dict]]] = {}
    for label, summary in rows:
        parsed = parse_label(label)
        if parsed is None:
            print(f"[skip] label {label!r} not in <NN>_<model>_slo<NNNN> form")
            continue
        model, slo = parsed
        grouped.setdefault(model, []).append((slo, summary))
    for model in grouped:
        grouped[model].sort(key=lambda x: x[0])
    return grouped


def pim_share(s: dict) -> float:
    p = s.get("routes_pim", 0.0)
    g = s.get("routes_gpu", 0.0)
    total = p + g
    return 100.0 * p / total if total > 0 else 0.0


def plot_route_mix(grouped, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model, points in grouped.items():
        slos = [p[0] for p in points]
        shares = [pim_share(p[1]) for p in points]
        ax.plot(slos, shares, marker="o", linewidth=2, label=model)
    ax.set_xscale("log")
    ax.set_xlabel("SLO E2E (ms)")
    ax.set_ylabel("PIM route share (%)")
    ax.set_title("E3 — PIM share vs SLO under synthetic VLA")
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3, which="both")
    ax.axhline(50, color="gray", linewidth=0.7, linestyle=":")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[plot] {out_path}")


def plot_metrics(grouped, out_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    panels = [
        ("throughput",   "Throughput (req/s)",   axes[0][0], False),
        ("ttft_p99",     "TTFT p99 (ms)",        axes[0][1], True),
        ("e2e_p99",      "E2E p99 (ms)",         axes[1][0], True),
        ("kv_oom_holds", "KV-OOM holds (count)", axes[1][1], False),
    ]
    for key, ylabel, ax, ylog in panels:
        for model, points in grouped.items():
            slos = [p[0] for p in points]
            vals = [p[1].get(key, 0.0) for p in points]
            ax.plot(slos, vals, marker="o", linewidth=1.8, label=model)
        ax.set_xscale("log")
        if ylog:
            ax.set_yscale("symlog")
        ax.set_xlabel("SLO E2E (ms)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="best", fontsize=8)
    fig.suptitle("E3 — per-metric vs SLO under synthetic VLA", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[plot] {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-dir", type=Path, required=True)
    args = ap.parse_args()

    if not args.sweep_dir.is_dir():
        sys.exit(f"[error] sweep-dir not a directory: {args.sweep_dir}")

    rows = discover_runs(args.sweep_dir)
    if not rows:
        sys.exit("[error] no sub-runs with policy_compare_summary.txt found")
    print(f"[scan] {len(rows)} sub-runs found")

    grouped = group_by_model(rows)
    if not grouped:
        sys.exit("[error] no labels matched <NN>_<model>_slo<NNNN>")

    for model, points in grouped.items():
        slos = [p[0] for p in points]
        shares = [round(pim_share(p[1]), 1) for p in points]
        print(f"  {model:<8}  slos={slos}  pim_share%={shares}")

    plots_dir = args.sweep_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_route_mix(grouped, plots_dir / "slo_route_mix.png")
    plot_metrics(grouped, plots_dir / "slo_metrics.png")


if __name__ == "__main__":
    main()
