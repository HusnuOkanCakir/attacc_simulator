#!/usr/bin/env python3
"""Scheduler-value experiment renderer (4 configs on the bootstrap trace).

Four panels:
  (a) achieved vs offered RPS
  (b) E2E p99 vs offered
  (c) TTFT p99 vs offered
  (d) energy per token (mJ/tok, from tools/compute_serving_energy.py
      convention recomputed inline) as grouped bars per scale

Configs: gpu_only, pim_only, hyb_mf (min_finish), hyb_lge
(latency_guarded_energy guard=200ms). KV pool 0.5 GiB cells only; the
64 GiB control pair is reported in the sanity print.

Usage:
    python thesis_plotting/scripts/thesis_scheduler_value_bootstrap.py \
        --sweep-dir cluster_outputs/online_serving_runs/sched_value_bootstrap_<TS> \
        --cost-dir cluster_outputs/cost_tables_energyfix_pi0_a6000
"""

import argparse
import csv as _csv
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO))

from plot_sweep_compare import parse_summary  # noqa: E402
from compute_serving_energy import EnergyTable, analyse_cell  # noqa: E402

from thesis_plotting.style import (  # noqa: E402
    configure_plotting, save_fig, set_spines, bold_legend,
    FONT_ANNOTATE, FONT_LABEL, FONT_TICK, FONT_TITLE,
)

CELL_RE = re.compile(
    r"^\d+_(?P<cfg>gpu_only|pim_only|hyb_mf|hyb_lge)_p(?P<pool>\d+)_s(?P<s>\d+)$")

CFG_LABEL = {
    "gpu_only": "GPU-only",
    "pim_only": "PIM-only",
    "hyb_mf":   "Hybrid (min_finish)",
    "hyb_lge":  "Hybrid (energy-guarded)",
}
CFG_COLOR = {
    "gpu_only": "#9aa5b1",
    "pim_only": "#5a9b5a",
    "hyb_mf":   "#e8a86c",
    "hyb_lge":  "#406ea3",
}
CFG_MARKER = {"gpu_only": "X", "pim_only": "s", "hyb_mf": "^", "hyb_lge": "o"}
CFG_ORDER = ["gpu_only", "pim_only", "hyb_mf", "hyb_lge"]


def load_cells(sweep_dir: Path, tables):
    rows = []
    for cell in sorted(sweep_dir.iterdir()):
        if not cell.is_dir():
            continue
        m = CELL_RE.match(cell.name)
        if not m:
            continue
        sumfile = cell / "policy_compare_summary.txt"
        if not sumfile.is_file():
            continue
        stats = parse_summary(sumfile)
        req_csv = next(iter(cell.glob("*/requests_out.csv")), None)
        if req_csv is None:
            continue
        n, max_arr = 0, 0.0
        with open(req_csv) as fh:
            for r in _csv.DictReader(fh):
                a = float(r["arrival_ms"])
                if a > max_arr:
                    max_arr = a
                n += 1
        if max_arr <= 0:
            continue
        energy = analyse_cell(cell, tables, bs_decode=4)
        rows.append({
            "cfg": m.group("cfg"),
            "pool": m.group("pool"),
            "offered": n / (max_arr / 1000.0),
            "achieved": stats.get("throughput"),
            "e2e_p99_s": stats.get("e2e_p99", 0) / 1000.0,
            "ttft_p99_s": stats.get("ttft_p99", 0) / 1000.0,
            "mj_per_tok": energy["mj_per_tok"] if energy else float("nan"),
            "waste_share": energy["waste_share"] if energy else float("nan"),
            "routes": energy["routes"] if energy else {},
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", type=Path, required=True)
    ap.add_argument("--cost-dir", type=Path,
                    default=REPO / "cluster_outputs/cost_tables_energyfix_pi0_a6000")
    ap.add_argument("--out-name", default="fig_scheduler_value_bootstrap")
    args = ap.parse_args()

    tables = {
        "gpu_only": EnergyTable(args.cost_dir / "gpu_only.csv", 4),
        "lpddr5_pim_bank": EnergyTable(args.cost_dir / "lpddr5_pim_bank.csv", 4),
    }
    rows = load_cells(args.sweep_dir, tables)
    if not rows:
        sys.exit(f"[error] no cells in {args.sweep_dir}")

    print(f"{'cfg':<10} {'pool':>5} {'offered':>8} {'achieved':>9} "
          f"{'e2e_p99':>9} {'ttft_p99':>9} {'mJ/tok':>8} {'waste%':>7}")
    for r in sorted(rows, key=lambda r: (r["pool"], r["cfg"], r["offered"])):
        print(f"{r['cfg']:<10} {r['pool']:>5} {r['offered']:>8.2f} "
              f"{r['achieved']:>9.2f} {r['e2e_p99_s']:>8.1f}s "
              f"{r['ttft_p99_s']:>8.1f}s {r['mj_per_tok']:>8.2f} "
              f"{100*r['waste_share']:>6.1f}%")

    # Main panels: pressure cells only (pool != 64).
    main_rows = [r for r in rows if r["pool"] != "64"]

    configure_plotting()
    FL  = FONT_LABEL    + 8
    FTI = FONT_TITLE    + 8
    FTK = FONT_TICK     + 8
    FAN = FONT_ANNOTATE + 8

    fig, axes = plt.subplots(2, 2, figsize=(15.0, 12.0))
    (ax_a, ax_b), (ax_c, ax_d) = axes

    for cfg in CFG_ORDER:
        pts = sorted([r for r in main_rows if r["cfg"] == cfg],
                     key=lambda r: r["offered"])
        if not pts:
            continue
        xs = [p["offered"] for p in pts]
        kw = dict(marker=CFG_MARKER[cfg], ms=11, lw=2.5,
                  color=CFG_COLOR[cfg], label=CFG_LABEL[cfg],
                  markeredgecolor="black", markeredgewidth=0.6)
        ax_a.plot(xs, [p["achieved"] for p in pts], **kw)
        ax_b.plot(xs, [p["e2e_p99_s"] for p in pts], **kw)
        ax_c.plot(xs, [p["ttft_p99_s"] for p in pts], **kw)

    lim = max(r["offered"] for r in main_rows) * 1.1
    ax_a.plot([0, lim], [0, lim], ls="--", color="#999", lw=1.0, zorder=1)
    ax_a.set_xlabel("Offered mean load  (RPS)", fontsize=FL, fontweight="bold")
    ax_a.set_ylabel("Achieved throughput  (RPS)", fontsize=FL, fontweight="bold")
    ax_a.set_title("(a) Offered vs achieved", fontsize=FTI,
                   fontweight="bold", loc="left")

    ax_b.set_xlabel("Offered mean load  (RPS)", fontsize=FL, fontweight="bold")
    ax_b.set_ylabel("E2E p99  (s)", fontsize=FL, fontweight="bold")
    ax_b.set_title("(b) E2E p99", fontsize=FTI, fontweight="bold", loc="left")

    ax_c.set_xlabel("Offered mean load  (RPS)", fontsize=FL, fontweight="bold")
    ax_c.set_ylabel("TTFT p99  (s)", fontsize=FL, fontweight="bold")
    ax_c.set_title("(c) TTFT p99", fontsize=FTI, fontweight="bold", loc="left")
    ax_c.set_yscale("log")

    # (d) energy per token bars, grouped by scale.
    scales = sorted({round(r["offered"], 2) for r in main_rows})
    width = 0.18
    for ci, cfg in enumerate(CFG_ORDER):
        vals, xpos = [], []
        for si, sc in enumerate(scales):
            match = [r for r in main_rows
                     if r["cfg"] == cfg and round(r["offered"], 2) == sc]
            if match:
                vals.append(match[0]["mj_per_tok"])
                xpos.append(si + (ci - 1.5) * width)
        ax_d.bar(xpos, vals, width, color=CFG_COLOR[cfg],
                 edgecolor="black", linewidth=0.6, label=CFG_LABEL[cfg])
    ax_d.set_xticks(range(len(scales)))
    ax_d.set_xticklabels([f"{s:.2f}" for s in scales], fontsize=FTK)
    ax_d.set_xlabel("Offered mean load  (RPS)", fontsize=FL, fontweight="bold")
    ax_d.set_ylabel("Energy per token  (mJ/tok)", fontsize=FL, fontweight="bold")
    ax_d.set_title("(d) Energy per generated token (incl. recompute waste)",
                   fontsize=FTI, fontweight="bold", loc="left")

    for ax in (ax_a, ax_b, ax_c, ax_d):
        ax.tick_params(axis="both", which="major", labelsize=FTK)
        ax.grid(True, alpha=0.4)
        set_spines(ax)
    leg = ax_a.legend(loc="lower right", fontsize=FTK - 1, framealpha=0.95)
    bold_legend(leg)

    fig.suptitle("Scheduler value under KV-pool pressure "
                 "(azure_bootstrap, 0.5 GiB pool, bs=4, max_active=5000)",
                 fontsize=FTI + 2, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_fig(fig, args.out_name, REPO / "thesis_plotting/figures")
    plt.close(fig)


if __name__ == "__main__":
    main()
