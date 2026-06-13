#!/usr/bin/env python3
"""SLO target sweep: at fixed budget=off, sweep SLO_E2E from 500ms to 100s.

Demonstrates the chapter claim that `min_finish` is SLO-blind:
throughput, route mix, TTFT, and E2E are bit-identical across all
SLO targets. Only the post-hoc violator count changes (different
threshold applied to the same latency distribution).

Reads cluster_outputs/online_serving_runs/knee_vs_slo_e2e_20260609_165957/.
Output: thesis_plotting/figures/fig_slo_target_sweep.{pdf,png}
"""

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

from plot_sweep_compare import parse_summary  # noqa: E402

from thesis_plotting.style import (  # noqa: E402
    configure_plotting, save_fig, set_spines,
    FONT_LABEL, FONT_TITLE, FONT_TICK, FONT_ANNOTATE,
)

SWEEP = REPO / "cluster_outputs/online_serving_runs/knee_vs_slo_e2e_20260609_165957"
OUT_NAME = "fig_slo_target_sweep"

CELL_RE = re.compile(r"^\d+_slo_e2e_(?P<slo>\d+)_s\d+$")


def load_cells(sweep_dir):
    rows = []
    for cell in sorted(sweep_dir.iterdir()):
        if not cell.is_dir():
            continue
        m = CELL_RE.match(cell.name)
        if not m:
            continue
        slo = int(m.group("slo"))
        sumfile = cell / "policy_compare_summary.txt"
        if not sumfile.is_file():
            continue
        metrics = parse_summary(sumfile)
        pim_gpu = (0, 0)
        slo_e2e = (0, 0)
        slo_ttft = (0, 0)
        for line in sumfile.read_text().splitlines():
            if line.startswith("Routes (PIM/GPU)"):
                try:
                    p, g = line.split()[-1].split("/")
                    pim_gpu = (int(p), int(g))
                except Exception:
                    pass
            elif line.startswith("SLO E2E violators"):
                try:
                    v, t = line.split()[-1].split("/")
                    slo_e2e = (int(v), int(t))
                except Exception:
                    pass
            elif line.startswith("SLO TTFT violators"):
                try:
                    v, t = line.split()[-1].split("/")
                    slo_ttft = (int(v), int(t))
                except Exception:
                    pass
        rows.append({
            "slo": slo,
            "throughput": metrics["throughput"],
            "ttft_p99": metrics["ttft_p99"],
            "e2e_p99": metrics["e2e_p99"],
            "pim_share": 100.0 * pim_gpu[0] / max(sum(pim_gpu), 1),
            "slo_e2e_pct": 100.0 * slo_e2e[0] / max(slo_e2e[1], 1),
            "slo_ttft_pct": 100.0 * slo_ttft[0] / max(slo_ttft[1], 1),
        })
    rows.sort(key=lambda r: r["slo"])
    return rows


def render(rows, out_dir):
    configure_plotting()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))

    slos = [r["slo"] for r in rows]

    ax_a = axes[0]
    throughput = [r["throughput"] for r in rows]
    pim_share = [r["pim_share"] for r in rows]
    ttft = [r["ttft_p99"] for r in rows]

    ax_a2 = ax_a.twinx()
    l1 = ax_a.plot(slos, throughput, "-o", color="#406ea3", lw=2.4, ms=9,
                   markeredgecolor="black", markeredgewidth=0.6,
                   label="Throughput (RPS)")
    l2 = ax_a.plot(slos, [t/1000.0 for t in ttft], "-s",
                   color="#5a9b5a", lw=2.4, ms=9,
                   markeredgecolor="black", markeredgewidth=0.6,
                   label="TTFT p99 (s)")
    l3 = ax_a2.plot(slos, pim_share, "-^", color="#c44e52", lw=2.4, ms=9,
                    markeredgecolor="black", markeredgewidth=0.6,
                    label="PIM share (%)")
    ax_a.set_xscale("log")
    ax_a.set_xlabel("SLO E2E target  (ms, log scale)",
                    fontsize=FONT_LABEL + 1, fontweight="bold")
    ax_a.set_ylabel("Throughput (RPS) / TTFT p99 (s)",
                    fontsize=FONT_LABEL + 1, fontweight="bold")
    ax_a2.set_ylabel("PIM share (%)",
                     fontsize=FONT_LABEL + 1, fontweight="bold",
                     color="#c44e52")
    ax_a2.tick_params(axis="y", labelcolor="#c44e52")
    ax_a.set_ylim(0, max(max(throughput), max(ttft)/1000.0) * 1.3)
    ax_a2.set_ylim(0, 100)
    ax_a.set_title("(a) System metrics — invariant under SLO change",
                   fontsize=FONT_TITLE, fontweight="bold", loc="left")
    ax_a.tick_params(axis="both", which="major", labelsize=FONT_TICK)
    ax_a2.tick_params(axis="both", which="major", labelsize=FONT_TICK)
    ax_a.grid(True, axis="y", alpha=0.4)
    set_spines(ax_a)
    lines = l1 + l2 + l3
    ax_a.legend(lines, [l.get_label() for l in lines],
                loc="center right", fontsize=FONT_TICK)

    ax_b = axes[1]
    e2e_viol = [r["slo_e2e_pct"] for r in rows]
    ttft_viol = [r["slo_ttft_pct"] for r in rows]
    ax_b.plot(slos, e2e_viol, "-o", color="#c44e52", lw=2.4, ms=9,
              markeredgecolor="black", markeredgewidth=0.6,
              label="E2E violators")
    ax_b.plot(slos, ttft_viol, "-s", color="#e8a86c", lw=2.4, ms=9,
              markeredgecolor="black", markeredgewidth=0.6,
              label="TTFT violators")
    ax_b.set_xscale("log")
    ax_b.set_xlabel("SLO E2E target  (ms, log scale)",
                    fontsize=FONT_LABEL + 1, fontweight="bold")
    ax_b.set_ylabel("SLO violators  (% of requests)",
                    fontsize=FONT_LABEL + 1, fontweight="bold")
    ax_b.set_title("(b) Violator counts — only the threshold moves",
                   fontsize=FONT_TITLE, fontweight="bold", loc="left")
    ax_b.tick_params(axis="both", which="major", labelsize=FONT_TICK)
    ax_b.set_ylim(0, 100)
    ax_b.grid(True, axis="y", alpha=0.4)
    set_spines(ax_b)
    ax_b.legend(loc="upper right", fontsize=FONT_TICK)

    for s, v in zip(slos, e2e_viol):
        ax_b.annotate(f"{v:.0f}%", (s, v),
                      xytext=(0, 8), textcoords="offset points",
                      fontsize=FONT_ANNOTATE, fontweight="bold",
                      ha="center", color="#c44e52")

    fig.suptitle("SLO target sweep at fixed budget=off "
                 "(offered ≈ 1.91 RPS, min_finish policy is SLO-blind)",
                 fontsize=FONT_TITLE + 1, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, OUT_NAME, out_dir)
    plt.close(fig)


def main():
    rows = load_cells(SWEEP)
    print(f"[info] {len(rows)} cells loaded")
    print(f"  {'slo':>7}  {'thru':>5}  {'ttft_p99':>9}  {'e2e_p99(s)':>11}  "
          f"{'PIM%':>6}  {'slo_e2e%':>9}  {'slo_ttft%':>9}")
    for r in rows:
        print(f"  {r['slo']:>7d}  {r['throughput']:>5.2f}  "
              f"{r['ttft_p99']:>9.0f}  {r['e2e_p99']/1000:>11.1f}  "
              f"{r['pim_share']:>5.1f}%  {r['slo_e2e_pct']:>8.1f}%  "
              f"{r['slo_ttft_pct']:>8.1f}%")
    render(rows, REPO / "thesis_plotting/figures")


if __name__ == "__main__":
    main()
