#!/usr/bin/env python3
"""Admission-violation-budget probe at calibrated SLOs (E2E=3000ms, TTFT=500ms).

5 cells at the safe op (scale=2.85, offered ≈1.91 RPS), budget values
{-1, 0, 1, 5, 20}. Compares throughput, TTFT p99, E2E p99,
admission-violation holds, and SLO violator counts.

Output: thesis_plotting/figures/fig_admbudget_calibrated.{pdf,png}
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

SWEEP = REPO / "cluster_outputs/online_serving_runs/knee_vs_admbudget_20260609_121228"
OUT_NAME = "fig_admbudget_calibrated"

# Pretty labels for the 5 budget values.
# Order: strictest → off (so the "off" baseline sits at the right edge).
LABELS = {"m1": "off  (-1)", "0": "0  (strict)", "1": "1", "5": "5", "20": "20"}
ORDER  = ["0", "1", "5", "20", "m1"]


def load_cells(sweep_dir: Path):
    out = {}
    for cell in sorted(sweep_dir.iterdir()):
        if not cell.is_dir():
            continue
        m = re.match(r"^\d+_admbudget_(?P<v>m?\d+)_s\d+$", cell.name)
        if not m:
            continue
        key = m.group("v")
        sumfile = cell / "policy_compare_summary.txt"
        if not sumfile.is_file():
            continue
        metrics = parse_summary(sumfile)
        adm_holds = 0
        pim_gpu = (0, 0)
        slo_e2e = slo_ttft = (0, 0)
        for line in sumfile.read_text().splitlines():
            if line.startswith("Admission viol. holds"):
                try:
                    adm_holds = int(line.split()[-1])
                except Exception:
                    pass
            elif line.startswith("Routes (PIM/GPU)"):
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
        out[key] = {
            "throughput": metrics["throughput"],
            "ttft_p99": metrics["ttft_p99"],
            "e2e_p99": metrics["e2e_p99"],
            "adm_holds": adm_holds,
            "pim_share": 100.0 * pim_gpu[0] / max(sum(pim_gpu), 1),
            "slo_e2e_pct": 100.0 * slo_e2e[0] / max(slo_e2e[1], 1),
            "slo_ttft_pct": 100.0 * slo_ttft[0] / max(slo_ttft[1], 1),
        }
    return out


def render(cells, out_dir):
    configure_plotting()
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.0))

    xs = np.arange(len(ORDER))
    xlabels = [LABELS[k] for k in ORDER]
    # Color gradient: strict red → loosening (orange, green, blue) → off (gray)
    colors = ["#c44e52", "#e8a86c", "#5a9b5a", "#406ea3", "#888888"]

    def bar(ax, key, title, ylabel, fmt="{:.2f}"):
        vals = [cells[k][key] for k in ORDER]
        ax.bar(xs, vals, color=colors, edgecolor="black", linewidth=0.5)
        for x, v in zip(xs, vals):
            ax.text(x, v, fmt.format(v), ha="center", va="bottom",
                    fontsize=FONT_ANNOTATE, fontweight="bold")
        ax.set_xticks(xs)
        ax.set_xticklabels(xlabels, fontsize=FONT_TICK)
        ax.set_ylabel(ylabel, fontsize=FONT_LABEL, fontweight="bold")
        ax.set_title(title, fontsize=FONT_TITLE, fontweight="bold")
        ax.set_ylim(top=max(vals) * 1.18 if max(vals) > 0 else 1)
        ax.grid(True, axis="y", alpha=0.4)
        set_spines(ax)

    bar(axes[0][0], "throughput", "(a) Throughput",
        "RPS", "{:.2f}")
    bar(axes[0][1], "ttft_p99", "(b) TTFT p99",
        "ms", "{:.0f}")
    bar(axes[0][2], "e2e_p99", "(c) E2E p99",
        "ms", "{:.0f}")
    bar(axes[1][0], "adm_holds", "(d) Admission-violation holds",
        "count", "{:.0f}")
    bar(axes[1][1], "slo_e2e_pct", "(e) SLO E2E violators",
        "% of requests", "{:.1f}%")
    bar(axes[1][2], "slo_ttft_pct", "(f) SLO TTFT violators",
        "% of requests", "{:.1f}%")

    for ax_row in axes:
        for ax in ax_row:
            ax.set_xlabel("Admission-violation budget",
                          fontsize=FONT_LABEL, fontweight="bold")

    fig.suptitle("Admission-violation budget at safe op (offered ≈ 1.91 RPS) "
                 "with calibrated SLOs (E2E=3000 ms, TTFT=500 ms)",
                 fontsize=FONT_TITLE + 1, fontweight="bold", y=1.00)
    fig.tight_layout()
    save_fig(fig, OUT_NAME, out_dir)
    plt.close(fig)


def main():
    cells = load_cells(SWEEP)
    if not cells:
        sys.exit(f"[error] no cells loaded from {SWEEP}")
    print(f"[info] {len(cells)} cells loaded")
    print(f"  {'budget':>10}  {'thru':>5}  {'ttft_p99':>9}  {'e2e_p99(s)':>11}  "
          f"{'adm_holds':>10}  {'PIM%':>6}  {'slo_e2e%':>9}  {'slo_ttft%':>9}")
    for k in ORDER:
        c = cells[k]
        print(f"  {LABELS[k]:>10}  {c['throughput']:>5.2f}  {c['ttft_p99']:>9.0f}  "
              f"{c['e2e_p99']/1000:>11.1f}  {c['adm_holds']:>10d}  "
              f"{c['pim_share']:>5.1f}%  {c['slo_e2e_pct']:>8.1f}%  "
              f"{c['slo_ttft_pct']:>8.1f}%")
    render(cells, REPO / "thesis_plotting/figures")


if __name__ == "__main__":
    main()
