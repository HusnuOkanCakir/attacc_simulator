#!/usr/bin/env python3
"""Wide-synthetic vs real-shape-bootstrap knee comparison.

Two E2E-p99-vs-offered curves for the Pi0 hybrid at max_active=5000:
  - azure_poisson_wide (synthetic 4-mode mixture; Lin mean 1560,
    Lin-Lout correlation +0.99)
  - azure_bootstrap (rows resampled jointly from the real Azure 19k
    conv trace; Lin mean 1138, correlation -0.11)

Quantifies how much of the wide trace's E2E tail comes from its
heavier-than-real shape structure vs pure utilisation.

Sources:
  cluster_outputs/online_serving_runs/rps_knee_pi0_wide_fine_n5k_a5000_20260605_212614
  cluster_outputs/online_serving_runs/rps_knee_pi0_20260612_191543
Output:
  thesis_plotting/figures/fig_wide_vs_bootstrap_knee.{pdf,png}
"""

import csv
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO))

from plot_sweep_compare import parse_summary  # noqa: E402

from thesis_plotting.style import (  # noqa: E402
    configure_plotting, save_fig, set_spines, bold_legend,
    FONT_ANNOTATE, FONT_LABEL, FONT_TICK, FONT_TITLE,
)

WIDE_SWEEP = REPO / ("cluster_outputs/online_serving_runs/"
                     "rps_knee_pi0_wide_fine_n5k_a5000_20260605_212614")
BOOT_SWEEPS = [
    REPO / "cluster_outputs/online_serving_runs/rps_knee_pi0_20260612_191543",
    REPO / "cluster_outputs/online_serving_runs/rps_knee_pi0_20260612_231639",
]
BOOT_RE = re.compile(r"^\d+_azure_bootstrap_s\d+$")


def load_cells(sweep_dir, cell_re=None):
    rows = []
    for cell in sorted(sweep_dir.iterdir()):
        if not cell.is_dir() or cell.name == "plots":
            continue
        if cell_re and not cell_re.match(cell.name):
            continue
        sumfile = cell / "policy_compare_summary.txt"
        if not sumfile.is_file():
            inner = next((c / "policy_compare_summary.txt"
                          for c in cell.iterdir() if c.is_dir()), None)
            if inner and inner.is_file():
                sumfile = inner
        req_csv = next(iter(cell.glob("*/requests_out.csv")), None)
        if not sumfile.is_file() or req_csv is None:
            continue
        stats = parse_summary(sumfile)
        e2e_p99 = stats.get("e2e_p99")
        if e2e_p99 is None:
            continue
        n, max_arr = 0, 0.0
        with open(req_csv) as fh:
            for row in csv.DictReader(fh):
                a = float(row["arrival_ms"])
                if a > max_arr:
                    max_arr = a
                n += 1
        if max_arr <= 0:
            continue
        rows.append((n / (max_arr / 1000.0), e2e_p99 / 1000.0))
    return sorted(rows)


def main():
    configure_plotting()
    FL  = FONT_LABEL    + 8
    FTI = FONT_TITLE    + 8
    FTK = FONT_TICK     + 8
    FAN = FONT_ANNOTATE + 9

    wide = load_cells(WIDE_SWEEP)
    boot = sorted(sum((load_cells(s, BOOT_RE) for s in BOOT_SWEEPS), []))
    if not wide or not boot:
        sys.exit("[error] missing cells")

    print(f"[info] wide: {len(wide)} cells, bootstrap: {len(boot)} cells")
    for name, pts in [("wide", wide), ("bootstrap", boot)]:
        for off, p99 in pts:
            print(f"  {name:>9}: offered={off:5.2f}  e2e_p99={p99:7.1f} s")

    fig, ax = plt.subplots(figsize=(12.0, 7.0))

    ax.plot([p[0] for p in wide], [p[1] for p in wide],
            "-o", color="#c44e52", lw=2.6, ms=10,
            markeredgecolor="black", markeredgewidth=0.6,
            label="azure_poisson_wide  (synthetic mixture)", zorder=5)
    ax.plot([p[0] for p in boot], [p[1] for p in boot],
            "-s", color="#406ea3", lw=2.6, ms=10,
            markeredgecolor="black", markeredgewidth=0.6,
            label="azure_bootstrap  (rows resampled from real 19k)",
            zorder=6)

    # Matched-load annotation at offered ~1.9
    ax.annotate("469 s",
                xy=(1.91, 469), xytext=(-55, 18),
                textcoords="offset points",
                fontsize=FAN, fontweight="bold", color="#c44e52",
                arrowprops=dict(arrowstyle="->", color="#c44e52", lw=1.2))
    ax.annotate("124 s  (3.8x lower\nat the same load)",
                xy=(1.89, 124), xytext=(25, -10),
                textcoords="offset points",
                fontsize=FAN, fontweight="bold", color="#406ea3",
                arrowprops=dict(arrowstyle="->", color="#406ea3", lw=1.2))

    ax.set_xlabel("Offered mean load  (RPS)",
                  fontsize=FL, fontweight="bold")
    ax.set_ylabel("Raw E2E p99  (s)",
                  fontsize=FL, fontweight="bold")
    ax.set_title("Synthetic wide mixture vs real-shape bootstrap "
                 "(Pi0 hybrid, max_active=5000, bs=4)",
                 fontsize=FTI, fontweight="bold")
    ax.set_xlim(0.4, 3.1)
    ax.set_ylim(0, 2600)
    ax.tick_params(axis="both", which="major", labelsize=FTK)
    ax.grid(True, alpha=0.45)
    set_spines(ax)
    leg = ax.legend(loc="upper left", fontsize=FTK,
                    title="workload", title_fontsize=FTK,
                    framealpha=0.95)
    bold_legend(leg)

    fig.tight_layout()
    save_fig(fig, "fig_wide_vs_bootstrap_knee",
             REPO / "thesis_plotting/figures")
    plt.close(fig)


if __name__ == "__main__":
    main()
