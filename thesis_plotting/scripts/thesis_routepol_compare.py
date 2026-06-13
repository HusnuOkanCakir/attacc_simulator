#!/usr/bin/env python3
"""Route policy comparison: min_finish vs latency_guarded_energy.

Spoiler: bit-identical results at every scale, because at the
calibrated SLO every request is predicted to violate and
latency_guarded_energy falls through to its min_finish backup.

Reads cluster_outputs/online_serving_runs/knee_vs_routepol_20260609_121241/.
Output: thesis_plotting/figures/fig_routepol_compare.{pdf,png}
"""

import csv
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

from plot_sweep_compare import parse_summary  # noqa: E402

from thesis_plotting.style import (  # noqa: E402
    configure_plotting, save_fig, set_spines, bold_legend,
    COLOR_HEAVY, FONT_LABEL, FONT_TITLE, FONT_TICK, FONT_ANNOTATE,
)

SWEEP = REPO / "cluster_outputs/online_serving_runs/knee_vs_routepol_20260609_121241"
OUT_NAME = "fig_routepol_compare"

CEILING = 2.13
SAFE_RPS = 1.91

POLICIES = {"min_finish": COLOR_HEAVY[1], "latency_guarded_energy": COLOR_HEAVY[2]}
MARKERS  = {"min_finish": "o", "latency_guarded_energy": "s"}

CELL_RE = re.compile(r"^\d+_routepol_(?P<pol>min_finish|latency_guarded_energy)_s(?P<sx10>\d+)$")


def load_cells(sweep_dir):
    rows = {p: [] for p in POLICIES}
    for cell in sorted(sweep_dir.iterdir()):
        if not cell.is_dir():
            continue
        m = CELL_RE.match(cell.name)
        if not m:
            continue
        pol = m.group("pol")
        scale = int(m.group("sx10")) / 10.0
        sumfile = cell / "policy_compare_summary.txt"
        req_csv = cell / "max_util_full" / "requests_out.csv"
        if not sumfile.is_file() or not req_csv.is_file():
            continue
        metrics = parse_summary(sumfile)
        achieved = metrics.get("throughput")
        e2e_p99 = metrics.get("e2e_p99")
        if achieved is None or e2e_p99 is None:
            continue
        n = 0
        max_arrival = 0.0
        with open(req_csv) as fh:
            for row in csv.DictReader(fh):
                a = float(row["arrival_ms"])
                if a > max_arrival:
                    max_arrival = a
                n += 1
        if max_arrival <= 0:
            continue
        offered = n / (max_arrival / 1000.0)
        rows[pol].append((scale, offered, achieved, e2e_p99))
    for k in rows:
        rows[k].sort(key=lambda r: r[1])
    return rows


def render(rows, out_dir):
    configure_plotting()
    fig, ax = plt.subplots(figsize=(10.5, 6.0))

    ax.axvline(SAFE_RPS, color="#1f77b4", linestyle="--", lw=1.3,
               alpha=0.7, zorder=2)
    ax.text(SAFE_RPS, 0.02, "  safe op  1.91",
            transform=ax.get_xaxis_transform(),
            ha="left", va="bottom", color="#1f77b4",
            fontsize=FONT_ANNOTATE, fontweight="bold")
    ax.axvline(CEILING, color="#c0392b", linestyle=":", lw=1.3,
               alpha=0.7, zorder=2)
    ax.text(CEILING, 0.02, "  ceiling  2.13",
            transform=ax.get_xaxis_transform(),
            ha="left", va="bottom", color="#c0392b",
            fontsize=FONT_ANNOTATE, fontweight="bold")

    for pol, color in POLICIES.items():
        pts = rows[pol]
        if not pts:
            continue
        xs = [p[1] for p in pts]
        ys = [p[3] / 1000.0 for p in pts]
        # Offset the second curve slightly so both are visible.
        offset = 1.0 if pol == "min_finish" else 1.03
        ax.plot(xs, [y * offset for y in ys], marker=MARKERS[pol], ms=10,
                lw=2.4, color=color,
                markeredgecolor="black", markeredgewidth=0.7,
                label=pol.replace("_", r"\_"), zorder=5)

    ax.set_xlabel("Offered mean load  (RPS)",
                  fontsize=FONT_LABEL + 1, fontweight="bold")
    ax.set_ylabel("Raw E2E p99  (s)",
                  fontsize=FONT_LABEL + 1, fontweight="bold")
    ax.set_title("Route policy comparison: identical results\n"
                 "(curves slightly offset for visibility)",
                 fontsize=FONT_TITLE, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=FONT_TICK + 1)
    ax.grid(True, alpha=0.4)
    set_spines(ax)
    leg = ax.legend(loc="upper left", fontsize=FONT_TICK + 1,
                    title="route policy", title_fontsize=FONT_TICK + 1)
    bold_legend(leg)

    # Annotate the null-finding overlay.
    ax.text(0.65, 0.55,
            "All metrics bit-identical:\n"
            "throughput, TTFT, E2E,\n"
            "route mix, SLO violators.\n"
            "→ latency_guarded_energy\n"
            "  reduces to min_finish at\n"
            "  calibrated SLO=3000 ms.",
            transform=ax.transAxes, fontsize=FONT_ANNOTATE + 1,
            ha="left", va="top", fontweight="bold",
            bbox=dict(facecolor="#fff7e6", edgecolor="#c08050",
                      boxstyle="round,pad=0.4", linewidth=0.8))

    save_fig(fig, OUT_NAME, out_dir)
    plt.close(fig)


def main():
    rows = load_cells(SWEEP)
    print(f"[info] {sum(len(v) for v in rows.values())} cells, "
          f"{len(rows)} policies")
    for pol, pts in rows.items():
        print(f"  {pol}:")
        for s, off, a, p99 in pts:
            print(f"    s={s:>5.2f}  offered={off:>5.2f}  "
                  f"achieved={a:>5.2f}  e2e_p99={p99/1000:>7.1f}s")
    render(rows, REPO / "thesis_plotting/figures")


if __name__ == "__main__":
    main()
