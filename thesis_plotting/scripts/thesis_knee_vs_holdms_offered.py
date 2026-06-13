#!/usr/bin/env python3
"""Offered-RPS knee curve for the KV-OOM hold-window sweep at 0.25 GiB pool.

Mirrors thesis_knee_vs_kvpolicy_offered.py / thesis_knee_vs_kvoom_offered.py.
Five hold-window values (10, 50, 200, 1000, 2000 ms) on the same x-axis
(offered mean RPS per cell). Per-knob safe-op lines at 0.9 x each setting's
achieved-throughput ceiling.

Source: cluster_outputs/online_serving_runs/knee_vs_holdms_0p25gb_20260609_223728/
Output: thesis_plotting/figures/fig_knee_vs_holdms_0p25gb_offered.{pdf,png}
        + a --linear variant if invoked with --linear.
"""

import argparse
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
    FONT_LABEL, FONT_TITLE, FONT_TICK, FONT_ANNOTATE,
)

SWEEP = REPO / "cluster_outputs/online_serving_runs/knee_vs_holdms_0p25gb_20260609_223728"
OUT_NAME = "fig_knee_vs_holdms_0p25gb_offered"

CELL_RE = re.compile(r"^\d+_holdms_0p25gb(?:_fill)?_(?P<v>\d+)_s(?P<sx10>\d+)$")

HOLDMS_VALUES = [10, 50, 200, 1000, 2000]
MARKERS = ["o", "s", "^", "D", "v"]
LABEL_OFFSETS = [0.03, 0.10, 0.17, 0.24, 0.31]


def load_cells(sweep_dirs):
    rows = []
    for sweep_dir in sweep_dirs:
        if not sweep_dir.is_dir():
            print(f"[warn] sweep dir missing: {sweep_dir}", file=sys.stderr)
            continue
        for cell in sorted(sweep_dir.iterdir()):
            if not cell.is_dir():
                continue
            m = CELL_RE.match(cell.name)
            if not m:
                continue
            holdms = int(m.group("v"))
            scale = int(m.group("sx10")) / 10.0
            sumfile = cell / "policy_compare_summary.txt"
            req_csv = cell / "max_util_full" / "requests_out.csv"
            if not sumfile.is_file():
                continue
            if not req_csv.is_file():
                for child in cell.iterdir():
                    cand = child / "requests_out.csv"
                    if cand.is_file():
                        req_csv = cand
                        break
                if not req_csv.is_file():
                    continue
            m_stats = parse_summary(sumfile)
            achieved = m_stats.get("throughput")
            e2e_p99 = m_stats.get("e2e_p99")
            if achieved is None or e2e_p99 is None:
                continue
            n = 0
            max_arrival = 0.0
            with open(req_csv) as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    a = float(row["arrival_ms"])
                    if a > max_arrival:
                        max_arrival = a
                    n += 1
            if max_arrival <= 0 or n == 0:
                continue
            offered = n / (max_arrival / 1000.0)
            rows.append((holdms, scale, offered, achieved, e2e_p99))
    return rows


def render(rows, out_dir: Path, out_name: str = OUT_NAME, linear: bool = False):
    configure_plotting()
    by_holdms = {}
    for holdms, scale, offered, achieved, p99 in rows:
        by_holdms.setdefault(holdms, []).append((offered, achieved, p99, scale))

    holdms_values = sorted(by_holdms.keys())
    cmap = plt.get_cmap("viridis")
    colors = {h: cmap(i / max(1, len(holdms_values) - 1))
              for i, h in enumerate(holdms_values)}

    fig, ax = plt.subplots(figsize=(11.0, 6.2))

    for i, holdms in enumerate(holdms_values):
        pts = sorted(by_holdms[holdms], key=lambda r: r[0])
        xs = [p[0] for p in pts]
        ys = [p[2] / 1000.0 for p in pts]
        ax.plot(xs, ys, marker=MARKERS[i % len(MARKERS)], ms=10, lw=2.4,
                color=colors[holdms], label=f"{holdms} ms",
                markeredgecolor="black", markeredgewidth=0.6, zorder=4)

    # Per-knob safe operating point: 0.9 x knob-specific achieved-RPS ceiling
    for i, holdms in enumerate(holdms_values):
        achieved_vals = [p[1] for p in by_holdms[holdms]]
        ceiling = max(achieved_vals)
        safe = 0.9 * ceiling
        color = colors[holdms]
        ax.axvline(safe, color=color, linestyle="--", linewidth=1.2,
                   zorder=3, alpha=0.65)
        offset = LABEL_OFFSETS[i % len(LABEL_OFFSETS)]
        ax.text(safe, offset,
                f"  safe {holdms}ms {safe:.2f}",
                transform=ax.get_xaxis_transform(),
                ha="left", va="bottom",
                color=color, fontsize=FONT_ANNOTATE,
                fontweight="bold")

    if not linear:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel("Offered mean load  (RPS"
                  + ("" if linear else ", log scale") + ")",
                  fontsize=FONT_LABEL + 1, fontweight="bold")
    ax.set_ylabel("E2E p99  (s"
                  + ("" if linear else ", log scale") + ")",
                  fontsize=FONT_LABEL + 1, fontweight="bold")
    ax.set_title("Pi0 @ azure_poisson_wide  —  KV-OOM hold window at 0.25 GiB pool",
                 fontsize=FONT_TITLE + 1, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=FONT_TICK + 1)
    ax.grid(True, which="major", alpha=0.45)
    if not linear:
        ax.grid(True, which="minor", alpha=0.18)
    set_spines(ax)
    leg = ax.legend(loc="upper left", fontsize=FONT_TICK + 1,
                    title="hold limit",
                    title_fontsize=FONT_TICK + 1, framealpha=0.95)
    bold_legend(leg)

    save_fig(fig, out_name, out_dir)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--linear", action="store_true")
    ap.add_argument("--out-name", default=OUT_NAME)
    ap.add_argument("--sweep-dirs", type=Path, nargs="+", default=[SWEEP],
                    help="One or more sweep dirs to merge.")
    args = ap.parse_args()

    rows = load_cells(args.sweep_dirs)
    if not rows:
        sys.exit(f"[error] no cells found in {args.sweep_dirs}")
    print(f"[info] {len(rows)} cells loaded")
    for holdms in sorted({r[0] for r in rows}):
        cells = sorted([r for r in rows if r[0] == holdms], key=lambda r: r[2])
        print(f"  holdms={holdms}:")
        for _, scale, offered, achieved, p99 in cells:
            print(f"    s={scale:>5.2f}  offered={offered:>5.2f}  "
                  f"achieved={achieved:>5.2f}  e2e_p99={p99/1000.0:>8.1f}s")
    name = args.out_name + ("_linear" if args.linear else "")
    render(rows, REPO / "thesis_plotting/figures", name, args.linear)


if __name__ == "__main__":
    main()
