#!/usr/bin/env python3
"""OpenVLA RPS knee on azure_poisson_wide — chapter Fig 6.11.

Plots Raw E2E p99 (s, log) versus OFFERED mean load (RPS, log).
Offered RPS is computed per cell from arrival timestamps in
requests_out.csv (n / max_arrival_time), NOT from the policy_compare
summary's span field (which conflates with achieved throughput at
post-cliff cells and looks like a fold-back/saturation cluster).

Source: cluster_outputs/online_serving_runs/rps_knee_openvla_20260601_175834/
        (azure_poisson_wide subset, max_active=32 era; the new
        max_active=5000 sweep is still in progress).

Output: thesis_plotting/figures/fig_rps_knee_openvla_azure_poisson_wide.{pdf,png}
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
    configure_plotting, save_fig, set_spines,
    COLOR_HEAVY,
    FONT_ANNOTATE, FONT_LABEL, FONT_TICK, FONT_TITLE,
)

DEFAULT_SWEEP = (
    REPO
    / "cluster_outputs/online_serving_runs/rps_knee_openvla_20260601_175834"
)
CELL_RE = re.compile(r"^\d+_azure_poisson_wide_s(?P<sx10>\d+)$")
LINE_COLOR = COLOR_HEAVY[2]


def load_cells(sweep_dir: Path):
    rows = []
    for cell in sorted(sweep_dir.iterdir()):
        if not cell.is_dir():
            continue
        m = CELL_RE.match(cell.name)
        if not m:
            continue
        scale = int(m.group("sx10")) / 10.0
        sumfile = cell / "policy_compare_summary.txt"
        if not sumfile.is_file():
            for child in cell.iterdir():
                cand = child / "policy_compare_summary.txt"
                if cand.is_file():
                    sumfile = cand
                    break
        if not sumfile.is_file():
            continue
        req_csv = cell / "max_util_full" / "requests_out.csv"
        if not req_csv.is_file():
            for child in cell.iterdir():
                cand = child / "requests_out.csv"
                if cand.is_file():
                    req_csv = cand
                    break
        if not req_csv.is_file():
            continue
        stats = parse_summary(sumfile)
        achieved = stats.get("throughput")
        e2e_p99 = stats.get("e2e_p99")
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
        if max_arrival <= 0 or n == 0:
            continue
        offered = n / (max_arrival / 1000.0)
        rows.append((scale, offered, achieved, e2e_p99))
    return sorted(rows, key=lambda r: r[1])


def render(rows, out_dir: Path, out_name: str, linear: bool = False):
    configure_plotting()
    # Local font overrides — defaults render ~3 pt after LaTeX scales to
    # \linewidth. Bump for readability across the chapter.
    FL  = FONT_LABEL    + 8   # 15  axis labels
    FTI = FONT_TITLE    + 8   # 16  title
    FTK = FONT_TICK     + 8   # 14  ticks
    FAN = FONT_ANNOTATE + 9   # 14  annotations

    fig, ax = plt.subplots(figsize=(11.0, 6.5))

    xs = [r[1] for r in rows]              # offered RPS
    ys = [r[3] / 1000.0 for r in rows]     # E2E p99 ms → s
    ax.plot(xs, ys, "-X", color=LINE_COLOR, lw=2.8, ms=12,
            markeredgecolor="black", markeredgewidth=0.7, zorder=5)

    # OpenVLA saturation ceiling (peak achieved throughput across cells)
    # and the 0.9 x reference operating point.
    ceiling = max(r[2] for r in rows)
    ref_op = 0.9 * ceiling

    ax.axvline(ref_op, color="#1f77b4", linestyle="--", lw=1.6,
               alpha=0.8, zorder=3)
    ax.text(ref_op, 0.04, f"ref op {ref_op:.2f}  ",
            transform=ax.get_xaxis_transform(),
            ha="right", va="bottom",
            color="#1f77b4", fontsize=FAN, fontweight="bold")

    ax.axvline(ceiling, color="#c0392b", linestyle=":", lw=1.6,
               alpha=0.8, zorder=3)
    ax.text(ceiling, 0.04, f"  ceiling {ceiling:.2f}",
            transform=ax.get_xaxis_transform(),
            ha="left", va="bottom",
            color="#c0392b", fontsize=FAN, fontweight="bold")

    if not linear:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel("Offered mean load  (RPS"
                  + ("" if linear else ", log scale") + ")",
                  fontsize=FL, fontweight="bold")
    ax.set_ylabel("Raw E2E p99  (s"
                  + ("" if linear else ", log scale") + ")",
                  fontsize=FL, fontweight="bold")
    ax.set_title("OpenVLA @ azure_poisson_wide  —  raw E2E p99 knee "
                 "(max_active=32 source)",
                 fontsize=FTI, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=FTK)
    ax.grid(True, which="major", alpha=0.45)
    if not linear:
        ax.grid(True, which="minor", alpha=0.18)
    set_spines(ax)
    fig.tight_layout()

    save_fig(fig, out_name, out_dir)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", type=Path, default=DEFAULT_SWEEP)
    ap.add_argument("--linear", action="store_true")
    ap.add_argument("--out-name",
                    default="fig_rps_knee_openvla_azure_poisson_wide")
    args = ap.parse_args()

    if not args.sweep_dir.is_dir():
        sys.exit(f"[error] sweep dir not found: {args.sweep_dir}")

    rows = load_cells(args.sweep_dir)
    if not rows:
        sys.exit("[error] no parseable cells")
    print(f"[info] {len(rows)} cells loaded")
    ceiling = max(r[2] for r in rows)
    print(f"[info] max achieved (saturation ceiling) = {ceiling:.3f} RPS")
    print(f"[info] 0.9 x ref op = {0.9*ceiling:.3f} RPS")
    for scale, offered, achieved, p99 in rows:
        print(f"  s={scale:>5.2f}  offered={offered:>6.3f}  "
              f"achieved={achieved:>5.2f}  e2e_p99={p99/1000:>8.1f} s")
    name = args.out_name + ("_linear" if args.linear else "")
    render(rows, REPO / "thesis_plotting/figures", name, args.linear)


if __name__ == "__main__":
    main()
