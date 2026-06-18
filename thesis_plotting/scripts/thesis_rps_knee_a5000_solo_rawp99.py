#!/usr/bin/env python3
"""Solo a5000 knee, raw E2E p99 (seconds) on y, offered RPS on x.

Variant of thesis_rps_knee_a5000_solo.py that plots **raw e2e p99**
instead of normalized ms/token. The cliff still sits at the same
offered-RPS location, but the post-cliff plateau (~2300 s) becomes
visible because raw p99 is bounded by the finite-trace drain time
(N / μ ≈ 5000 / 2.10 ≈ 2380 s) rather than dividing it out.

Source:
    cluster_outputs/online_serving_runs/rps_knee_pi0_wide_fine_n5k_a5000_20260605_212614

Output:
    thesis_plotting/figures/fig_rps_knee_a5000_solo_rawp99.{pdf,png}
        + a --linear variant if invoked with --linear.
"""

import argparse
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
    configure_plotting, save_fig, set_spines,
    COLOR_HEAVY, FONT_LABEL, FONT_TITLE, FONT_TICK, FONT_ANNOTATE,
)

SWEEP = REPO / "cluster_outputs/online_serving_runs/rps_knee_pi0_wide_fine_n5k_a5000_20260605_212614"
LINE_COLOR = COLOR_HEAVY[2]

CEILING = 2.10
SAFE_RPS = 1.91


def collect(sweep_dir):
    rows = []
    for cell in sorted(sweep_dir.iterdir()):
        if not cell.is_dir():
            continue
        run_info = cell / "run_info.txt"
        try:
            scale = float(re.search(r"arrival_scale=([\d.]+)",
                                    run_info.read_text()).group(1))
        except Exception:
            continue
        sumfile = cell / "policy_compare_summary.txt"
        req_csv = cell / "max_util_full" / "requests_out.csv"
        if not sumfile.is_file() or not req_csv.is_file():
            continue
        m = parse_summary(sumfile)
        e2e_p99 = m.get("e2e_p99")
        if e2e_p99 is None:
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
        rows.append((scale, offered, e2e_p99))
    return sorted(rows, key=lambda t: t[1])


def render(pts, out_dir, out_name, linear=False, x_max=None, drop_last=0,
           show_ceiling=True):
    configure_plotting()
    # Local font overrides — defaults render ~3 pt after LaTeX scales the
    # figure to \linewidth. Bump everything so the rendered text is readable.
    FL  = FONT_LABEL    + 8   # 15
    FTI = FONT_TITLE    + 8   # 16
    FTK = FONT_TICK     + 8   # 14
    FAN = FONT_ANNOTATE + 9   # 14

    fig, ax = plt.subplots(figsize=(11.0, 6.5))

    if x_max is not None:
        pts = [p for p in pts if p[1] <= x_max]
    if drop_last:
        pts = pts[:-drop_last]  # pts are sorted ascending by offered RPS

    xs = [p[1] for p in pts]
    ys = [p[2] / 1000.0 for p in pts]  # ms → s
    ax.plot(xs, ys, "-o", color=LINE_COLOR, lw=2.8, ms=11,
            markeredgecolor="black", markeredgewidth=0.7,
            label="raw E2E p99", zorder=5)

    ax.axvline(SAFE_RPS, color="#1f77b4", linestyle="--", lw=1.6,
               alpha=0.75, zorder=3)
    ax.text(SAFE_RPS, 0.04, "ref op 1.91  ",
            transform=ax.get_xaxis_transform(),
            ha="right", va="bottom",
            color="#1f77b4", fontsize=FAN, fontweight="bold")
    if show_ceiling:
        ax.axvline(CEILING, color="#c0392b", linestyle=":", lw=1.6,
                   alpha=0.75, zorder=3)
        ax.text(CEILING, 0.04, "  ceiling 2.10",
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
    ax.set_title("Pi0 @ azure_poisson_wide  —  raw E2E p99 knee at max_active=5000",
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
    ap.add_argument("--linear", action="store_true")
    ap.add_argument("--x-max", type=float, default=None,
                    help="Clip cells past this offered RPS to hide the "
                         "finite-trace plateau (e.g. --x-max 2.1).")
    ap.add_argument("--drop-last", type=int, default=0,
                    help="Drop the N right-most (highest offered RPS) points "
                         "after the --x-max filter.")
    ap.add_argument("--no-ceiling", dest="show_ceiling", action="store_false",
                    help="Hide the red ceiling line and its label.")
    ap.add_argument("--out-name", default="fig_rps_knee_a5000_solo_rawp99")
    args = ap.parse_args()

    pts = collect(SWEEP)
    if not pts:
        sys.exit(f"[error] no cells found in {SWEEP}")
    print(f"[info] {len(pts)} cells loaded")
    for scale, offered, p99 in pts:
        print(f"  s={scale:>5.2f}  offered={offered:>5.2f}  "
              f"e2e_p99={p99/1000.0:>8.1f}s")

    name = args.out_name + ("_linear" if args.linear else "")
    render(pts, REPO / "thesis_plotting/figures", name,
           linear=args.linear, x_max=args.x_max, drop_last=args.drop_last,
           show_ceiling=args.show_ceiling)


if __name__ == "__main__":
    main()
