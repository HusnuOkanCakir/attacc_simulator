#!/usr/bin/env python3
"""Solo version of the chaos-comparison right panel: only the
max_active=5000 probe curve, normalized-latency knee, offered-RPS
x-axis. No chaos-zone band, no baseline curve, no two-panel layout.

Source:
    cluster_outputs/online_serving_runs/rps_knee_pi0_wide_fine_n5k_a5000_20260605_212614

Output:
    thesis_plotting/figures/fig_rps_knee_a5000_solo.{pdf,png}
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from thesis_plotting.style import (  # noqa: E402
    configure_plotting, save_fig, set_spines, bold_legend,
    COLOR_HEAVY, FONT_LABEL, FONT_TITLE, FONT_TICK, FONT_ANNOTATE,
)

SWEEP = REPO / "cluster_outputs/online_serving_runs/rps_knee_pi0_wide_fine_n5k_a5000_20260605_212614"
LINE_COLOR = COLOR_HEAVY[2]

CEILING = 2.10  # measured ceiling at max_active=5000
SAFE_RPS = 1.91


def percentile(vals, q):
    s = sorted(vals)
    if not s:
        return None
    k = (len(s) - 1) * q
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


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
        req_csv = cell / "max_util_full" / "requests_out.csv"
        if not req_csv.is_file():
            continue
        n = 0
        max_arrival = 0.0
        norms = []
        with open(req_csv) as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                a = float(row["arrival_ms"])
                if a > max_arrival:
                    max_arrival = a
                n += 1
                if row["state"] != "done":
                    continue
                gt = max(int(row["generated_tokens"]), 1)
                e2e = float(row["e2e_ms"])
                norms.append(e2e / gt)
        if max_arrival <= 0 or n == 0 or not norms:
            continue
        offered = n / (max_arrival / 1000.0)
        rows.append((scale, offered,
                     sum(norms) / len(norms), percentile(norms, 0.99)))
    return sorted(rows, key=lambda t: t[1])


def render(pts, out_dir, out_name, linear=False, show_p99=False):
    configure_plotting()
    fig, ax = plt.subplots(figsize=(10.5, 6.0))

    xs = [p[1] for p in pts]
    ys_mean = [p[2] for p in pts]
    ax.plot(xs, ys_mean, "-o", color=LINE_COLOR, lw=2.4, ms=9,
            markeredgecolor="black", markeredgewidth=0.7,
            label="mean normalized latency", zorder=5)

    if show_p99:
        ys_p99 = [p[3] for p in pts]
        ax.plot(xs, ys_p99, "--s", color=LINE_COLOR, lw=1.8, ms=7,
                markeredgecolor="black", markeredgewidth=0.5, alpha=0.8,
                label="p99 normalized latency", zorder=4)

    # Reference markers.
    ax.axvline(SAFE_RPS, color="#1f77b4", linestyle="--", lw=1.3,
               alpha=0.7, zorder=3)
    ax.text(SAFE_RPS, 0.02, "  safe op  1.91",
            transform=ax.get_xaxis_transform(),
            ha="left", va="bottom",
            color="#1f77b4", fontsize=FONT_ANNOTATE, fontweight="bold")
    ax.axvline(CEILING, color="#c0392b", linestyle=":", lw=1.3,
               alpha=0.7, zorder=3)
    ax.text(CEILING, 0.02, "  ceiling  2.10",
            transform=ax.get_xaxis_transform(),
            ha="left", va="bottom",
            color="#c0392b", fontsize=FONT_ANNOTATE, fontweight="bold")

    if not linear:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel("Offered mean load  (RPS"
                  + ("" if linear else ", log scale") + ")",
                  fontsize=FONT_LABEL + 1, fontweight="bold")
    ax.set_ylabel("ms / generated token"
                  + ("" if linear else "  (log scale)"),
                  fontsize=FONT_LABEL + 1, fontweight="bold")
    ax.set_title("Pi0 @ azure_poisson_wide  —  normalized knee at max_active=5000",
                 fontsize=FONT_TITLE + 1, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=FONT_TICK + 1)
    ax.grid(True, which="major", alpha=0.45)
    if not linear:
        ax.grid(True, which="minor", alpha=0.18)
    set_spines(ax)
    if show_p99:
        leg = ax.legend(loc="upper left", fontsize=FONT_TICK + 1,
                        framealpha=0.95)
        bold_legend(leg)

    save_fig(fig, out_name, out_dir)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--linear", action="store_true",
                    help="linear axes instead of log-log")
    ap.add_argument("--p99", action="store_true",
                    help="also overlay p99 normalized latency")
    ap.add_argument("--out-name", default="fig_rps_knee_a5000_solo")
    args = ap.parse_args()

    pts = collect(SWEEP)
    if not pts:
        sys.exit(f"[error] no cells found in {SWEEP}")
    print(f"[info] {len(pts)} cells loaded")
    for scale, offered, mean_n, p99_n in pts:
        print(f"  s={scale:>5.2f}  offered={offered:>5.2f}  "
              f"mean_norm={mean_n:>8.2f}  p99_norm={p99_n:>9.2f} ms/tok")

    name = args.out_name + ("_linear" if args.linear else "")
    render(pts, REPO / "thesis_plotting/figures", name,
           linear=args.linear, show_p99=args.p99)


if __name__ == "__main__":
    main()
