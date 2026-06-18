#!/usr/bin/env python3
"""Offered-RPS knee curve for the route sweep (hybrid vs gpu_only).

Mirror of thesis_knee_vs_bs_offered.py for the route knob. Two
curves: hybrid (PIM + GPU available) vs gpu_only (PIM route
sentinel-disabled). x-axis is offered mean RPS computed per cell
from arrival timestamps, eliminating the back-bending artifact of
the achieved-RPS rendering.

Source: cluster_outputs/online_serving_runs/knee_vs_route_20260606_001144/
Output: thesis_plotting/figures/fig_knee_vs_route_offered.{pdf,png}
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
    COLOR_ROUTE,
)

SWEEP = REPO / "cluster_outputs/online_serving_runs/knee_vs_route_20260606_001144"

CEILING_HYBRID = 2.10
CEILING_GPUONLY = 0.93
SAFE_HYBRID = 1.91
SAFE_GPUONLY = 0.79  # 0.85 * 0.93 (informational; not annotated by default)

CELL_RE = re.compile(r"^\d+_route_(?P<v>hybrid|gpu_only)_s(?P<sx10>\d+)$")

ROUTE_LABEL = {"hybrid": "Hybrid  (PIM + GPU)",
               "gpu_only": "GPU-only"}
ROUTE_COLOR = {"hybrid": COLOR_ROUTE["lpddr5_pim_bank"],
               "gpu_only": COLOR_ROUTE["gpu_only"]}
ROUTE_MARKER = {"hybrid": "o", "gpu_only": "s"}


def load_cells(sweep_dir: Path):
    rows = []
    for cell in sorted(sweep_dir.iterdir()):
        if not cell.is_dir():
            continue
        m = CELL_RE.match(cell.name)
        if not m:
            continue
        route = m.group("v")
        scale = int(m.group("sx10")) / 10.0
        sumfile = cell / "policy_compare_summary.txt"
        req_csv = cell / "max_util_full" / "requests_out.csv"
        if not sumfile.is_file() or not req_csv.is_file():
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
        rows.append((route, scale, offered, achieved, e2e_p99))
    return rows


def render(rows, out_dir: Path, out_name: str, linear: bool,
           title=None, ref_rps=SAFE_HYBRID,
           hybrid_xmax=None, gpuonly_xmax=None):
    # Local font overrides — defaults (FONT_BASE=7) render ~3 pt after LaTeX
    # scales the figure to \linewidth. Bump for readability.
    FL  = FONT_LABEL    + 8   # 15  axis labels
    FTI = FONT_TITLE    + 8   # 16  title
    FTK = FONT_TICK     + 8   # 14  ticks / legend
    FAN = FONT_ANNOTATE + 9   # 14  safe-op / ceiling annotations
    configure_plotting()
    by_route = {}
    for route, scale, offered, achieved, p99 in rows:
        by_route.setdefault(route, []).append((offered, achieved, p99, scale))

    fig, ax = plt.subplots(figsize=(10.5, 6.0))

    route_xmax = {"hybrid": hybrid_xmax, "gpu_only": gpuonly_xmax}
    for route in ["gpu_only", "hybrid"]:
        if route not in by_route:
            continue
        pts = sorted(by_route[route], key=lambda r: r[0])
        xm = route_xmax.get(route)
        if xm is not None:
            pts = [p for p in pts if p[0] <= xm]
        xs = [p[0] for p in pts]
        ys = [p[2] / 1000.0 for p in pts]
        ax.plot(xs, ys, marker=ROUTE_MARKER[route], ms=10, lw=2.5,
                color=ROUTE_COLOR[route], label=ROUTE_LABEL[route],
                markeredgecolor="black", markeredgewidth=0.7, zorder=4)

    if ref_rps and ref_rps > 0:
        ax.axvline(ref_rps, color="#1f77b4", linestyle="--",
                   linewidth=1.6, alpha=0.8, zorder=3)
        ax.text(ref_rps, 0.04, f"ref op {ref_rps:.2f} (hybrid)  ",
                transform=ax.get_xaxis_transform(),
                ha="right", va="bottom",
                color="#1f77b4", fontsize=FAN, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none",
                          pad=1.5, alpha=0.85))

    if not linear:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel("Offered mean load  (RPS"
                  + ("" if linear else ", log scale") + ")",
                  fontsize=FL, fontweight="bold")
    ax.set_ylabel("E2E p99  (s"
                  + ("" if linear else ", log scale") + ")",
                  fontsize=FL, fontweight="bold")
    ax.set_title(title or "Pi0 @ azure_poisson_wide  —  knee vs route",
                 fontsize=FTI, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=FTK)
    ax.grid(True, which="major", alpha=0.45)
    if not linear:
        ax.grid(True, which="minor", alpha=0.18)
    set_spines(ax)
    leg = ax.legend(loc="upper left", fontsize=FTK,
                    title="route mode",
                    title_fontsize=FTK, framealpha=0.95)
    bold_legend(leg)

    save_fig(fig, out_name, out_dir)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--linear", action="store_true",
                    help="linear axes instead of log-log")
    ap.add_argument("--out-name", default="fig_knee_vs_route_offered")
    ap.add_argument("--sweep-dirs", type=Path, nargs="+", default=[SWEEP],
                    help="one or more route knee sweep dirs to merge")
    ap.add_argument("--title", default=None,
                    help="override plot title (e.g. for the bootstrap trace)")
    ap.add_argument("--ref-rps", type=float, default=SAFE_HYBRID,
                    help="hybrid ref-op vertical line; <=0 hides it")
    ap.add_argument("--hybrid-xmax", type=float, default=None,
                    help="drop hybrid points with offered RPS above this")
    ap.add_argument("--gpuonly-xmax", type=float, default=None,
                    help="drop gpu_only points with offered RPS above this")
    args = ap.parse_args()

    rows = []
    for d in args.sweep_dirs:
        rows += load_cells(d)
    if not rows:
        sys.exit(f"[error] no cells found in {args.sweep_dirs}")
    print(f"[info] {len(rows)} cells loaded")
    for route in sorted({r[0] for r in rows}):
        print(f"  {route}:")
        for _, scale, offered, achieved, p99 in sorted(
                [r for r in rows if r[0] == route], key=lambda r: r[2]):
            print(f"    s={scale:>5.2f}  offered={offered:>5.2f}  "
                  f"achieved={achieved:>5.2f}  e2e_p99={p99/1000.0:>8.1f}s")

    name = args.out_name + ("_linear" if args.linear else "")
    render(rows, REPO / "thesis_plotting/figures", name, args.linear,
           title=args.title, ref_rps=args.ref_rps,
           hybrid_xmax=args.hybrid_xmax, gpuonly_xmax=args.gpuonly_xmax)


if __name__ == "__main__":
    main()
