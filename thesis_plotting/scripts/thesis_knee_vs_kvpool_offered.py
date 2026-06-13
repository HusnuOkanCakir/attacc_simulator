#!/usr/bin/env python3
"""Offered-RPS knee curve for the KV-pool size sweep at max_active=5000.

Mirrors thesis_knee_vs_kvpolicy_offered.py. Four KV-pool sizes on the
same x-axis (offered mean RPS per cell, computed from arrival
timestamps in requests_out.csv, NOT the achieved throughput). Per-pool
safe-op vertical markers at 0.9 x each pool's achieved-throughput
ceiling.

Source: cluster_outputs/online_serving_runs/knee_vs_kvpool_20260608_192523/
Output: thesis_plotting/figures/fig_knee_vs_kvpool_offered.{pdf,png}
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

SWEEP = REPO / "cluster_outputs/online_serving_runs/knee_vs_kvpool_20260608_192523"
OUT_NAME = "fig_knee_vs_kvpool_offered"

# Cell label format: NN_kvpool_<value>_s<scale*10>, where <value> is
# "0_25", "1", "8", "64", etc. (underscores in floats).
CELL_RE = re.compile(r"^\d+_kvpool(?:_lowfill|_midfill)?_(?P<v>0_25|0_5|0_75|1|2|8|64)_s(?P<sx10>\d+)$")

# Display order — pool sizes in GiB; subset matches the chapter selection.
DEFAULT_KEEP = ["0_25", "1", "8", "64"]

POOL_LABEL = {
    "0_25": "0.25 GiB",
    "0_5":  "0.5 GiB",
    "0_75": "0.75 GiB",
    "1":    "1 GiB",
    "2":    "2 GiB",
    "8":    "8 GiB",
    "64":   "64 GiB",
}
POOL_COLOR = {
    "0_25": "#c44e52",   # red — severely starved
    "1":    "#e8a86c",   # orange
    "8":    "#5a9b5a",   # green — near-correct
    "64":   "#406ea3",   # blue — saturated
}
POOL_MARKER = {"0_25": "^", "1": "s", "8": "D", "64": "o"}


def load_cells(sweep_dirs, keep_values):
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
            pool = m.group("v")
            if pool not in keep_values:
                continue
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
            rows.append((pool, scale, offered, achieved, e2e_p99))
    return rows


def render(rows, out_dir: Path, out_name: str, keep_order, linear: bool = False):
    configure_plotting()
    # Local font overrides — defaults (FONT_BASE=7) render ~3 pt after LaTeX
    # scales the figure to \linewidth. Bump for readability.
    FL  = FONT_LABEL    + 8   # 15  axis labels
    FTI = FONT_TITLE    + 8   # 16  title
    FTK = FONT_TICK     + 8   # 14  ticks / legend
    FAN = FONT_ANNOTATE + 9   # 14  safe-op annotations
    by_pool = {}
    for pool, scale, offered, achieved, p99 in rows:
        by_pool.setdefault(pool, []).append((offered, achieved, p99, scale))

    fig, ax = plt.subplots(figsize=(12.0, 7.0))

    for pool in keep_order:
        if pool not in by_pool:
            continue
        pts = sorted(by_pool[pool], key=lambda r: r[0])
        xs = [p[0] for p in pts]
        ys = [p[2] / 1000.0 for p in pts]
        ax.plot(xs, ys, marker=POOL_MARKER[pool], ms=10, lw=2.4,
                color=POOL_COLOR[pool], label=POOL_LABEL[pool],
                markeredgecolor="black", markeredgewidth=0.6, zorder=4)

    # Per-pool ref operating point: 0.9 x pool-specific achieved-RPS ceiling.
    # Each pool size hits its own throughput plateau; the unconstrained ceiling
    # (2.10 RPS at max_active=5000) is only reached by the 64 GiB pool.
    label_offsets = {"0_25": 0.03, "1": 0.10, "8": 0.17, "64": 0.24}
    for pool in keep_order:
        if pool not in by_pool:
            continue
        achieved_vals = [p[1] for p in by_pool[pool]]
        ceiling = max(achieved_vals)
        safe = 0.9 * ceiling
        color = POOL_COLOR[pool]
        ax.axvline(safe, color=color, linestyle="--", linewidth=1.4,
                   zorder=3, alpha=0.75)
        ax.text(safe, label_offsets.get(pool, 0.03),
                f"  ref {POOL_LABEL[pool]} {safe:.2f}",
                transform=ax.get_xaxis_transform(),
                ha="left", va="bottom",
                color=color, fontsize=FAN,
                fontweight="bold")

    if not linear:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel("Offered mean load  (RPS"
                  + ("" if linear else ", log scale") + ")",
                  fontsize=FL, fontweight="bold")
    ax.set_ylabel("E2E p99  (s"
                  + ("" if linear else ", log scale") + ")",
                  fontsize=FL, fontweight="bold")
    ax.set_title("Pi0 @ azure_poisson_wide  —  knee vs KV pool size",
                 fontsize=FTI, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=FTK)
    ax.grid(True, which="major", alpha=0.45)
    if not linear:
        ax.grid(True, which="minor", alpha=0.18)
    set_spines(ax)
    leg = ax.legend(loc="upper left", fontsize=FTK,
                    title="PIM KV pool",
                    title_fontsize=FTK, framealpha=0.95)
    bold_legend(leg)
    fig.tight_layout()

    save_fig(fig, out_name, out_dir)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--linear", action="store_true")
    ap.add_argument("--out-name", default=OUT_NAME)
    ap.add_argument("--sweep-dirs", type=Path, nargs="+", default=[SWEEP])
    ap.add_argument("--keep-values", type=str,
                    default=",".join(DEFAULT_KEEP),
                    help="Comma-separated underscore-form pool values "
                         "(e.g. '0_25,1,8,64'). Default: %(default)s.")
    args = ap.parse_args()

    keep = [v.strip() for v in args.keep_values.split(",") if v.strip()]
    rows = load_cells(args.sweep_dirs, set(keep))
    if not rows:
        sys.exit(f"[error] no cells found in {args.sweep_dirs}")
    print(f"[info] {len(rows)} cells loaded")
    for pool in keep:
        cells = sorted([r for r in rows if r[0] == pool], key=lambda r: r[2])
        if not cells:
            continue
        print(f"  {pool} ({POOL_LABEL.get(pool, pool)}):")
        for _, scale, offered, achieved, p99 in cells:
            print(f"    s={scale:>5.2f}  offered={offered:>5.2f}  "
                  f"achieved={achieved:>5.2f}  e2e_p99={p99/1000.0:>8.1f}s")
    name = args.out_name + ("_linear" if args.linear else "")
    render(rows, REPO / "thesis_plotting/figures", name, keep, args.linear)


if __name__ == "__main__":
    main()
