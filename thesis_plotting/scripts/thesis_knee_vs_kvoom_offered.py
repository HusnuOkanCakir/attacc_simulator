#!/usr/bin/env python3
"""Offered-RPS knee curve for the KV-OOM-response sweep at 0.25 GiB pool.

Mirrors thesis_knee_vs_kvpolicy_offered.py. Three KV-OOM responses
on the same x-axis (offered mean RPS per cell, computed from
arrival timestamps in requests_out.csv). Per-policy safe-op lines
at 0.9 x each response's achieved-throughput ceiling.

Source: cluster_outputs/online_serving_runs/knee_vs_kvoom_0p25gb_20260609_223709/
Output: thesis_plotting/figures/fig_knee_vs_kvoom_0p25gb_offered.{pdf,png}
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

SWEEP = REPO / "cluster_outputs/online_serving_runs/knee_vs_kvoom_0p25gb_20260609_223709"
OUT_NAME = "fig_knee_vs_kvoom_0p25gb_offered"

# hold | fbg = fallback_gpu | htf = hold_then_fallback
CELL_RE = re.compile(r"^\d+_kvoom_0p25gb(?:_fill)?_(?P<v>hold|fbg|htf)_s(?P<sx10>\d+)$")

POLICY_LABEL = {
    "hold": "hold",
    "fbg": "fallback_gpu",
    "htf": "hold_then_fallback",
}
POLICY_COLOR = {
    "hold": "#c44e52",  # red
    "fbg": "#5a9b5a",   # green
    "htf": "#406ea3",   # blue
}
POLICY_MARKER = {"hold": "s", "fbg": "^", "htf": "o"}
POLICY_ORDER = ["hold", "fbg", "htf"]


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
            pol = m.group("v")
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
            rows.append((pol, scale, offered, achieved, e2e_p99))
    return rows


def render(rows, out_dir: Path, out_name: str = OUT_NAME, linear: bool = False):
    configure_plotting()
    by_pol = {}
    for pol, scale, offered, achieved, p99 in rows:
        by_pol.setdefault(pol, []).append((offered, achieved, p99, scale))

    fig, ax = plt.subplots(figsize=(11.0, 6.2))

    for pol in POLICY_ORDER:
        if pol not in by_pol:
            continue
        pts = sorted(by_pol[pol], key=lambda r: r[0])
        xs = [p[0] for p in pts]
        ys = [p[2] / 1000.0 for p in pts]
        ax.plot(xs, ys, marker=POLICY_MARKER[pol], ms=10, lw=2.4,
                color=POLICY_COLOR[pol], label=POLICY_LABEL[pol],
                markeredgecolor="black", markeredgewidth=0.6, zorder=4)

    # Per-policy safe operating point: 0.9 x policy-specific achieved-RPS ceiling
    label_offsets = {"hold": 0.03, "fbg": 0.10, "htf": 0.17}
    for pol in POLICY_ORDER:
        if pol not in by_pol:
            continue
        achieved_vals = [p[1] for p in by_pol[pol]]
        ceiling = max(achieved_vals)
        safe = 0.9 * ceiling
        color = POLICY_COLOR[pol]
        ax.axvline(safe, color=color, linestyle="--", linewidth=1.4,
                   zorder=3, alpha=0.75)
        ax.text(safe, label_offsets.get(pol, 0.03),
                f"  safe {pol} {safe:.2f}",
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
    ax.set_title("Pi0 @ azure_poisson_wide  —  KV-OOM response at 0.25 GiB pool",
                 fontsize=FONT_TITLE + 1, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=FONT_TICK + 1)
    ax.grid(True, which="major", alpha=0.45)
    if not linear:
        ax.grid(True, which="minor", alpha=0.18)
    set_spines(ax)
    leg = ax.legend(loc="upper left", fontsize=FONT_TICK + 1,
                    title="KV-OOM response",
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
    for pol in POLICY_ORDER:
        cells = sorted([r for r in rows if r[0] == pol], key=lambda r: r[2])
        print(f"  {pol}:")
        for _, scale, offered, achieved, p99 in cells:
            print(f"    s={scale:>5.2f}  offered={offered:>5.2f}  "
                  f"achieved={achieved:>5.2f}  e2e_p99={p99/1000.0:>8.1f}s")
    name = args.out_name + ("_linear" if args.linear else "")
    render(rows, REPO / "thesis_plotting/figures", name, args.linear)


if __name__ == "__main__":
    main()
