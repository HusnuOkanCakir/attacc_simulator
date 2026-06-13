#!/usr/bin/env python3
"""Normalized-latency knee curve for the bs sweep, offered-RPS x-axis.

Plots ms/token (mean and p99) instead of raw e2e p99. The
finite-trace plateau seen in the p99(e2e) version disappears here,
since normalized latency factors out the request-length scaling
that dominates the deep-saturation cells.

Source: cluster_outputs/online_serving_runs/knee_vs_bs_20260606_114716/
Output: thesis_plotting/figures/fig_knee_vs_bs_normalized.{pdf,png}
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from thesis_plotting.style import (  # noqa: E402
    configure_plotting, save_fig, set_spines, bold_legend,
    FONT_LABEL, FONT_TITLE, FONT_TICK, FONT_ANNOTATE,
)

SWEEP = REPO / "cluster_outputs/online_serving_runs/knee_vs_bs_20260606_114716"
OUT_NAME = "fig_knee_vs_bs_normalized"

CEILING_RPS = 2.13
SAFE_RPS = 1.91

CELL_RE = re.compile(r"^\d+_bs_(?P<v>\d+)_s(?P<sx10>\d+)$")


def percentile(vals, q):
    if not vals:
        return None
    s = sorted(vals)
    k = (len(s) - 1) * q
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def load_cells(sweep_dir: Path):
    rows = []
    for cell in sorted(sweep_dir.iterdir()):
        if not cell.is_dir():
            continue
        m = CELL_RE.match(cell.name)
        if not m:
            continue
        knob = int(m.group("v"))
        scale = int(m.group("sx10")) / 10.0
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
        mean_norm = sum(norms) / len(norms)
        p99_norm = percentile(norms, 0.99)
        rows.append((knob, scale, offered, mean_norm, p99_norm))
    return rows


def render(rows, out_dir: Path, out_name: str = OUT_NAME, linear: bool = False):
    configure_plotting()
    by_knob = {}
    for knob, scale, offered, mean_n, p99_n in rows:
        by_knob.setdefault(knob, []).append((offered, mean_n, p99_n, scale))
    knob_values = sorted(by_knob.keys())

    cmap = plt.get_cmap("viridis")
    colors = {kv: cmap(i / max(1, len(knob_values) - 1))
              for i, kv in enumerate(knob_values)}

    fig, (ax_mean, ax_p99) = plt.subplots(
        1, 2, figsize=(14.5, 6.2), sharex=True,
    )

    markers = ["o", "s", "^", "D", "v"]
    for ax, idx, title in [
        (ax_mean, 1, "Mean normalized latency"),
        (ax_p99,  2, "p99 normalized latency"),
    ]:
        for i, kv in enumerate(knob_values):
            pts = sorted(by_knob[kv], key=lambda r: r[0])
            xs = [p[0] for p in pts]
            ys = [p[idx] for p in pts]
            ax.plot(xs, ys, marker=markers[i % len(markers)], ms=9, lw=2.4,
                    color=colors[kv], label=f"bs = {kv}",
                    markeredgecolor="black", markeredgewidth=0.6, zorder=4)
        ax.axvline(SAFE_RPS, color="#1f77b4", linestyle="--",
                   linewidth=1.3, alpha=0.7, zorder=3)
        ax.text(SAFE_RPS, 0.02, "  safe op  1.91",
                transform=ax.get_xaxis_transform(),
                ha="left", va="bottom",
                color="#1f77b4", fontsize=FONT_ANNOTATE, fontweight="bold")
        ax.axvline(CEILING_RPS, color="#c0392b", linestyle=":",
                   linewidth=1.3, alpha=0.7, zorder=3)
        ax.text(CEILING_RPS, 0.02, "  ceiling  2.13",
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
        ax.set_title(title, fontsize=FONT_TITLE, fontweight="bold")
        ax.tick_params(axis="both", which="major", labelsize=FONT_TICK + 1)
        ax.grid(True, which="major", alpha=0.45)
        if not linear:
            ax.grid(True, which="minor", alpha=0.18)
        set_spines(ax)

    leg = ax_mean.legend(loc="upper left", fontsize=FONT_TICK + 1,
                         title="decode batch cap",
                         title_fontsize=FONT_TICK + 1, framealpha=0.95)
    bold_legend(leg)

    fig.suptitle("Pi0 @ azure_poisson_wide  —  normalized knee vs decode batch cap",
                 fontsize=FONT_TITLE + 2, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, out_name, out_dir)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--linear", action="store_true")
    ap.add_argument("--out-name", default=OUT_NAME)
    args = ap.parse_args()

    rows = load_cells(SWEEP)
    if not rows:
        sys.exit(f"[error] no cells found in {SWEEP}")
    print(f"[info] {len(rows)} cells loaded")
    for kv in sorted({r[0] for r in rows}):
        print(f"  bs={kv}:")
        for _, scale, offered, mean_n, p99_n in sorted(
                [r for r in rows if r[0] == kv], key=lambda r: r[2]):
            print(f"    s={scale:>5.2f}  offered={offered:>5.2f}  "
                  f"mean_norm={mean_n:>9.2f}  p99_norm={p99_n:>9.2f} ms/tok")
    name = args.out_name + ("_linear" if args.linear else "")
    render(rows, REPO / "thesis_plotting/figures", name, args.linear)


if __name__ == "__main__":
    main()
