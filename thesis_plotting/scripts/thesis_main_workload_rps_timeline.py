#!/usr/bin/env python3
"""RPS timeline for the main Chapter 6 Pi0 comparison workload.

The script uses the same request CSV as the main GPU-only versus hybrid
comparison and applies the documented time stretching internally. The rendered
figure reports only the resulting RPS statistics.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO))

from profile_trace_rps import load_rows, bucket_rps  # noqa: E402
from thesis_plotting.style import (  # noqa: E402
    configure_plotting, save_fig, set_spines,
    COLOR_CATEGORY, FONT_ANNOTATE, FONT_LABEL, FONT_TITLE, WIDE_FIGSIZE,
)


CSV_PATH = REPO / "cluster_outputs/synthetic_v2/azure_poisson_wide_n5000_seed0.csv"
OUT_NAME = "fig_rps_timeline_pi0_wide_main"
INTERNAL_TIME_STRETCH = 5.0
BUCKET_MS = 1000


def main() -> None:
    if not CSV_PATH.is_file():
        sys.exit(f"[error] CSV not found: {CSV_PATH}")

    configure_plotting()
    rows = list(load_rows(CSV_PATH))
    timestamps = [r[0] for r in rows]
    t0 = timestamps[0]
    scaled_ts = [t0 + (ts - t0) * INTERNAL_TIME_STRETCH for ts in timestamps]
    buckets = bucket_rps(scaled_ts, BUCKET_MS)
    ts_s = [t for t, _ in buckets]
    rps = [count * 1000.0 / BUCKET_MS for _, count in buckets]
    mean_rps = len(rows) / ((scaled_ts[-1] - scaled_ts[0]).total_seconds())
    peak_rps = max(rps)
    p99_rps = sorted(rps)[int(0.99 * (len(rps) - 1))]

    fig, ax = plt.subplots(figsize=(WIDE_FIGSIZE[0], 3.8))
    colors = [
        COLOR_CATEGORY["low"] if x < mean_rps else
        COLOR_CATEGORY["mid"] if x < peak_rps else
        COLOR_CATEGORY["high"]
        for x in rps
    ]
    ax.bar([t / 60.0 for t in ts_s], rps, width=(BUCKET_MS / 1000.0) / 60.0,
           color=colors, edgecolor="none", align="edge", zorder=3)
    ax.axhline(mean_rps, color="#c0392b", linestyle="--", linewidth=1.2, zorder=4)
    ax.text(0.99, mean_rps, f" mean offered load = {mean_rps:.2f} RPS",
            transform=ax.get_yaxis_transform(), ha="right", va="bottom",
            fontsize=FONT_ANNOTATE + 1, color="#c0392b", fontweight="bold")

    ax.text(0.01, 0.94,
            f"n = {len(rows):,} requests    peak = {peak_rps:.1f} RPS    p99 = {p99_rps:.1f} RPS",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=FONT_ANNOTATE + 1, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.86, pad=2))
    ax.set_xlabel("Time (min)", fontsize=FONT_LABEL, fontweight="bold")
    ax.set_ylabel(f"Instantaneous RPS ({BUCKET_MS // 1000} s buckets)",
                  fontsize=FONT_LABEL, fontweight="bold")
    ax.set_title("Main comparison input load over time",
                 fontsize=FONT_TITLE, fontweight="bold")
    ax.grid(axis="y", alpha=0.45)
    ax.grid(axis="x", visible=False)
    ax.set_xlim(0, ts_s[-1] / 60.0)
    ax.set_ylim(0, max(peak_rps * 1.18, mean_rps * 1.5))
    set_spines(ax)
    fig.tight_layout()
    save_fig(fig, OUT_NAME, REPO / "thesis_plotting/figures")
    plt.close(fig)
    print(f"[info] wrote {OUT_NAME}: mean={mean_rps:.3f} RPS peak={peak_rps:.1f} RPS")


if __name__ == "__main__":
    main()
