#!/usr/bin/env python3
"""Side-by-side KV pool occupancy: 0.5 GiB vs 8 GiB at two arrival scales.

Reads the filtered first-60s debug_first60s.log from each of 4 cells:
- kv_timeline_kvpool_0_5_s015 / _s020 (small pool, cliff + under-cliff)
- kv_timeline_kvpool_8_s015   / _s020 (over-provisioned baseline)

Produces a single 2x2 figure showing pool occupancy over time. The
small-pool cells should saturate (100% util, frequent evicts); the
8 GiB cells should leave ~90% of the pool free.

Output: thesis_plotting/figures/fig_kv_pool_timeline_compare.{pdf,png}
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

from plot_kv_pool_timeline import parse_log, replay, plot_timeline, read_pool_bytes  # noqa: E402

CELLS = [
    # (label, run_dir, row, col)
    ("0.5 GiB, scale=1.5", "kv_timeline_kvpool_0_5_s015", 0, 0),
    ("8 GiB, scale=1.5",   "kv_timeline_kvpool_8_s015",   0, 1),
    ("0.5 GiB, scale=2.0", "kv_timeline_kvpool_0_5_s020", 1, 0),
    ("8 GiB, scale=2.0",   "kv_timeline_kvpool_8_s020",   1, 1),
]

RUN_ROOT = REPO / "cluster_outputs/online_serving_runs"
OUT_DIR  = REPO / "thesis_plotting/figures"


def main():
    fig, axes = plt.subplots(2, 2, figsize=(18, 9), sharex=True)

    for label, cell_name, r, c in CELLS:
        ax = axes[r][c]
        subdir = RUN_ROOT / cell_name / "max_util_full"
        log = subdir / "debug_first60s.log"
        if not log.exists():
            print(f"[skip] {cell_name}: no debug_first60s.log")
            ax.text(0.5, 0.5, f"missing: {cell_name}", ha="center", va="center")
            continue
        events = parse_log(log)
        segments = replay(events)
        pool = read_pool_bytes(subdir)
        plot_timeline(ax, segments, pool, label, t_max=60000)

    fig.suptitle("Pi0 + azure_poisson_wide v2 — KV pool occupancy (first 60 s simulated)\n"
                 "Left: 0.5 GiB pool (stressed) • Right: 8 GiB pool (over-provisioned)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / "fig_kv_pool_timeline_compare.pdf"
    png = OUT_DIR / "fig_kv_pool_timeline_compare.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=140)
    plt.close(fig)
    print(f"  [pdf] {pdf}")
    print(f"  [png] {png}")


if __name__ == "__main__":
    main()
