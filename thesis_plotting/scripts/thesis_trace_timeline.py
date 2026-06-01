#!/usr/bin/env python3
"""Thesis figure: RPS-over-time bar chart for a trace.

Bins arrivals from a TIMESTAMP,ContextTokens,GeneratedTokens CSV into
fixed time buckets and plots instantaneous RPS as colored bars. Optional
horizontal `--a-rps` overlay marks the system's measured peak RPS so
the reader can see how much of the trace is above / near / below the
system's capacity.

Bar coloring (relative to `a`):
  - green  : < 0.7 × a    (under-stressing the system)
  - amber  : 0.7 × a ≤ … < 1.0 × a   (near capacity)
  - red    : ≥ 1.0 × a    (above capacity → expected to queue)

If `--a-rps` is not given the bars are colored uniformly and only the
distribution is shown.

Output: thesis_plotting/figures/fig_trace_timeline_<tag>.{pdf,png}

Usage:
    # Azure conv full trace, overlay knee at a=14 req/s
    python thesis_plotting/scripts/thesis_trace_timeline.py \\
        --csv cluster_outputs/azure/AzureLLMInferenceTrace_conv.csv \\
        --a-rps 14 --tag azure

    # Synthetic v2 robotic (no overlay)
    python thesis_plotting/scripts/thesis_trace_timeline.py \\
        --csv cluster_outputs/synthetic_v2/robotic_vla_n5000_r30_seed0.csv \\
        --tag robotic
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO))

from profile_trace_rps import load_rows, bucket_rps  # noqa: E402

from thesis_plotting.style import (  # noqa: E402
    configure_plotting, save_fig, set_spines, bold_legend,
    COLOR_CATEGORY, WIDE_FIGSIZE,
    FONT_ANNOTATE, FONT_LABEL, FONT_TITLE,
)


def color_for(rps: float, a: float | None) -> str:
    if a is None:
        return "#1f6e8c"   # deep teal — neutral
    if rps < 0.7 * a:
        return COLOR_CATEGORY["low"]    # lightgreen
    if rps < 1.0 * a:
        return COLOR_CATEGORY["mid"]    # orange
    return COLOR_CATEGORY["high"]       # lightcoral


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, required=True,
                    help="Trace CSV path.")
    ap.add_argument("--bucket-ms", type=int, default=1000,
                    help="Bucket size for the timeline (default 1000ms).")
    ap.add_argument("--a-rps", type=float, default=None,
                    help="System's peak RPS `a` to overlay as a "
                         "horizontal reference line. If omitted, the "
                         "bars get a uniform color and no line is drawn.")
    ap.add_argument("--tag", type=str, required=True,
                    help="Short tag for the output filename "
                         "(e.g. 'azure', 'robotic', 'conv').")
    ap.add_argument("--max-buckets", type=int, default=None,
                    help="If the trace has more than this many buckets, "
                         "subsample for readability. Default = no cap.")
    ap.add_argument("--out-dir", type=Path,
                    default=REPO / "thesis_plotting" / "figures")
    args = ap.parse_args()

    if not args.csv.is_file():
        sys.exit(f"[error] CSV not found: {args.csv}")

    configure_plotting()

    rows = list(load_rows(args.csv))
    if not rows:
        sys.exit(f"[error] empty CSV: {args.csv}")
    timestamps = [r[0] for r in rows]

    buckets = bucket_rps(timestamps, args.bucket_ms)
    ts_s = [t for t, _ in buckets]
    counts = [c for _, c in buckets]
    rps_series = [c * 1000.0 / args.bucket_ms for c in counts]

    if args.max_buckets and len(rps_series) > args.max_buckets:
        # Average-pool to fit max_buckets bins. Width per bin scales.
        step = len(rps_series) // args.max_buckets
        ts_s = ts_s[::step][:args.max_buckets]
        rps_series = [
            sum(rps_series[i:i + step]) / step
            for i in range(0, len(rps_series) - step + 1, step)
        ][:len(ts_s)]
        bar_w_s = (ts_s[1] - ts_s[0]) if len(ts_s) > 1 else args.bucket_ms / 1000.0
        print(f"[info] subsampled to {len(ts_s)} buckets "
              f"(avg-pool step {step})")
    else:
        bar_w_s = args.bucket_ms / 1000.0

    colors = [color_for(r, args.a_rps) for r in rps_series]
    peak = max(rps_series) if rps_series else 0.0
    mean = sum(rps_series) / max(len(rps_series), 1)

    fig, ax = plt.subplots(figsize=(WIDE_FIGSIZE[0], 3.6))

    # Use minutes if span >1h for readability.
    span_s = ts_s[-1] if ts_s else 0
    if span_s > 3600:
        ts_axis = [t / 60.0 for t in ts_s]
        x_unit = "min"
        bar_w_axis = bar_w_s / 60.0
    else:
        ts_axis = ts_s
        x_unit = "s"
        bar_w_axis = bar_w_s

    ax.bar(ts_axis, rps_series, width=bar_w_axis,
           color=colors, edgecolor="none", align="edge",
           zorder=3)

    if args.a_rps is not None:
        ax.axhline(args.a_rps, color="#c0392b", linestyle="--",
                   linewidth=1.0, zorder=5)
        ax.text(0.985, args.a_rps,
                f"  peak RPS  a = {args.a_rps:.1f}",
                transform=ax.get_yaxis_transform(),
                ha="right", va="bottom",
                fontsize=FONT_ANNOTATE + 1, color="#c0392b",
                fontweight="bold")

    ax.set_xlabel(f"Time ({x_unit})",
                  fontsize=FONT_LABEL, fontweight="bold")
    ax.set_ylabel(f"Instantaneous RPS  ({args.bucket_ms} ms buckets)",
                  fontsize=FONT_LABEL, fontweight="bold")
    title = f"Arrival timeline — {args.tag}"
    subtitle = (f"n={len(rows):,}    "
                f"peak={peak:.1f} req/s    "
                f"mean={mean:.2f} req/s    "
                f"span={span_s:,.0f} s")
    ax.set_title(f"{title}\n{subtitle}",
                 fontsize=FONT_TITLE, fontweight="bold")
    ax.set_xlim(0, ts_axis[-1] + bar_w_axis if ts_axis else 1)
    ax.set_ylim(0, max(peak * 1.18, (args.a_rps or 0) * 1.18, 1))
    ax.grid(axis="y", alpha=0.5)
    ax.grid(axis="x", visible=False)
    set_spines(ax)

    # Color legend if a_rps was provided.
    if args.a_rps is not None:
        handles = [
            plt.Rectangle((0, 0), 1, 1,
                          facecolor=COLOR_CATEGORY["low"],
                          edgecolor="black", linewidth=0.3),
            plt.Rectangle((0, 0), 1, 1,
                          facecolor=COLOR_CATEGORY["mid"],
                          edgecolor="black", linewidth=0.3),
            plt.Rectangle((0, 0), 1, 1,
                          facecolor=COLOR_CATEGORY["high"],
                          edgecolor="black", linewidth=0.3),
        ]
        labels = [f"under (< 0.7 a)",
                  f"near (0.7–1.0 a)",
                  f"over (≥ a)"]
        leg = ax.legend(handles, labels, loc="upper right",
                        ncol=3, fontsize=FONT_ANNOTATE + 1)
        bold_legend(leg)

    save_fig(fig, f"fig_trace_timeline_{args.tag}", args.out_dir)
    plt.close(fig)

    print(f"[info] {args.tag}: n={len(rows)}  span={span_s:.0f}s  "
          f"peak={peak:.2f} RPS  mean={mean:.2f} RPS")


if __name__ == "__main__":
    main()
