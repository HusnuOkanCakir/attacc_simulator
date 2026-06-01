#!/usr/bin/env python3
"""Thesis figure: RPS-over-time, before vs after scaling to peak `a`.

For a given trace CSV and a measured peak RPS `a` (from the knee
sweep), this produces a two-panel figure:

  - top:    original timeline (arrival_scale = 1.0). Bars colored
            under/near/over `a`; if the trace's intrinsic peak is far
            from `a`, most bars sit on one side.
  - bottom: same trace with timestamps multiplied by
            `scale = current_peak / a`, so the peak instantaneous RPS
            now touches the `a` line. This is what the cluster sweep
            would actually run at `--arrival-scale <scale>`.

Output: thesis_plotting/figures/fig_rps_scale_<tag>.{pdf,png}

Usage:
    python thesis_plotting/scripts/thesis_rps_scale_to_peak.py \\
        --csv cluster_outputs/synthetic_v2/vla_poisson_n5000_seed0.csv \\
        --a-rps 20.1 --tag vla_poisson --bucket-ms 1000
"""

import argparse
import datetime as _dt
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


def color_for(rps: float, a: float) -> str:
    if rps < 0.7 * a:
        return COLOR_CATEGORY["low"]
    if rps < 1.0 * a:
        return COLOR_CATEGORY["mid"]
    return COLOR_CATEGORY["high"]


def render_panel(ax, ts_s: list[float], rps_series: list[float],
                 a: float, bucket_ms: int, x_unit: str,
                 title: str, subtitle: str):
    bar_w_s = (ts_s[1] - ts_s[0]) if len(ts_s) > 1 else bucket_ms / 1000.0
    if x_unit == "min":
        ts_axis = [t / 60.0 for t in ts_s]
        bar_w_axis = bar_w_s / 60.0
    else:
        ts_axis = ts_s
        bar_w_axis = bar_w_s

    colors = [color_for(r, a) for r in rps_series]
    ax.bar(ts_axis, rps_series, width=bar_w_axis,
           color=colors, edgecolor="none", align="edge",
           zorder=3)

    ax.axhline(a, color="#c0392b", linestyle="--",
               linewidth=1.0, zorder=5)
    ax.text(0.985, a, f"  a = {a:.1f}",
            transform=ax.get_yaxis_transform(),
            ha="right", va="bottom",
            fontsize=FONT_ANNOTATE + 1, color="#c0392b",
            fontweight="bold")

    peak = max(rps_series) if rps_series else 0.0
    mean = sum(rps_series) / max(len(rps_series), 1)

    ax.set_ylabel(f"RPS  ({bucket_ms} ms bucket)",
                  fontsize=FONT_LABEL, fontweight="bold")
    # Short title — just panel label. Stats annotated inside panel.
    ax.set_title(title, fontsize=FONT_TITLE - 1,
                 fontweight="bold", loc="left")
    # Stats line inside the panel (upper-left, just below the title).
    ax.text(0.005, 0.92,
            f"peak = {peak:.1f}    mean = {mean:.2f}    {subtitle}",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=FONT_ANNOTATE + 1, fontweight="bold",
            color="#333",
            bbox=dict(facecolor="white", edgecolor="none",
                      pad=1.5, alpha=0.85))
    ax.set_xlim(0, ts_axis[-1] + bar_w_axis if ts_axis else 1)
    ax.set_ylim(0, max(peak * 1.18, a * 1.18, 1))
    ax.grid(axis="y", alpha=0.5)
    ax.grid(axis="x", visible=False)
    set_spines(ax)
    return ts_axis


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, required=True,
                    help="Original trace CSV (arrival_scale = 1.0).")
    ap.add_argument("--a-rps", type=float, required=True,
                    help="System's measured peak RPS — both panels "
                         "overlay this as the red dashed reference.")
    ap.add_argument("--tag", type=str, required=True,
                    help="Short tag for the output filename "
                         "(e.g. 'vla_poisson').")
    ap.add_argument("--bucket-ms", type=int, default=1000,
                    help="Bucket size for RPS counting (default 1000ms). "
                         "Larger buckets give wider bars but average over "
                         "more time, shrinking the apparent peak.")
    ap.add_argument("--arrival-scale", type=float, default=None,
                    help="Force a specific arrival_scale value (overrides "
                         "the auto-computed peak/a). Use this to keep the "
                         "visualization aligned with a real run's scale "
                         "when bucket-ms changes the apparent peak.")
    ap.add_argument("--out-dir", type=Path,
                    default=REPO / "thesis_plotting" / "figures")
    args = ap.parse_args()

    if not args.csv.is_file():
        sys.exit(f"[error] CSV not found: {args.csv}")
    if args.a_rps <= 0:
        sys.exit("[error] --a-rps must be positive")

    configure_plotting()

    rows = list(load_rows(args.csv))
    if not rows:
        sys.exit(f"[error] empty CSV: {args.csv}")
    timestamps = [r[0] for r in rows]
    t0 = timestamps[0]

    # ── Panel A: original ──────────────────────────────────────────────
    buckets_orig = bucket_rps(timestamps, args.bucket_ms)
    ts_orig = [t for t, _ in buckets_orig]
    rps_orig = [c * 1000.0 / args.bucket_ms for _, c in buckets_orig]
    peak_orig = max(rps_orig) if rps_orig else 0.0

    # ── Compute scale so peak → a (or use override) ────────────────────
    if peak_orig <= 0:
        sys.exit("[error] zero peak in original trace")
    if args.arrival_scale is not None:
        scale = args.arrival_scale
    else:
        scale = peak_orig / args.a_rps
    span_orig_s = ts_orig[-1] if ts_orig else 0

    # Apply scale = multiply (ts - t0) by `scale`. scale > 1 spreads
    # arrivals (lower RPS); scale < 1 compresses them (higher RPS).
    scaled_ts = [t0 + (ts - t0) * scale for ts in timestamps]
    # Scale the bucket size proportionally so BOTH panels end up with
    # the same number of bars (same visual width). RPS values still
    # comparable since we report req/s either way — the scaled bucket
    # just averages over a proportionally longer time window.
    bucket_ms_scaled = int(round(args.bucket_ms * scale))
    buckets_scaled = bucket_rps(scaled_ts, bucket_ms_scaled)
    ts_scaled = [t for t, _ in buckets_scaled]
    rps_scaled = [c * 1000.0 / bucket_ms_scaled for _, c in buckets_scaled]
    span_scaled_s = ts_scaled[-1] if ts_scaled else 0

    # Choose x-axis unit per panel based on each panel's span.
    use_min_orig = span_orig_s > 3600
    use_min_scaled = span_scaled_s > 3600
    x_unit_orig = "min" if use_min_orig else "s"
    x_unit_scaled = "min" if use_min_scaled else "s"

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(WIDE_FIGSIZE[0], 4.8),
        sharex=False, gridspec_kw={"hspace": 0.40},
    )

    render_panel(
        ax_top, ts_orig, rps_orig, args.a_rps, args.bucket_ms,
        x_unit_orig,
        title="(a) Original trace  (arrival_scale = 1.0)",
        subtitle=f"span = {span_orig_s:,.0f} s",
    )
    ax_top.set_xlabel(f"Time ({x_unit_orig})",
                      fontsize=FONT_LABEL, fontweight="bold")

    peak_scaled = max(rps_scaled) if rps_scaled else 0.0
    # "peak ≈ a" only when the scaled peak is within ±15% of a.
    # Otherwise the override pushed the system below or above a;
    # show the actual peak value instead.
    if abs(peak_scaled - args.a_rps) / max(args.a_rps, 1e-9) <= 0.15:
        peak_note = "peak ≈ a"
    else:
        peak_note = f"peak = {peak_scaled:.2f} req/s  (a = {args.a_rps:.2f})"
    render_panel(
        ax_bot, ts_scaled, rps_scaled, args.a_rps, bucket_ms_scaled,
        x_unit_scaled,
        title=f"(b) Scaled  (--arrival-scale = {scale:.3f})",
        subtitle=f"span = {span_scaled_s:,.0f} s   {peak_note}",
    )
    ax_bot.set_xlabel(f"Time ({x_unit_scaled})",
                      fontsize=FONT_LABEL, fontweight="bold")

    # Color legend at upper center — sits between the suptitle and the
    # (short) panel titles. Now the panel titles are 1 line, no overlap.
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=COLOR_CATEGORY["low"],
                      edgecolor="black", linewidth=0.3),
        plt.Rectangle((0, 0), 1, 1, facecolor=COLOR_CATEGORY["mid"],
                      edgecolor="black", linewidth=0.3),
        plt.Rectangle((0, 0), 1, 1, facecolor=COLOR_CATEGORY["high"],
                      edgecolor="black", linewidth=0.3),
    ]
    labels = ["under (< 0.7 a)", "near (0.7–1.0 a)", "over (≥ a)"]
    leg = fig.legend(handles, labels, loc="upper center",
                     ncol=3, fontsize=FONT_ANNOTATE + 1,
                     bbox_to_anchor=(0.5, 1.06))
    bold_legend(leg)

    fig.suptitle(
        f"RPS over time — {args.tag}   (a = {args.a_rps:.1f} req/s)",
        fontsize=FONT_TITLE + 1, fontweight="bold", y=1.13,
    )

    save_fig(fig, f"fig_rps_scale_{args.tag}", args.out_dir)
    plt.close(fig)

    print(f"[info] {args.tag}:  original peak={peak_orig:.1f}  "
          f"scale={scale:.4f}  scaled peak={max(rps_scaled):.1f}")


if __name__ == "__main__":
    main()
