#!/usr/bin/env python3
"""Thesis figure: per-request shape (Lin / Lout) over time.

For each request in the CSV, plot (arrival_time, ContextTokens) as a
small scatter point, with a second translucent layer for
GeneratedTokens. Reveals how the trace's *shape distribution* evolves
over the trace — clusters of long-context requests, bursts, etc.

Distinct from `thesis_trace_timeline.py` (which shows ARRIVAL RATE
over time): this one shows REQUEST SHAPE over time.

Usage:
    python thesis_plotting/scripts/thesis_shape_timeline.py \\
        --csv cluster_outputs/azure/AzureLLMInferenceTrace_conv.csv \\
        --tag azure
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO))

from profile_trace_rps import load_rows  # noqa: E402

from thesis_plotting.style import (  # noqa: E402
    configure_plotting, save_fig, set_spines, bold_legend,
    COLOR_MODEL, WIDE_FIGSIZE,
    FONT_ANNOTATE, FONT_LABEL, FONT_TITLE,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, required=True,
                    help="Trace CSV path.")
    ap.add_argument("--tag", type=str, required=True,
                    help="Short tag for the output filename "
                         "(e.g. 'azure', 'vla_poisson').")
    ap.add_argument("--max-points", type=int, default=20000,
                    help="If the trace has more than N rows, randomly "
                         "subsample for plotting speed. Default 20k.")
    ap.add_argument("--out-dir", type=Path,
                    default=REPO / "thesis_plotting" / "figures")
    args = ap.parse_args()

    if not args.csv.is_file():
        sys.exit(f"[error] CSV not found: {args.csv}")

    configure_plotting()

    rows = list(load_rows(args.csv))
    if not rows:
        sys.exit(f"[error] empty CSV: {args.csv}")

    t0 = rows[0][0]
    times_s = [(r[0] - t0).total_seconds() for r in rows]
    lins    = [r[1] for r in rows]
    louts   = [r[2] for r in rows]

    if len(rows) > args.max_points:
        import random
        idxs = random.Random(0).sample(range(len(rows)), args.max_points)
        idxs.sort()
        times_s = [times_s[i] for i in idxs]
        lins    = [lins[i]    for i in idxs]
        louts   = [louts[i]   for i in idxs]
        print(f"[info] subsampled {len(rows)} → {len(times_s)} points")

    span_s = times_s[-1] if times_s else 0
    use_minutes = span_s > 3600
    x_unit = "min" if use_minutes else "s"
    x_axis = [t / 60.0 for t in times_s] if use_minutes else times_s

    # Local font overrides — defaults render ~3 pt after LaTeX scales the
    # figure to \linewidth. Bump everything so the rendered text is readable.
    FL  = FONT_LABEL    + 7   # axis label
    FTI = FONT_TITLE    + 7   # suptitle
    FAN = FONT_ANNOTATE + 8   # legend / subtitle / tick labels

    fig, (ax_lin, ax_lout) = plt.subplots(
        2, 1, figsize=(WIDE_FIGSIZE[0], 6.0),
        sharex=True, gridspec_kw={"hspace": 0.08},
    )

    ax_lin.scatter(x_axis, lins,
                   s=4.0, alpha=0.45,
                   color=COLOR_MODEL["pi0"],
                   edgecolors="none",
                   label=f"ContextTokens (n={len(lins):,})")
    ax_lin.set_ylabel("ContextTokens (Lin)",
                      fontsize=FL, fontweight="bold")
    ax_lin.tick_params(axis="both", which="major", labelsize=FAN)
    ax_lin.grid(axis="y", alpha=0.5)
    ax_lin.grid(axis="x", visible=False)
    set_spines(ax_lin)
    leg = ax_lin.legend(loc="upper right", fontsize=FAN)
    bold_legend(leg)

    ax_lout.scatter(x_axis, louts,
                    s=4.0, alpha=0.45,
                    color=COLOR_MODEL["openvla"],
                    edgecolors="none",
                    label=f"GeneratedTokens (n={len(louts):,})")
    ax_lout.set_ylabel("GeneratedTokens (Lout)",
                       fontsize=FL, fontweight="bold")
    ax_lout.set_xlabel(f"Arrival time ({x_unit})",
                       fontsize=FL, fontweight="bold")
    ax_lout.tick_params(axis="both", which="major", labelsize=FAN)
    ax_lout.grid(axis="y", alpha=0.5)
    ax_lout.grid(axis="x", visible=False)
    set_spines(ax_lout)
    leg = ax_lout.legend(loc="upper right", fontsize=FAN)
    bold_legend(leg)

    # Headline stats line.
    lin_mean  = sum(lins) / len(lins)
    lin_max   = max(lins)
    lout_mean = sum(louts) / len(louts)
    lout_max  = max(louts)
    subtitle = (f"n={len(rows):,}   span={span_s:,.0f} s   "
                f"Lin mean/max = {lin_mean:.0f}/{lin_max:,}   "
                f"Lout mean/max = {lout_mean:.0f}/{lout_max:,}")
    fig.suptitle(f"Per-request shape over time: {args.tag}\n{subtitle}",
                 fontsize=FTI, fontweight="bold", y=0.995)

    ax_lin.set_xlim(0, x_axis[-1] if x_axis else 1)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    save_fig(fig, f"fig_shape_timeline_{args.tag}", args.out_dir)
    plt.close(fig)

    print(f"[info] {args.tag}: Lin mean/max = {lin_mean:.0f}/{lin_max}, "
          f"Lout mean/max = {lout_mean:.0f}/{lout_max}")


if __name__ == "__main__":
    main()
