#!/usr/bin/env python3
"""Two-panel before/after RPS-over-time figure for the chapter-6 main
comparison workload.

Methodology convention:
  - "1.09 RPS" is the *mean* offered load (matching vLLM/Splitwise/
    ThrottLLeM convention and the queueing-theory definition of λ).
  - The saturation ceiling is also a mean throughput
    (achieved completions / trace span).
  - The visualization therefore plots the mean as a horizontal
    reference line (an average, not a threshold), and additionally
    plots the saturation ceiling for context. Poisson arrivals
    naturally produce instantaneous bursts above the mean; that is
    not a sizing violation under mean-based queueing analysis.

Top panel:    original wide-Poisson trace (arrival_scale = 1.0):
              peak \(\approx 14\) RPS, mean \(\approx 5.45\) RPS.
Bottom panel: same trace stretched so the mean offered load is
              \(1.09\) RPS (arrival_scale = 5.0); this is the actual
              workload the main GPU-only-vs-hybrid comparison was run on.

Output: thesis_plotting/figures/fig_rps_scale_main_comparison_clean.{pdf,png}
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO))

from profile_trace_rps import load_rows, bucket_rps  # noqa: E402
from thesis_plotting.style import (  # noqa: E402
    configure_plotting, save_fig, set_spines, bold_legend,
    WIDE_FIGSIZE,
    FONT_ANNOTATE, FONT_LABEL, FONT_TITLE,
)

CSV          = REPO / "cluster_outputs/synthetic_v2/azure_poisson_wide_n5000_seed0.csv"
SCALE        = 2.85    # arrival_scale used by the main comparison
TAG          = "main_comparison_clean"

BAR_COLOR    = "#6593c4"   # one neutral colour — bars are not classified
MEAN_COLOR   = "#1f77b4"   # mean offered load for this panel
PEAK_COLOR   = "#c0392b"   # peak offered load for this panel

# Local font overrides — defaults (FONT_BASE=7) become ~3pt after LaTeX scales
# the figure to \linewidth. Bump everything so the rendered text is readable.
FL  = FONT_LABEL    + 7   # axis label
FTI = FONT_TITLE    + 7   # panel title / suptitle
FAN = FONT_ANNOTATE + 8   # in-panel annotations / line labels


def render_panel(ax, ts_s, rps_series, bucket_ms, x_unit, title, subtitle):
    bar_w_s = (ts_s[1] - ts_s[0]) if len(ts_s) > 1 else bucket_ms / 1000.0
    if x_unit == "min":
        ts_axis = [t / 60.0 for t in ts_s]
        bar_w_axis = bar_w_s / 60.0
    else:
        ts_axis = ts_s
        bar_w_axis = bar_w_s

    ax.bar(ts_axis, rps_series, width=bar_w_axis,
           color=BAR_COLOR, edgecolor="none", align="edge", zorder=3, alpha=0.8)

    peak = max(rps_series) if rps_series else 0.0
    mean = sum(rps_series) / max(len(rps_series), 1)

    # Mean offered load for THIS panel (not the global operating point).
    ax.axhline(mean, color=MEAN_COLOR, linestyle="--",
               linewidth=1.8, zorder=5)
    ax.text(0.012, mean, f" mean = {mean:.2f} RPS",
            transform=ax.get_yaxis_transform(),
            ha="left", va="bottom",
            fontsize=FAN, color=MEAN_COLOR, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", pad=1.0, alpha=0.85))

    # Peak offered load for THIS panel.
    ax.axhline(peak, color=PEAK_COLOR, linestyle=":",
               linewidth=1.8, zorder=5)
    ax.text(0.012, peak, f" peak = {peak:.1f} RPS",
            transform=ax.get_yaxis_transform(),
            ha="left", va="top",
            fontsize=FAN, color=PEAK_COLOR, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", pad=1.0, alpha=0.85))

    ax.set_ylabel(f"Instantaneous RPS\n({bucket_ms} ms bucket)",
                  fontsize=FL, fontweight="bold")
    ax.set_title(title, fontsize=FTI, fontweight="bold", loc="left")
    ax.text(0.99, 0.94,
            f"{subtitle}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=FAN, fontweight="bold", color="#333",
            bbox=dict(facecolor="white", edgecolor="#cccccc",
                      boxstyle="round,pad=0.3", linewidth=0.7, alpha=0.9))
    ax.set_xlim(0, ts_axis[-1] + bar_w_axis if ts_axis else 1)
    ax.set_ylim(0, peak * 1.22)
    ax.tick_params(axis="both", which="major", labelsize=FAN)
    ax.grid(axis="y", alpha=0.5)
    ax.grid(axis="x", visible=False)
    set_spines(ax)


def main():
    configure_plotting()

    rows = list(load_rows(CSV))
    if not rows:
        sys.exit(f"[error] empty CSV: {CSV}")
    timestamps = [r[0] for r in rows]
    t0 = timestamps[0]

    BUCKET_MS_ORIG = 1000
    buckets_orig = bucket_rps(timestamps, BUCKET_MS_ORIG)
    ts_orig  = [t for t, _ in buckets_orig]
    rps_orig = [c * 1000.0 / BUCKET_MS_ORIG for _, c in buckets_orig]
    span_orig_s = ts_orig[-1] if ts_orig else 0

    scaled_ts = [t0 + (ts - t0) * SCALE for ts in timestamps]
    BUCKET_MS_SCALED = int(round(BUCKET_MS_ORIG * SCALE))
    buckets_scaled = bucket_rps(scaled_ts, BUCKET_MS_SCALED)
    ts_scaled = [t for t, _ in buckets_scaled]
    rps_scaled = [c * 1000.0 / BUCKET_MS_SCALED for _, c in buckets_scaled]
    span_scaled_s = ts_scaled[-1] if ts_scaled else 0

    use_min_orig   = span_orig_s   > 3600
    use_min_scaled = span_scaled_s > 3600
    x_unit_orig    = "min" if use_min_orig   else "s"
    x_unit_scaled  = "min" if use_min_scaled else "s"

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(WIDE_FIGSIZE[0], 6.5),
        sharex=False, gridspec_kw={"hspace": 0.55},
    )

    render_panel(
        ax_top, ts_orig, rps_orig, BUCKET_MS_ORIG, x_unit_orig,
        title="(a) Original wide-Poisson trace  (arrival_scale = 1.0)",
        subtitle=f"span = {span_orig_s:,.0f} s",
    )
    ax_top.set_xlabel(f"Time ({x_unit_orig})",
                      fontsize=FL, fontweight="bold")

    render_panel(
        ax_bot, ts_scaled, rps_scaled, BUCKET_MS_SCALED, x_unit_scaled,
        title=f"(b) Main comparison input  (arrival_scale = {SCALE})",
        subtitle=f"span = {span_scaled_s:,.0f} s",
    )
    ax_bot.set_xlabel(f"Time ({x_unit_scaled})",
                      fontsize=FL, fontweight="bold")

    fig.suptitle(
        "Main comparison workload: offered RPS over time",
        fontsize=FTI + 2, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_dir = REPO / "thesis_plotting" / "figures"
    save_fig(fig, f"fig_rps_scale_{TAG}", out_dir)
    plt.close(fig)

    print(f"[info] original peak={max(rps_orig):.2f} RPS  "
          f"mean={sum(rps_orig)/len(rps_orig):.2f} RPS")
    print(f"[info] scaled   peak={max(rps_scaled):.2f} RPS  "
          f"mean={sum(rps_scaled)/len(rps_scaled):.3f} RPS")


if __name__ == "__main__":
    main()
