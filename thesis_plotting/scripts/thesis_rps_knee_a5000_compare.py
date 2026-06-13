#!/usr/bin/env python3
"""Side-by-side knee comparison: baseline max_active=32 vs probe max_active=5000.

Reads both sweep dirs, plots offered RPS vs achieved RPS and offered RPS vs
normalized latency for the two configurations on the same axes. Highlights
how the non-monotonic switching region (s ∈ [2.65, 2.85]) collapses in the
baseline run but stays smooth at max_active=5000.

Output:
    thesis_plotting/figures/fig_max_active_chaos_comparison.{pdf,png}
"""

import sys
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from thesis_plotting.style import (  # noqa: E402
    configure_plotting, save_fig, set_spines, bold_legend,
    COLOR_HEAVY, FONT_LABEL, FONT_TITLE, FONT_ANNOTATE,
)

SWEEPS = [
    ("baseline (max_active=32)",
     REPO / "cluster_outputs/online_serving_runs/rps_knee_pi0_wide_fine_n5k_20260602_183013",
     COLOR_HEAVY[1], "#c44e52"),
    ("probe (max_active=5000)",
     REPO / "cluster_outputs/online_serving_runs/rps_knee_pi0_wide_fine_n5k_a5000_20260605_212614",
     COLOR_HEAVY[2], "#1f5b1f"),
]
OUT_NAME = "fig_max_active_chaos_comparison"
CEILING  = 2.13


def collect(sweep_dir):
    rows = []
    for cell in sorted(sweep_dir.iterdir()):
        if not cell.is_dir():
            continue
        run_info = cell / "run_info.txt"
        try:
            scale = float(re.search(r"arrival_scale=([\d.]+)",
                                    run_info.read_text()).group(1))
        except Exception:
            continue
        csv = cell / "max_util_full" / "requests_out.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        if df.empty:
            continue
        offered = len(df) / (df["arrival_ms"].max() / 1000.0)
        done = df[df["state"] == "done"]
        if done.empty:
            continue
        achieved = len(done) / (done["completion_ms"].max() / 1000.0)
        done = done.copy()
        done["norm"] = done["e2e_ms"] / done["generated_tokens"].clip(lower=1)
        rows.append((scale, offered, achieved,
                     done["norm"].mean(), done["norm"].quantile(0.99)))
    return sorted(rows, key=lambda t: t[1])  # sort by offered


def main():
    configure_plotting()
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14.0, 6.5),
                                      constrained_layout=True)

    max_offered = 0
    for label, sweep_dir, line_color, marker_color in SWEEPS:
        pts = collect(sweep_dir)
        if not pts:
            print(f"[warn] empty sweep: {sweep_dir.name}")
            continue
        offered  = [p[1] for p in pts]
        achieved = [p[2] for p in pts]
        norm_mean = [p[3] for p in pts]
        max_offered = max(max_offered, max(offered))

        # Left panel: offered vs achieved
        ax_l.plot(offered, achieved, "-o", color=line_color, lw=1.8,
                  markersize=6, markeredgecolor="black", markeredgewidth=0.4,
                  label=label, zorder=5)

        # Right panel: offered vs normalized latency (mean)
        ax_r.plot(offered, norm_mean, "-o", color=line_color, lw=1.8,
                  markersize=6, markeredgecolor="black", markeredgewidth=0.4,
                  label=label, zorder=5)

    max_offered *= 1.05

    # Left panel: identity + ceiling references
    ax_l.plot([0, max_offered], [0, max_offered],
              ls="--", color="#888888", lw=1.0, zorder=2,
              label="$y=x$  (achieved $=$ offered)")
    ax_l.axhline(CEILING, color="#c0392b", linestyle=":", lw=1.3, zorder=3)
    ax_l.text(0.985, CEILING, "  ceiling = 2.13 RPS",
              transform=ax_l.get_yaxis_transform(),
              ha="right", va="bottom",
              fontsize=FONT_ANNOTATE + 1, color="#c0392b", fontweight="bold")

    # Shaded chaos-zone band on both panels.
    for ax in (ax_l, ax_r):
        ax.axvspan(1.93, 2.08, color="#ffa500", alpha=0.18, zorder=1,
                   label="_nolegend_")
        ax.text(2.005, 0.95, "non-monotonic\nswitching region",
                transform=ax.get_xaxis_transform(),
                ha="center", va="top",
                fontsize=FONT_ANNOTATE, fontweight="bold", color="#a07050",
                bbox=dict(facecolor="white", edgecolor="#c08050",
                          boxstyle="round,pad=0.3", linewidth=0.7, alpha=0.95))

    ax_l.set_xlim(0, max_offered)
    ax_l.set_ylim(0, max(CEILING, 2.2) * 1.12)
    ax_l.set_xlabel("Offered RPS  (arrivals / arrival span)",
                    fontsize=FONT_LABEL + 1, fontweight="bold")
    ax_l.set_ylabel("Achieved RPS  (completions / completion span)",
                    fontsize=FONT_LABEL + 1, fontweight="bold")
    ax_l.set_title("(a) offered vs achieved RPS\n"
                   "(baseline collapses in orange band; probe stays smooth)",
                   fontsize=FONT_TITLE, fontweight="bold", loc="left")
    ax_l.grid(True, alpha=0.4)
    set_spines(ax_l)
    leg = ax_l.legend(loc="lower right", fontsize=FONT_LABEL - 2)
    bold_legend(leg)

    # Right panel: normalized latency
    ax_r.set_xscale("log")
    ax_r.set_yscale("log")
    ax_r.set_xlim(0.4, max_offered)
    ax_r.set_xlabel("Mean offered load (RPS)",
                    fontsize=FONT_LABEL + 1, fontweight="bold")
    ax_r.set_ylabel("Normalized latency  (ms / output token; log scale)",
                    fontsize=FONT_LABEL + 1, fontweight="bold")
    ax_r.set_title("(b) normalized-latency knee curve\n"
                   "(baseline V-dives in orange band; probe is monotonic)",
                   fontsize=FONT_TITLE, fontweight="bold", loc="left")
    ax_r.grid(True, which="both", alpha=0.4)
    set_spines(ax_r)
    leg = ax_r.legend(loc="upper left", fontsize=FONT_LABEL - 2)
    bold_legend(leg)

    fig.suptitle(
        "Pi0 + azure_poisson_wide v2 — chaos zone collapses at max_active=5000",
        fontsize=FONT_TITLE + 2, fontweight="bold",
    )

    save_fig(fig, OUT_NAME, REPO / "thesis_plotting/figures")
    plt.close(fig)

    print(f"[info] {len(SWEEPS)} sweeps overlaid; output → {OUT_NAME}.{{pdf,png}}")


if __name__ == "__main__":
    main()
