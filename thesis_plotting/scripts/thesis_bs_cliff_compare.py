#!/usr/bin/env python3
"""OpenVLA bs cliff: saturation (E6) vs knee (J).

E6 found a starvation cliff at bs=16 when running at deep saturation
(arrival_scale=4.24). J reruns the same matrix at the knee scale and
shows the cliff disappears. This figure overlays both regimes for
OpenVLA so the cliff-is-an-artifact finding reads in one glance.

Usage:
    python thesis_plotting/scripts/thesis_bs_cliff_compare.py
"""

import re
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO))

from plot_sweep_compare import parse_summary  # noqa: E402

from thesis_plotting.style import (  # noqa: E402
    configure_plotting, save_fig, set_spines, bold_legend,
    COLOR_HEAVY, FONT_LABEL, FONT_TITLE, FONT_TICK,
)


DEFAULTS = {
    "saturation":  REPO / "cluster_outputs/online_serving_runs/"
                           "e6_bs_sweep_azure_20260530_173802",
    "knee":        REPO / "cluster_outputs/online_serving_runs/"
                           "j_knee_bs_azure_poisson_20260530_174904",
}


def discover_openvla_bs(sweep_dir: Path) -> list:
    """Returns sorted [(bs, metrics_dict)] for OpenVLA cells."""
    out = []
    label_re = re.compile(r"^\d+_openvla_bs(?P<bs>\d+)$")
    for sub in sorted(sweep_dir.iterdir()):
        if not sub.is_dir():
            continue
        m = label_re.match(sub.name)
        if not m:
            continue
        sumfile = sub / "policy_compare_summary.txt"
        if not sumfile.is_file():
            for child in sub.iterdir():
                if child.is_dir():
                    cand = child / "policy_compare_summary.txt"
                    if cand.is_file():
                        sumfile = cand
                        break
        if not sumfile.is_file():
            continue
        out.append((int(m["bs"]), parse_summary(sumfile)))
    out.sort(key=lambda x: x[0])
    return out


def render(sat_pts, knee_pts, out_dir: Path, out_name: str):
    configure_plotting()
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4))

    bs_sat,  thru_sat  = zip(*[(p[0], p[1].get("throughput", 0)) for p in sat_pts])
    bs_knee, thru_knee = zip(*[(p[0], p[1].get("throughput", 0)) for p in knee_pts])
    mb_sat   = [p[1].get("decode_mean_bs", 0) for p in sat_pts]
    mb_knee  = [p[1].get("decode_mean_bs", 0) for p in knee_pts]

    SAT_COLOR  = COLOR_HEAVY[1]   # rust
    KNEE_COLOR = COLOR_HEAVY[0]   # deep teal

    # ─ Panel A: throughput vs bs config ─
    ax = axes[0]
    ax.plot(bs_sat, thru_sat, "o-", color=SAT_COLOR, lw=2.4, ms=10,
            label="Saturation (E6, scale=4.24)", zorder=3)
    ax.plot(bs_knee, thru_knee, "s-", color=KNEE_COLOR, lw=2.4, ms=10,
            label="Knee (J, scale=3.75, azure_poisson)", zorder=3)

    # Highlight the cliff: vertical drop at bs=16 in saturation.
    if len(bs_sat) >= 2 and thru_sat[-1] < thru_sat[-2] * 0.5:
        ax.annotate("saturation cliff\n(bs=16 starved)",
                    xy=(bs_sat[-1], thru_sat[-1]),
                    xytext=(bs_sat[-1] - 4.5, thru_sat[-1] + 0.45),
                    fontsize=FONT_LABEL, fontweight="bold",
                    color=SAT_COLOR,
                    arrowprops=dict(arrowstyle="->", color=SAT_COLOR,
                                    lw=1.5))

    ax.set_xscale("log", base=2)
    ax.set_xticks(bs_knee)
    ax.set_xticklabels([str(b) for b in bs_knee],
                       fontsize=FONT_TICK)
    ax.set_xlabel("max_decode_batch_size  (configured)",
                  fontsize=FONT_LABEL, fontweight="bold")
    ax.set_ylabel("Throughput  (req/s)",
                  fontsize=FONT_LABEL, fontweight="bold")
    ax.set_title("(a) Throughput",
                 fontsize=FONT_TITLE - 1, fontweight="bold")
    ax.set_ylim(0, 1.7)
    ax.grid(True, axis="y", alpha=0.3)
    bold_legend(ax.legend(loc="lower right", fontsize=FONT_LABEL,
                          frameon=True))
    set_spines(ax)

    # ─ Panel B: mean_bs (the cohort-fill signal) ─
    ax = axes[1]
    ax.plot(bs_sat, mb_sat, "o-", color=SAT_COLOR, lw=2.4, ms=10,
            label="Saturation", zorder=3)
    ax.plot(bs_knee, mb_knee, "s-", color=KNEE_COLOR, lw=2.4, ms=10,
            label="Knee", zorder=3)
    ax.plot([1, 16], [1, 16], "--", color="grey", lw=1.4, alpha=0.6,
            zorder=2, label="y = configured")

    if len(bs_sat) >= 2 and mb_sat[-1] < mb_sat[-2] * 0.5:
        ax.annotate("collapses to bs≈1\n(queue too shallow)",
                    xy=(bs_sat[-1], mb_sat[-1]),
                    xytext=(bs_sat[-1] - 4.5, mb_sat[-1] + 4),
                    fontsize=FONT_LABEL, fontweight="bold",
                    color=SAT_COLOR,
                    arrowprops=dict(arrowstyle="->", color=SAT_COLOR,
                                    lw=1.5))

    ax.set_xscale("log", base=2)
    ax.set_xticks(bs_knee)
    ax.set_xticklabels([str(b) for b in bs_knee],
                       fontsize=FONT_TICK)
    ax.set_xlabel("max_decode_batch_size  (configured)",
                  fontsize=FONT_LABEL, fontweight="bold")
    ax.set_ylabel("Mean actual batch size",
                  fontsize=FONT_LABEL, fontweight="bold")
    ax.set_title("(b) Cohort fill (mean_bs)",
                 fontsize=FONT_TITLE - 1, fontweight="bold")
    ax.set_ylim(0, 17)
    ax.grid(True, axis="y", alpha=0.3)
    bold_legend(ax.legend(loc="upper left", fontsize=FONT_LABEL,
                          frameon=True))
    set_spines(ax)

    fig.suptitle(
        "OpenVLA — bs=16 starvation cliff is a saturation artifact, "
        "absent at the knee",
        fontsize=FONT_TITLE, fontweight="bold", y=1.00)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    save_fig(fig, out_name, out_dir)
    plt.close(fig)


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--saturation-dir", type=Path,
                    default=DEFAULTS["saturation"])
    ap.add_argument("--knee-dir", type=Path,
                    default=DEFAULTS["knee"])
    ap.add_argument("--out-dir", type=Path,
                    default=REPO / "thesis_plotting/figures")
    ap.add_argument("--out-name", default="fig_bs_cliff_sat_vs_knee")
    args = ap.parse_args()

    sat_pts = discover_openvla_bs(args.saturation_dir)
    knee_pts = discover_openvla_bs(args.knee_dir)
    print(f"[sat ] OpenVLA bs={[p[0] for p in sat_pts]}  "
          f"thru={[round(p[1].get('throughput',0), 2) for p in sat_pts]}  "
          f"mean_bs={[round(p[1].get('decode_mean_bs',0), 2) for p in sat_pts]}")
    print(f"[knee] OpenVLA bs={[p[0] for p in knee_pts]}  "
          f"thru={[round(p[1].get('throughput',0), 2) for p in knee_pts]}  "
          f"mean_bs={[round(p[1].get('decode_mean_bs',0), 2) for p in knee_pts]}")

    render(sat_pts, knee_pts, args.out_dir, args.out_name)


if __name__ == "__main__":
    main()
