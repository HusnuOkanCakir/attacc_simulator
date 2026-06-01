#!/usr/bin/env python3
"""Pi0 hybrid advantage vs workload shape (the "tail-density ladder").

Plots TTFT p99 (hybrid vs gpu-only) and the hybrid speedup across
four Pi0 datasets, ordered by *tail density* — the fraction of
requests with Lin > 4 000. This turned out to be a cleaner
discriminator than Lin CV: azure_poisson_wide has high CV (0.81) but
sparse tail (1.7% > 4k) and shows NO hybrid advantage, while
azure_empirical (8.2% > 4k) shows 232×. The PIM advantage is a step
function of tail density, not a smooth function of variance.

Usage:
    python thesis_plotting/scripts/thesis_pi0_shape_axis.py
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO))

from plot_sweep_compare import parse_summary  # noqa: E402

from thesis_plotting.style import (  # noqa: E402
    configure_plotting, save_fig, set_spines, bold_legend,
    COLOR_ROUTE, COLOR_HEAVY, FONT_LABEL, FONT_TITLE, FONT_TICK,
)


# (dataset_slug, label, lin_cv, tail_pct_gt4k, sweep_dir_basename, scale)
# tail_pct_gt4k = % of requests with Lin > 4000 (measured per-trace).
PI0_DATASETS = [
    ("azure_poisson",      "azure_poisson\n(Poisson, narrow)",       0.03, 0.0,
     "g_knee_azure_poisson_20260530_174416",  3.75),
    ("azure_poisson_wide", "azure_poisson_wide\n(Poisson, mixture v2)", 0.91, 4.6,
     "g_knee_azure_poisson_wide_20260531_173946", 6.5),
    ("azure_empirical",    "azure_empirical\n(CDF heavy-tail synth)", 0.96, 8.2,
     "g_knee_azure_empirical_20260531_152009", 3.77),
    ("azure_full19k",      "azure_full19k\n(real Azure heavy-tail)",  0.96, 8.3,
     "g_knee_main_claim_20260530_164544",     4.24),
]


def discover_pi0(sweep_dir: Path) -> dict:
    """Returns {mode: metrics_dict} for pi0 cells."""
    out = {}
    label_re = re.compile(r"^\d+_pi0_(?P<mode>hybrid|gpuonly)$")
    if not sweep_dir.is_dir():
        return out
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
        if sumfile.is_file():
            out[m["mode"]] = parse_summary(sumfile)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path,
                    default=REPO / "thesis_plotting/figures")
    ap.add_argument("--out-name", default="fig_pi0_shape_axis_ladder")
    args = ap.parse_args()

    runs_root = REPO / "cluster_outputs/online_serving_runs"
    rows = []  # (slug, label, cv, tail_pct, hyb_ttft, gpu_ttft, hyb_e2e, gpu_e2e)
    for slug, label, cv, tail_pct, basename, _scale in PI0_DATASETS:
        sweep_dir = runs_root / basename
        cells = discover_pi0(sweep_dir)
        if not cells.get("hybrid") or not cells.get("gpuonly"):
            print(f"[skip] {slug}: sweep {basename} not found / missing cells")
            continue
        rows.append((slug, label, cv, tail_pct,
                     cells["hybrid"].get("ttft_p99"),
                     cells["gpuonly"].get("ttft_p99"),
                     cells["hybrid"].get("e2e_p99"),
                     cells["gpuonly"].get("e2e_p99")))

    if not rows:
        sys.exit("[error] no rows discovered")

    print(f"[info] {len(rows)} datasets:")
    for slug, _label, cv, tp, ht, gt, he, ge in rows:
        print(f"  {slug:<20}  CV={cv:.2f}  tail%>4k={tp:.1f}  "
              f"hyb_TTFT={ht}  gpu_TTFT={gt}  hyb_E2E={he}  gpu_E2E={ge}")

    configure_plotting()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    n = len(rows)
    xs = np.arange(n)
    labels = [r[1] for r in rows]
    HYB = COLOR_ROUTE["lpddr5_pim_bank"]
    GPU = COLOR_ROUTE["gpu_only"]

    # ── Panel A: TTFT p99 ──
    ax = axes[0]
    hyb_ttft = [r[4] for r in rows]
    gpu_ttft = [r[5] for r in rows]
    width = 0.36
    ax.bar(xs - width/2, gpu_ttft, width=width, color=GPU,
           label="GPU-only", edgecolor="black", linewidth=0.5, zorder=3)
    ax.bar(xs + width/2, hyb_ttft, width=width, color=HYB,
           label="Hybrid", edgecolor="black", linewidth=0.5, zorder=3)
    ax.set_yscale("log")
    for i, (gt, ht) in enumerate(zip(gpu_ttft, hyb_ttft)):
        for x, v in [(i - width/2, gt), (i + width/2, ht)]:
            ax.text(x, v * 1.15, f"{v:.0f}", ha="center", va="bottom",
                    fontsize=FONT_LABEL - 1, fontweight="bold")
        # speedup annotation
        if gt > 0 and ht > 0:
            ratio = gt / ht
            ax.text(i, max(gt, ht) * 4.5,
                    f"{ratio:.1f}×" if ratio >= 2 else f"{ratio:.2f}×",
                    ha="center", va="bottom",
                    fontsize=FONT_LABEL + 1, fontweight="bold",
                    color="#2ca02c" if ratio > 1 else "#c44e52")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=FONT_TICK - 1, fontweight="bold")
    ax.set_ylabel("TTFT p99  (ms, log)", fontsize=FONT_LABEL,
                  fontweight="bold")
    ax.set_title("(a) TTFT p99 — hybrid vs gpu-only across tail density",
                 fontsize=FONT_TITLE - 2, fontweight="bold")
    ax.set_ylim(top=max(max(gpu_ttft), max(hyb_ttft)) * 30)
    ax.grid(True, axis="y", alpha=0.3)
    bold_legend(ax.legend(loc="upper left", fontsize=FONT_LABEL))
    set_spines(ax)

    # ── Panel B: speedup ratio vs tail density ──
    ax = axes[1]
    tail_pcts = [r[3] for r in rows]
    ratios = [(r[5] / r[4]) if (r[5] and r[4]) else 1.0 for r in rows]
    ax.plot(tail_pcts, ratios, "o-", color=HYB, lw=2.5, ms=12,
            markeredgecolor="black", markeredgewidth=0.6, zorder=3)
    for tp, ratio, (slug, *_rest) in zip(tail_pcts, ratios, rows):
        ax.annotate(slug.replace("azure_", "az_").replace("_", " "),
                    xy=(tp, ratio),
                    xytext=(8, -4), textcoords="offset points",
                    fontsize=FONT_LABEL - 1, fontweight="bold",
                    color="#333")
    ax.axhline(1.0, color="#888", lw=1.2, ls="--", zorder=2)
    ax.set_yscale("log")
    ax.set_xlabel("Tail density  (% of requests with Lin > 4 000)",
                  fontsize=FONT_LABEL, fontweight="bold")
    ax.set_ylabel("Hybrid TTFT speedup  (gpu_only / hybrid, log)",
                  fontsize=FONT_LABEL, fontweight="bold")
    ax.set_title("(b) Hybrid advantage is set by long-context tail density",
                 fontsize=FONT_TITLE - 2, fontweight="bold")
    ax.set_ylim(0.5, max(ratios) * 4)
    ax.set_xlim(-0.5, 10)
    ax.grid(True, alpha=0.3)
    set_spines(ax)

    fig.suptitle(
        "Pi0 at the knee — hybrid's gain is a tail-density story",
        fontsize=FONT_TITLE, fontweight="bold", y=1.00)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    save_fig(fig, args.out_name, args.out_dir)
    plt.close(fig)


if __name__ == "__main__":
    main()
