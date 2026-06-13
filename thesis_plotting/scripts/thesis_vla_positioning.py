#!/usr/bin/env python3
"""VLA-serving positioning plot for the related-work section (Path 2).

Plots E2E p99 (s) vs achieved throughput (req/s) on log axes. Our
\pi_0 simulator on a VLA-shape trace sits in a fundamentally different
operating regime than the chat-serving literature: short outputs
(L_out ~ 7 tokens), small backbones, action-latency-bounded SLOs.

The point of the figure is to show that VLA serving is its own niche
with its own RPS and latency envelope, distinct from chat-style LLM
serving where vLLM / Splitwise / ThrottLL'eM operate.

The literature reference regions are INDICATIVE and derived from each
paper's headline plot (operating range, not exact single points). They
are not head-to-head comparisons — different models, hardware, and
workloads — and are labelled explicitly as such.

Sources:
  Pi0 + vla_poisson knee:
    cluster_outputs/online_serving_runs/rps_knee_pi0_20260529_224452/
    (max_active=32; pre-chapter-migration default. The ceiling shifts
    a few percent at max_active=5000 but the regime is unchanged.)

Output: thesis_plotting/figures/fig_vla_positioning.{pdf,png}
"""

import csv
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO))

from plot_sweep_compare import parse_summary  # noqa: E402

from thesis_plotting.style import (  # noqa: E402
    configure_plotting, save_fig, set_spines, bold_legend,
    FONT_LABEL, FONT_TITLE, FONT_TICK, FONT_ANNOTATE,
)

VLA_SWEEP = REPO / "cluster_outputs/online_serving_runs/rps_knee_pi0_20260529_224452"
CELL_RE = re.compile(r"^\d+_vla_poisson_s(?P<sx10>\d+)$")


def load_vla_curve(sweep_dir):
    rows = []
    for cell in sorted(sweep_dir.iterdir()):
        if not cell.is_dir():
            continue
        m = CELL_RE.match(cell.name)
        if not m:
            continue
        sumfile = cell / "policy_compare_summary.txt"
        if not sumfile.is_file():
            # Look one level down (multi-policy runs nest the summary)
            for child in cell.iterdir():
                cand = child / "policy_compare_summary.txt"
                if cand.is_file():
                    sumfile = cand
                    break
        if not sumfile.is_file():
            continue
        stats = parse_summary(sumfile)
        thru = stats.get("throughput")
        e2e_p99_ms = stats.get("e2e_p99")
        if thru is None or e2e_p99_ms is None:
            continue
        rows.append((thru, e2e_p99_ms / 1000.0))  # → seconds
    return sorted(rows, key=lambda r: r[0])


def render(curve, out_dir):
    configure_plotting()
    FL  = FONT_LABEL    + 8
    FTI = FONT_TITLE    + 6
    FTK = FONT_TICK     + 8
    FAN = FONT_ANNOTATE + 9

    fig, ax = plt.subplots(figsize=(12.5, 7.5))

    # ── Our VLA curve ───────────────────────────────────────────────────
    xs = [r[0] for r in curve]
    ys = [r[1] for r in curve]
    ax.plot(xs, ys, "-o", color="#406ea3", lw=2.6, ms=11,
            markeredgecolor="black", markeredgewidth=0.7,
            label=r"Ours: $\pi_0$ hybrid (PIM+GPU), VLA-shape trace "
                  r"(L_in$\approx$306, L_out$\approx$7)",
            zorder=6)

    # Mark sub-cliff usable operating region: scale-10 to scale-1.5
    # (RPS 3 to ~20 at sub-second E2E p99).
    usable_pts = [(r[0], r[1]) for r in curve if r[1] < 5.0]
    if usable_pts:
        ax.axvspan(usable_pts[0][0] / 1.05, usable_pts[-1][0] * 1.05,
                   ymin=0, ymax=0.6,
                   color="#406ea3", alpha=0.08, zorder=1,
                   label="VLA sub-cliff operating region")

    # ── Robotic-control latency budget reference band ───────────────────
    ax.axhspan(0.1, 0.5, color="#5cb85c", alpha=0.10, zorder=1)
    ax.text(0.6, 0.21,
            "robotic-control action budget\n(100--500 ms typical)",
            fontsize=FAN, fontweight="bold", color="#2d6e2d",
            va="center", ha="left",
            bbox=dict(facecolor="white", edgecolor="#5cb85c",
                      boxstyle="round,pad=0.3", linewidth=0.8, alpha=0.95))

    # ── Literature reference regions (indicative) ───────────────────────
    # vLLM (Kwon et al., SOSP 2023): LLaMA-13B / OPT-13B, A100, ShareGPT.
    # Fig 8 typical sweet-spot ~4-8 RPS at p99 E2E ~2-6 s.
    rect_vllm = Rectangle((4.0, 1.5), 4.0, 5.0, facecolor="#d6e9c6",
                          edgecolor="#5cb85c", alpha=0.55, linewidth=1.8,
                          zorder=2)
    ax.add_patch(rect_vllm)
    ax.text(5.7, 3.2, "vLLM\n(LLaMA-13B / OPT-13B,\nA100, ShareGPT chat)",
            ha="center", va="center", fontsize=FAN - 1, fontweight="bold",
            color="#2d6e2d",
            bbox=dict(facecolor="white", edgecolor="#5cb85c",
                      boxstyle="round,pad=0.25", linewidth=0.7, alpha=0.95))

    # Splitwise (Patel et al., ISCA 2024): LLaMA-2 70B disaggregated
    # prefill+decode on A100/H100 cluster. RPS 5-12 at E2E p99 ~5-30 s.
    rect_split = Rectangle((5.0, 8.0), 7.0, 22.0, facecolor="#fcf0d0",
                           edgecolor="#e0a040", alpha=0.55, linewidth=1.8,
                           zorder=2)
    ax.add_patch(rect_split)
    ax.text(8.2, 15.5, "Splitwise\n(LLaMA-2 70B, A100/H100,\nprod. trace, chat)",
            ha="center", va="center", fontsize=FAN - 1, fontweight="bold",
            color="#9a6f1d",
            bbox=dict(facecolor="white", edgecolor="#e0a040",
                      boxstyle="round,pad=0.25", linewidth=0.7, alpha=0.95))

    # ThrottLL'eM (Kakolyris et al., 2025): LLaMA-2 7B/13B, A100,
    # ShareGPT+Azure conv. Throughput-at-SLO controller. Operating
    # range ~10-40 RPS at E2E p99 around 1-3 s.
    rect_thr = Rectangle((10.0, 1.0), 30.0, 2.0, facecolor="#f7d8d8",
                         edgecolor="#c44e52", alpha=0.55, linewidth=1.8,
                         zorder=2)
    ax.add_patch(rect_thr)
    ax.text(20.0, 1.5, "ThrottLL'eM\n(LLaMA-2 7B/13B, A100,\nShareGPT+Azure chat)",
            ha="center", va="center", fontsize=FAN - 1, fontweight="bold",
            color="#8e2528",
            bbox=dict(facecolor="white", edgecolor="#c44e52",
                      boxstyle="round,pad=0.25", linewidth=0.7, alpha=0.95))

    # ── Sub-cliff highlight annotation on our curve ─────────────────────
    # Pick the s=1.5 cell (RPS ~20, E2E p99 ~0.94 s)
    for rps, e2e_s in curve:
        if abs(rps - 20.1) < 0.5:
            ax.annotate(f"  {rps:.1f} RPS / {e2e_s*1000:.0f} ms p99\n"
                        f"  (sub-cliff VLA operating point)",
                        xy=(rps, e2e_s), xytext=(45, -25),
                        textcoords="offset points",
                        fontsize=FAN, fontweight="bold", color="#1f3f6f",
                        bbox=dict(facecolor="white", edgecolor="#406ea3",
                                  boxstyle="round,pad=0.35", linewidth=0.9,
                                  alpha=0.95),
                        arrowprops=dict(arrowstyle="->", color="#406ea3", lw=1.3))
            break

    # ── Axes & cosmetics ────────────────────────────────────────────────
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.5, 60.0)
    ax.set_ylim(0.1, 250.0)
    ax.set_xlabel("Achieved throughput  (req/s, log scale)",
                  fontsize=FL, fontweight="bold")
    ax.set_ylabel("End-to-end p99 latency  (s, log scale)",
                  fontsize=FL, fontweight="bold")
    ax.set_title("Workload positioning: VLA serving vs chat-style LLM serving\n"
                 "(literature regions are indicative, different model/hardware/workload)",
                 fontsize=FTI, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=FTK)
    ax.grid(True, which="major", alpha=0.45)
    ax.grid(True, which="minor", alpha=0.18)
    set_spines(ax)
    leg = ax.legend(loc="upper left", fontsize=FTK,
                    framealpha=0.95)
    bold_legend(leg)

    fig.tight_layout()
    save_fig(fig, "fig_vla_positioning", out_dir)
    plt.close(fig)


def main():
    print(f"[info] loading {VLA_SWEEP}")
    curve = load_vla_curve(VLA_SWEEP)
    print(f"[info]  {len(curve)} VLA cells loaded")
    for rps, e2e_s in curve:
        print(f"   rps={rps:>6.2f}  e2e_p99={e2e_s:>8.2f} s")
    render(curve, REPO / "thesis_plotting/figures")


if __name__ == "__main__":
    main()
