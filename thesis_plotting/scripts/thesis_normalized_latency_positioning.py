#!/usr/bin/env python3
"""Normalized-latency positioning plot for the related-work section.

Uses the vLLM convention (Kwon et al., SOSP 2023): normalized latency
(per-token end-to-end latency, ms/token) against achieved throughput
(req/s). Plots our simulator's Pi0 hybrid full knee curve plus the
single Pi0 gpu-only safe-op point, and overlays loosely-labeled
"reference regions" from published serving papers.

IMPORTANT METHODOLOGICAL CAVEAT: every published paper cited here uses
a different model, hardware, and workload. The literature markers are
indicative regions, NOT head-to-head comparisons. The figure is
intended to position our PIM-serving simulator within the broader
serving-systems metric landscape, not to claim a benchmark win.

Output: thesis_plotting/figures/fig_normalized_latency_positioning.{pdf,png}
"""

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO))

from plot_sweep_compare import parse_summary  # noqa: E402

from thesis_plotting.style import (  # noqa: E402
    configure_plotting, save_fig, set_spines, bold_legend,
    FONT_LABEL, FONT_TITLE, FONT_TICK, FONT_ANNOTATE,
)

KNEE_SWEEP = REPO / "cluster_outputs/online_serving_runs/rps_knee_pi0_wide_fine_n5k_a5000_20260605_212614"
MAIN_COMPARISON = REPO / "cluster_outputs/online_serving_runs/main_comparison_a5000_20260605_235416"


def load_knee_curve(sweep_dir):
    rows = []
    for cell in sorted(sweep_dir.iterdir()):
        if not cell.is_dir():
            continue
        sumfile = cell / "policy_compare_summary.txt"
        req_csv = cell / "max_util_full" / "requests_out.csv"
        if not sumfile.is_file() or not req_csv.is_file():
            continue
        m = parse_summary(sumfile)
        achieved = m.get("throughput")
        if achieved is None:
            continue
        n = 0
        sum_e2e = 0.0
        sum_lout = 0.0
        with open(req_csv) as fh:
            for row in csv.DictReader(fh):
                try:
                    sum_e2e += float(row["e2e_ms"])
                    sum_lout += float(row["generated_tokens"])
                    n += 1
                except (KeyError, ValueError):
                    continue
        if n == 0 or sum_lout <= 0:
            continue
        mean_e2e_ms = sum_e2e / n
        mean_lout = sum_lout / n
        norm_lat = mean_e2e_ms / mean_lout
        rows.append((achieved, norm_lat, mean_lout))
    return sorted(rows, key=lambda r: r[0])


def load_point(cell_dir, policy="max_util_full"):
    sumfile = cell_dir / "policy_compare_summary.txt"
    req_csv = cell_dir / policy / "requests_out.csv"
    if not sumfile.is_file() or not req_csv.is_file():
        return None
    m = parse_summary(sumfile)
    achieved = m.get("throughput")
    n = 0
    sum_e2e = 0.0
    sum_lout = 0.0
    with open(req_csv) as fh:
        for row in csv.DictReader(fh):
            sum_e2e += float(row["e2e_ms"])
            sum_lout += float(row["generated_tokens"])
            n += 1
    mean_e2e_ms = sum_e2e / n
    mean_lout = sum_lout / n
    return achieved, mean_e2e_ms / mean_lout


def render(curve, hybrid_point, gpu_point, out_dir):
    configure_plotting()
    FL  = FONT_LABEL    + 8
    FTI = FONT_TITLE    + 8
    FTK = FONT_TICK     + 8
    FAN = FONT_ANNOTATE + 9

    fig, ax = plt.subplots(figsize=(12.0, 7.5))

    # ── Our work: Pi0 hybrid full knee curve ──────────────────────────────
    xs = [p[0] for p in curve]
    ys = [p[1] for p in curve]
    ax.plot(xs, ys, "-o", color="#406ea3", lw=2.5, ms=9,
            markeredgecolor="black", markeredgewidth=0.6,
            label=r"Ours: $\pi_0$ proxy, hybrid (PIM + GPU), A6000 cost model, "
                  r"$\mathtt{azure\_poisson\_wide}$",
            zorder=6)

    # Our safe operating point — star
    h_rps, h_norm = hybrid_point
    ax.scatter([h_rps], [h_norm], marker="*", s=520,
               color="#c44e52", edgecolor="black", linewidth=1.0,
               zorder=8, label=r"Our safe operating point  (1.91 RPS, $\pi_0$ hybrid)")
    ax.annotate(f"  safe op\n  {h_rps:.2f} RPS, {h_norm:.0f} ms/tok",
                xy=(h_rps, h_norm), xytext=(15, 10),
                textcoords="offset points",
                fontsize=FAN, fontweight="bold", color="#c44e52",
                bbox=dict(facecolor="white", edgecolor="#c44e52",
                          boxstyle="round,pad=0.3", linewidth=0.9, alpha=0.95))

    # Pi0 gpu-only single point at its own saturated cell
    g_rps, g_norm = gpu_point
    ax.scatter([g_rps], [g_norm], marker="X", s=320,
               color="#888", edgecolor="black", linewidth=0.8,
               zorder=8,
               label=r"Ours: $\pi_0$ gpu-only (saturated at its own knee, 0.93 RPS)")
    ax.annotate(f"  gpu-only saturated\n  {g_rps:.2f} RPS, {g_norm:.0f} ms/tok",
                xy=(g_rps, g_norm), xytext=(15, -25),
                textcoords="offset points",
                fontsize=FAN, fontweight="bold", color="#444",
                bbox=dict(facecolor="white", edgecolor="#888",
                          boxstyle="round,pad=0.3", linewidth=0.9, alpha=0.95))

    # ── Literature reference regions (indicative, NOT head-to-head) ───────
    # Each rectangle covers the rough operating range that paper reports
    # for its primary model/hardware/workload.

    # vLLM (Kwon et al., SOSP 2023): OPT-13B / Llama-13B, A100 40GB,
    # ShareGPT. Fig 8 shows normalized lat 20-50 ms/tok in the 4-8 RPS
    # band; saturates beyond ~10 RPS.
    rect_vllm = Rectangle((4.0, 20.0), 4.0, 30.0, facecolor="#d6e9c6",
                          edgecolor="#5cb85c", alpha=0.5, linewidth=1.5,
                          zorder=2)
    ax.add_patch(rect_vllm)
    ax.text(6.0, 36.0, "vLLM\n(OPT-13B / LLaMA-13B, A100, ShareGPT)",
            ha="center", va="center", fontsize=FAN - 1, fontweight="bold",
            color="#2d6e2d",
            bbox=dict(facecolor="white", edgecolor="#5cb85c",
                      boxstyle="round,pad=0.25", linewidth=0.7, alpha=0.95))

    # Splitwise (Patel et al., ISCA 2024): LLaMA-2 70B, A100/H100 cluster,
    # production trace. Disaggregated prefill+decode improves throughput
    # ~1.4× over co-located at fixed SLA. Operating range ~5-12 RPS,
    # ~50-150 ms/tok per their Fig 5/6.
    rect_split = Rectangle((5.0, 60.0), 7.0, 90.0, facecolor="#fcf0d0",
                           edgecolor="#e0a040", alpha=0.5, linewidth=1.5,
                           zorder=2)
    ax.add_patch(rect_split)
    ax.text(8.5, 110.0, "Splitwise\n(LLaMA-2 70B, A100/H100, prod. trace)",
            ha="center", va="center", fontsize=FAN - 1, fontweight="bold",
            color="#9a6f1d",
            bbox=dict(facecolor="white", edgecolor="#e0a040",
                      boxstyle="round,pad=0.25", linewidth=0.7, alpha=0.95))

    # ThrottLL'eM (Kakolyris et al., 2025): LLaMA-2 7B/13B, A100,
    # ShareGPT+Azure. Energy-aware throughput throttling. Operating
    # range ~10-40 RPS for 7B at 1 s SLO, ~20-40 ms/tok.
    rect_thr = Rectangle((10.0, 15.0), 30.0, 25.0, facecolor="#d8e6f8",
                         edgecolor="#406ea3", alpha=0.5, linewidth=1.5,
                         zorder=2)
    ax.add_patch(rect_thr)
    ax.text(25.0, 27.5, "ThrottLL'eM\n(LLaMA-2 7B/13B, A100, ShareGPT+Azure)",
            ha="center", va="center", fontsize=FAN - 1, fontweight="bold",
            color="#1f3f6f",
            bbox=dict(facecolor="white", edgecolor="#406ea3",
                      boxstyle="round,pad=0.25", linewidth=0.7, alpha=0.95))

    # ── Axes & cosmetics ─────────────────────────────────────────────────
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.4, 60.0)
    ax.set_ylim(5.0, 50000.0)
    ax.set_xlabel("Achieved throughput  (req/s, log scale)",
                  fontsize=FL, fontweight="bold")
    ax.set_ylabel(r"Mean normalized latency  ($T_{\rm E2E} / L_{\rm out}$,"
                  r" ms/token, log scale)",
                  fontsize=FL, fontweight="bold")
    ax.set_title("Positioning of this work against published serving baselines\n"
                 "(vLLM convention; literature regions are indicative, "
                 "different model/HW/workload)",
                 fontsize=FTI, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=FTK)
    ax.grid(True, which="major", alpha=0.45)
    ax.grid(True, which="minor", alpha=0.18)
    set_spines(ax)
    leg = ax.legend(loc="upper left", fontsize=FTK,
                    framealpha=0.95)
    bold_legend(leg)

    fig.tight_layout()
    save_fig(fig, "fig_normalized_latency_positioning", out_dir)
    plt.close(fig)


def main():
    print("[info] loading Pi0 hybrid knee curve ...")
    curve = load_knee_curve(KNEE_SWEEP)
    print(f"[info]  {len(curve)} cells loaded")
    for rps, nlat, lout in curve[:5] + [("...", "...", "...")] + curve[-3:]:
        if isinstance(rps, str):
            print(f"  {rps}")
        else:
            print(f"  rps={rps:.2f}  norm_lat={nlat:.1f} ms/tok  "
                  f"lout_mean={lout:.0f}")

    print("[info] loading main-comparison hybrid + gpu-only points ...")
    h = load_point(MAIN_COMPARISON / "A_hybrid_a5000")
    g = load_point(MAIN_COMPARISON / "B_gpu_only_a5000")
    print(f"[info]  hybrid:   {h[0]:.2f} RPS  {h[1]:.1f} ms/tok")
    print(f"[info]  gpu-only: {g[0]:.2f} RPS  {g[1]:.1f} ms/tok")

    render(curve, h, g, REPO / "thesis_plotting/figures")


if __name__ == "__main__":
    main()
