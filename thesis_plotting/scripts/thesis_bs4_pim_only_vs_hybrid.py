#!/usr/bin/env python3
"""Compare bs=4 hybrid (knee_vs_bs) vs bs=4 PIM-only (droop probe).

Same 7 arrival scales, same trace, same kv_pool=64 GiB. The two
curves answer the question: does the deep-saturation droop visible
in hybrid persist when we forcibly disable the min_finish route
migration (--pim-only + kv_oom=hold)?

Two-panel output:
    (a) Offered RPS vs Achieved RPS  — droop = curve dropping below y=x ceiling
    (b) Offered RPS vs raw E2E p99   — cliff visualisation

Output: thesis_plotting/figures/fig_bs4_pim_only_vs_hybrid{,_linear}.{pdf,png}
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

from plot_sweep_compare import parse_summary  # noqa: E402

from thesis_plotting.style import (  # noqa: E402
    configure_plotting, save_fig, set_spines, bold_legend,
    COLOR_HEAVY, FONT_LABEL, FONT_TITLE, FONT_TICK, FONT_ANNOTATE,
)

HYBRID_SWEEP = REPO / "cluster_outputs/online_serving_runs/knee_vs_bs_20260606_114716"
PIM_SWEEP    = REPO / "cluster_outputs/online_serving_runs/bs4_pim_only_droop_probe_20260608_161357"

CEILING = 2.13
SAFE_RPS = 1.91

HYBRID_COLOR = COLOR_HEAVY[1]   # warm
PIM_COLOR    = COLOR_HEAVY[2]   # green


def load_hybrid_bs4(sweep_dir: Path):
    """Read the seven bs=4 cells (15..21) from the hybrid bs sweep."""
    rows = []
    pattern = re.compile(r"^\d+_bs_4_s(?P<sx10>\d+)$")
    for cell in sorted(sweep_dir.iterdir()):
        if not cell.is_dir():
            continue
        m = pattern.match(cell.name)
        if not m:
            continue
        scale = int(m.group("sx10")) / 10.0
        rows.append(load_cell(cell, scale))
    return [r for r in rows if r is not None]


def load_pim(sweep_dir: Path):
    rows = []
    pattern = re.compile(r"^\d+_bs_4_pim_s(?P<sx10>\d+)$")
    for cell in sorted(sweep_dir.iterdir()):
        if not cell.is_dir():
            continue
        m = pattern.match(cell.name)
        if not m:
            continue
        # scale labels are like s010, s005, s003, s285, s025, s002, s015
        # which encode {10.0, 5.0, 3.0, 2.85, 2.5, 2.0, 1.5}
        raw = m.group("sx10")
        # heuristic: 3-digit codes where leading char is 2 are scale * 100
        # (e.g. "285" → 2.85, "025" → 2.5? no, 025 → 2.5/2.50)
        # Easier: read scale from run_info.txt
        run_info = (cell / "run_info.txt")
        try:
            scale = float(re.search(r"arrival_scale=([\d.]+)",
                                    run_info.read_text()).group(1))
        except Exception:
            scale = int(raw) / 10.0
        rows.append(load_cell(cell, scale))
    return [r for r in rows if r is not None]


def load_cell(cell: Path, scale: float):
    sumfile = cell / "policy_compare_summary.txt"
    req_csv = cell / "max_util_full" / "requests_out.csv"
    if not sumfile.is_file() or not req_csv.is_file():
        return None
    m = parse_summary(sumfile)
    achieved = m.get("throughput")
    e2e_p99 = m.get("e2e_p99")
    if achieved is None or e2e_p99 is None:
        return None
    # Parse "Routes (PIM/GPU)  N/M" from the summary text.
    pim_count = gpu_count = 0
    for line in sumfile.read_text().splitlines():
        if line.startswith("Routes (PIM/GPU)"):
            try:
                split = line.split()[-1]
                pim_count, gpu_count = (int(x) for x in split.split("/"))
            except Exception:
                pass
            break
    n = 0
    max_arrival = 0.0
    with open(req_csv) as fh:
        for row in csv.DictReader(fh):
            a = float(row["arrival_ms"])
            if a > max_arrival:
                max_arrival = a
            n += 1
    if max_arrival <= 0 or n == 0:
        return None
    offered = n / (max_arrival / 1000.0)
    return (scale, offered, achieved, e2e_p99, pim_count, gpu_count)


def render(hybrid, pim, out_dir: Path, out_name: str, linear: bool):
    configure_plotting()
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(18.5, 6.4))

    # Sort by offered.
    hybrid = sorted(hybrid, key=lambda r: r[1])
    pim    = sorted(pim,    key=lambda r: r[1])
    hx = [r[1] for r in hybrid]
    ha = [r[2] for r in hybrid]
    hp = [r[3] / 1000.0 for r in hybrid]
    h_pim_count = [r[4] for r in hybrid]
    h_gpu_count = [r[5] for r in hybrid]
    h_pim_share = [100.0 * pc / max(pc + gc, 1)
                   for pc, gc in zip(h_pim_count, h_gpu_count)]
    h_gpu_share = [100.0 - s for s in h_pim_share]
    px = [r[1] for r in pim]
    pa = [r[2] for r in pim]
    pp = [r[3] / 1000.0 for r in pim]
    p_pim_count = [r[4] for r in pim]
    p_gpu_count = [r[5] for r in pim]
    p_pim_share = [100.0 * pc / max(pc + gc, 1)
                   for pc, gc in zip(p_pim_count, p_gpu_count)]

    # ── Panel (a): Offered vs Achieved RPS — droop visualisation ──
    max_offered = max(max(hx), max(px)) * 1.05
    ax_a.plot([0, max_offered], [0, max_offered],
              ls="--", color="#999999", lw=1.0, zorder=1,
              label="$y=x$  (achieved = offered)")
    ax_a.axhline(CEILING, color="#c0392b", linestyle=":", lw=1.3,
                 alpha=0.7, zorder=2)
    ax_a.text(0.99, CEILING + 0.03, "ceiling = 2.13 RPS",
              transform=ax_a.get_yaxis_transform(),
              ha="right", va="bottom",
              color="#c0392b", fontsize=FONT_ANNOTATE, fontweight="bold")
    ax_a.axvline(SAFE_RPS, color="#1f77b4", linestyle="--", lw=1.1,
                 alpha=0.6, zorder=2)
    ax_a.text(SAFE_RPS, 0.02, "  safe op  1.91",
              transform=ax_a.get_xaxis_transform(),
              ha="left", va="bottom",
              color="#1f77b4", fontsize=FONT_ANNOTATE, fontweight="bold")
    ax_a.plot(hx, ha, "-o", color=HYBRID_COLOR, lw=2.4, ms=10,
              markeredgecolor="black", markeredgewidth=0.6,
              label="Hybrid  (PIM + GPU, min_finish)", zorder=5)
    ax_a.plot(px, pa, "-s", color=PIM_COLOR, lw=2.4, ms=10,
              markeredgecolor="black", markeredgewidth=0.6,
              label="PIM-only  (scheduler bypassed)", zorder=6)

    # Annotate each hybrid point with GPU share / count.
    for x, y, gc, gs in zip(hx, ha, h_gpu_count, h_gpu_share):
        if gs < 1:
            continue
        ax_a.annotate(f"GPU: {gs:.0f}%\n({gc})",
                      xy=(x, y), xytext=(8, -28), textcoords="offset points",
                      fontsize=FONT_ANNOTATE - 1, fontweight="bold",
                      color=HYBRID_COLOR,
                      bbox=dict(facecolor="white", edgecolor=HYBRID_COLOR,
                                boxstyle="round,pad=0.2", linewidth=0.7,
                                alpha=0.95))

    ax_a.set_xlim(0, max_offered)
    ax_a.set_ylim(0, max(CEILING, max(ha), max(pa)) * 1.18)
    ax_a.set_xlabel("Offered mean load  (RPS)",
                    fontsize=FONT_LABEL + 1, fontweight="bold")
    ax_a.set_ylabel("Achieved throughput  (RPS)",
                    fontsize=FONT_LABEL + 1, fontweight="bold")
    ax_a.set_title("(a) Offered vs Achieved RPS — droop visualisation",
                    fontsize=FONT_TITLE, fontweight="bold", loc="left")
    ax_a.tick_params(axis="both", which="major", labelsize=FONT_TICK + 1)
    ax_a.grid(True, alpha=0.4)
    set_spines(ax_a)
    leg = ax_a.legend(loc="lower right", fontsize=FONT_TICK)
    bold_legend(leg)

    # ── Panel (b): Offered vs E2E p99 — cliff visualisation ──
    ax_b.plot(hx, hp, "-o", color=HYBRID_COLOR, lw=2.4, ms=10,
              markeredgecolor="black", markeredgewidth=0.6,
              label="Hybrid  (PIM + GPU)", zorder=5)
    ax_b.plot(px, pp, "-s", color=PIM_COLOR, lw=2.4, ms=10,
              markeredgecolor="black", markeredgewidth=0.6,
              label="PIM-only  (scheduler bypassed)", zorder=6)

    ax_b.axvline(SAFE_RPS, color="#1f77b4", linestyle="--", lw=1.1,
                 alpha=0.6, zorder=2)
    ax_b.text(SAFE_RPS, 0.02, "  safe op  1.91",
              transform=ax_b.get_xaxis_transform(),
              ha="left", va="bottom",
              color="#1f77b4", fontsize=FONT_ANNOTATE, fontweight="bold")
    ax_b.axvline(CEILING, color="#c0392b", linestyle=":", lw=1.1,
                 alpha=0.6, zorder=2)
    ax_b.text(CEILING, 0.02, "  ceiling  2.13",
              transform=ax_b.get_xaxis_transform(),
              ha="left", va="bottom",
              color="#c0392b", fontsize=FONT_ANNOTATE, fontweight="bold")

    if not linear:
        ax_b.set_xscale("log")
        ax_b.set_yscale("log")
    ax_b.set_xlabel("Offered mean load  (RPS"
                     + ("" if linear else ", log scale") + ")",
                     fontsize=FONT_LABEL + 1, fontweight="bold")
    ax_b.set_ylabel("E2E p99  (s"
                     + ("" if linear else ", log scale") + ")",
                     fontsize=FONT_LABEL + 1, fontweight="bold")
    ax_b.set_title("(b) Offered RPS vs raw E2E p99 — cliff",
                    fontsize=FONT_TITLE, fontweight="bold", loc="left")
    ax_b.tick_params(axis="both", which="major", labelsize=FONT_TICK + 1)
    ax_b.grid(True, which="major", alpha=0.45)
    if not linear:
        ax_b.grid(True, which="minor", alpha=0.18)
    set_spines(ax_b)
    leg = ax_b.legend(loc="upper left", fontsize=FONT_TICK)
    bold_legend(leg)

    # ── Panel (c): PIM share % vs Offered RPS — route mix ──
    ax_c.plot(hx, h_pim_share, "-o", color=HYBRID_COLOR, lw=2.4, ms=10,
              markeredgecolor="black", markeredgewidth=0.6,
              label="Hybrid PIM share %", zorder=5)
    ax_c.plot(px, p_pim_share, "-s", color=PIM_COLOR, lw=2.4, ms=10,
              markeredgecolor="black", markeredgewidth=0.6,
              label="PIM-only PIM share %", zorder=6)

    # Annotate hybrid points with PIM/GPU split counts.
    for x, y, pc, gc in zip(hx, h_pim_share, h_pim_count, h_gpu_count):
        ax_c.annotate(f"{pc}/{gc}",
                      xy=(x, y), xytext=(8, -22), textcoords="offset points",
                      fontsize=FONT_ANNOTATE - 1, fontweight="bold",
                      color=HYBRID_COLOR,
                      bbox=dict(facecolor="white", edgecolor=HYBRID_COLOR,
                                boxstyle="round,pad=0.2", linewidth=0.7,
                                alpha=0.95))
    # All PIM-only cells are 5000/0; annotate once on the last point.
    if px and p_pim_count:
        ax_c.annotate("5000/0\n(all cells)",
                      xy=(px[-1], p_pim_share[-1]),
                      xytext=(-65, 12), textcoords="offset points",
                      fontsize=FONT_ANNOTATE - 1, fontweight="bold",
                      color=PIM_COLOR,
                      bbox=dict(facecolor="white", edgecolor=PIM_COLOR,
                                boxstyle="round,pad=0.2", linewidth=0.7,
                                alpha=0.95))

    ax_c.axvline(SAFE_RPS, color="#1f77b4", linestyle="--", lw=1.1,
                 alpha=0.6, zorder=2)
    ax_c.text(SAFE_RPS, 0.02, "  safe op  1.91",
              transform=ax_c.get_xaxis_transform(),
              ha="left", va="bottom",
              color="#1f77b4", fontsize=FONT_ANNOTATE, fontweight="bold")
    ax_c.axvline(CEILING, color="#c0392b", linestyle=":", lw=1.1,
                 alpha=0.6, zorder=2)
    ax_c.text(CEILING, 0.02, "  ceiling  2.13",
              transform=ax_c.get_xaxis_transform(),
              ha="left", va="bottom",
              color="#c0392b", fontsize=FONT_ANNOTATE, fontweight="bold")

    ax_c.set_xlabel("Offered mean load  (RPS)",
                    fontsize=FONT_LABEL + 1, fontweight="bold")
    ax_c.set_ylabel("Requests routed to PIM  (%)",
                    fontsize=FONT_LABEL + 1, fontweight="bold")
    ax_c.set_title("(c) Route mix vs offered RPS\n"
                   "(labels are PIM/GPU request counts)",
                    fontsize=FONT_TITLE, fontweight="bold", loc="left")
    ax_c.set_ylim(50, 105)
    ax_c.tick_params(axis="both", which="major", labelsize=FONT_TICK + 1)
    ax_c.grid(True, alpha=0.4)
    set_spines(ax_c)
    leg = ax_c.legend(loc="lower left", fontsize=FONT_TICK)
    bold_legend(leg)

    fig.suptitle("bs=4 droop probe — hybrid (scheduler routes to GPU under saturation) "
                 "vs PIM-only (scheduler bypassed)",
                 fontsize=FONT_TITLE + 1, fontweight="bold", y=1.01)
    fig.tight_layout()
    save_fig(fig, out_name, out_dir)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--linear", action="store_true")
    ap.add_argument("--out-name", default="fig_bs4_pim_only_vs_hybrid")
    args = ap.parse_args()

    hybrid = load_hybrid_bs4(HYBRID_SWEEP)
    pim    = load_pim(PIM_SWEEP)
    if not hybrid or not pim:
        sys.exit("[error] no cells loaded")

    print(f"[info] hybrid: {len(hybrid)} cells, pim_only: {len(pim)} cells")
    print(f"  {'scale':>6} {'offered':>8} {'h_PIM':>6} {'h_GPU':>6} "
          f"{'h_PIM%':>7} {'p_PIM':>6} {'p_GPU':>6}")
    by_off_h = {round(r[1], 2): r for r in hybrid}
    by_off_p = {round(r[1], 2): r for r in pim}
    keys = sorted(set(by_off_h) | set(by_off_p))
    for k in keys:
        h = by_off_h.get(k)
        p = by_off_p.get(k)
        if h and p:
            tot = h[4] + h[5]
            pct = 100.0 * h[4] / max(tot, 1)
            print(f"  {h[0]:>6.2f} {h[1]:>8.2f} {h[4]:>6d} {h[5]:>6d} "
                  f"{pct:>6.1f}% {p[4]:>6d} {p[5]:>6d}")

    name = args.out_name + ("_linear" if args.linear else "")
    render(hybrid, pim, REPO / "thesis_plotting/figures", name, args.linear)


if __name__ == "__main__":
    main()
