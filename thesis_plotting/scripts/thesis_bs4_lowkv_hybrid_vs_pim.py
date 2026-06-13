#!/usr/bin/env python3
"""bs=4 low-KV (0.25 GiB) hybrid vs PIM-only.

Companion to thesis_bs4_pim_only_vs_hybrid.py (which compared at
64 GiB pool). This variant reads from the single sweep dir
bs4_lowkv_hybrid_vs_pim_<TS>/ which contains both hybrid (`01_hybrid_*`)
and PIM-only (`08_pim_*`) cells at KV pool = 0.25 GiB.

Output: fig_bs4_lowkv_hybrid_vs_pim{,_linear}.{pdf,png}
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

SWEEP = REPO / "cluster_outputs/online_serving_runs/bs4_lowkv_hybrid_vs_pim_20260609_005822"

CEILING = 2.10
SAFE_RPS = 1.91

HYBRID_COLOR = COLOR_HEAVY[1]
PIM_COLOR    = COLOR_HEAVY[2]

CELL_RE = re.compile(r"^\d+_(?P<kind>hybrid|pim)_bs_4_s(?P<sx10>\d+)$")


def load_cells(sweep_dir: Path):
    hybrid = []
    pim = []
    for cell in sorted(sweep_dir.iterdir()):
        if not cell.is_dir():
            continue
        m = CELL_RE.match(cell.name)
        if not m:
            continue
        kind = m.group("kind")
        raw = m.group("sx10")
        run_info = cell / "run_info.txt"
        try:
            scale = float(re.search(r"arrival_scale=([\d.]+)",
                                    run_info.read_text()).group(1))
        except Exception:
            scale = int(raw) / 10.0
        rec = load_cell(cell, scale)
        if rec is None:
            continue
        (hybrid if kind == "hybrid" else pim).append(rec)
    return hybrid, pim


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
    pim_count = gpu_count = 0
    holds = 0
    for line in sumfile.read_text().splitlines():
        if line.startswith("Routes (PIM/GPU)"):
            try:
                split = line.split()[-1]
                pim_count, gpu_count = (int(x) for x in split.split("/"))
            except Exception:
                pass
        elif line.startswith("KV-OOM holds"):
            try:
                holds = int(line.split()[-1])
            except Exception:
                pass
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
    return (scale, offered, achieved, e2e_p99, pim_count, gpu_count, holds)


def render(hybrid, pim, out_dir: Path, out_name: str, linear: bool):
    configure_plotting()
    # Stacked vertically (3 rows x 1 col) for a tall, page-spanning layout
    # with much larger fonts than the default thesis style. Local scale-up
    # so that after LaTeX scales the figure to \linewidth the text remains
    # readable.
    FL = FONT_LABEL + 12      # 19
    FTI = FONT_TITLE + 12     # 20
    FTK = FONT_TICK + 10      # 16
    FAN = FONT_ANNOTATE + 8   # 13
    fig, (ax_a, ax_b, ax_c) = plt.subplots(3, 1, figsize=(11.0, 18.0))

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

    # ── Panel (a): Achieved vs offered ──
    max_offered = max(max(hx), max(px)) * 1.05
    ax_a.plot([0, max_offered], [0, max_offered],
              ls="--", color="#999999", lw=1.0, zorder=1,
              label="$y=x$  (achieved = offered)")
    ax_a.plot(hx, ha, "-o", color=HYBRID_COLOR, lw=2.4, ms=10,
              markeredgecolor="black", markeredgewidth=0.6,
              label="Hybrid  (PIM + GPU, min_finish)", zorder=5)
    ax_a.plot(px, pa, "-s", color=PIM_COLOR, lw=2.4, ms=10,
              markeredgecolor="black", markeredgewidth=0.6,
              label="PIM-only  (scheduler bypassed)", zorder=6)

    # Annotate only the light-load cells where GPU% > 5% is genuinely
    # informative; suppress the noisy ~1% labels at higher loads.
    for x, y, gc, gs in zip(hx, ha, h_gpu_count, h_gpu_share):
        if gs < 5:
            continue
        ax_a.annotate(f"GPU: {gs:.0f}%\n({gc})",
                      xy=(x, y), xytext=(10, -38), textcoords="offset points",
                      fontsize=FAN, fontweight="bold",
                      color=HYBRID_COLOR,
                      bbox=dict(facecolor="white", edgecolor=HYBRID_COLOR,
                                boxstyle="round,pad=0.25", linewidth=0.8,
                                alpha=0.95))

    ax_a.set_xlim(0, max_offered)
    ax_a.set_ylim(0, max(CEILING, max(ha), max(pa)) * 1.18)
    ax_a.set_xlabel("Offered mean load  (RPS)",
                    fontsize=FL, fontweight="bold")
    ax_a.set_ylabel("Achieved throughput  (RPS)",
                    fontsize=FL, fontweight="bold")
    ax_a.set_title("(a) Offered vs Achieved RPS",
                    fontsize=FTI, fontweight="bold", loc="left")
    ax_a.tick_params(axis="both", which="major", labelsize=FTK)
    ax_a.grid(True, alpha=0.4)
    set_spines(ax_a)
    leg = ax_a.legend(loc="lower right", fontsize=FTK)
    bold_legend(leg)

    # ── Panel (b): Offered vs E2E p99 ──
    ax_b.plot(hx, hp, "-o", color=HYBRID_COLOR, lw=2.4, ms=10,
              markeredgecolor="black", markeredgewidth=0.6,
              label="Hybrid  (PIM + GPU)", zorder=5)
    ax_b.plot(px, pp, "-s", color=PIM_COLOR, lw=2.4, ms=10,
              markeredgecolor="black", markeredgewidth=0.6,
              label="PIM-only  (scheduler bypassed)", zorder=6)

    if not linear:
        ax_b.set_xscale("log")
        ax_b.set_yscale("log")
    ax_b.set_xlabel("Offered mean load  (RPS"
                     + ("" if linear else ", log scale") + ")",
                     fontsize=FL, fontweight="bold")
    ax_b.set_ylabel("E2E p99  (s"
                     + ("" if linear else ", log scale") + ")",
                     fontsize=FL, fontweight="bold")
    ax_b.set_title("(b) Offered vs E2E p99",
                    fontsize=FTI, fontweight="bold", loc="left")
    ax_b.tick_params(axis="both", which="major", labelsize=FTK)
    ax_b.grid(True, which="major", alpha=0.45)
    if not linear:
        ax_b.grid(True, which="minor", alpha=0.18)
    set_spines(ax_b)
    leg = ax_b.legend(loc="upper left", fontsize=FTK)
    bold_legend(leg)

    # ── Panel (c): Route mix ──
    ax_c.plot(hx, h_pim_share, "-o", color=HYBRID_COLOR, lw=2.4, ms=10,
              markeredgecolor="black", markeredgewidth=0.6,
              label="Hybrid PIM share %", zorder=5)
    ax_c.plot(px, p_pim_share, "-s", color=PIM_COLOR, lw=2.4, ms=10,
              markeredgecolor="black", markeredgewidth=0.6,
              label="PIM-only PIM share %", zorder=6)

    # Annotate only the cells where the PIM/GPU split is meaningfully
    # informative (low-load region) plus the first cell in the saturated
    # cluster. Stagger up/down to avoid overlap.
    annotate_idx = []
    last_share = None
    for i, share in enumerate(h_pim_share):
        if last_share is None or abs(share - last_share) > 1.0:
            annotate_idx.append(i)
            last_share = share
    for k, i in enumerate(annotate_idx):
        offset_y = -34 if (k % 2 == 0) else 22
        ax_c.annotate(f"{h_pim_count[i]}/{h_gpu_count[i]}",
                      xy=(hx[i], h_pim_share[i]),
                      xytext=(10, offset_y), textcoords="offset points",
                      fontsize=FAN, fontweight="bold",
                      color=HYBRID_COLOR,
                      bbox=dict(facecolor="white", edgecolor=HYBRID_COLOR,
                                boxstyle="round,pad=0.25", linewidth=0.8,
                                alpha=0.95))
    if px and p_pim_count:
        ax_c.annotate("PIM-only: 5000/0\n(all cells)",
                      xy=(px[-2] if len(px) > 1 else px[-1],
                          p_pim_share[-2] if len(p_pim_share) > 1 else p_pim_share[-1]),
                      xytext=(0, -55), textcoords="offset points",
                      fontsize=FAN, fontweight="bold", ha="center",
                      color=PIM_COLOR,
                      bbox=dict(facecolor="white", edgecolor=PIM_COLOR,
                                boxstyle="round,pad=0.25", linewidth=0.8,
                                alpha=0.95))

    ax_c.set_xlabel("Offered mean load  (RPS)",
                    fontsize=FL, fontweight="bold")
    ax_c.set_ylabel("Requests routed to PIM  (%)",
                    fontsize=FL, fontweight="bold")
    ax_c.set_title("(c) Route mix vs offered RPS\n"
                   "(labels are PIM/GPU request counts)",
                    fontsize=FTI, fontweight="bold", loc="left")
    ax_c.set_ylim(0, 105)
    ax_c.tick_params(axis="both", which="major", labelsize=FTK)
    ax_c.grid(True, alpha=0.4)
    set_spines(ax_c)
    leg = ax_c.legend(loc="lower left", fontsize=FTK)
    bold_legend(leg)

    fig.suptitle("bs=4 low-KV (0.25 GiB) — hybrid (PIM + GPU fallback) "
                 "vs PIM-only (no fallback)",
                 fontsize=FTI + 2, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.965], h_pad=3.5)
    save_fig(fig, out_name, out_dir)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--linear", action="store_true")
    ap.add_argument("--out-name", default="fig_bs4_lowkv_hybrid_vs_pim")
    args = ap.parse_args()

    hybrid, pim = load_cells(SWEEP)
    if not hybrid or not pim:
        sys.exit(f"[error] no cells loaded from {SWEEP}")

    print(f"[info] hybrid: {len(hybrid)} cells, pim_only: {len(pim)} cells")
    print(f"  {'scale':>6} {'offered':>8} {'h_thru':>7} {'p_thru':>7} "
          f"{'h_PIM%':>7} {'h_p99(s)':>9} {'p_p99(s)':>9} {'p_holds':>10}")
    by_off_h = {round(r[1], 2): r for r in hybrid}
    by_off_p = {round(r[1], 2): r for r in pim}
    keys = sorted(set(by_off_h) | set(by_off_p))
    for k in keys:
        h = by_off_h.get(k)
        p = by_off_p.get(k)
        if h and p:
            tot = h[4] + h[5]
            pct = 100.0 * h[4] / max(tot, 1)
            print(f"  {h[0]:>6.2f} {h[1]:>8.2f} {h[2]:>7.2f} {p[2]:>7.2f} "
                  f"{pct:>6.1f}% {h[3]/1000:>9.1f} {p[3]/1000:>9.1f} {p[6]:>10d}")

    name = args.out_name + ("_linear" if args.linear else "")
    render(hybrid, pim, REPO / "thesis_plotting/figures", name, args.linear)


if __name__ == "__main__":
    main()
