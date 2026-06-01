#!/usr/bin/env python3
"""Admission-violation-budget X at the knee — null finding figure.

All 5 budgets {-1, 0, 1, 5, 20} produced bit-identical results on
Pi0 + azure_poisson @ scale=3.75 (knee). The thesis-friendly framing
of this null result: show throughput and TTFT p99 across X values
as flat lines, accompanied by a one-line annotation explaining why
the knob is inert at the knee (no prediction crosses the
violation threshold).

Usage:
    python thesis_plotting/scripts/thesis_k_admission_null.py
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
    configure_plotting, save_fig, set_spines,
    COLOR_HEAVY, FONT_LABEL, FONT_TITLE, FONT_TICK, FONT_ANNOTATE,
)


DEFAULT = REPO / "cluster_outputs/online_serving_runs/" \
                  "k_knee_admit_azure_poisson_20260530_175019"


def discover_x_cells(sweep_dir: Path) -> list:
    """Returns sorted [(X_value, metrics_dict)]."""
    out = []
    label_re = re.compile(r"^\d+_pi0_x(?P<label>m?\d+)$")
    for sub in sorted(sweep_dir.iterdir()):
        if not sub.is_dir():
            continue
        m = label_re.match(sub.name)
        if not m:
            continue
        raw = m["label"]
        x = -int(raw[1:]) if raw.startswith("m") else int(raw)
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
        out.append((x, parse_summary(sumfile)))
    out.sort(key=lambda r: r[0])
    return out


def render(cells: list, out_dir: Path, out_name: str):
    configure_plotting()
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))

    xs   = [c[0] for c in cells]
    thru = [c[1].get("throughput", 0) for c in cells]
    ttft = [c[1].get("ttft_p99", 0) for c in cells]
    e2e  = [c[1].get("e2e_p99", 0) / 1000.0 for c in cells]

    labels = ["off (−1)" if x == -1 else f"{x}" for x in xs]
    positions = np.arange(len(xs))

    BAR_COLOR = COLOR_HEAVY[0]

    # ── Panel A: throughput ──
    ax = axes[0]
    ax.bar(positions, thru, width=0.55, color=BAR_COLOR,
           edgecolor="black", linewidth=0.5, zorder=3)
    for i, v in enumerate(thru):
        ax.text(i, v + 0.04, f"{v:.2f}",
                ha="center", va="bottom",
                fontsize=FONT_LABEL, fontweight="bold")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=FONT_TICK)
    ax.set_xlabel("Admission-violation budget X",
                  fontsize=FONT_LABEL, fontweight="bold")
    ax.set_ylabel("Throughput  (req/s)",
                  fontsize=FONT_LABEL, fontweight="bold")
    ax.set_title("(a) Throughput",
                 fontsize=FONT_TITLE - 1, fontweight="bold")
    ax.set_ylim(0, max(thru) * 1.25)
    ax.grid(True, axis="y", alpha=0.3)
    set_spines(ax)

    # ── Panel B: TTFT p99 ──
    ax = axes[1]
    ax.bar(positions, ttft, width=0.55, color=BAR_COLOR,
           edgecolor="black", linewidth=0.5, zorder=3)
    for i, v in enumerate(ttft):
        ax.text(i, v + max(ttft) * 0.015, f"{v:.0f}",
                ha="center", va="bottom",
                fontsize=FONT_LABEL, fontweight="bold")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=FONT_TICK)
    ax.set_xlabel("Admission-violation budget X",
                  fontsize=FONT_LABEL, fontweight="bold")
    ax.set_ylabel("TTFT p99  (ms)",
                  fontsize=FONT_LABEL, fontweight="bold")
    ax.set_title("(b) TTFT p99",
                 fontsize=FONT_TITLE - 1, fontweight="bold")
    ax.set_ylim(0, max(ttft) * 1.4)
    ax.grid(True, axis="y", alpha=0.3)
    set_spines(ax)

    # Single null-finding annotation across both panels.
    fig.text(0.5, 0.01,
             "All 5 budgets produce bit-identical results — the knob "
             "doesn't fire at the knee\n"
             "(no admission decision crosses the predicted-violation threshold).",
             ha="center", va="bottom",
             fontsize=FONT_ANNOTATE + 1, fontweight="bold",
             color="#c44e52")

    fig.suptitle(
        "K — Admission-violation budget X at the knee  (Pi0, azure_poisson)",
        fontsize=FONT_TITLE, fontweight="bold", y=1.00)

    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.95))
    save_fig(fig, out_name, out_dir)
    plt.close(fig)


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep-dir", type=Path, default=DEFAULT)
    ap.add_argument("--out-dir", type=Path,
                    default=REPO / "thesis_plotting/figures")
    ap.add_argument("--out-name", default="fig_k_admission_null_knee")
    args = ap.parse_args()

    cells = discover_x_cells(args.sweep_dir)
    print(f"[info] {len(cells)} cells:")
    for x, m in cells:
        print(f"  X={x:>3}  thru={m.get('throughput')}  "
              f"ttft_p99={m.get('ttft_p99')}ms  "
              f"e2e_p99={m.get('e2e_p99')}ms")
    render(cells, args.out_dir, args.out_name)


if __name__ == "__main__":
    main()
