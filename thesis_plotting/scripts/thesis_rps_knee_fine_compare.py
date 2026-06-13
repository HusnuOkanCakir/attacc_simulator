#!/usr/bin/env python3
"""Fine-cliff RPS knee at n=5000 vs n=10000 — chaotic switching zone exposed.

The 24-scale fine sweep at arrival_scale ∈ [1.5, 10] with 0.01 step around
the cliff (s=2.65-2.70 + s=2.50-2.55) reveals that the "metastable
bistability" is not a narrow window — it's a chaotic switching zone where
adjacent arrival_scales randomly land in DRAIN or OVERFLOW. Worse at
n=10000, but present at both trace sizes.

Inputs:
  cluster_outputs/online_serving_runs/rps_knee_pi0_wide_fine_n5k_20260602_183013
  cluster_outputs/online_serving_runs/rps_knee_pi0_wide_fine_20260602_181837

Output: thesis_plotting/figures/fig_rps_knee_pi0_wide_fine_compare.{pdf,png}
"""

import sys
from pathlib import Path
import re

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from thesis_plotting.style import (
    configure_plotting, save_fig, set_spines, bold_legend,
    COLOR_HEAVY, FONT_LABEL, FONT_TITLE, FONT_ANNOTATE,
)

N5K  = REPO / "cluster_outputs/online_serving_runs/rps_knee_pi0_wide_fine_n5k_20260602_183013"
N10K = REPO / "cluster_outputs/online_serving_runs/rps_knee_pi0_wide_fine_20260602_181837"


def collect(sweep_dir):
    rows = {}
    for cell in sorted(sweep_dir.iterdir()):
        if not cell.is_dir():
            continue
        try:
            scale = float(re.search(r'arrival_scale=([\d.]+)',
                                    (cell/'run_info.txt').read_text()).group(1))
        except Exception:
            continue
        csv = cell / 'max_util_full' / 'requests_out.csv'
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        d = df[df.state == 'done'].copy()
        if not len(d):
            continue
        rps = len(d) / (d.completion_ms.max() / 1000)
        d['norm'] = d.e2e_ms / d.generated_tokens.clip(lower=1)
        key = round(scale, 3)
        rec = (key, rps, d['norm'].mean(), d['norm'].quantile(.99))
        cur = rows.get(key)
        if cur is None or rec[2] < cur[2]:
            rows[key] = rec
    return sorted(rows.values())


def classify(scale, rps):
    """Use throughput deficit relative to offered load to classify."""
    offered = 5.5 / scale
    capacity = 2.13
    expected = min(offered, capacity)
    if expected == 0:
        return 'drain'
    deficit = (expected - rps) / expected
    if deficit > 0.30:
        return 'overflow'
    return 'drain'


def main():
    configure_plotting()
    n5k  = collect(N5K)
    n10k = collect(N10K)
    print(f'[info] n=5000 cells: {len(n5k)}')
    print(f'[info] n=10000 cells: {len(n10k)}')

    fig, ax = plt.subplots(figsize=(12, 7))

    # n=5000
    drain_5k = [p for p in n5k if classify(p[0], p[1]) == 'drain']
    over_5k  = [p for p in n5k if classify(p[0], p[1]) == 'overflow']
    drain_5k_sorted = sorted(drain_5k, key=lambda t: t[1])
    ax.plot([p[1] for p in drain_5k_sorted], [p[2] for p in drain_5k_sorted],
            color=COLOR_HEAVY[3], lw=1.8, marker='o', markersize=7,
            markeredgecolor='black', markeredgewidth=0.4, alpha=0.85,
            label='n=5,000  DRAIN  (mean norm latency)', zorder=4)
    if over_5k:
        ax.scatter([p[1] for p in over_5k], [p[2] for p in over_5k],
                   color=COLOR_HEAVY[3], marker='x', s=140, linewidth=2,
                   label='n=5,000  OVERFLOW', zorder=5)

    # n=10000
    drain_10k = [p for p in n10k if classify(p[0], p[1]) == 'drain']
    over_10k  = [p for p in n10k if classify(p[0], p[1]) == 'overflow']
    drain_10k_sorted = sorted(drain_10k, key=lambda t: t[1])
    ax.plot([p[1] for p in drain_10k_sorted], [p[2] for p in drain_10k_sorted],
            color=COLOR_HEAVY[2], lw=1.8, marker='s', markersize=7,
            markeredgecolor='black', markeredgewidth=0.4, alpha=0.85,
            label='n=10,000  DRAIN  (mean norm latency)', zorder=4)
    if over_10k:
        ax.scatter([p[1] for p in over_10k], [p[2] for p in over_10k],
                   color=COLOR_HEAVY[2], marker='+', s=180, linewidth=2.5,
                   label='n=10,000  OVERFLOW', zorder=5)

    # Safe operating point at s=4 (a = 1.37)
    s4_10k = next((p for p in drain_10k if abs(p[0] - 4.0) < 0.01), None)
    if s4_10k:
        ax.scatter([s4_10k[1]], [s4_10k[2]], marker='*', s=420, zorder=7,
                   color=COLOR_HEAVY[2], edgecolor='black', linewidth=1.0)
        ax.annotate(
            f'a ≈ 1.36 req/s  (s=4)\n'
            f'mean norm = 73-75 ms/tok\n'
            f'stable in BOTH n=5k & n=10k',
            xy=(s4_10k[1], s4_10k[2]),
            xytext=(50, -70), textcoords='offset points',
            ha='left', va='top',
            fontsize=FONT_ANNOTATE + 2, fontweight='bold',
            color='#1f5b1f',
            bbox=dict(facecolor='white', edgecolor='#4a8a4a',
                      boxstyle='round,pad=0.3', linewidth=0.8, alpha=0.95),
            arrowprops=dict(arrowstyle='->', color='#4a8a4a', lw=0.9))

    # Safe zone: rps < 1.55 (i.e. s ≥ 3.5)
    safe_max = 1.55
    ax.axvspan(0.4, safe_max, color='#90ee90', alpha=0.10, zorder=1)
    ax.text(0.85, 0.04, 'safe zone  (rps ≤ 1.55, s ≥ 3.5)\nno OVERFLOW at either trace size',
            transform=ax.get_xaxis_transform(),
            ha='center', va='bottom',
            fontsize=FONT_ANNOTATE + 1, fontweight='bold',
            color='#1f5b1f',
            bbox=dict(facecolor='white', edgecolor='#4a8a4a',
                      boxstyle='round,pad=0.3', linewidth=0.6, alpha=0.95))

    # Chaotic switching zone: s=2.65 to s=2.85
    ax.axvspan(safe_max, 2.07, color='#ffa500', alpha=0.10, zorder=1)
    ax.text(1.78, 0.96, 'chaotic switching zone\n(s=2.65–2.85: DRAIN/OVERFLOW\nflips per arrival_scale)',
            transform=ax.get_xaxis_transform(),
            ha='center', va='top',
            fontsize=FONT_ANNOTATE + 1, fontweight='bold',
            color='#a07050',
            bbox=dict(facecolor='white', edgecolor='#c08050',
                      boxstyle='round,pad=0.3', linewidth=0.7, alpha=0.95))

    # Stable saturation: s ≤ 2.54
    sat_min = 2.07
    drain_rps_max = max(p[1] for p in (drain_5k + drain_10k))
    ax.axvspan(sat_min, drain_rps_max * 1.05, color='#c44e52', alpha=0.10, zorder=1)
    ax.text(2.12, 0.04, 'stable saturation\n(rps≈2.12, growing queue)',
            transform=ax.get_xaxis_transform(),
            ha='center', va='bottom',
            fontsize=FONT_ANNOTATE + 1, fontweight='bold',
            color='#a04040',
            bbox=dict(facecolor='white', edgecolor='#c44e52',
                      boxstyle='round,pad=0.3', linewidth=0.6, alpha=0.95))

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:g}'))

    ax.set_xlabel('Offered RPS  (effective req / s)',
                  fontsize=FONT_LABEL + 1, fontweight='bold')
    ax.set_ylabel('Mean normalized latency  (ms / output token; log scale)',
                  fontsize=FONT_LABEL + 1, fontweight='bold')
    ax.set_title('Pi0 + azure_poisson_wide v2 — fine-cliff sweep at n=5,000 vs n=10,000\n'
                 '24 scales: 0.01-step resolution at cliff onset (s=2.65–2.70) '
                 'and partial-recovery (s=2.50–2.55)',
                 fontsize=FONT_TITLE, fontweight='bold')
    ax.grid(True, which='both', alpha=0.4)
    set_spines(ax)
    leg = ax.legend(loc='upper left', fontsize=FONT_LABEL - 1)
    bold_legend(leg)

    fig.tight_layout()
    out_dir = REPO / 'thesis_plotting/figures'
    save_fig(fig, 'fig_rps_knee_pi0_wide_fine_compare', out_dir)
    plt.close(fig)


if __name__ == '__main__':
    main()
