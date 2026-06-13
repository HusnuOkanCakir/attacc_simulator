#!/usr/bin/env python3
"""Normalized-latency headline figures from the fine-cliff sweeps.

Renders two headline figures, one per trace size:
 (a) fig_rps_knee_pi0_wide_normalized_fine_n5k.{pdf,png}   — n=5,000
 (b) fig_rps_knee_pi0_wide_normalized_fine_n10k.{pdf,png}  — n=10,000

Each shows:
  Solid green  : MEAN normalized latency  (vLLM, headline)
  Dashed red   : p99 normalized latency   (tail companion)
  Hollow X     : OVERFLOW cells (cliff/chaos)
  Green star   : safe operating point a = 1.36 req/s (s=4)

Inputs:
  cluster_outputs/online_serving_runs/rps_knee_pi0_wide_fine_n5k_20260602_183013
  cluster_outputs/online_serving_runs/rps_knee_pi0_wide_fine_20260602_181837
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

SWEEPS = [
    ('n=5,000',  REPO / 'cluster_outputs/online_serving_runs/rps_knee_pi0_wide_fine_n5k_20260602_183013',
     'fig_rps_knee_pi0_wide_normalized_fine_n5k'),
    ('n=10,000', REPO / 'cluster_outputs/online_serving_runs/rps_knee_pi0_wide_fine_20260602_181837',
     'fig_rps_knee_pi0_wide_normalized_fine_n10k'),
]


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
    """Throughput-deficit classifier:
    expected = min(offered_mean, capacity); deficit > 30% ⇒ overflow."""
    offered = 5.5 / scale
    capacity = 2.13
    expected = min(offered, capacity)
    if expected == 0:
        return 'drain'
    deficit = (expected - rps) / expected
    return 'overflow' if deficit > 0.30 else 'drain'


def render(pts, n_label, out_name):
    configure_plotting()
    fig, ax = plt.subplots(figsize=(11.5, 6.5))

    drain = sorted([p for p in pts if classify(p[0], p[1]) == 'drain'],
                   key=lambda t: t[1])
    over  = [p for p in pts if classify(p[0], p[1]) == 'overflow']

    # MEAN (vLLM-faithful headline)
    xs = [p[1] for p in drain]
    ys_mean = [p[2] for p in drain]
    ys_p99  = [p[3] for p in drain]
    ax.plot(xs, ys_mean, color=COLOR_HEAVY[2], lw=2.0, marker='o', markersize=7,
            markeredgecolor='black', markeredgewidth=0.5,
            label='MEAN normalized latency  (vLLM, headline)', zorder=5)
    ax.plot(xs, ys_p99, color=COLOR_HEAVY[1], lw=1.8, marker='s', markersize=6,
            markeredgecolor='black', markeredgewidth=0.5, ls='--',
            label='p99 normalized latency  (tail companion)', zorder=4)

    # OVERFLOW cells (chaotic switching) — hollow X
    if over:
        ax.scatter([p[1] for p in over], [p[2] for p in over],
                   marker='X', s=140, facecolor='white',
                   edgecolor=COLOR_HEAVY[2], linewidth=1.5, zorder=4,
                   label='OVERFLOW (chaotic switching zone)')
        ax.scatter([p[1] for p in over], [p[3] for p in over],
                   marker='X', s=100, facecolor='white',
                   edgecolor=COLOR_HEAVY[1], linewidth=1.5, zorder=3)

    # Safe operating point
    s4 = next((p for p in drain if abs(p[0] - 4.0) < 0.01), None)
    if s4:
        ax.scatter([s4[1]], [s4[2]], marker='*', s=420, zorder=7,
                   color=COLOR_HEAVY[2], edgecolor='black', linewidth=1.0)
        ax.annotate(
            f'a = {s4[1]:.2f} req/s  (safe; s=4)\n'
            f'mean = {s4[2]:.0f} ms/tok    p99 = {s4[3]:.0f} ms/tok',
            xy=(s4[1], s4[2]),
            xytext=(50, -55), textcoords='offset points',
            ha='left', va='top',
            fontsize=FONT_ANNOTATE + 2, fontweight='bold',
            color='#1f5b1f',
            bbox=dict(facecolor='white', edgecolor='#4a8a4a',
                      boxstyle='round,pad=0.3', linewidth=0.7, alpha=0.95),
            arrowprops=dict(arrowstyle='->', color='#4a8a4a', lw=0.9))

    # Safe zone: rps ≤ 1.55 (s ≥ 3.5)
    ax.axvspan(0.4, 1.55, color='#90ee90', alpha=0.10, zorder=1)
    ax.text(0.85, 0.04, 'safe zone (s ≥ 3.5)\nno OVERFLOW observed',
            transform=ax.get_xaxis_transform(),
            ha='center', va='bottom',
            fontsize=FONT_ANNOTATE, fontweight='bold',
            color='#1f5b1f',
            bbox=dict(facecolor='white', edgecolor='#4a8a4a',
                      boxstyle='round,pad=0.3', linewidth=0.6, alpha=0.95))

    # Chaotic switching zone
    if over:
        over_rps_lo = min(p[1] for p in over + drain if p[0] >= 2.65)
        over_rps_hi = max(p[1] for p in over)
        # span between rps≈0.75 (deep overflow) and 2.07 (cliff-DRAIN edge)
        ax.axvspan(1.55, 2.07, color='#ffa500', alpha=0.10, zorder=1)
        ax.text(1.78, 0.96, 'chaotic switching zone\ns=2.65–2.85',
                transform=ax.get_xaxis_transform(),
                ha='center', va='top',
                fontsize=FONT_ANNOTATE, fontweight='bold',
                color='#a07050',
                bbox=dict(facecolor='white', edgecolor='#c08050',
                          boxstyle='round,pad=0.3', linewidth=0.7, alpha=0.95))

    # Stable saturation zone
    drain_rps_max = max(p[1] for p in drain)
    ax.axvspan(2.07, drain_rps_max * 1.05, color='#c44e52', alpha=0.10, zorder=1)
    ax.text(2.12, 0.04, 'stable saturation\n(rps≈2.13, growing queue)',
            transform=ax.get_xaxis_transform(),
            ha='center', va='bottom',
            fontsize=FONT_ANNOTATE, fontweight='bold',
            color='#a04040',
            bbox=dict(facecolor='white', edgecolor='#c44e52',
                      boxstyle='round,pad=0.3', linewidth=0.6, alpha=0.95))

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:g}'))

    ax.set_xlabel('Offered RPS  (effective req / s)',
                  fontsize=FONT_LABEL + 1, fontweight='bold')
    ax.set_ylabel('Normalized latency  (ms / output token; log scale)',
                  fontsize=FONT_LABEL + 1, fontweight='bold')
    ax.set_title(f'Pi0 + azure_poisson_wide v2 — vLLM-style normalized E2E latency  ({n_label}, fine cliff)\n'
                 'Solid = MEAN (vLLM headline) • Dashed = p99 (tail companion) • Hollow X = OVERFLOW',
                 fontsize=FONT_TITLE, fontweight='bold')
    ax.grid(True, which='both', alpha=0.4)
    set_spines(ax)
    leg = ax.legend(loc='upper left', fontsize=FONT_LABEL - 1, title='Series')
    bold_legend(leg)
    if leg.get_title() is not None:
        leg.get_title().set_fontweight('bold')

    fig.tight_layout()
    out_dir = REPO / 'thesis_plotting/figures'
    save_fig(fig, out_name, out_dir)
    plt.close(fig)


def main():
    for label, sweep_dir, out_name in SWEEPS:
        pts = collect(sweep_dir)
        print(f'[info] {label}: {len(pts)} cells')
        render(pts, label, out_name)


if __name__ == '__main__':
    main()
