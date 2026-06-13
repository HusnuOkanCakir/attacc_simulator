#!/usr/bin/env python3
"""Cleaned-up normalized-latency knee figure for the n=5k Pi0 cell."""

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
     'fig_rps_knee_pi0_wide_normalized_fine_n5k_clean'),
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
        if df.empty:
            continue
        offered = len(df) / (df.arrival_ms.max() / 1000.0)
        d = df[df.state == 'done'].copy()
        if not len(d):
            continue
        achieved = len(d) / (d.completion_ms.max() / 1000.0)
        d['norm'] = d.e2e_ms / d.generated_tokens.clip(lower=1)
        key = round(scale, 3)
        rec = (key, offered, achieved, d['norm'].mean(), d['norm'].quantile(.99))
        cur = rows.get(key)
        if cur is None or rec[3] < cur[3]:
            rows[key] = rec
    return sorted(rows.values())


def classify(offered, achieved):
    """Throughput-deficit classifier:
    expected = min(offered, capacity); deficit > 30% ⇒ overflow."""
    capacity = 2.13
    expected = min(offered, capacity)
    if expected == 0:
        return 'drain'
    deficit = (expected - achieved) / expected
    return 'overflow' if deficit > 0.30 else 'drain'


def render(pts, n_label, out_name):
    configure_plotting()
    fig, ax = plt.subplots(figsize=(11.5, 6.5))

    # Sort ALL cells by offered RPS so the line shows the zig-zag through
    # the non-monotonic switching region (no cells are removed).
    cells = sorted(pts, key=lambda t: t[1])
    xs       = [p[1] for p in cells]                  # offered RPS
    ys_mean  = [p[3] for p in cells]
    ys_p99   = [p[4] for p in cells]
    modes    = [classify(p[1], p[2]) for p in cells]  # offered, achieved

    # MEAN — single connected line through all cells.
    ax.plot(xs, ys_mean, color=COLOR_HEAVY[2], lw=2.0,
            label='MEAN normalized latency  (vLLM, headline)', zorder=5)
    # p99 — single connected dashed line through all cells.
    ax.plot(xs, ys_p99, color=COLOR_HEAVY[1], lw=1.8, ls='--',
            label='p99 normalized latency  (tail companion)', zorder=4)

    # Markers coloured by attractor (drain green / overflow red).
    drain_mask    = [m == 'drain' for m in modes]
    overflow_mask = [m == 'overflow' for m in modes]

    def sel(arr, mask):
        return [v for v, m in zip(arr, mask) if m]

    ax.scatter(sel(xs, drain_mask), sel(ys_mean, drain_mask),
               marker='o', s=55, color=COLOR_HEAVY[2],
               edgecolor='black', linewidth=0.5, zorder=6,
               label='drain attractor (mean)')
    ax.scatter(sel(xs, drain_mask), sel(ys_p99, drain_mask),
               marker='s', s=42, facecolor=COLOR_HEAVY[1],
               edgecolor='black', linewidth=0.4, zorder=5)
    if any(overflow_mask):
        ax.scatter(sel(xs, overflow_mask), sel(ys_mean, overflow_mask),
                   marker='o', s=85, color='#c44e52',
                   edgecolor='black', linewidth=0.6, zorder=8,
                   label='overflow attractor  (achieved $\\approx$ 0.8 RPS)')
        ax.scatter(sel(xs, overflow_mask), sel(ys_p99, overflow_mask),
                   marker='s', s=60, color='#a04040',
                   edgecolor='black', linewidth=0.4, zorder=7)

    # Safe operating point — find cell closest to scale=4.0 (the standard
    # safe-a anchor in the chapter).
    s4 = next((p for p in cells if abs(p[0] - 4.0) < 0.01), None)
    if s4:
        ax.scatter([s4[1]], [s4[3]], marker='*', s=420, zorder=9,
                   color=COLOR_HEAVY[2], edgecolor='black', linewidth=1.0)
        ax.annotate(
            f'mean offered load = {s4[1]:.2f} RPS\n'
            f'mean = {s4[3]:.0f} ms/tok    p99 = {s4[4]:.0f} ms/tok',
            xy=(s4[1], s4[3]),
            xytext=(-180, 30), textcoords='offset points',
            ha='left', va='bottom',
            fontsize=FONT_ANNOTATE + 2, fontweight='bold',
            color='#1f5b1f',
            bbox=dict(facecolor='white', edgecolor='#4a8a4a',
                      boxstyle='round,pad=0.3', linewidth=0.7, alpha=0.95),
            arrowprops=dict(arrowstyle='->', color='#4a8a4a', lw=0.9))

    # Safe zone: up to the stable operating point marked by the star.
    safe_boundary = s4[1] if s4 else 1.36
    ax.axvspan(0.4, safe_boundary, color='#90ee90', alpha=0.10, zorder=1)
    ax.text(0.85, 0.04, 'safe zone',
            transform=ax.get_xaxis_transform(),
            ha='center', va='bottom',
            fontsize=FONT_ANNOTATE, fontweight='bold',
            color='#1f5b1f',
            bbox=dict(facecolor='white', edgecolor='#4a8a4a',
                      boxstyle='round,pad=0.3', linewidth=0.6, alpha=0.95))

    # Non-monotonic switching zone (band only; no overflow markers)
    ax.axvspan(safe_boundary, 2.07, color='#ffa500', alpha=0.10, zorder=1)
    ax.text(1.78, 0.96, 'non-monotonic\nswitching region',
            transform=ax.get_xaxis_transform(),
            ha='center', va='top',
            fontsize=FONT_ANNOTATE, fontweight='bold',
            color='#a07050',
            bbox=dict(facecolor='white', edgecolor='#c08050',
                      boxstyle='round,pad=0.3', linewidth=0.7, alpha=0.95))

    # Stable saturation zone
    offered_max = max(xs) * 1.05
    ax.axvspan(2.07, offered_max, color='#c44e52', alpha=0.10, zorder=1)
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

    ax.set_xlabel('Mean offered load (RPS)',
                  fontsize=FONT_LABEL + 1, fontweight='bold')
    ax.set_ylabel('Normalized latency  (ms / output token; log scale)',
                  fontsize=FONT_LABEL + 1, fontweight='bold')
    ax.set_title(f'Pi0 + azure_poisson_wide v2: normalized E2E latency  ({n_label}, fine cliff)\n'
                 'Solid = MEAN (vLLM headline) • Dashed = p99 (tail companion)',
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
