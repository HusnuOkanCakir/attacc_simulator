#!/usr/bin/env python3
"""Pi0 + azure_poisson_wide v2 — knee curve under vLLM-style normalized latency.

Metric: normalized latency  =  e2e_ms / output_length  (ms/token, per Kwon et al.
SOSP'23 §6 Key metrics).
Two curves:
 (a) MEAN normalized latency  — vLLM headline metric
 (b) p99  normalized latency  — tail metric (companion, our addition)

Both curves share one log-log panel. Knee marked at the chosen safe operating
point a = 1.36 req/s (scale=4). Cliff/metastability region shaded as
"danger zone".

Output: thesis_plotting/figures/fig_rps_knee_pi0_wide_normalized.{pdf,png}
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

DENSE = REPO / "cluster_outputs/online_serving_runs/rps_knee_pi0_20260601_175358"
META  = REPO / "cluster_outputs/online_serving_runs/rps_knee_pi0_20260601_183211"

OVERFLOW_SCALES = {2.60, 2.65}
PARTIAL_SCALES  = {2.55}


def collect():
    rows = {}
    for sweep_dir in [DENSE, META]:
        for cell in sorted(sweep_dir.iterdir()):
            if not cell.is_dir() or 'azure_poisson_wide' not in cell.name:
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
            key = round(scale, 2)
            cand = (key, rps, d['norm'].mean(), d['norm'].quantile(.99))
            cur = rows.get(key)
            if cur is None or cand[2] < cur[2]:
                rows[key] = cand
    return sorted(rows.values())


def classify(scale):
    if scale in OVERFLOW_SCALES: return 'overflow'
    if scale in PARTIAL_SCALES:  return 'partial'
    return 'drain'


def main():
    configure_plotting()
    pts = collect()
    drain = sorted([p for p in pts if classify(p[0]) == 'drain'], key=lambda t: t[1])
    over  = [p for p in pts if classify(p[0]) == 'overflow']
    part  = [p for p in pts if classify(p[0]) == 'partial']

    fig, ax = plt.subplots(figsize=(11.5, 6.5))

    # MEAN (vLLM-faithful, headline)
    xs = [p[1] for p in drain]
    ys_mean = [p[2] for p in drain]
    ax.plot(xs, ys_mean, color=COLOR_HEAVY[2], lw=2.0, marker='o', markersize=7,
            markeredgecolor='black', markeredgewidth=0.5,
            label='MEAN normalized latency  (vLLM, headline)',
            zorder=5)

    # p99 (companion, tail awareness)
    ys_p99 = [p[3] for p in drain]
    ax.plot(xs, ys_p99, color=COLOR_HEAVY[1], lw=1.8, marker='s', markersize=6,
            markeredgecolor='black', markeredgewidth=0.5, ls='--',
            label='p99 normalized latency  (our tail companion)',
            zorder=4)

    # Off-curve cells (OVERFLOW + partial recovery) are intentionally NOT
    # rendered — the danger-zone shading already conveys "do not operate here"
    # and removing them keeps the eye on the two clean DRAIN curves.

    # Safe operating point marker
    s4 = next((p for p in drain if abs(p[0] - 4.0) < 0.01), None)
    if s4:
        ax.scatter([s4[1]], [s4[2]],
                   marker='*', s=420, zorder=7,
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

    # Danger-zone shading past a
    drain_rps_max = max(p[1] for p in drain)
    if s4:
        ax.axvspan(s4[1], drain_rps_max * 1.05,
                   color='#c44e52', alpha=0.08, zorder=1)
        x_mid = (s4[1] * drain_rps_max) ** 0.5
        ax.text(x_mid, 0.96, 'danger zone  (cliff + metastable bistability)',
                transform=ax.get_xaxis_transform(),
                ha='center', va='top',
                fontsize=FONT_ANNOTATE + 1, fontweight='bold',
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
    ax.set_title('Pi0 + azure_poisson_wide v2 — vLLM-style normalized E2E latency knee\n'
                 'Solid = MEAN (vLLM-faithful headline) • Dashed = p99 (tail companion)',
                 fontsize=FONT_TITLE, fontweight='bold')
    ax.grid(True, which='both', alpha=0.4)
    set_spines(ax)

    leg = ax.legend(loc='upper left', fontsize=FONT_LABEL - 1,
                    title='Series')
    bold_legend(leg)
    if leg.get_title() is not None:
        leg.get_title().set_fontweight('bold')

    fig.tight_layout()
    out_dir = REPO / 'thesis_plotting/figures'
    save_fig(fig, 'fig_rps_knee_pi0_wide_normalized', out_dir)
    plt.close(fig)


if __name__ == '__main__':
    main()
