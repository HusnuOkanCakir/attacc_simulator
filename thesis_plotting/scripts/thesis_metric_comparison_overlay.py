#!/usr/bin/env python3
"""Single-panel overlay of 4 latency-metric variants on the Pi0
azure_poisson_wide knee curve.

All 4 curves share the same x-axis (offered RPS) and y-axis (log scale).
Units differ (E2E in seconds, normalized in ms/token) so each curve is
labelled with its units; comparison is shape-based, not absolute-value-
based.

Output: thesis_plotting/figures/fig_metric_overlay_pi0_wide.{pdf,png}
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
            e2e_p99       = d.e2e_ms.quantile(.99) / 1000.0
            arr_cutoff    = d.arrival_ms.quantile(0.80)
            d_trunc       = d[d.arrival_ms <= arr_cutoff]
            e2e_p99_trunc = d_trunc.e2e_ms.quantile(.99) / 1000.0
            d['norm']     = d.e2e_ms / d.generated_tokens.clip(lower=1)
            norm_mean     = d['norm'].mean()
            ttft_p99      = d.ttft_ms.quantile(.99) / 1000.0
            key = round(scale, 2)
            cand = (key, rps, e2e_p99, e2e_p99_trunc, norm_mean, ttft_p99)
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
    print(f'[info] {len(pts)} cells')

    drain = sorted([p for p in pts if classify(p[0]) == 'drain'], key=lambda t: t[1])
    over  = [p for p in pts if classify(p[0]) == 'overflow']
    part  = [p for p in pts if classify(p[0]) == 'partial']

    fig, ax = plt.subplots(figsize=(11.5, 6.5))

    series = [
        ('E2E p99 — RAW (s)',           2, COLOR_HEAVY[1], 'o', '-'),
        ('E2E p99 — TRUNC 80% (s)',     3, COLOR_HEAVY[3], 's', '-'),
        ('Normalized mean (ms/token)',  4, COLOR_HEAVY[0], '^', '-'),
        ('TTFT p99 (s)',                5, COLOR_HEAVY[2], 'D', '-'),
    ]

    for label, idx, color, marker, ls in series:
        xs = [p[1] for p in drain]
        ys = [p[idx] for p in drain]
        ax.plot(xs, ys, color=color, lw=1.6, marker=marker, markersize=6,
                ls=ls, alpha=0.85,
                markeredgecolor='black', markeredgewidth=0.4,
                label=label, zorder=4)
        # Overlay the off-curve (overflow + partial) as smaller hollow markers
        for grp, mk, sz in [(over, 'X', 100), (part, 'D', 70)]:
            if grp:
                ax.scatter([p[1] for p in grp], [p[idx] for p in grp],
                           marker=mk, s=sz, facecolor='white',
                           edgecolor=color, linewidth=1.0, zorder=3)

    # Safe-operating-point line + annotation
    s4 = next((p for p in drain if abs(p[0] - 4.0) < 0.01), None)
    if s4:
        ax.axvline(s4[1], color='#4a8a4a', ls=':', lw=1.5, alpha=0.7)
        ax.text(s4[1] * 1.04, ax.get_ylim()[1] if False else 1.05,
                f'safe operating point\na = {s4[1]:.2f} req/s  (s=4)',
                transform=ax.get_xaxis_transform(),
                ha='left', va='top',
                fontsize=FONT_ANNOTATE + 1, fontweight='bold',
                color='#1f5b1f',
                bbox=dict(facecolor='white', edgecolor='#4a8a4a',
                          boxstyle='round,pad=0.3', linewidth=0.7, alpha=0.95))

    # Danger zone shading past s=4 (rps > 1.36)
    drain_rps_max = max(p[1] for p in drain)
    ax.axvspan(s4[1], drain_rps_max * 1.05,
               color='#c44e52', alpha=0.06, zorder=1)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:g}'))

    ax.set_xlabel('Offered RPS  (effective req / s)',
                  fontsize=FONT_LABEL + 1, fontweight='bold')
    ax.set_ylabel('Latency metric  (s or ms/token — see legend; log scale)',
                  fontsize=FONT_LABEL + 1, fontweight='bold')
    ax.set_title('Pi0 + azure_poisson_wide v2 — 4 latency-metric variants on one axis\n'
                 'Solid = DRAIN cells • Hollow X = OVERFLOW • Hollow ◇ = partial recovery',
                 fontsize=FONT_TITLE, fontweight='bold')
    ax.grid(True, which='both', alpha=0.4)
    set_spines(ax)

    leg = ax.legend(loc='upper left', fontsize=FONT_LABEL,
                    title='Metric (unit)')
    bold_legend(leg)
    if leg.get_title() is not None:
        leg.get_title().set_fontweight('bold')

    fig.tight_layout()
    out_dir = REPO / 'thesis_plotting/figures'
    save_fig(fig, 'fig_metric_overlay_pi0_wide', out_dir)
    plt.close(fig)


if __name__ == '__main__':
    main()
