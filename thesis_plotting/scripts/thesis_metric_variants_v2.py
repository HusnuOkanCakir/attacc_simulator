#!/usr/bin/env python3
"""Pi0 + azure_poisson_wide v2 — overlay of 4 new metric variants.

Variants (all use the dense + metastability sweeps):
  1. E2E p99 — truncated at 60% (keep first 60% of arrivals, drop last 40%)
  2. E2E p99 — middle 60% (drop first 20% + last 20%)
  3. Normalized mean — middle 60%
  4. Normalized mean — first 80%

Output: thesis_plotting/figures/fig_metric_variants_v2_pi0_wide.{pdf,png}
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

            # Sub-slices
            arr_60 = d[d.arrival_ms <= d.arrival_ms.quantile(0.60)]
            arr_80 = d[d.arrival_ms <= d.arrival_ms.quantile(0.80)]
            arr_mid_60 = d[(d.arrival_ms >= d.arrival_ms.quantile(0.20)) &
                           (d.arrival_ms <= d.arrival_ms.quantile(0.80))]

            d['norm']     = d.e2e_ms / d.generated_tokens.clip(lower=1)
            arr_80['norm']     = arr_80.e2e_ms / arr_80.generated_tokens.clip(lower=1)
            arr_mid_60['norm'] = arr_mid_60.e2e_ms / arr_mid_60.generated_tokens.clip(lower=1)

            metrics = {
                'e2e_p99_trunc60':  arr_60.e2e_ms.quantile(.99) / 1000.0,
                'e2e_p99_mid60':    arr_mid_60.e2e_ms.quantile(.99) / 1000.0,
                'norm_mean_mid60':  arr_mid_60['norm'].mean(),
                'norm_mean_80':     arr_80['norm'].mean(),
            }
            key = round(scale, 2)
            cur = rows.get(key)
            if cur is None or metrics['e2e_p99_trunc60'] < cur[1]['e2e_p99_trunc60']:
                rows[key] = ((key, rps), metrics)
    out = []
    for k in sorted(rows.keys()):
        (scale, rps), m = rows[k]
        out.append((scale, rps, m))
    return out


def classify(scale):
    if scale in OVERFLOW_SCALES: return 'overflow'
    if scale in PARTIAL_SCALES:  return 'partial'
    return 'drain'


def main():
    configure_plotting()
    pts = collect()
    print(f'[info] {len(pts)} cells')

    drain = [p for p in pts if classify(p[0]) == 'drain']
    drain = sorted(drain, key=lambda t: t[1])
    over  = [p for p in pts if classify(p[0]) == 'overflow']
    part  = [p for p in pts if classify(p[0]) == 'partial']

    print(f'{"scale":>6} {"rps":>5} {"e2e_p99 60%":>13} {"e2e_p99 mid60":>15} '
          f'{"norm mean mid60":>17} {"norm mean 80%":>15}')
    for s, r, m in pts:
        print(f'{s:>6.2f} {r:>5.2f} {m["e2e_p99_trunc60"]:>13.1f} '
              f'{m["e2e_p99_mid60"]:>15.1f} {m["norm_mean_mid60"]:>17.1f} '
              f'{m["norm_mean_80"]:>15.1f}')

    fig, ax = plt.subplots(figsize=(11.5, 6.5))

    series = [
        ('E2E p99 — TRUNC 60%  (first 60%, s)',           'e2e_p99_trunc60',
         COLOR_HEAVY[1], 'o'),
        ('E2E p99 — MIDDLE 60%  (drop 20%+20%, s)',       'e2e_p99_mid60',
         COLOR_HEAVY[3], 's'),
        ('Normalized mean — MIDDLE 60%  (ms/token)',       'norm_mean_mid60',
         COLOR_HEAVY[0], '^'),
        ('Normalized mean — TRUNC 80%  (first 80%, ms/token)', 'norm_mean_80',
         COLOR_HEAVY[2], 'D'),
    ]

    for label, key, color, marker in series:
        xs = [p[1] for p in drain]
        ys = [p[2][key] for p in drain]
        ax.plot(xs, ys, color=color, lw=1.6, marker=marker, markersize=6,
                alpha=0.85,
                markeredgecolor='black', markeredgewidth=0.4,
                label=label, zorder=4)
        for grp, mk, sz in [(over, 'X', 100), (part, 'D', 70)]:
            if grp:
                ax.scatter([p[1] for p in grp], [p[2][key] for p in grp],
                           marker=mk, s=sz, facecolor='white',
                           edgecolor=color, linewidth=1.0, zorder=3)

    # Safe-operating-point line + annotation
    s4 = next((p for p in drain if abs(p[0] - 4.0) < 0.01), None)
    if s4:
        ax.axvline(s4[1], color='#4a8a4a', ls=':', lw=1.5, alpha=0.7)
        ax.text(s4[1] * 1.04, 1.05,
                f'safe operating point\na = {s4[1]:.2f} req/s  (s=4)',
                transform=ax.get_xaxis_transform(),
                ha='left', va='top',
                fontsize=FONT_ANNOTATE + 1, fontweight='bold',
                color='#1f5b1f',
                bbox=dict(facecolor='white', edgecolor='#4a8a4a',
                          boxstyle='round,pad=0.3', linewidth=0.7, alpha=0.95))

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
    ax.set_title('Pi0 + azure_poisson_wide v2 — 4 new metric variants (truncation × normalization)\n'
                 'Solid = DRAIN cells • Hollow X = OVERFLOW • Hollow ◇ = partial recovery',
                 fontsize=FONT_TITLE, fontweight='bold')
    ax.grid(True, which='both', alpha=0.4)
    set_spines(ax)

    leg = ax.legend(loc='upper left', fontsize=FONT_LABEL - 1,
                    title='Metric (unit)')
    bold_legend(leg)
    if leg.get_title() is not None:
        leg.get_title().set_fontweight('bold')

    fig.tight_layout()
    out_dir = REPO / 'thesis_plotting/figures'
    save_fig(fig, 'fig_metric_variants_v2_pi0_wide', out_dir)
    plt.close(fig)


if __name__ == '__main__':
    main()
