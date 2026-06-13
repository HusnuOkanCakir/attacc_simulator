#!/usr/bin/env python3
"""Compare 4 latency-metric variants on the Pi0 azure_poisson_wide knee.

Reads the dense knee + metastability sweeps and computes 4 metrics per cell:
  1. E2E p99 (raw — current default)         — vulnerable to drain artifact
  2. E2E p99 truncated (last 20% dropped)     — partial drain mitigation
  3. Normalized mean = mean(e2e / out_len)    — vLLM/Orca style
  4. TTFT p99                                 — Splitwise/ThrottLLeM secondary

Output: thesis_plotting/figures/fig_metric_comparison_pi0_wide.{pdf,png}
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

A_REQ_S = 1.36         # chosen safe operating point (scale=4)


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
            # Metrics
            e2e_p99       = d.e2e_ms.quantile(.99) / 1000.0
            arr_cutoff    = d.arrival_ms.quantile(0.80)
            d_trunc       = d[d.arrival_ms <= arr_cutoff]
            e2e_p99_trunc = d_trunc.e2e_ms.quantile(.99) / 1000.0
            d['norm']     = d.e2e_ms / d.generated_tokens.clip(lower=1)
            norm_mean     = d['norm'].mean()
            ttft_p99      = d.ttft_ms.quantile(.99) / 1000.0
            key = round(scale, 2)
            cur = rows.get(key)
            cand = (key, rps, e2e_p99, e2e_p99_trunc, norm_mean, ttft_p99)
            if cur is None or cand[2] < cur[2]:
                rows[key] = cand
    return sorted(rows.values())


def classify(scale):
    if scale in OVERFLOW_SCALES:
        return 'overflow'
    if scale in PARTIAL_SCALES:
        return 'partial'
    return 'drain'


def panel(ax, pts, idx, title, ylabel):
    drain = [p for p in pts if classify(p[0]) == 'drain']
    drain = sorted(drain, key=lambda t: t[1])
    over  = [p for p in pts if classify(p[0]) == 'overflow']
    part  = [p for p in pts if classify(p[0]) == 'partial']

    xs = [p[1] for p in drain]
    ys = [p[idx] for p in drain]
    ax.plot(xs, ys, color=COLOR_HEAVY[2], lw=1.6, alpha=0.6, zorder=2)
    ax.scatter(xs, ys, color=COLOR_HEAVY[2], marker='o', s=58,
               edgecolor='black', linewidth=0.5, label='DRAIN', zorder=4)

    if over:
        ax.scatter([p[1] for p in over], [p[idx] for p in over],
                   color=COLOR_HEAVY[1], marker='X', s=120,
                   edgecolor='black', linewidth=0.5, label='OVERFLOW', zorder=5)
    if part:
        ax.scatter([p[1] for p in part], [p[idx] for p in part],
                   color=COLOR_HEAVY[3], marker='D', s=90,
                   edgecolor='black', linewidth=0.5, label='partial', zorder=5)

    # Mark chosen a
    s4 = next((p for p in drain if abs(p[0] - 4.0) < 0.01), None)
    if s4:
        ax.axvline(s4[1], color='#4a8a4a', ls=':', lw=1.2, alpha=0.7)
        ax.scatter([s4[1]], [s4[idx]], marker='*', s=240,
                   color=COLOR_HEAVY[2], edgecolor='black', linewidth=0.8,
                   zorder=6, label=f'a={s4[1]:.2f}')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:g}'))
    ax.set_xlabel('Offered RPS  (req/s)', fontsize=FONT_LABEL,
                  fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=FONT_LABEL, fontweight='bold')
    ax.set_title(title, fontsize=FONT_TITLE - 1, fontweight='bold')
    ax.grid(True, which='both', alpha=0.4)
    set_spines(ax)
    leg = ax.legend(loc='upper left', fontsize=FONT_LABEL - 2)
    bold_legend(leg)


def main():
    configure_plotting()
    pts = collect()
    print(f'[info] {len(pts)} cells')

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    panel(axes[0][0], pts, 2,
          '(a) E2E p99  — RAW  (current default; has end-of-trace drain artifact)',
          'E2E p99  (s, log)')
    panel(axes[0][1], pts, 3,
          '(b) E2E p99 — TRUNCATED  (drop last 20% of arrivals)',
          'E2E p99 — last 20% dropped  (s, log)')
    panel(axes[1][0], pts, 4,
          '(c) Normalized mean  =  mean(e2e / out_len)  — vLLM/Orca style',
          'Mean ms per output token (log)')
    panel(axes[1][1], pts, 5,
          '(d) TTFT p99  — Splitwise/ThrottLLeM secondary metric',
          'TTFT p99  (s, log)')

    fig.suptitle('Pi0 + azure_poisson_wide v2 — 4 metric variants over the same data\n'
                 'Vertical dotted line + green star = chosen safe operating point  a = 1.36 req/s  (scale = 4)',
                 fontsize=FONT_TITLE + 1, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out_dir = REPO / 'thesis_plotting/figures'
    save_fig(fig, 'fig_metric_comparison_pi0_wide', out_dir)
    plt.close(fig)


if __name__ == '__main__':
    main()
