#!/usr/bin/env python3
"""Pi0 + azure_poisson_wide v2 — normalized-latency knee at n=5000 vs n=10000.

Renders two figures:
 (a) fig_rps_knee_pi0_wide_normalized_n10k.{pdf,png}  — n=10000 version of
     the headline normalized-latency knee. Same shape as the existing
     n=5000 figure but with the longer (vLLM §6 1-hour-equivalent) trace.
 (b) fig_rps_knee_pi0_wide_normalized_compare.{pdf,png}  — side-by-side
     overlay of n=5000 and n=10000 (mean only) so the trace-length
     sensitivity is visible at a glance.

Inputs:
  cluster_outputs/online_serving_runs/rps_knee_pi0_20260601_175358   (n=5000 dense)
  cluster_outputs/online_serving_runs/rps_knee_pi0_20260601_183211   (n=5000 meta)
  cluster_outputs/online_serving_runs/rps_knee_pi0_wide_n10k_20260602_173950  (n=10000)
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

N5K_DENSE = REPO / "cluster_outputs/online_serving_runs/rps_knee_pi0_20260601_175358"
N5K_META  = REPO / "cluster_outputs/online_serving_runs/rps_knee_pi0_20260601_183211"
N10K      = REPO / "cluster_outputs/online_serving_runs/rps_knee_pi0_wide_n10k_20260602_173950"

N5K_OVERFLOW_SCALES = {2.60, 2.65}
N5K_PARTIAL_SCALES  = {2.55}


def collect(dirs, dataset_filter='azure_poisson_wide'):
    rows = {}
    for sweep_dir in dirs:
        if not sweep_dir.exists():
            continue
        for cell in sorted(sweep_dir.iterdir()):
            if not cell.is_dir() or dataset_filter not in cell.name:
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


def classify_5k(scale):
    if scale in N5K_OVERFLOW_SCALES: return 'overflow'
    if scale in N5K_PARTIAL_SCALES:  return 'partial'
    return 'drain'


def classify_10k(scale, rps):
    """Classify by EFFECTIVE THROUGHPUT relative to OFFERED LOAD.
    The trace's natural mean rate is ~5.5 rps, so offered mean at arrival_scale
    s is 5.5/s. If achieved rps is much less than offered, the system is in
    OVERFLOW (queue grew faster than it drained). Tolerance: 30% deficit = OF."""
    offered_mean = 5.5 / scale
    capacity = 2.13   # max sustained throughput observed at stable saturation
    expected = min(offered_mean, capacity)
    if expected == 0: return 'drain'
    deficit = (expected - rps) / expected
    if deficit > 0.30: return 'overflow'
    return 'drain'


def render_single(pts, classifier, out_name, n_label):
    configure_plotting()
    drain = sorted([p for p in pts if classifier(p[0], p[1] if isinstance(p, tuple) else None) == 'drain']
                   if classifier.__code__.co_argcount == 2 else
                   [p for p in pts if classifier(p[0]) == 'drain'],
                   key=lambda t: t[1])
    fig, ax = plt.subplots(figsize=(11.5, 6.5))

    xs = [p[1] for p in drain]
    ys_mean = [p[2] for p in drain]
    ys_p99  = [p[3] for p in drain]

    ax.plot(xs, ys_mean, color=COLOR_HEAVY[2], lw=2.0, marker='o', markersize=7,
            markeredgecolor='black', markeredgewidth=0.5,
            label='MEAN normalized latency  (vLLM, headline)', zorder=5)
    ax.plot(xs, ys_p99, color=COLOR_HEAVY[1], lw=1.8, marker='s', markersize=6,
            markeredgecolor='black', markeredgewidth=0.5, ls='--',
            label='p99 normalized latency  (tail companion)', zorder=4)

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
    ax.set_title(f'Pi0 + azure_poisson_wide v2 — vLLM-style normalized E2E latency  ({n_label})\n'
                 'Solid = MEAN (vLLM-faithful headline) • Dashed = p99 (tail companion)',
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


def render_compare(pts_5k, pts_10k):
    configure_plotting()
    fig, ax = plt.subplots(figsize=(11.5, 6.5))

    drain_5k = sorted([p for p in pts_5k if classify_5k(p[0]) == 'drain'],
                     key=lambda t: t[1])
    drain_10k = sorted([p for p in pts_10k if classify_10k(p[0], p[1]) == 'drain'],
                      key=lambda t: t[1])

    ax.plot([p[1] for p in drain_5k], [p[2] for p in drain_5k],
            color=COLOR_HEAVY[3], lw=2.0, marker='o', markersize=7,
            markeredgecolor='black', markeredgewidth=0.5,
            label='n = 5,000  (mean normalized latency)', zorder=4)
    ax.plot([p[1] for p in drain_10k], [p[2] for p in drain_10k],
            color=COLOR_HEAVY[2], lw=2.0, marker='s', markersize=7,
            markeredgecolor='black', markeredgewidth=0.5,
            label='n = 10,000  (mean normalized latency)  — vLLM-scale', zorder=5)

    # Mark safe a on the n=10k curve
    s4_10k = next((p for p in drain_10k if abs(p[0] - 4.0) < 0.01), None)
    if s4_10k:
        ax.scatter([s4_10k[1]], [s4_10k[2]], marker='*', s=420, zorder=7,
                   color=COLOR_HEAVY[2], edgecolor='black', linewidth=1.0)
        ax.annotate(
            f'a = {s4_10k[1]:.2f} req/s\n'
            f'(stable in BOTH n=5k and n=10k)',
            xy=(s4_10k[1], s4_10k[2]),
            xytext=(50, -60), textcoords='offset points',
            ha='left', va='top',
            fontsize=FONT_ANNOTATE + 2, fontweight='bold',
            color='#1f5b1f',
            bbox=dict(facecolor='white', edgecolor='#4a8a4a',
                      boxstyle='round,pad=0.3', linewidth=0.7, alpha=0.95),
            arrowprops=dict(arrowstyle='->', color='#4a8a4a', lw=0.9))

    # Annotate the cliff expansion
    ax.annotate(
        'Cliff EXPANDS LEFT with longer trace:\n'
        'cells at rps≈1.4–2.0 (s=2.7–3.0) drop\n'
        'into OVERFLOW at n=10k, were stable at n=5k',
        xy=(1.81, 8154),  # s=3.0 at n=10k
        xytext=(1.6, 2000),
        ha='center', va='top',
        fontsize=FONT_ANNOTATE + 1, fontweight='bold',
        color='#a04040',
        bbox=dict(facecolor='white', edgecolor='#c44e52',
                  boxstyle='round,pad=0.3', linewidth=0.8, alpha=0.95),
        arrowprops=dict(arrowstyle='->', color='#c44e52', lw=1.0))

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:g}'))

    ax.set_xlabel('Offered RPS  (effective req / s)',
                  fontsize=FONT_LABEL + 1, fontweight='bold')
    ax.set_ylabel('Mean normalized latency  (ms / output token; log scale)',
                  fontsize=FONT_LABEL + 1, fontweight='bold')
    ax.set_title('Pi0 + azure_poisson_wide v2 — trace-length sensitivity\n'
                 'Same config, same workload, two trace sizes: n=5k vs n=10k',
                 fontsize=FONT_TITLE, fontweight='bold')
    ax.grid(True, which='both', alpha=0.4)
    set_spines(ax)
    leg = ax.legend(loc='upper left', fontsize=FONT_LABEL, title='Trace size')
    bold_legend(leg)
    if leg.get_title() is not None:
        leg.get_title().set_fontweight('bold')

    fig.tight_layout()
    out_dir = REPO / 'thesis_plotting/figures'
    save_fig(fig, 'fig_rps_knee_pi0_wide_normalized_compare', out_dir)
    plt.close(fig)


def main():
    pts_5k  = collect([N5K_DENSE, N5K_META])
    pts_10k = collect([N10K])
    print(f'[info] n=5k cells: {len(pts_5k)}')
    print(f'[info] n=10k cells: {len(pts_10k)}')

    render_single(pts_10k, classify_10k, 'fig_rps_knee_pi0_wide_normalized_n10k', 'n = 10,000')
    render_compare(pts_5k, pts_10k)


if __name__ == '__main__':
    main()
