#!/usr/bin/env python3
"""
Cross-sub-run comparison plot for any sweep directory.

For a sweep dir like:

    cluster_outputs/online_serving_runs/sweep_pi0_n2000_20260513_184949/
        01_slo500_guaranteed_no_evict/
            policy_compare_summary.txt
        02_slo500_max_util_full/
            ...

this script reads every `<NN>_<label>/policy_compare_summary.txt` and produces
a multi-panel comparison figure showing all sub-runs side-by-side on key
metrics (TTFT, E2E, throughput, route mix, scheduler events).

Works with any sweep convention as long as each sub-run directory contains a
`policy_compare_summary.txt` (the runner writes this automatically).

Usage:
    python tools/plot_sweep_compare.py --sweep-dir <path>
    python tools/plot_sweep_compare.py --sweep-dir <path> --out plots/foo.png
    python tools/plot_sweep_compare.py --sweep-dir <path> --policy max_util_full

The optional `--policy` filter pins which column of the summary to read when
sub-runs ran multiple policies. Without it, the first non-header column is used.
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── parsing ──────────────────────────────────────────────────────────────────

# Each row in policy_compare_summary.txt is "Metric name<whitespace>value column(s)".
# We use anchored regexes per metric so we don't break if column widths drift.
METRIC_PATTERNS = {
    # name              regex                                              extractor
    "routes_pim":       (r"Routes \(PIM/GPU\)\s+(\d+)/(\d+)",              lambda m: int(m.group(1))),
    "routes_gpu":       (r"Routes \(PIM/GPU\)\s+(\d+)/(\d+)",              lambda m: int(m.group(2))),
    "admitted":         (r"Admitted/Dropped\s+(\d+)/(\d+)",                lambda m: int(m.group(1))),
    "dropped":          (r"Admitted/Dropped\s+(\d+)/(\d+)",                lambda m: int(m.group(2))),
    "kv_oom_holds":     (r"KV-OOM holds\s+([\d]+)",                        lambda m: int(m.group(1))),
    "admission_violation_holds": (r"Admission viol\. holds\s+([\d]+)",     lambda m: int(m.group(1))),
    "preempt_total":    (r"Preempt count \(tail/full\)\s+([\d]+)",         lambda m: int(m.group(1))),
    "preempt_tail":     (r"Preempt count \(tail/full\)\s+\d+\s+\((\d+)/\d+\)",      lambda m: int(m.group(1))),
    "preempt_full":     (r"Preempt count \(tail/full\)\s+\d+\s+\(\d+/(\d+)\)",      lambda m: int(m.group(1))),
    "tokens_recomp":    (r"Tokens recomputed\s+([\d]+)",                   lambda m: int(m.group(1))),
    "predictive_holds": (r"Predictive holds\s+([\d]+)",                    lambda m: int(m.group(1))),
    "ttft_p50":         (r"TTFT p50/p90 \(ms\)\s+([\d.]+)/[\d.]+",         lambda m: float(m.group(1))),
    "ttft_p90":         (r"TTFT p50/p90 \(ms\)\s+[\d.]+/([\d.]+)",         lambda m: float(m.group(1))),
    "ttft_p99":         (r"TTFT p99 \(ms\)\s+([\d.]+)",                    lambda m: float(m.group(1))),
    "e2e_p50":          (r"E2E p50/p90 \(ms\)\s+([\d.]+)/[\d.]+",          lambda m: float(m.group(1))),
    "e2e_p90":          (r"E2E p50/p90 \(ms\)\s+[\d.]+/([\d.]+)",          lambda m: float(m.group(1))),
    "e2e_p99":          (r"E2E p99 \(ms\)\s+([\d.]+)",                     lambda m: float(m.group(1))),
    "span_ms":          (r"Span \(ms\)\s+([\d.]+)",                        lambda m: float(m.group(1))),
    "throughput":       (r"Throughput \(req/s\)\s+([\d.]+)",               lambda m: float(m.group(1))),
    # SLO violation accounting (added 2026-05-19; n/a if SLO disabled)
    "slo_e2e_violators": (r"SLO E2E violators\s+(\d+)/\d+",                 lambda m: int(m.group(1))),
    "slo_e2e_done":      (r"SLO E2E violators\s+\d+/(\d+)",                 lambda m: int(m.group(1))),
    "slo_ttft_violators":(r"SLO TTFT violators\s+(\d+)/\d+",                lambda m: int(m.group(1))),
    "slo_ttft_done":     (r"SLO TTFT violators\s+\d+/(\d+)",                lambda m: int(m.group(1))),
    # Phase E decode-batching telemetry (added 2026-05-27)
    "decode_mean_bs":    (r"Decode batch mean/max/cfg\s+([\d.]+)/\d+/\d+",  lambda m: float(m.group(1))),
    "decode_max_bs":     (r"Decode batch mean/max/cfg\s+[\d.]+/(\d+)/\d+",  lambda m: int(m.group(1))),
    "decode_max_bs_cfg": (r"Decode batch mean/max/cfg\s+[\d.]+/\d+/(\d+)",  lambda m: int(m.group(1))),
    "decode_steps":      (r"Decode steps\s+(\d+)",                          lambda m: int(m.group(1))),
}


def parse_summary(path: Path) -> dict | None:
    if not path.is_file():
        return None
    text = path.read_text()
    out = {}
    for name, (pattern, extract) in METRIC_PATTERNS.items():
        m = re.search(pattern, text, re.MULTILINE)
        out[name] = extract(m) if m else None
    return out


def discover_runs(sweep_dir: Path) -> list[tuple[str, dict]]:
    """Return [(label, metrics_dict), ...] in directory-sort order.

    Sub-run directories are expected to be named `<NN>_<rest>` so that
    natural sort puts them in the sweep order. We strip the leading
    `<NN>_` from the label for the plot axes.
    """
    rows = []
    for sub in sorted(sweep_dir.iterdir()):
        if not sub.is_dir():
            continue
        # Allow summary at either <sub>/policy_compare_summary.txt
        # or <sub>/<policy>/policy_compare_summary.txt
        candidates = [
            sub / "policy_compare_summary.txt",
        ]
        # Also accept files inside any subdir of <sub> in case the sweep
        # nests by policy.
        for child in sub.iterdir() if sub.is_dir() else []:
            if child.is_dir():
                candidates.append(child / "policy_compare_summary.txt")
        summary = next((c for c in candidates if c.is_file()), None)
        if summary is None:
            print(f"  [warn] no summary in {sub.name}, skipping", file=sys.stderr)
            continue
        metrics = parse_summary(summary)
        if metrics is None or metrics.get("throughput") is None:
            print(f"  [warn] parse failed for {sub.name}", file=sys.stderr)
            continue
        # Strip the leading 2-digit + underscore prefix for the label.
        label = re.sub(r"^\d+_", "", sub.name)
        rows.append((label, metrics))
    return rows


# ── plotting ─────────────────────────────────────────────────────────────────

def _bar_group(ax, labels: list[str], series: dict[str, list[float]],
               title: str, ylabel: str, log: bool = False,
               annotate: bool = True, ymax_pad: float = 1.18) -> None:
    """Grouped bar chart. `series` is {name: values_list}, one bar per label."""
    n_groups = len(labels)
    n_series = len(series)
    width = 0.8 / max(n_series, 1)
    xs = np.arange(n_groups)
    colors = plt.get_cmap("tab10").colors

    max_val = 0.0
    for i, (name, vals) in enumerate(series.items()):
        offsets = xs + (i - (n_series - 1) / 2) * width
        clean_vals = [v if v is not None else 0.0 for v in vals]
        ax.bar(offsets, clean_vals, width, label=name, color=colors[i % len(colors)],
               edgecolor="black", linewidth=0.4)
        for j, v in enumerate(clean_vals):
            if v is None or v == 0:
                continue
            max_val = max(max_val, v)
            if annotate:
                txt = (f"{v:.2g}" if v < 1000 else
                       f"{v/1000:.1f}k" if v < 1e6 else
                       f"{v/1e6:.1f}M")
                ax.text(offsets[j], v, txt, ha="center", va="bottom",
                        fontsize=6, rotation=0)
    if log and max_val > 0:
        ax.set_yscale("log")
    elif max_val > 0:
        ax.set_ylim(0, max_val * ymax_pad)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    if n_series > 1:
        ax.legend(fontsize=7, loc="best")
    ax.grid(True, axis="y", alpha=0.3)


def _stacked_route_mix(ax, labels: list[str],
                       pim: list[float], gpu: list[float]) -> None:
    xs = np.arange(len(labels))
    totals = [(p or 0) + (g or 0) for p, g in zip(pim, gpu)]
    pim_pct = [100 * (p or 0) / t if t > 0 else 0 for p, t in zip(pim, totals)]
    gpu_pct = [100 - p for p in pim_pct]
    ax.bar(xs, pim_pct, 0.62, label="PIM", color="#1f77b4",
           edgecolor="black", linewidth=0.4)
    ax.bar(xs, gpu_pct, 0.62, bottom=pim_pct, label="GPU", color="#ff7f0e",
           edgecolor="black", linewidth=0.4)
    for j, (p, g, t) in enumerate(zip(pim, gpu, totals)):
        if t == 0:
            continue
        ax.text(xs[j], 50, f"{p}/{g}", ha="center", va="center",
                fontsize=7, color="white" if pim_pct[j] > 25 else "black")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_title("Route mix (PIM / GPU)", fontsize=10)
    ax.set_ylabel("Routes (%)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)


def make_plot(rows: list[tuple[str, dict]], sweep_dir: Path,
              out_path: Path, title_suffix: str = "",
              linear_y: bool = False) -> None:
    labels = [lbl for lbl, _ in rows]
    metrics = [m for _, m in rows]
    # When linear_y is set, panels that would use log Y switch to linear
    # (with annotated bar values + headroom padding). Useful for sweeps
    # where the bars are within ~1 order of magnitude and log obscures
    # the differences.
    use_log = not linear_y

    fig, axes = plt.subplots(2, 3, figsize=(max(10, 1.4 * len(labels) + 5), 9))
    fig.suptitle(f"{sweep_dir.name}{title_suffix}", fontsize=12, y=0.995)

    # (0, 0) TTFT
    _bar_group(axes[0, 0], labels,
               {"p50": [m.get("ttft_p50") for m in metrics],
                "p90": [m.get("ttft_p90") for m in metrics],
                "p99": [m.get("ttft_p99") for m in metrics]},
               title="TTFT (ms)", ylabel="ms",
               log=use_log, annotate=not use_log)

    # (0, 1) E2E
    _bar_group(axes[0, 1], labels,
               {"p50": [m.get("e2e_p50") for m in metrics],
                "p90": [m.get("e2e_p90") for m in metrics],
                "p99": [m.get("e2e_p99") for m in metrics]},
               title="E2E latency (ms)", ylabel="ms",
               log=use_log, annotate=not use_log)

    # (0, 2) Throughput — already linear
    _bar_group(axes[0, 2], labels,
               {"throughput": [m.get("throughput") for m in metrics]},
               title="Throughput (req/s)", ylabel="req/s",
               log=False, annotate=True)

    # (1, 0) Route mix stacked
    _stacked_route_mix(axes[1, 0], labels,
                       [m.get("routes_pim") for m in metrics],
                       [m.get("routes_gpu") for m in metrics])

    # (1, 1) KV-OOM holds + preempts
    _bar_group(axes[1, 1], labels,
               {"KV-OOM holds":   [m.get("kv_oom_holds") for m in metrics],
                "Preempt (full)": [m.get("preempt_full") for m in metrics],
                "Preempt (tail)": [m.get("preempt_tail") for m in metrics]},
               title="KV-OOM + preempts (counts)",
               ylabel="count" + (" (symlog)" if use_log else ""),
               log=use_log, annotate=not use_log)

    # (1, 2) Tokens recomputed + predictive holds + dropped
    _bar_group(axes[1, 2], labels,
               {"Tokens recomp":     [m.get("tokens_recomp") for m in metrics],
                "Predictive holds":  [m.get("predictive_holds") for m in metrics],
                "Dropped":           [m.get("dropped") for m in metrics]},
               title="Recomp / hold / drop",
               ylabel="count" + (" (symlog)" if use_log else ""),
               log=use_log, annotate=not use_log)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-dir", required=True, type=Path,
                    help="Sweep root directory containing <NN>_<label>/ sub-runs.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output PNG path. Default: <sweep-dir>/plots/sweep_compare.png")
    ap.add_argument("--linear-y", action="store_true",
                    help="Use linear Y axes (default: log Y on TTFT/E2E/counts). "
                         "Useful when bars are within ~1 order of magnitude and "
                         "log compresses the differences.")
    args = ap.parse_args()

    sweep_dir = args.sweep_dir.resolve()
    if not sweep_dir.is_dir():
        sys.exit(f"[error] sweep-dir not found: {sweep_dir}")

    print(f"[sweep_compare] scanning {sweep_dir}")
    rows = discover_runs(sweep_dir)
    if not rows:
        sys.exit("[error] no usable sub-runs found "
                 "(expected <NN>_<label>/policy_compare_summary.txt)")

    print(f"[sweep_compare] {len(rows)} sub-runs:")
    for lbl, m in rows:
        print(f"  - {lbl:<35} throughput={m.get('throughput')}  "
              f"TTFT p50={m.get('ttft_p50')}  p99={m.get('ttft_p99')}  "
              f"routes={m.get('routes_pim')}/{m.get('routes_gpu')}")

    # If --linear-y, write to a separate file by default to avoid clobbering
    # the canonical log-Y plot.
    default_name = "sweep_compare_linear.png" if args.linear_y else "sweep_compare.png"
    out_path = args.out or (sweep_dir / "plots" / default_name)
    make_plot(rows, sweep_dir, out_path, linear_y=args.linear_y)


if __name__ == "__main__":
    main()
