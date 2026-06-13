#!/usr/bin/env python3
"""Per-cell tail anatomy — who is in the e2e tail and why?

For one knee-sweep cell, this answers three diagnostic questions:

  Q1 (arrival-time hypothesis): are the worst-e2e requests dominated by
     LATE arrivers (queue-drain artifact), or scattered across the run?
  Q2 (request-shape hypothesis): are the worst-e2e requests the ones
     with the largest CONTEXT (or output) tokens?
  Q3 (route-fallback hypothesis): are the worst-e2e requests the ones
     that got REROUTED to GPU (kv_oom_fallback_gpu or
     kv_max_preempt_tries_exceeded)?

CLI:
    python tools/analyze_request_tail.py \\
        --run-dir cluster_outputs/online_serving_runs/<sweep>/<cell> \\
        --policy max_util_full \\
        --top-n 50 \\
        --out-dir <path>

Outputs:
  <out-dir>/<cell_name>_tail_summary.txt   — human-readable summary
  <out-dir>/<cell_name>_tail_anatomy.{pdf,png} — 3-panel figure

The 3 panels:
  A. (arrival_ms, e2e_ms) scatter, colored by context_tokens (log).
  B. (context_tokens, e2e_ms) scatter, colored by route.
  C. boxplot of e2e_ms by arrival decile.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

from plot_sweep_compare import parse_summary  # noqa: E402


# String values of admission_last_reject_reason that indicate the
# request was REROUTED to GPU (not just held-and-retried).
REROUTE_REASONS = {
    "kv_oom_fallback_gpu",
    "kv_max_preempt_tries_exceeded",
}

# Strings that indicate the request was held at admission at some point.
# (The reason field stores only the LAST reason — so this is a lower
# bound on "ever-held".)
HOLD_REASONS = {
    "kv_oom",
    "predictive_pressure",
    "admission_violation_budget",
}


def load_cell(run_dir: Path, policy: str) -> tuple[pd.DataFrame, dict, dict]:
    """Return (per-request DataFrame, system summary dict, meta dict)."""
    csv = run_dir / policy / "requests_out.csv"
    if not csv.is_file():
        sys.exit(f"[error] requests_out.csv not found: {csv}")

    df = pd.read_csv(csv)
    d = df[df.state == "done"].copy()
    if not len(d):
        sys.exit(f"[error] no 'done' requests in {csv}")

    # Decile by arrival time
    d["arr_decile"] = pd.qcut(d.arrival_ms, q=10, labels=False, duplicates="drop")

    # Reason-string buckets
    reason = d.get("admission_last_reject_reason", pd.Series("", index=d.index))
    reason = reason.fillna("")
    d["was_rerouted"] = reason.isin(REROUTE_REASONS)
    d["was_held"] = reason.isin(HOLD_REASONS) | d["was_rerouted"]

    summary = parse_summary(run_dir / "policy_compare_summary.txt") or {}

    # Read run_info.txt for scale / n_requests / kv_pool
    meta = {}
    info_path = run_dir / "run_info.txt"
    if info_path.is_file():
        for line in info_path.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                meta[k.strip()] = v.strip()
    return d, summary, meta


def write_summary(
    d: pd.DataFrame,
    summary: dict,
    meta: dict,
    top_n: int,
    out_txt: Path,
) -> None:
    top = d.nlargest(top_n, "e2e_ms")
    arrival_span_s = d.arrival_ms.max() / 1000.0

    lines = []
    lines.append(f"=== tail anatomy: {out_txt.stem} ===")
    lines.append("")
    lines.append("# meta")
    for key in ("model", "n_requests", "arrival_scale", "kv_pool_bytes"):
        if key in meta:
            lines.append(f"  {key}={meta[key]}")
    lines.append("")
    lines.append("# population")
    lines.append(f"  n_done                  = {len(d)}")
    lines.append(f"  arrival span (s)        = {arrival_span_s:.1f}")
    lines.append(f"  throughput (req/s)      = {summary.get('throughput', '?')}")
    lines.append(f"  e2e p50 / p90 / p99 (s) ="
                 f" {d.e2e_ms.quantile(.5)/1000:.2f}"
                 f" / {d.e2e_ms.quantile(.9)/1000:.2f}"
                 f" / {d.e2e_ms.quantile(.99)/1000:.2f}")
    lines.append(f"  route mix (overall)     = "
                 f"PIM {(d.route=='lpddr5_pim_bank').sum()}"
                 f" / GPU {(d.route=='gpu_only').sum()}")
    lines.append("")
    lines.append("# system counters (from policy_compare_summary.txt)")
    for k in (
        "kv_oom_holds", "predictive_holds", "admission_violation_holds",
        "preempt_total", "preempt_full", "preempt_tail", "tokens_recomp",
    ):
        if k in summary and summary[k] is not None:
            lines.append(f"  {k:25s}= {summary[k]}")
    lines.append("")
    lines.append(f"# top-{top_n} worst-e2e requests")
    lines.append(f"  arrival_ms     range: [{top.arrival_ms.min():.0f}, {top.arrival_ms.max():.0f}]")
    lines.append(f"  e2e_ms         range: [{top.e2e_ms.min():.0f}, {top.e2e_ms.max():.0f}]")
    lines.append(f"  context_tokens range: [{top.context_tokens.min()}, {top.context_tokens.max()}]"
                 f"  median={top.context_tokens.median():.0f}")
    lines.append(f"  generated_tokens     : [{top.generated_tokens.min()}, {top.generated_tokens.max()}]"
                 f"  median={top.generated_tokens.median():.0f}")
    lines.append(f"  route mix in top-{top_n}: "
                 f"PIM {(top.route=='lpddr5_pim_bank').sum()}"
                 f" / GPU {(top.route=='gpu_only').sum()}"
                 f"  (overall GPU share = {(d.route=='gpu_only').mean()*100:.1f}%)")
    lines.append(f"  preempt_count  median={top.preempt_count.median():.0f}"
                 f"  max={top.preempt_count.max()}")
    lines.append(f"  is_lost (SLO miss admit) = {int(top.is_lost.sum())}/{len(top)}")
    lines.append(f"  was_rerouted (kv_oom_fallback_gpu or preempt-budget) "
                 f"= {int(top.was_rerouted.sum())}/{len(top)}"
                 f"  (overall: {int(d.was_rerouted.sum())}/{len(d)})")
    lines.append(f"  was_held (any hold reason) "
                 f"= {int(top.was_held.sum())}/{len(top)}"
                 f"  (overall: {int(d.was_held.sum())}/{len(d)})")
    lines.append("")
    lines.append(f"# top-{top_n} worst-e2e arrival decile histogram")
    deciles = top.arr_decile.value_counts().sort_index()
    for i in range(10):
        cnt = int(deciles.get(i, 0))
        bar = "#" * cnt
        lines.append(f"  decile {i}: {cnt:3d}  {bar}")
    lines.append("")
    lines.append("# whole-population per-decile e2e (median / p99) in seconds")
    lines.append(f"  {'decile':>6}  {'n':>5}  {'arr_lo (s)':>11}  {'arr_hi (s)':>11}"
                 f"  {'e2e p50 (s)':>13}  {'e2e p99 (s)':>13}")
    for i in range(10):
        sub = d[d.arr_decile == i]
        if not len(sub):
            continue
        lines.append(
            f"  {i:>6}  {len(sub):>5}  "
            f"{sub.arrival_ms.min()/1000:>11.1f}  {sub.arrival_ms.max()/1000:>11.1f}  "
            f"{sub.e2e_ms.quantile(.5)/1000:>13.1f}  {sub.e2e_ms.quantile(.99)/1000:>13.1f}"
        )
    lines.append("")
    lines.append("# reason-bucket counts among the worst-N")
    if "admission_last_reject_reason" in top.columns:
        reasons = top.admission_last_reject_reason.fillna("").value_counts()
        for r, c in reasons.items():
            label = r if r else "(none / clean admit)"
            lines.append(f"  {label:40s}  {c}")
    lines.append("")
    out_txt.write_text("\n".join(lines))
    print(f"[txt] {out_txt}")


def render(
    d: pd.DataFrame, top_n: int, out_dir: Path, cell_name: str, title_extra: str
) -> None:
    top = d.nlargest(top_n, "e2e_ms")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel A: (arrival_ms, e2e_ms) colored by context_tokens
    ax = axes[0]
    arr_s = d.arrival_ms / 1000
    e2e_s = d.e2e_ms / 1000
    sc = ax.scatter(
        arr_s, e2e_s,
        c=d.context_tokens, s=12, alpha=0.55,
        cmap="viridis",
        norm=plt.matplotlib.colors.LogNorm(
            vmin=max(1, d.context_tokens.min()),
            vmax=max(2, d.context_tokens.max()),
        ),
        edgecolors="none", zorder=2,
    )
    # Highlight the top-N
    ax.scatter(
        top.arrival_ms / 1000, top.e2e_ms / 1000,
        marker="o", facecolor="none", edgecolor="#c44e52",
        s=60, linewidth=1.0, zorder=3, label=f"top-{top_n} worst e2e",
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("context_tokens (log)")
    ax.set_xlabel("arrival_ms  (s)", fontweight="bold")
    ax.set_ylabel("e2e_ms  (s)", fontweight="bold")
    ax.set_title("A. (arrival, e2e) — color = context_tokens", fontweight="bold")
    ax.grid(True, alpha=0.4)
    ax.legend(loc="upper left", fontsize=9)

    # Panel B: (context_tokens, e2e_ms) colored by route
    ax = axes[1]
    pim = d[d.route == "lpddr5_pim_bank"]
    gpu = d[d.route == "gpu_only"]
    ax.scatter(
        pim.context_tokens, pim.e2e_ms / 1000,
        c="#1f77b4", s=12, alpha=0.5, label=f"PIM (n={len(pim)})",
        edgecolors="none", zorder=2,
    )
    ax.scatter(
        gpu.context_tokens, gpu.e2e_ms / 1000,
        c="#d62728", s=12, alpha=0.5, label=f"GPU (n={len(gpu)})",
        edgecolors="none", zorder=2,
    )
    ax.scatter(
        top.context_tokens, top.e2e_ms / 1000,
        marker="o", facecolor="none", edgecolor="black",
        s=55, linewidth=0.8, zorder=3, label=f"top-{top_n} worst e2e",
    )
    ax.set_xlabel("context_tokens", fontweight="bold")
    ax.set_ylabel("e2e_ms  (s)", fontweight="bold")
    ax.set_title("B. (context, e2e) — color = route", fontweight="bold")
    ax.grid(True, alpha=0.4)
    ax.legend(loc="upper left", fontsize=9)

    # Panel C: boxplot of e2e by arrival decile
    ax = axes[2]
    data_per_decile = [d[d.arr_decile == i].e2e_ms / 1000 for i in range(10)]
    bp = ax.boxplot(
        data_per_decile, positions=range(10),
        widths=0.7, showfliers=True, patch_artist=True,
    )
    for box in bp["boxes"]:
        box.set_facecolor("#90c0e0")
        box.set_alpha(0.7)
    # Overlay top-N markers
    for i in range(10):
        in_decile = top[top.arr_decile == i]
        if len(in_decile):
            ax.scatter(
                np.full(len(in_decile), i), in_decile.e2e_ms / 1000,
                color="#c44e52", marker="o", s=28, zorder=3, alpha=0.8,
            )
    ax.set_yscale("log")
    ax.set_xlabel("arrival decile (0=first 10% of arrivals)", fontweight="bold")
    ax.set_ylabel("e2e_ms  (s, log)", fontweight="bold")
    ax.set_title(f"C. e2e by arrival decile  (red dots = top-{top_n})",
                 fontweight="bold")
    ax.grid(True, alpha=0.4, which="both")

    fig.suptitle(f"Tail anatomy — {cell_name}   {title_extra}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    pdf = out_dir / f"{cell_name}_tail_anatomy.pdf"
    png = out_dir / f"{cell_name}_tail_anatomy.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=140)
    plt.close(fig)
    print(f"[pdf] {pdf}")
    print(f"[png] {png}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True,
                    help="Path to a cell run dir, e.g. "
                         "cluster_outputs/online_serving_runs/<sweep>/<cell>")
    ap.add_argument("--policy", default="max_util_full",
                    help="Policy subdir under the run dir (default: max_util_full)")
    ap.add_argument("--top-n", type=int, default=50)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Output dir for txt + figure. "
                         "Default: <run-dir>/plots")
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        sys.exit(f"[error] run-dir not found: {run_dir}")
    out_dir = (args.out_dir or run_dir / "plots").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    d, summary, meta = load_cell(run_dir, args.policy)
    cell_name = run_dir.name
    title_extra = ""
    if "arrival_scale" in meta:
        title_extra = (f"(scale={meta['arrival_scale']}  "
                       f"rps={summary.get('throughput', '?')}  "
                       f"n={meta.get('n_requests', len(d))})")
    write_summary(d, summary, meta, args.top_n, out_dir / f"{cell_name}_tail_summary.txt")
    render(d, args.top_n, out_dir, cell_name, title_extra)


if __name__ == "__main__":
    main()
