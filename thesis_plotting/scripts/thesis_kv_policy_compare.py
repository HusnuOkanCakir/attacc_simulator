#!/usr/bin/env python3
"""Thesis figure: KV-allocation policy comparison.

Compares the three KV scheduler policies head-to-head on the same
sweep:
  - guaranteed_no_evict  — admit only when full footprint reserves
  - max_util_full        — chunked growth + full-evict on pressure
  - max_util_tail        — chunked growth + tail-trim on pressure

Four panels (2×2):
  (a) Throughput          — req/s
  (b) E2E p99 latency     — ms, log scale (saturation-sensitive)
  (c) Admitted requests   — how aggressive the policy is
  (d) Preemption cost     — tokens recomputed (work wasted)

Output: thesis_plotting/figures/fig_kv_policy_compare.{pdf,png}
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO))

from plot_sweep_compare import parse_summary  # noqa: E402

from thesis_plotting.style import (  # noqa: E402
    configure_plotting, save_fig, set_spines, bold_legend,
    COLOR_HEAVY, WIDE_FIGSIZE, FONT_LABEL, FONT_TITLE,
)


# Scale=3.0 + SLO=500ms is the regime where max_util_full clearly wins
# (vs scale=1.0 deep-overload where everything saturates). OpenVLA
# n=2000, kv_pool=4GiB, max_active=32.
DEFAULT_RUN  = REPO / "cluster_outputs/online_serving_runs/" \
                      "sweep_openvla_n2000_20260513_161108"
DEFAULT_SLO  = 500

# Order the three policies left-to-right by "aggression":
#   safe → mid → aggressive.
POLICIES = ["guaranteed_no_evict", "max_util_full", "max_util_tail"]
POLICY_LABEL = {
    "guaranteed_no_evict": "guaranteed\nno_evict",
    "max_util_full":       "max_util\nfull",
    "max_util_tail":       "max_util\ntail",
}
POLICY_COLOR = {
    "guaranteed_no_evict": COLOR_HEAVY[0],   # deep teal — conservative
    "max_util_full":       COLOR_HEAVY[3],   # amber     — moderate
    "max_util_tail":       COLOR_HEAVY[1],   # rust      — aggressive
}


def _parse_table(text: str) -> dict:
    """Parse one policy_compare_summary.txt body — could be single-column
    or multi-column. Returns {col_name: {metric: value, ...}}."""
    import re

    cols: list[str] | None = None
    out: dict[str, dict] = {}

    def store(metric: str, vals: list):
        for c, v in zip(cols, vals):
            out.setdefault(c, {})[metric] = v

    for line in text.splitlines():
        stripped = line.strip()
        if cols is None:
            if "guaranteed_no_evict" in stripped or \
               "max_util_full" in stripped or \
               "max_util_tail" in stripped:
                toks = stripped.split()
                cols = [t for t in toks if t != "Metric"]
            continue

        toks = line.split()
        if line.startswith("Throughput (req/s)"):
            store("throughput", [float(t) for t in toks[-len(cols):]])
        elif line.startswith("E2E p99 (ms)"):
            store("e2e_p99", [float(t) for t in toks[-len(cols):]])
        elif line.startswith("TTFT p99 (ms)"):
            store("ttft_p99", [float(t) for t in toks[-len(cols):]])
        elif line.startswith("Admitted/Dropped"):
            for c, v in zip(cols, toks[-len(cols):]):
                a, d = v.split("/")
                out.setdefault(c, {})["admitted"] = int(a)
                out.setdefault(c, {})["dropped"]  = int(d)
        elif line.startswith("KV-OOM holds"):
            store("kv_oom_holds", [int(t) for t in toks[-len(cols):]])
        elif line.startswith("Tokens recomputed"):
            store("tokens_recomp", [int(t) for t in toks[-len(cols):]])
        elif line.startswith("Preempt count"):
            pairs = re.findall(r"(\d+)\s*\(\d+/\d+\)", line)
            for c, v in zip(cols, pairs):
                out.setdefault(c, {})["preempt_total"] = int(v)

    return out


def load_metrics(run_dir: Path, slo_filter: int | None = None) -> dict:
    """Load policy metrics from one of two supported layouts:

    Layout A — combined root file:
        <run>/policy_compare_summary.txt   (all policies as columns)

    Layout B — per-cell subdirs:
        <run>/NN_<...>_<policy>/policy_compare_summary.txt
        e.g. NN_slo<X>_<policy>. If `slo_filter` is given, restrict to
        cells containing `slo<filter>`.
    """
    root_sum = run_dir / "policy_compare_summary.txt"
    if root_sum.is_file():
        return _parse_table(root_sum.read_text())

    # Layout B
    import re
    cell_re = re.compile(
        r"^(?:\d+_)?(?:[a-z]+_)*?(?P<policy>"
        + "|".join(POLICIES) + r")$"
    )
    merged: dict[str, dict] = {}
    for sub in sorted(run_dir.iterdir()):
        if not sub.is_dir():
            continue
        if slo_filter is not None and f"slo{slo_filter}" not in sub.name:
            continue
        m = cell_re.match(sub.name)
        if not m:
            # also accept any subdir whose name ends with `_<policy>`
            for p in POLICIES:
                if sub.name.endswith("_" + p) or sub.name == p:
                    m = type("M", (), {"group": lambda self, k: p})()
                    break
        if not m:
            continue
        sumfile = sub / "policy_compare_summary.txt"
        if not sumfile.is_file():
            continue
        parsed = _parse_table(sumfile.read_text())
        # Single-column summary keyed by policy name.
        policy = m.group("policy") if hasattr(m, "groupdict") else m.group(0)
        for col_name, metrics in parsed.items():
            if col_name in POLICIES:
                merged[col_name] = metrics
    return merged


def _bar_panel(ax, metrics, key, title, ylabel, log=False,
               value_fmt=lambda v: f"{v:.0f}",
               winning_direction="higher"):
    xs = np.arange(len(POLICIES))
    vals = [metrics.get(p, {}).get(key, 0.0) for p in POLICIES]
    colors = [POLICY_COLOR[p] for p in POLICIES]

    bars = ax.bar(xs, vals, width=0.55, color=colors,
                  edgecolor="black", linewidth=0.5, zorder=3)

    # Highlight the best with a star above it.
    if any(v > 0 for v in vals):
        if winning_direction == "higher":
            best_i = int(np.argmax(vals))
        else:
            best_i = int(np.argmin(v if v > 0 else float("inf") for v in vals))
        # workaround: explicit re-eval
        if winning_direction == "lower":
            best_i = int(min(range(len(vals)),
                             key=lambda i: vals[i] if vals[i] > 0 else float("inf")))
        # mark winning value
        # we'll just put a small bold "★ best" caption above the winning bar later

    # Determine best.
    if winning_direction == "higher":
        best_i = int(np.argmax(vals))
    elif log:
        # Log axis can't represent 0; treat 0 as not-a-candidate.
        finite = [(i, v) for i, v in enumerate(vals) if v > 0]
        best_i = min(finite, key=lambda iv: iv[1])[0] if finite else 0
    else:
        # Linear scale, lower-is-better — zero IS a valid (and best) value.
        best_i = int(np.argmin(vals))

    top = max(vals) if max(vals) > 0 else 1.0
    for i, (bar, v) in enumerate(zip(bars, vals)):
        is_best = (i == best_i)
        label = value_fmt(v)
        if is_best:
            label = r"$\bigstar$ " + label

        if log:
            y_lab = max(bar.get_height(), top * 0.001) * 1.10
        else:
            y_lab = bar.get_height() + top * 0.03

        ax.text(bar.get_x() + bar.get_width() / 2,
                y_lab, label,
                ha="center", va="bottom",
                fontsize=FONT_LABEL + 1,
                color=(POLICY_COLOR[POLICIES[i]] if is_best else "black"),
                fontweight="bold",
                zorder=4)
        # Highlight the best bar with a thicker dark outline.
        if is_best:
            bar.set_linewidth(1.4)
            bar.set_edgecolor("#222")

    ax.set_xticks(xs)
    ax.set_xticklabels([POLICY_LABEL[p] for p in POLICIES],
                       fontsize=FONT_LABEL, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=FONT_LABEL + 1, fontweight="bold")
    ax.set_title(title, fontsize=FONT_TITLE + 1, fontweight="bold")
    if log:
        ax.set_yscale("log")
        ax.set_ylim(top=max(vals) * 4.0, bottom=min(vals) * 0.6)
    else:
        if max(vals) > 0:
            ax.set_ylim(0, max(vals) * 1.30)
    ax.grid(axis="y", alpha=0.5)
    ax.grid(axis="x", visible=False)
    set_spines(ax)


def make_figure(metrics: dict, out_dir: Path, out_name: str,
                title_suffix: str):
    fig, axes = plt.subplots(2, 2, figsize=(WIDE_FIGSIZE[0], 5.2),
                             sharex=False)

    _bar_panel(axes[0, 0], metrics, "throughput",
               "(a) Throughput",
               "Requests / second",
               value_fmt=lambda v: f"{v:.2f}",
               winning_direction="higher")

    _bar_panel(axes[0, 1], metrics, "e2e_p99",
               "(b) E2E latency p99",
               "ms (log scale)",
               log=True,
               value_fmt=lambda v: f"{v:.0f}",
               winning_direction="lower")

    _bar_panel(axes[1, 0], metrics, "admitted",
               "(c) Admitted requests  (aggression)",
               "count",
               value_fmt=lambda v: f"{int(v)}",
               winning_direction="higher")

    _bar_panel(axes[1, 1], metrics, "tokens_recomp",
               "(d) Tokens recomputed  (wasted work)",
               "count",
               value_fmt=lambda v: f"{int(v):,}",
               winning_direction="lower")

    # Bottom legend with policy color key.
    handles = [plt.Rectangle((0, 0), 1, 1,
                             facecolor=POLICY_COLOR[p],
                             edgecolor="black", linewidth=0.4)
               for p in POLICIES]
    labels = [p for p in POLICIES]
    leg = fig.legend(handles, labels, loc="lower center",
                     ncol=len(POLICIES),
                     bbox_to_anchor=(0.5, -0.02))
    bold_legend(leg)

    fig.suptitle(
        f"KV allocation policy comparison  {title_suffix}",
        fontsize=FONT_TITLE + 2, fontweight="bold",
    )

    save_fig(fig, out_name, out_dir)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    ap.add_argument("--slo", type=int, default=DEFAULT_SLO,
                    help="SLO filter for per-cell layouts (matches "
                         "subdirs containing slo<N>). Default 500.")
    ap.add_argument("--out-dir", type=Path,
                    default=REPO / "thesis_plotting" / "figures")
    ap.add_argument("--out-name", type=str,
                    default="fig_kv_policy_compare")
    ap.add_argument("--title-suffix", type=str,
                    default="(OpenVLA, Azure conv, n=2000, scale=3.0, "
                            "SLO=500ms — moderate-load regime)")
    args = ap.parse_args()

    if not args.run_dir.is_dir():
        sys.exit(f"[error] run dir not found: {args.run_dir}")

    configure_plotting()
    metrics = load_metrics(args.run_dir, slo_filter=args.slo)
    if not metrics:
        sys.exit("[error] no policy metrics parsed")

    print(f"[info] policies: {list(metrics.keys())}")
    for p in POLICIES:
        m = metrics.get(p, {})
        print(f"  {p:>22}  thru={m.get('throughput', 0):5.2f}  "
              f"e2e_p99={m.get('e2e_p99', 0):8.0f}  "
              f"admitted={m.get('admitted', 0):4d}  "
              f"recomp={m.get('tokens_recomp', 0):6d}")

    make_figure(metrics, args.out_dir, args.out_name, args.title_suffix)


if __name__ == "__main__":
    main()
