#!/usr/bin/env python3
"""Knee curve under varied system knobs.

Companion to tools/run_knee_vs_knob_sweep.sbatch. Discovers cells
labelled `NN_<knob>_<value>_s<scale*10>`, groups by knob value, and
draws one log-log E2E-p99 vs effective-RPS curve per knob value with
the knee star annotated. Output: fig_knee_vs_<knob>.{pdf,png} (or
_linear variants).

Usage:
    python thesis_plotting/scripts/thesis_knee_vs_knob.py \\
        --sweep-dir cluster_outputs/online_serving_runs/knee_vs_bs_<TS> \\
        --knob bs

    # Linear-axes variant (same matrix):
    python thesis_plotting/scripts/thesis_knee_vs_knob.py \\
        --sweep-dir <...> --knob bs --linear
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO))

from plot_sweep_compare import parse_summary  # noqa: E402

from thesis_plotting.style import (  # noqa: E402
    configure_plotting, save_fig, set_spines, bold_legend,
    COLOR_HEAVY, FONT_LABEL, FONT_TITLE, FONT_TICK,
)


# Per-knob display labels for the figure title + y-units. Add new knobs
# here as the corresponding sweeps land.
KNOB_DISPLAY = {
    "bs":            ("Batch size",                "max_decode_batch_size"),
    "kvpolicy":      ("KV allocation policy",      "policy"),
    "kvoom":         ("KV OOM policy",             "kv_oom_policy"),
    "holdms":        ("KV OOM hold limit",         "ms"),
    "route":         ("Route mode",                "route"),
    "slo":           ("SLO E2E target",            "ms"),
    "maxactive":     ("max_active",                "concurrent requests"),
    "maxconsec":     ("max_consecutive_decode_batches", "consecutive decode"),
    "admbudget":     ("Admission violation budget", "X"),
    "promptprio":    ("Prompt priority",           "on/off"),
    "energyguard":   ("Energy-latency guard",      "ms"),
}

# Per-knob value display labels (where the safe-label encoded form
# would otherwise be cryptic). Keys are the safe-label encodings from
# the sbatch.
VALUE_DISPLAY = {
    "gne": "guaranteed_no_evict",
    "muf": "max_util_full",
    "mut": "max_util_tail",
    "htf": "hold_then_fallback",
    "fbg": "fallback_gpu",
    "hold": "hold",
    "hybrid": "hybrid",
    "gpu_only": "gpu-only",
    "on": "on",
    "off": "off",
}


LABEL_RE = re.compile(
    r"^(?:\d+_)?(?P<knob>[a-zA-Z]+)_(?P<value>[^_]+(?:_[^_]+)*?)"
    r"_s(?P<sx10>\d+)$"
)


def cell_metrics(sweep_dir: Path):
    """Yield (knob_value, rps, e2e_p99_ms, scale) for each cell."""
    for sub in sorted(sweep_dir.iterdir()):
        if not sub.is_dir():
            continue
        m = LABEL_RE.match(sub.name)
        if not m:
            continue
        knob_value = m.group("value")
        scale = int(m.group("sx10")) / 10.0
        # Per-cell summary lives at <sub>/policy_compare_summary.txt
        # or one level deeper if the runner emitted it inside the
        # policy subdir.
        sumfile = sub / "policy_compare_summary.txt"
        if not sumfile.is_file():
            for child in sub.iterdir():
                if child.is_dir():
                    cand = child / "policy_compare_summary.txt"
                    if cand.is_file():
                        sumfile = cand
                        break
        if not sumfile.is_file():
            continue
        metrics = parse_summary(sumfile)
        rps = metrics.get("throughput")
        e2e_p99 = metrics.get("e2e_p99")
        if rps is None or e2e_p99 is None:
            continue
        yield knob_value, rps, e2e_p99, scale


def group_by_knob_value(records):
    out = {}
    for kv, rps, p99, scale in records:
        out.setdefault(kv, []).append((rps, p99, scale))
    for kv in out:
        out[kv].sort()
    return out


def detect_saturation(points, rps_tol=0.02):
    """Identify the saturation cluster (consecutive points sharing RPS).
    Returns (cluster_pts, non_cluster_pts)."""
    if len(points) < 3:
        return [], list(points)
    sorted_pts = sorted(points)
    rps_vals = [p[0] for p in sorted_pts]
    max_rps = max(rps_vals)
    # Any point within `rps_tol` of max_rps is in the cluster.
    cluster = [p for p in sorted_pts if abs(p[0] - max_rps) <= max_rps * rps_tol]
    if len(cluster) < 2:
        return [], sorted_pts
    non_cluster = [p for p in sorted_pts if p not in cluster]
    return cluster, non_cluster


def detect_knee(points):
    """Knee = last non-cluster point with highest RPS. Returns
    (rps, p99) or None."""
    cluster, non_cluster = detect_saturation(points)
    if cluster and non_cluster:
        knee_pt = max(non_cluster, key=lambda p: p[0])
        return (knee_pt[0], knee_pt[1])
    # Fallback: the highest-RPS point.
    if points:
        knee_pt = max(points, key=lambda p: p[0])
        return (knee_pt[0], knee_pt[1])
    return None


def render(grouped: dict, knob_name: str, out_dir: Path, out_name: str,
           linear: bool = False):
    configure_plotting()
    fig, ax = plt.subplots(figsize=(9.0, 5.2))

    title_word, _unit = KNOB_DISPLAY.get(knob_name, (knob_name, ""))

    # Stable ordering of knob values: try numeric ascending first, fall
    # back to alphabetical.
    def _sort_key(v):
        # Handle "mN" prefix (negative ints).
        try:
            if v.startswith("m") and v[1:].isdigit():
                return -int(v[1:])
            return float(v)
        except (ValueError, TypeError):
            return float("inf")

    values_sorted = sorted(grouped.keys(),
                           key=lambda v: (_sort_key(v), v))

    colors = COLOR_HEAVY * 3
    markers = ["o", "s", "^", "D", "X", "P", "v", "<", ">", "*", "h"]

    for i, value in enumerate(values_sorted):
        points = grouped[value]
        rps = [p[0] for p in points]
        p99_s = [p[1] / 1000.0 for p in points]
        display = VALUE_DISPLAY.get(value, value)
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        ax.plot(rps, p99_s, marker=marker, color=color, lw=2.0, ms=9,
                markeredgecolor="black", markeredgewidth=0.5,
                label=display, zorder=4 + i)

        knee = detect_knee(points)
        if knee:
            knee_rps, knee_p99_ms = knee
            knee_p99_s = knee_p99_ms / 1000.0
            ax.scatter([knee_rps], [knee_p99_s], marker="*", s=220,
                       color=color, edgecolor="black", linewidth=0.6,
                       zorder=10)

    if not linear:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel("Offered RPS  (effective req / s)",
                  fontsize=FONT_LABEL, fontweight="bold")
    ax.set_ylabel("E2E p99  (s, log)" if not linear else "E2E p99  (s)",
                  fontsize=FONT_LABEL, fontweight="bold")
    ax.set_title(f"Pi0 @ azure_poisson_wide v2 — knee vs {title_word}",
                 fontsize=FONT_TITLE, fontweight="bold")
    ax.grid(True, alpha=0.4)
    bold_legend(ax.legend(loc="upper left", fontsize=FONT_LABEL,
                          frameon=True, ncol=1,
                          title=title_word,
                          title_fontsize=FONT_LABEL))
    set_spines(ax)
    fig.tight_layout()
    save_fig(fig, out_name, out_dir)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep-dir", type=Path, required=True)
    ap.add_argument("--knob", type=str, required=True,
                    help="Knob short name (e.g. bs, kvpolicy, slo).")
    ap.add_argument("--out-dir", type=Path,
                    default=REPO / "thesis_plotting/figures")
    ap.add_argument("--out-name", type=str, default=None)
    ap.add_argument("--linear", action="store_true")
    args = ap.parse_args()

    if not args.sweep_dir.is_dir():
        sys.exit(f"[error] sweep dir not found: {args.sweep_dir}")

    records = list(cell_metrics(args.sweep_dir))
    if not records:
        sys.exit("[error] no parseable cells found")

    grouped = group_by_knob_value(records)
    print(f"[info] {len(records)} cells, {len(grouped)} knob values")
    for v, pts in sorted(grouped.items()):
        print(f"  {v}: {len(pts)} points")
        for rps, p99, scale in pts:
            print(f"    s={scale:>5.2f}  rps={rps:>6.2f}  "
                  f"e2e_p99={p99:>10.0f}ms")

    out_name = args.out_name or f"fig_knee_vs_{args.knob}"
    if args.linear:
        out_name = f"{out_name}_linear"
    render(grouped, args.knob, args.out_dir, out_name, linear=args.linear)


if __name__ == "__main__":
    main()
