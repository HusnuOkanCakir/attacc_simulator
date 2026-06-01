#!/usr/bin/env python3
"""Thesis figure: RPS knee curve — E2E p99 vs offered RPS.

Reads each cell of an RPS-sweep dir (one cell per
(dataset, arrival_scale) combination) and plots E2E p99 (log y)
against effective RPS (log x). The "knee" — the smallest RPS where
p99 starts rising super-linearly — defines the system's sustainable
peak `a`.

Cell labels expected: `<NN>_<dataset>_s<NNN>` where dataset is one of
`vla_poisson`, `azure_poisson`, `azure_empirical`, and the 3-digit
suffix is arrival_scale × 10 (`s100` = 10.0 ... `s001` = 0.1).
Produced by tools/run_rps_knee_sweep.sbatch.

Output: thesis_plotting/figures/fig_rps_knee.{pdf,png}
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO))

from plot_sweep_compare import parse_summary  # noqa: E402

from thesis_plotting.style import (  # noqa: E402
    configure_plotting, save_fig, set_spines, bold_legend,
    COLOR_MODEL, COLOR_CATEGORY, COLOR_HEAVY,
    STD_FIGSIZE, WIDE_FIGSIZE,
    FONT_ANNOTATE, FONT_LABEL, FONT_TITLE,
)


LABEL_RE = re.compile(
    r"^(?:\d+_)?(?P<dataset>"
    # Longer names FIRST so 'azure_poisson_wide' matches before 'azure_poisson'.
    r"azure_poisson_wide|azure_empirical|azure_full19k|"
    r"vla_poisson|azure_poisson|azure_real"
    r")_s(?P<sx10>\d+)$"
)
DATASET_DISPLAY = {
    "vla_poisson":        "VLA  (Poisson shape)",
    "azure_poisson":      "Azure  (Poisson, narrow)",
    "azure_poisson_wide": "Azure  (Poisson, wide mixture)",
    "azure_empirical":    "Azure  (empirical CDF)",
    "azure_real":         "Azure  (real, first 5k)",
    "azure_full19k":      "Azure  (real, full 19k)",
}
DATASET_COLOR = {
    "vla_poisson":        COLOR_HEAVY[0],   # deep teal
    "azure_poisson":      COLOR_HEAVY[3],   # amber
    "azure_poisson_wide": COLOR_HEAVY[2],   # green
    "azure_empirical":    COLOR_HEAVY[1],   # rust
    "azure_real":         COLOR_HEAVY[4],   # purple — ground-truth (5k)
    "azure_full19k":      COLOR_HEAVY[5],   # neutral grey — full 19k
}
DATASET_MARKER = {
    "vla_poisson":        "o",
    "azure_poisson":      "s",
    "azure_poisson_wide": "X",
    "azure_empirical":    "^",
    "azure_real":         "D",              # diamond — ground-truth
    "azure_full19k":      "P",              # plus — full ground-truth
}


def parse_label(label: str):
    m = LABEL_RE.match(label)
    if not m:
        return None
    return m.group("dataset"), int(m.group("sx10")) / 10.0


def cell_metrics(sweep_dir: Path):
    """Yield (preset, scale, effective_rps, e2e_p99_ms) per cell."""
    for sub in sorted(sweep_dir.iterdir()):
        if not sub.is_dir():
            continue
        parsed = parse_label(sub.name)
        if not parsed:
            continue
        dataset, scale = parsed

        sumfile = sub / "policy_compare_summary.txt"
        if not sumfile.is_file():
            # fallback: <sub>/<policy>/policy_compare_summary.txt
            inner = next((c / "policy_compare_summary.txt"
                          for c in sub.iterdir() if c.is_dir()), None)
            if inner and inner.is_file():
                sumfile = inner
        if not sumfile.is_file():
            print(f"[warn] no summary in {sub.name}", file=sys.stderr)
            continue

        m = parse_summary(sumfile)
        if not m or m.get("e2e_p99") is None or m.get("span_ms") is None:
            continue
        # n_admitted as proxy for n_requests served
        n = (m.get("admitted") or 0) + (m.get("dropped") or 0)
        if n <= 0:
            continue
        span_s = m["span_ms"] / 1000.0
        if span_s <= 0:
            continue
        # The sim already plays back at scaled timestamps, so
        # span_s ≈ original_span × scale. Effective offered RPS over the
        # scaled span is simply n / span_s.
        eff_rps = n / span_s
        yield dataset, scale, eff_rps, m["e2e_p99"], m


def group_by_dataset(records):
    grouped: dict[str, list[tuple]] = {}
    for dataset, scale, rps, p99, _m in records:
        grouped.setdefault(dataset, []).append((rps, p99, scale))
    for k in grouped:
        grouped[k].sort()  # sort by RPS ascending
    return grouped


def detect_saturation_cluster(points: list[tuple[float, float, float]],
                               rps_tol: float = 0.05):
    """Return the set of points belonging to the saturation cluster — the
    group of HIGHEST-RPS points whose RPS values are within `rps_tol`
    (relative) of each other. These appear visually as a vertical stack
    because the system caps throughput while p99 keeps climbing.

    Returns (cluster_points, non_cluster_points). Cluster is empty if no
    saturation is detected (single max-RPS point).
    """
    if len(points) < 2:
        return [], list(points)
    max_rps = max(p[0] for p in points)
    if max_rps <= 0:
        return [], list(points)
    cluster = [p for p in points
               if abs(p[0] - max_rps) / max_rps <= rps_tol]
    non_cluster = [p for p in points if p not in cluster]
    if len(cluster) < 2:
        return [], list(points)
    return cluster, non_cluster


def resolve_knee(dataset: str, points: list,
                 overrides: dict[str, float] | None = None) -> tuple[float, float] | None:
    """Use a manual override if present; otherwise auto-detect.
    Override format: {dataset_slug: rps_value}. The override picks the
    measured cell whose RPS is closest to the requested value so the
    star sits on a real data point, not a synthetic one.
    """
    if overrides and dataset in overrides:
        target = overrides[dataset]
        # Cells that share an RPS (e.g. a saturated cluster) have
        # slightly different float RPS values from divergent
        # span_s. Use a 5% relative window so all "same-RPS" cells
        # are candidates, then prefer the one with the LOWEST p99 —
        # that's the cliff-start cell, not a later saturated sibling.
        tol = max(0.05, 0.05 * target)
        near = [p for p in points if abs(p[0] - target) <= tol]
        if near:
            closest = min(near, key=lambda p: p[1])
        else:
            closest = min(points, key=lambda p: abs(p[0] - target))
        return (closest[0], closest[1])
    return detect_knee(points)


def detect_knee(points: list[tuple[float, float, float]],
                rise_factor: float = 2.0) -> tuple[float, float] | None:
    """Return (rps_at_knee, p99_at_knee).

    Primary strategy: identify the throughput-saturation cluster (where
    multiple consecutive points share the same RPS within tolerance)
    and pick the LAST point BEFORE that cluster as the knee. This is
    visually where the curve transitions from linear-on-log to vertical.

    Fallback (when no saturation cluster exists): smallest RPS where
    p99 ≥ `rise_factor` × low-RPS plateau.
    """
    if len(points) < 3:
        return None
    cluster, non_cluster = detect_saturation_cluster(points)
    if cluster and non_cluster:
        # Last point below the saturation cluster (highest non-saturated RPS).
        knee_pt = max(non_cluster, key=lambda p: p[0])
        return (knee_pt[0], knee_pt[1])
    # Fallback heuristic.
    rps_sorted = [p[0] for p in points]
    p99_sorted = [p[1] for p in points]
    half = max(2, len(points) // 2)
    plateau = float(np.median(p99_sorted[:half]))
    threshold = plateau * rise_factor
    for rps, p99, _ in points:
        if p99 >= threshold:
            return (rps, p99)
    return None


def _make_combined_linear(grouped: dict, out_dir: Path, out_name: str,
                          manual_overrides: dict[str, float] | None = None):
    """Combined figure on LINEAR axes — one subplot per dataset because
    the RPS ranges differ by ~10× across datasets."""
    n = len(grouped)
    fig, axes = plt.subplots(1, n,
                             figsize=(4.2 * n, 4.0),
                             squeeze=False)
    axes = axes[0]
    knee_for_legend = []

    for ax, (dataset, points) in zip(axes, grouped.items()):
        points = sorted(points)
        rps = [p[0] for p in points]
        p99_s = [p[1] / 1000.0 for p in points]
        color = DATASET_COLOR[dataset]
        marker = DATASET_MARKER[dataset]

        # Cluster vs non-cluster
        cluster, non_cluster = detect_saturation_cluster(points)
        for r, p99_ms, _ in non_cluster:
            ax.scatter([r], [p99_ms / 1000.0],
                       marker=marker, s=58, zorder=5,
                       color=color, edgecolor="black", linewidth=0.5)
        for r, p99_ms, _ in cluster:
            ax.scatter([r], [p99_ms / 1000.0],
                       marker=marker, s=36, zorder=4,
                       facecolor="white", edgecolor=color, linewidth=1.0)
        ax.plot(rps, p99_s, linewidth=1.4, color=color, alpha=0.55,
                zorder=3)

        knee = resolve_knee(dataset, points, manual_overrides)
        if knee:
            knee_rps, knee_p99_ms = knee
            knee_p99_s = knee_p99_ms / 1000.0
            ax.scatter([knee_rps], [knee_p99_s],
                       marker="*", s=300, zorder=6,
                       color=color, edgecolor="black", linewidth=0.9)
            ax.annotate(
                f"a ≈ {knee_rps:.1f} req/s",
                xy=(knee_rps, knee_p99_s),
                xytext=(-10, 20), textcoords="offset points",
                ha="right", va="bottom",
                fontsize=FONT_ANNOTATE + 2,
                color=color, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor=color,
                          boxstyle="round,pad=0.25", linewidth=0.6,
                          alpha=0.95),
                arrowprops=dict(arrowstyle="->", color=color, lw=0.7))
            knee_for_legend.append((dataset, knee_rps))

        # Cap y-axis so the saturation cluster doesn't dominate
        # (its values are 100x or more above the knee).
        if knee:
            ax.set_ylim(0, knee[1] / 1000.0 * 2.5)
        ax.set_xlim(0, max(rps) * 1.1)
        ax.set_xlabel("Offered RPS  (req/s)",
                      fontsize=FONT_LABEL, fontweight="bold")
        ax.set_ylabel("E2E p99 (s)",
                      fontsize=FONT_LABEL, fontweight="bold")
        ax.set_title(f"{DATASET_DISPLAY[dataset]}",
                     fontsize=FONT_TITLE, fontweight="bold")
        ax.grid(True, alpha=0.5)
        set_spines(ax)

    fig.suptitle("Peak-RPS knee  (linear axes — y capped at 2.5× knee p99)",
                 fontsize=FONT_TITLE + 1, fontweight="bold", y=1.02)

    save_fig(fig, out_name, out_dir)
    plt.close(fig)
    return knee_for_legend


def make_figure(grouped: dict, sweep_dir: Path,
                out_dir: Path, out_name: str,
                linear: bool = False,
                manual_overrides: dict[str, float] | None = None):
    """Combined knee figure. With linear=False (default), uses log-log
    axes — works for the whole RPS range (0.5–25) in one panel.
    With linear=True, generates a multi-panel figure (one subplot per
    dataset) because the linear axes can't sensibly accommodate VLA
    (RPS up to 25) and Azure (RPS up to 2.6) in the same panel."""
    if linear:
        return _make_combined_linear(grouped, out_dir, out_name, manual_overrides)
    fig, ax = plt.subplots(figsize=(WIDE_FIGSIZE[0] * 0.85, 5.0))

    knee_for_legend = []
    # Stagger annotation offsets per dataset so they don't pile up.
    annot_offsets = [(-18, -28), (-18, 22), (22, -28), (22, 22), (-50, 0)]

    for idx, (dataset, points) in enumerate(grouped.items()):
        # Sort + dedupe markers for cluster vs non-cluster.
        points = sorted(points)
        rps    = [p[0] for p in points]
        p99_s  = [p[1] / 1000.0 for p in points]
        color  = DATASET_COLOR[dataset]
        marker = DATASET_MARKER[dataset]
        label  = DATASET_DISPLAY[dataset]
        ax.plot(rps, p99_s,
                marker=marker, linewidth=1.5, markersize=5,
                color=color, label=label, zorder=3,
                markeredgecolor="black", markeredgewidth=0.4)

        # Annotate the knee (placed using staggered offset).
        knee = resolve_knee(dataset, points, manual_overrides)
        if knee:
            knee_rps, knee_p99_ms = knee
            knee_p99_s = knee_p99_ms / 1000.0
            ax.scatter([knee_rps], [knee_p99_s],
                       marker="*", s=260, zorder=6,
                       color=color, edgecolor="black", linewidth=0.8)
            dx, dy = annot_offsets[idx % len(annot_offsets)]
            ha = "right" if dx < 0 else "left"
            va = "top" if dy < 0 else "bottom"
            ax.annotate(
                f"a ≈ {knee_rps:.1f} req/s",
                xy=(knee_rps, knee_p99_s),
                xytext=(dx, dy), textcoords="offset points",
                ha=ha, va=va,
                fontsize=FONT_ANNOTATE + 2,
                color=color, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor=color,
                          boxstyle="round,pad=0.25", linewidth=0.6,
                          alpha=0.95),
                arrowprops=dict(arrowstyle="->", color=color, lw=0.7))
            knee_for_legend.append((dataset, knee_rps))

    ax.set_xscale("log")
    ax.set_yscale("log")
    fmt = mticker.FuncFormatter(lambda x, _p: f"{x:g}")
    ax.xaxis.set_major_formatter(fmt)

    ax.set_xlabel("Offered RPS  (effective requests / second)",
                  fontsize=FONT_LABEL, fontweight="bold")
    ax.set_ylabel("E2E p99 latency  (s, log scale)",
                  fontsize=FONT_LABEL, fontweight="bold")
    ax.set_title("Peak-RPS knee — where the system saturates",
                 fontsize=FONT_TITLE + 1, fontweight="bold")
    ax.grid(True, which="both", alpha=0.5)
    set_spines(ax)

    leg = ax.legend(loc="upper left", title="Dataset",
                    fontsize=FONT_LABEL)
    bold_legend(leg)
    if leg.get_title() is not None:
        leg.get_title().set_fontweight("bold")

    save_fig(fig, out_name, out_dir)
    plt.close(fig)

    return knee_for_legend


def make_per_dataset_figure(dataset: str, points: list,
                             out_dir: Path, out_name: str,
                             linear: bool = False,
                             manual_overrides: dict[str, float] | None = None):
    """Single-dataset knee figure — one line, one knee marker, clear
    saturation-cluster annotation."""
    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    # Sort points by RPS for line ordering
    points = sorted(points)
    rps    = [p[0] for p in points]
    p99_s  = [p[1] / 1000.0 for p in points]
    color  = DATASET_COLOR[dataset]
    marker = DATASET_MARKER[dataset]

    # Identify saturation cluster + non-saturated points
    cluster, non_cluster = detect_saturation_cluster(points)
    cluster_rps_set = {p[0] for p in cluster}

    # Plot the line in full (visually shows the trajectory) but mark
    # cluster vs non-cluster points differently.
    ax.plot(rps, p99_s,
            linewidth=1.6, color=color, alpha=0.55, zorder=3)

    # Non-cluster points = regular markers with scale label
    for r, p99_ms, scale in non_cluster:
        ax.scatter([r], [p99_ms / 1000.0],
                   marker=marker, s=64, zorder=5,
                   color=color, edgecolor="black", linewidth=0.5)
        ax.annotate(f"s={scale:g}",
                    xy=(r, p99_ms / 1000.0),
                    xytext=(6, -3), textcoords="offset points",
                    fontsize=FONT_ANNOTATE, color="#444",
                    ha="left", va="center", fontweight="bold")

    # Cluster points = smaller hollow markers (visually grouped)
    if cluster:
        for r, p99_ms, _scale in cluster:
            ax.scatter([r], [p99_ms / 1000.0],
                       marker=marker, s=38, zorder=4,
                       facecolor="white", edgecolor=color, linewidth=1.0)
        # Single annotation for the whole cluster.
        cluster_scales = sorted({p[2] for p in cluster}, reverse=True)
        scales_str = ", ".join(f"{s:g}" for s in cluster_scales[:4])
        if len(cluster_scales) > 4:
            scales_str += ", …"
        cluster_rps = cluster[0][0]
        cluster_p99_max = max(p[1] for p in cluster) / 1000.0
        cluster_p99_min = min(p[1] for p in cluster) / 1000.0
        ax.annotate(
            f"saturated cluster\n({len(cluster)} cells: s ∈ {{{scales_str}}})\n"
            f"throughput stuck at {cluster_rps:.2f} req/s",
            xy=(cluster_rps, cluster_p99_min),
            xytext=(-90, -30),
            textcoords="offset points",
            ha="right", va="top",
            fontsize=FONT_ANNOTATE + 1, color="#444",
            arrowprops=dict(arrowstyle="-", color="#888", lw=0.5),
            bbox=dict(facecolor="white", edgecolor="#888",
                      boxstyle="round,pad=0.3", linewidth=0.5,
                      alpha=0.95))

    # Knee marker + annotation (placed below-left of point to clear title)
    knee = resolve_knee(dataset, points, manual_overrides)
    knee_str = ""
    if knee:
        knee_rps, knee_p99_ms = knee
        knee_p99_s = knee_p99_ms / 1000.0
        ax.scatter([knee_rps], [knee_p99_s],
                   marker="*", s=320, zorder=6,
                   color=color, edgecolor="black", linewidth=0.9)
        ax.annotate(
            f"a ≈ {knee_rps:.1f} req/s",
            xy=(knee_rps, knee_p99_s),
            xytext=(-15, -28), textcoords="offset points",
            ha="right", va="top",
            fontsize=FONT_ANNOTATE + 3,
            color=color, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor=color,
                      boxstyle="round,pad=0.3", linewidth=0.7,
                      alpha=0.95),
            arrowprops=dict(arrowstyle="->", color=color, lw=0.9))
        knee_str = f"  (a ≈ {knee_rps:.1f} req/s)"

    if linear:
        # Linear axes: cap y so the saturation cluster doesn't compress
        # the visible curve into the bottom 1%.
        if knee:
            ax.set_ylim(0, knee[1] / 1000.0 * 3.0)
        ax.set_xlim(0, max(rps) * 1.1)
        x_label = "Offered RPS  (effective req / s)"
        y_label = "E2E p99  (s)"
        scale_note = "linear axes — y capped at 3× knee p99"
    else:
        ax.set_xscale("log")
        ax.set_yscale("log")
        fmt = mticker.FuncFormatter(lambda x, _p: f"{x:g}")
        ax.xaxis.set_major_formatter(fmt)
        x_label = "Offered RPS  (effective req / s)"
        y_label = "E2E p99  (s, log scale)"
        scale_note = "s = arrival_scale; a = knee/sustainable peak RPS"

    ax.set_xlabel(x_label, fontsize=FONT_LABEL + 1, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=FONT_LABEL + 1, fontweight="bold")
    ax.set_title(
        f"{DATASET_DISPLAY[dataset]} — peak-RPS knee\n({scale_note})",
        fontsize=FONT_TITLE + 1, fontweight="bold")
    ax.grid(True, which="both" if not linear else "major", alpha=0.5)
    set_spines(ax)

    save_fig(fig, out_name, out_dir)
    plt.close(fig)
    return knee_str


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-dir", type=Path, required=True,
                    help="Sweep dir from tools/run_rps_knee_sweep.sbatch")
    ap.add_argument("--out-dir", type=Path,
                    default=REPO / "thesis_plotting" / "figures")
    ap.add_argument("--out-name", type=str, default="fig_rps_knee")
    ap.add_argument("--no-per-dataset", action="store_true",
                    help="Skip the individual per-dataset figures (only "
                         "produce the combined one).")
    ap.add_argument("--linear", action="store_true",
                    help="Use linear axes instead of log-log. The "
                         "combined figure becomes a multi-panel layout "
                         "(one subplot per dataset) because RPS ranges "
                         "differ ~10x across datasets. Output suffix: "
                         "fig_rps_knee_linear.{pdf,png}.")
    ap.add_argument("--manual-knee", action="append", default=[],
                    metavar="DATASET=RPS",
                    help="Override the auto-detected knee for one dataset. "
                         "Format: dataset_slug=rps_value. Repeat for multiple "
                         "datasets. Used when the saturation-cluster heuristic "
                         "picks the wrong point (e.g. when cliff-start and "
                         "throughput ceiling coincide).")
    args = ap.parse_args()

    if not args.sweep_dir.is_dir():
        sys.exit(f"[error] sweep dir not found: {args.sweep_dir}")

    configure_plotting()

    records = list(cell_metrics(args.sweep_dir))
    if not records:
        sys.exit("[error] no parseable cells found")

    grouped = group_by_dataset(records)
    print(f"[info] {sum(len(v) for v in grouped.values())} cells, "
          f"{len(grouped)} datasets")
    for dataset, points in grouped.items():
        print(f"  {DATASET_DISPLAY[dataset]}:")
        for rps, p99, scale in points:
            print(f"    scale={scale:>5.2f}  rps={rps:>6.2f}  e2e_p99={p99:>10.0f}ms")

    # Parse --manual-knee dataset=rps[,dataset=rps...] into a dict.
    manual_overrides = {}
    for item in args.manual_knee:
        if "=" not in item:
            sys.exit(f"[error] --manual-knee expects DATASET=RPS, got {item!r}")
        dset, val = item.split("=", 1)
        manual_overrides[dset.strip()] = float(val.strip())
    if manual_overrides:
        print(f"[manual knee overrides] {manual_overrides}")

    out_name = args.out_name + ("_linear" if args.linear else "")
    knees = make_figure(grouped, args.sweep_dir, args.out_dir, out_name,
                        linear=args.linear,
                        manual_overrides=manual_overrides)
    print()
    print("[knee detection]")
    for dataset, knee_rps in knees:
        print(f"  {DATASET_DISPLAY[dataset]:<28}  a ≈ {knee_rps:.2f} req/s")

    if not args.no_per_dataset:
        print()
        print("[per-dataset figures]")
        for dataset, points in grouped.items():
            suffix = "_linear" if args.linear else ""
            name = f"{args.out_name}_{dataset}{suffix}"
            make_per_dataset_figure(dataset, points, args.out_dir, name,
                                     linear=args.linear,
                                     manual_overrides=manual_overrides)


if __name__ == "__main__":
    main()
