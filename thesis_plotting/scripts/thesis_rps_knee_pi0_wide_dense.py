#!/usr/bin/env python3
"""Pi0 + azure_poisson_wide v2 — fine-grained knee curve.

Merges two sweep dirs:
- `rps_knee_pi0_20260601_175358` — dense 12-point sweep (scales 0.5–10).
- `rps_knee_pi0_20260601_183211` — metastability sweep at scales
  2.40 / 2.45 / 2.50 / 2.55 / 2.60 / 2.65 / 2.70 (the discontinuous
  bistability window).

Reads each cell's `run_info.txt` for the true `arrival_scale` (the
sweep label is truncated to int(scale*10), which can't disambiguate
2.6 from 2.65). All data points are plotted; the bistability window
is colored distinctly so the DRAIN ↔ OVERFLOW flip is visible.

Output: thesis_plotting/figures/fig_rps_knee_pi0_wide_azure_poisson_wide.{pdf,png}
(overwrites the existing single-source rendering).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO))

from plot_sweep_compare import parse_summary  # noqa: E402

from thesis_plotting.style import (  # noqa: E402
    configure_plotting, save_fig, set_spines, bold_legend,
    COLOR_HEAVY, FONT_LABEL, FONT_TITLE, FONT_ANNOTATE,
)


SWEEP_DENSE = (REPO / "cluster_outputs/online_serving_runs"
               / "rps_knee_pi0_20260601_175358")
SWEEP_META = (REPO / "cluster_outputs/online_serving_runs"
              / "rps_knee_pi0_20260601_183211")

DATASET = "azure_poisson_wide"
DISPLAY = "Pi0 + Azure (Poisson, wide-mixture v2)"

# Color the bistability window (these scales hit the OVERFLOW or partial
# recovery attractors instead of the DRAIN ceiling).
META_SCALES = {2.40, 2.45, 2.50, 2.55, 2.60, 2.65, 2.70}
OVERFLOW_SCALES = {2.60, 2.65}        # rps collapses to ~1.25
PARTIAL_SCALES = {2.55}                # rps ~1.53

COLOR_LINE     = COLOR_HEAVY[2]        # green — main dense curve
COLOR_DRAIN    = COLOR_HEAVY[2]        # green — DRAIN points
COLOR_OVERFLOW = COLOR_HEAVY[1]        # rust  — OVERFLOW attractor
COLOR_PARTIAL  = COLOR_HEAVY[3]        # amber — partial recovery


def read_cell(cell_dir: Path):
    """Returns (scale, rps, p99_ms) or None."""
    run_info = cell_dir / "run_info.txt"
    if not run_info.is_file():
        return None
    scale = None
    for line in run_info.read_text().splitlines():
        if line.startswith("arrival_scale="):
            try:
                scale = float(line.split("=", 1)[1].strip())
            except ValueError:
                pass
            break
    if scale is None:
        return None

    sumfile = cell_dir / "policy_compare_summary.txt"
    if not sumfile.is_file():
        inner = next((c / "policy_compare_summary.txt"
                      for c in cell_dir.iterdir() if c.is_dir()), None)
        if inner and inner.is_file():
            sumfile = inner
    if not sumfile.is_file():
        return None

    m = parse_summary(sumfile)
    if not m or m.get("e2e_p99") is None or m.get("span_ms") is None:
        return None
    n = (m.get("admitted") or 0) + (m.get("dropped") or 0)
    span_s = m["span_ms"] / 1000.0
    if n <= 0 or span_s <= 0:
        return None
    return scale, n / span_s, m["e2e_p99"]


def collect(sweep_dir: Path):
    pts = []
    if not sweep_dir.is_dir():
        print(f"[warn] sweep not found: {sweep_dir}", file=sys.stderr)
        return pts
    for sub in sorted(sweep_dir.iterdir()):
        if not sub.is_dir():
            continue
        if DATASET not in sub.name:
            continue
        rec = read_cell(sub)
        if rec is None:
            print(f"[warn] no data in {sub.name}", file=sys.stderr)
            continue
        pts.append(rec)
    return pts


def classify(scale):
    if scale in OVERFLOW_SCALES:
        return "overflow"
    if scale in PARTIAL_SCALES:
        return "partial"
    return "drain"


def render(points, out_dir: Path, out_name: str, linear: bool = False):
    fig, ax = plt.subplots(figsize=(9.5, 5.6))

    drain_pts = sorted([(s, r, p) for s, r, p in points if classify(s) == "drain"],
                       key=lambda t: t[1])

    # Main curve = DRAIN only, sorted by RPS — gives a clean monotonic knee.
    drain_rps  = [t[1] for t in drain_pts]
    drain_p99s = [t[2] / 1000.0 for t in drain_pts]
    ax.plot(drain_rps, drain_p99s, color=COLOR_LINE, lw=1.6, alpha=0.55,
            zorder=2)

    def scatter(group, color, marker, label, size=72):
        if not group:
            return
        xs = [g[1] for g in group]
        ys = [g[2] / 1000.0 for g in group]
        ax.scatter(xs, ys, color=color, marker=marker, s=size,
                   edgecolor="black", linewidth=0.6, label=label, zorder=5)

    scatter(drain_pts, COLOR_DRAIN, "o", "Pi0 + Azure wide-mixture v2")

    # Annotate scale next to DRAIN points only (ceiling cluster gets a
    # single representative label to avoid stacking).
    seen_ceiling = 0
    for s, r, p in drain_pts:
        if abs(r - 2.13) < 0.01:
            seen_ceiling += 1
            if seen_ceiling > 1:
                continue
        ax.annotate(f"s={s:g}",
                    xy=(r, p / 1000.0),
                    xytext=(6, -3), textcoords="offset points",
                    fontsize=FONT_ANNOTATE - 1, color="#444",
                    ha="left", va="center")

    # Knee marker — fixed at a = 2.13 (the DRAIN ceiling).
    drain_rps_only = [t[1] for t in drain_pts]
    drain_p99_s_only = [t[2] / 1000.0 for t in drain_pts]
    knee_pt = None
    if drain_rps_only:
        # The knee is the highest-RPS DRAIN point (lowest p99 at the
        # ceiling RPS — that's the cliff-start cell).
        ceiling_rps = max(drain_rps_only)
        ceiling_pts = [(r, y) for r, y in zip(drain_rps_only, drain_p99_s_only)
                       if abs(r - ceiling_rps) / ceiling_rps <= 0.05]
        knee_pt = min(ceiling_pts, key=lambda p: p[1])
    if knee_pt:
        ax.scatter([knee_pt[0]], [knee_pt[1]],
                   marker="*", s=420, zorder=7,
                   color=COLOR_DRAIN, edgecolor="black", linewidth=1.0)
        ax.annotate(
            f"a ≈ {knee_pt[0]:.2f} req/s",
            xy=knee_pt,
            xytext=(40, -8), textcoords="offset points",
            ha="left", va="top",
            fontsize=FONT_ANNOTATE + 3, fontweight="bold",
            color=COLOR_DRAIN,
            bbox=dict(facecolor="white", edgecolor=COLOR_DRAIN,
                      boxstyle="round,pad=0.3", linewidth=0.8,
                      alpha=0.95),
            arrowprops=dict(arrowstyle="->", color=COLOR_DRAIN, lw=0.9))

    if linear:
        if knee_pt:
            ax.set_ylim(0, knee_pt[1] * 3.0)
        all_rps = [p[1] for p in points]
        ax.set_xlim(0, max(all_rps) * 1.1)
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
        scale_note = "s = arrival_scale; dense 0.05-step sampling in 2.4–2.7 window"

    ax.set_xlabel(x_label, fontsize=FONT_LABEL + 1, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=FONT_LABEL + 1, fontweight="bold")
    ax.set_title(
        f"{DISPLAY} — fine-grained peak-RPS knee\n({scale_note})",
        fontsize=FONT_TITLE + 1, fontweight="bold")
    ax.grid(True, which="both" if not linear else "major", alpha=0.5)
    set_spines(ax)

    save_fig(fig, out_name, out_dir)
    plt.close(fig)


def main():
    configure_plotting()

    pts = collect(SWEEP_DENSE) + collect(SWEEP_META)
    # Dedup by (rounded) scale → take the lower-p99 point per scale.
    by_scale: dict[float, tuple] = {}
    for s, r, p in pts:
        key = round(s, 2)
        cur = by_scale.get(key)
        if cur is None or p < cur[2]:
            by_scale[key] = (key, r, p)
    points = list(by_scale.values())
    points.sort(key=lambda p: p[0])

    print(f"[info] {len(points)} unique scales")
    for s, r, p in points:
        tag = classify(s).upper()
        print(f"  scale={s:>5.2f}  rps={r:>5.2f}  e2e_p99={p:>9.0f}ms  [{tag}]")

    out_dir = REPO / "thesis_plotting/figures"
    render(points, out_dir, "fig_rps_knee_pi0_wide_azure_poisson_wide",
           linear=False)
    render(points, out_dir,
           "fig_rps_knee_pi0_wide_azure_poisson_wide_linear", linear=True)


if __name__ == "__main__":
    main()
