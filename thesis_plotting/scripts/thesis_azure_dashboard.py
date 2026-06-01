#!/usr/bin/env python3
"""ThrottLLM-style multi-panel dashboard for one simulator run.

Reads a `requests_out.csv` (the per-request CSV the runner writes
under `<sweep>/<cell>/<policy>/requests_out.csv`) and produces a
single figure with stacked time-series panels that share an x-axis.

Panels (top → bottom):

  1. **Arrival RPS**      — requests admitted per bucket
  2. **Shape (Lin, Lout)** — per-request scatter
  3. **Route mix**         — stacked PIM vs GPU per bucket
  4. **TTFT p99 (rolling)**— sliding-window p99 over completed requests
  5. **E2E p99 (rolling)** — sliding-window p99 with SLO line

Inspired by Figure 5 of the ThrottLLM paper.

Usage:
    python thesis_plotting/scripts/thesis_azure_dashboard.py \\
        --results-csv cluster_outputs/.../requests_out.csv \\
        --tag f_pi0_hybrid --bucket-ms 60000 --slo-ms 500
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from thesis_plotting.style import (  # noqa: E402
    configure_plotting, save_fig, set_spines, bold_legend,
    COLOR_MODEL, COLOR_ROUTE, COLOR_CATEGORY, WIDE_FIGSIZE,
    FONT_ANNOTATE, FONT_LABEL, FONT_TITLE,
)


def load_rows(csv_path: Path):
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                rows.append({
                    "arrival_s":      float(r["arrival_ms"]) / 1000.0,
                    "lin":            int(r["context_tokens"]),
                    "lout":           int(r["generated_tokens"]),
                    "route":          r["route"],
                    "state":          r["state"],
                    "ttft_ms":        float(r["ttft_ms"]) if r["ttft_ms"] else float("nan"),
                    "e2e_ms":         float(r["e2e_ms"]) if r["e2e_ms"] else float("nan"),
                    "completion_s":   float(r["completion_ms"]) / 1000.0
                                       if r["completion_ms"] else float("nan"),
                })
            except (ValueError, KeyError):
                continue
    return rows


def bucket_counts(values_s: list[float], bucket_s: float, last_t: float):
    """Return list of (t_start_s, count) per bucket up to last_t."""
    nb = int(math.ceil(last_t / bucket_s)) + 1
    counts = [0] * nb
    for v in values_s:
        idx = int(v / bucket_s)
        if 0 <= idx < nb:
            counts[idx] += 1
    return [(i * bucket_s, counts[i]) for i in range(nb)]


def rolling_p99(times_s: list[float], values: list[float],
                window_s: float, step_s: float, last_t: float):
    """For each window-end t, compute p99 of values whose times fall
    in [t - window_s, t]. Returns (centers_s, p99_values)."""
    paired = sorted(zip(times_s, values))
    n_steps = int(math.ceil(last_t / step_s)) + 1
    centers = []
    p99s = []
    i_lo = 0
    i_hi = 0
    for k in range(n_steps):
        t_end = k * step_s
        t_start = t_end - window_s
        while i_lo < len(paired) and paired[i_lo][0] < t_start:
            i_lo += 1
        while i_hi < len(paired) and paired[i_hi][0] <= t_end:
            i_hi += 1
        sub = [v for _, v in paired[i_lo:i_hi] if not math.isnan(v)]
        centers.append(t_end)
        if not sub:
            p99s.append(float("nan"))
            continue
        sub.sort()
        idx = max(0, min(len(sub) - 1, int(round(0.99 * (len(sub) - 1)))))
        p99s.append(sub[idx])
    return centers, p99s


def fmt_t(secs: float) -> str:
    if secs > 7200:
        return f"{secs/3600:.1f}h"
    if secs > 120:
        return f"{secs/60:.1f}m"
    return f"{secs:.0f}s"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-csv", type=Path, required=True,
                    help="requests_out.csv from a sim run.")
    ap.add_argument("--tag", type=str, required=True,
                    help="Short tag for the output filename.")
    ap.add_argument("--bucket-ms", type=int, default=60000,
                    help="Bucket width for arrival-rate + route-mix "
                         "panels (default 60 s).")
    ap.add_argument("--rolling-window-ms", type=int, default=120000,
                    help="Sliding window for p99 panels (default 120 s).")
    ap.add_argument("--rolling-step-ms", type=int, default=10000,
                    help="Stride between rolling-window samples "
                         "(default 10 s).")
    ap.add_argument("--slo-ms", type=float, default=500.0,
                    help="E2E SLO target — drawn as a dashed line.")
    ap.add_argument("--max-scatter", type=int, default=15000,
                    help="If more than N requests, subsample the "
                         "Lin/Lout scatter for plotting speed.")
    ap.add_argument("--out-dir", type=Path,
                    default=REPO / "thesis_plotting" / "figures")
    args = ap.parse_args()

    if not args.results_csv.is_file():
        sys.exit(f"[error] CSV not found: {args.results_csv}")

    configure_plotting()
    rows = load_rows(args.results_csv)
    if not rows:
        sys.exit("[error] no usable rows")

    bucket_s  = args.bucket_ms / 1000.0
    window_s  = args.rolling_window_ms / 1000.0
    step_s    = args.rolling_step_ms / 1000.0
    last_t    = max(r["arrival_s"] for r in rows) + bucket_s

    use_min = last_t > 3600
    def to_x(s):
        return s / 60.0 if use_min else s
    x_unit = "min" if use_min else "s"

    # ── Panel 1: arrival RPS vs completion RPS ─────────────────────────
    buckets_arr = bucket_counts([r["arrival_s"] for r in rows],
                                bucket_s, last_t)
    rps_axis = [to_x(t) for t, _ in buckets_arr]
    rps_vals = [c / bucket_s for _, c in buckets_arr]
    peak_rps = max(rps_vals) if rps_vals else 0.0
    mean_rps = sum(rps_vals) / max(len(rps_vals), 1)

    # Completions per bucket — cell-dependent (the SYSTEM'S response).
    done_completion_s = [r["completion_s"] for r in rows
                         if r["state"] == "done"
                         and not math.isnan(r["completion_s"])]
    buckets_done = bucket_counts(done_completion_s, bucket_s, last_t)
    thru_vals = [c / bucket_s for _, c in buckets_done]
    n_done = len(done_completion_s)
    # Matches the runner's "Throughput" metric: n_done / full span from
    # first arrival to last completion (includes post-arrival drain).
    completion_span_s = (max(done_completion_s) if done_completion_s
                         else last_t)
    sustained_thru = n_done / completion_span_s if completion_span_s > 0 else 0.0

    # ── Panel 2: shape scatter ─────────────────────────────────────────
    sample_rows = rows
    if len(rows) > args.max_scatter:
        import random
        sample_rows = random.Random(0).sample(rows, args.max_scatter)
    sample_rows = sorted(sample_rows, key=lambda r: r["arrival_s"])
    scat_t  = [to_x(r["arrival_s"]) for r in sample_rows]
    scat_lin = [r["lin"] for r in sample_rows]
    scat_lout = [r["lout"] for r in sample_rows]

    # ── Panel 3: route mix ─────────────────────────────────────────────
    pim_t = [r["arrival_s"] for r in rows
             if r["route"] == "lpddr5_pim_bank"]
    gpu_t = [r["arrival_s"] for r in rows
             if r["route"] == "gpu_only"]
    bk_pim = bucket_counts(pim_t, bucket_s, last_t)
    bk_gpu = bucket_counts(gpu_t, bucket_s, last_t)
    rm_axis = [to_x(t) for t, _ in bk_pim]
    rm_pim  = [c for _, c in bk_pim]
    rm_gpu  = [c for _, c in bk_gpu]

    # ── Panels 4/5: rolling p99 ────────────────────────────────────────
    done_rows = [r for r in rows
                 if r["state"] == "done" and not math.isnan(r["e2e_ms"])]
    ttft_centers, ttft_p99 = rolling_p99(
        [r["completion_s"] for r in done_rows],
        [r["ttft_ms"] for r in done_rows],
        window_s, step_s, last_t)
    e2e_centers, e2e_p99 = rolling_p99(
        [r["completion_s"] for r in done_rows],
        [r["e2e_ms"] for r in done_rows],
        window_s, step_s, last_t)

    # ── Render ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        5, 1, figsize=(WIDE_FIGSIZE[0], 8.2),
        sharex=True, gridspec_kw={"hspace": 0.18,
                                  "height_ratios": [1, 1.4, 1, 1.1, 1.1]},
    )
    ax_rps, ax_shape, ax_route, ax_ttft, ax_e2e = axes

    bar_w_x = (rps_axis[1] - rps_axis[0]) if len(rps_axis) > 1 else 1.0

    # (1) arrival rate (bars) + completion rate (line)
    ax_rps.bar(rps_axis, rps_vals, width=bar_w_x,
               align="edge", color="#cfd8dc",
               edgecolor="none", zorder=3, label="arrivals")
    ax_rps.plot(rps_axis, thru_vals,
                color=COLOR_ROUTE["lpddr5_pim_bank"],
                lw=1.6, zorder=4, label="completions")
    ax_rps.axhline(peak_rps, color="#c0392b", lw=0.6, ls=":", zorder=5)
    ax_rps.text(0.985, peak_rps,
                f" peak arrivals {peak_rps:.1f}",
                transform=ax_rps.get_yaxis_transform(),
                ha="right", va="bottom",
                fontsize=FONT_ANNOTATE, color="#c0392b", fontweight="bold")
    ax_rps.set_ylabel(f"RPS  ({args.bucket_ms / 1000:.0f} s buckets)",
                      fontsize=FONT_LABEL, fontweight="bold")
    ax_rps.set_title(
        f"(a) Arrival rate vs throughput    "
        f"arrivals: mean={mean_rps:.2f}, peak={peak_rps:.1f} req/s    "
        f"throughput: {sustained_thru:.2f} req/s "
        f"(n_done={n_done:,}/{len(rows):,}, "
        f"completion span={fmt_t(completion_span_s)})",
        fontsize=FONT_TITLE - 1, fontweight="bold", loc="left")
    leg = ax_rps.legend(loc="upper right", fontsize=FONT_ANNOTATE, ncol=2)
    bold_legend(leg)

    # (2) shape scatter — two y-axes (Lin left, Lout right)
    ax_shape.scatter(scat_t, scat_lin, s=2, alpha=0.35,
                     color=COLOR_MODEL["pi0"], edgecolors="none",
                     label="ContextTokens (Lin)")
    ax_shape.set_ylabel("Lin (tokens)",
                       fontsize=FONT_LABEL, fontweight="bold",
                       color=COLOR_MODEL["pi0"])
    ax_shape.tick_params(axis="y", labelcolor=COLOR_MODEL["pi0"])
    ax_shape2 = ax_shape.twinx()
    ax_shape2.scatter(scat_t, scat_lout, s=2, alpha=0.35,
                      color=COLOR_MODEL["openvla"], edgecolors="none",
                      label="GeneratedTokens (Lout)")
    ax_shape2.set_ylabel("Lout (tokens)",
                        fontsize=FONT_LABEL, fontweight="bold",
                        color=COLOR_MODEL["openvla"])
    ax_shape2.tick_params(axis="y", labelcolor=COLOR_MODEL["openvla"])
    ax_shape2.spines["right"].set_visible(True)
    ax_shape2.spines["right"].set_linewidth(0.5)
    ax_shape.set_title(
        f"(b) Per-request shape  (Lin mean/p99 = "
        f"{sum(r['lin'] for r in rows)/len(rows):.0f}/"
        f"{sorted(r['lin'] for r in rows)[int(len(rows)*0.99)]} ; "
        f"Lout mean/p99 = "
        f"{sum(r['lout'] for r in rows)/len(rows):.0f}/"
        f"{sorted(r['lout'] for r in rows)[int(len(rows)*0.99)]})",
        fontsize=FONT_TITLE - 1, fontweight="bold", loc="left")

    # (3) route mix — stacked bars (PIM bottom, GPU on top)
    ax_route.bar(rm_axis, rm_pim, width=bar_w_x,
                 align="edge", color=COLOR_ROUTE["lpddr5_pim_bank"],
                 edgecolor="none", label="PIM", zorder=3)
    ax_route.bar(rm_axis, rm_gpu, width=bar_w_x,
                 align="edge", bottom=rm_pim,
                 color=COLOR_ROUTE["gpu_only"],
                 edgecolor="none", label="GPU", zorder=3)
    total_pim = sum(rm_pim)
    total_gpu = sum(rm_gpu)
    total = max(total_pim + total_gpu, 1)
    ax_route.set_ylabel(f"Requests  ({args.bucket_ms / 1000:.0f}s buckets)",
                        fontsize=FONT_LABEL, fontweight="bold")
    ax_route.set_title(
        f"(c) Route mix    "
        f"PIM={total_pim}/{total_pim + total_gpu} "
        f"({100 * total_pim / total:.1f}%)",
        fontsize=FONT_TITLE - 1, fontweight="bold", loc="left")
    leg = ax_route.legend(loc="upper right", fontsize=FONT_ANNOTATE,
                          ncol=2)
    bold_legend(leg)

    # (4) TTFT p99 rolling
    ttft_x = [to_x(t) for t in ttft_centers]
    ax_ttft.plot(ttft_x, [v / 1000.0 if not math.isnan(v) else float("nan")
                          for v in ttft_p99],
                 color=COLOR_MODEL["openvla"], lw=1.2,
                 marker=".", markersize=2, zorder=3)
    ax_ttft.set_yscale("log")
    ax_ttft.set_ylabel("TTFT p99 (s)",
                       fontsize=FONT_LABEL, fontweight="bold")
    ax_ttft.set_title(
        f"(d) TTFT p99 — sliding {window_s:.0f} s window",
        fontsize=FONT_TITLE - 1, fontweight="bold", loc="left")

    # (5) E2E p99 rolling
    e2e_x = [to_x(t) for t in e2e_centers]
    ax_e2e.plot(e2e_x, [v / 1000.0 if not math.isnan(v) else float("nan")
                        for v in e2e_p99],
                color=COLOR_ROUTE["lpddr5_pim_bank"], lw=1.2,
                marker=".", markersize=2, zorder=3)
    ax_e2e.set_yscale("log")
    if args.slo_ms > 0:
        ax_e2e.axhline(args.slo_ms / 1000.0, color="#c0392b",
                       ls="--", lw=1.0, zorder=4)
        ax_e2e.text(0.985, args.slo_ms / 1000.0,
                    f" SLO = {args.slo_ms:.0f} ms",
                    transform=ax_e2e.get_yaxis_transform(),
                    ha="right", va="bottom",
                    fontsize=FONT_ANNOTATE, color="#c0392b",
                    fontweight="bold")
    ax_e2e.set_ylabel("E2E p99 (s)",
                      fontsize=FONT_LABEL, fontweight="bold")
    ax_e2e.set_xlabel(f"Time ({x_unit})",
                      fontsize=FONT_LABEL, fontweight="bold")
    ax_e2e.set_title(
        f"(e) E2E p99 — sliding {window_s:.0f} s window",
        fontsize=FONT_TITLE - 1, fontweight="bold", loc="left")

    for ax in axes:
        ax.grid(axis="y", alpha=0.4)
        ax.grid(axis="x", visible=False)
        set_spines(ax)
        ax.set_xlim(0, to_x(last_t))

    fig.suptitle(
        f"{args.tag} — full-trace dashboard  "
        f"({len(rows):,} requests, "
        f"{fmt_t(last_t)} span)",
        fontsize=FONT_TITLE + 1, fontweight="bold", y=1.015,
    )

    save_fig(fig, f"fig_azure_dashboard_{args.tag}", args.out_dir)
    plt.close(fig)

    print(f"[info] {args.tag}: {len(rows)} requests, "
          f"{fmt_t(last_t)} span, peak {peak_rps:.1f} RPS, "
          f"PIM share {100 * total_pim / total:.1f}%")


if __name__ == "__main__":
    main()
