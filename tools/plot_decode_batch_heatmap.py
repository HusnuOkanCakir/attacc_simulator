#!/usr/bin/env python3
"""
Plot a (batch_size × max_consec_decode) heatmap from a dbatch_sweep run directory.

Each cell shows throughput (req/s); iso-TTFT-p99 contours are overlaid.
Sub-runs must be named  <NN>_bs<B>_mc<M>/  (new 2-D format).

Usage:
    python tools/plot_decode_batch_heatmap.py --sweep-dir <path> [--policy max_util_full]
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


# ── metric extraction ────────────────────────────────────────────────────────

def parse_summary(summary_path: Path, policy: str) -> dict | None:
    """Return {throughput, ttft_p99, e2e_p99, pim_frac} or None on parse failure."""
    text = summary_path.read_text()
    # find the column index for this policy in the header
    lines = text.splitlines()
    if not lines:
        return None

    def grab(pattern):
        m = re.search(pattern, text, re.MULTILINE)
        return float(m.group(1)) if m else None

    throughput = grab(r"Throughput \(req/s\)\s+([\d.]+)")
    ttft_p99   = grab(r"TTFT p99 \(ms\)\s+([\d.]+)")
    e2e_p99    = grab(r"E2E p99 \(ms\)\s+([\d.]+)")

    # Routes (PIM/GPU)   1530/470
    routes_m = re.search(r"Routes \(PIM/GPU\)\s+(\d+)/(\d+)", text)
    pim_frac = None
    if routes_m:
        pim = int(routes_m.group(1))
        gpu = int(routes_m.group(2))
        pim_frac = pim / (pim + gpu) if (pim + gpu) > 0 else None

    if throughput is None:
        return None
    return dict(throughput=throughput, ttft_p99=ttft_p99,
                e2e_p99=e2e_p99, pim_frac=pim_frac)


# ── directory discovery ──────────────────────────────────────────────────────

def discover_runs(sweep_dir: Path, policy: str) -> list[dict]:
    """Scan <sweep_dir>/<NN>_bs<B>_mc<M>/ sub-directories."""
    pattern = re.compile(r"^\d+_bs(\d+)_mc(\d+)$")
    rows = []
    for d in sorted(sweep_dir.iterdir()):
        if not d.is_dir():
            continue
        m = pattern.match(d.name)
        if not m:
            continue
        bs, mc = int(m.group(1)), int(m.group(2))
        summary = d / "policy_compare_summary.txt"
        if not summary.exists():
            # try inside policy sub-dir
            summary = d / policy / "policy_compare_summary.txt"
        if not summary.exists():
            print(f"  [warn] no summary in {d.name}, skipping", file=sys.stderr)
            continue
        metrics = parse_summary(summary, policy)
        if metrics is None:
            print(f"  [warn] parse failed for {d.name}", file=sys.stderr)
            continue
        rows.append(dict(bs=bs, mc=mc, **metrics))
    return rows


# ── plotting ─────────────────────────────────────────────────────────────────

def make_heatmap(rows: list[dict], out_path: Path, policy: str, sweep_label: str):
    bs_vals = sorted(set(r["bs"] for r in rows))
    mc_vals = sorted(set(r["mc"] for r in rows))

    nb, nm = len(bs_vals), len(mc_vals)
    thr   = np.full((nm, nb), np.nan)  # rows=mc, cols=bs
    ttft  = np.full((nm, nb), np.nan)

    for r in rows:
        i = mc_vals.index(r["mc"])
        j = bs_vals.index(r["bs"])
        thr[i, j]  = r["throughput"]
        if r["ttft_p99"] is not None:
            ttft[i, j] = r["ttft_p99"]

    fig, ax = plt.subplots(figsize=(max(5, nb * 1.4 + 1.5), max(4, nm * 1.2 + 1.5)))

    # ── heatmap (throughput) ──────────────────────────────────────────────────
    vmin, vmax = np.nanmin(thr), np.nanmax(thr)
    if np.isnan(vmin):
        vmin, vmax = 0, 1
    im = ax.imshow(thr, aspect="auto", origin="lower",
                   cmap="YlGn", vmin=vmin, vmax=vmax,
                   extent=[-0.5, nb - 0.5, -0.5, nm - 0.5])

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Throughput (req/s)", fontsize=10)

    # annotate cells
    for i in range(nm):
        for j in range(nb):
            if not np.isnan(thr[i, j]):
                ax.text(j, i, f"{thr[i,j]:.2f}",
                        ha="center", va="center", fontsize=8,
                        color="black" if thr[i, j] < 0.7 * vmax else "white")

    # ── TTFT-p99 contours ────────────────────────────────────────────────────
    if not np.all(np.isnan(ttft)):
        xs = np.arange(nb)
        ys = np.arange(nm)
        XX, YY = np.meshgrid(xs, ys)
        # fill NaN with column mean for contouring
        ttft_filled = ttft.copy()
        col_means = np.nanmean(ttft_filled, axis=0)
        for j in range(nb):
            mask = np.isnan(ttft_filled[:, j])
            ttft_filled[mask, j] = col_means[j]

        valid_levels = np.nanpercentile(ttft[~np.isnan(ttft)],
                                        [10, 30, 50, 70, 90])
        valid_levels = np.unique(valid_levels.round(0))
        if len(valid_levels) >= 2:
            cs = ax.contour(XX, YY, ttft_filled, levels=valid_levels,
                            colors="steelblue", linewidths=1.2, linestyles="--")
            ax.clabel(cs, fmt=lambda v: f"{v/1000:.1f}s", fontsize=7,
                      inline=True)

    # ── axes labels / ticks ───────────────────────────────────────────────────
    ax.set_xticks(range(nb))
    ax.set_xticklabels([str(b) for b in bs_vals])
    ax.set_yticks(range(nm))
    ax.set_yticklabels([str(m) for m in mc_vals])
    ax.set_xlabel("max_decode_batch_size", fontsize=11)
    ax.set_ylabel("max_consec_decode_batches", fontsize=11)
    ax.set_title(
        f"Decode-batch sweep — {sweep_label}\n"
        f"policy={policy}   color=throughput (req/s)   contours=TTFT-p99",
        fontsize=10)

    # iso-product guide lines (diagonal bands where bs*mc is constant)
    iso_products = sorted(set(b * m for b in bs_vals for m in mc_vals))
    for prod in iso_products:
        pts = [(bs_vals.index(b), mc_vals.index(m))
               for b in bs_vals for m in mc_vals if b * m == prod]
        if len(pts) >= 2:
            pts.sort()
            jj = [p[0] for p in pts]
            ii = [p[1] for p in pts]
            ax.plot(jj, ii, "r-", alpha=0.25, linewidth=0.8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[heatmap] {out_path}")


# ── TTFT-p99 vs throughput scatter (Pareto frontier) ────────────────────────

def make_pareto_scatter(rows: list[dict], out_path: Path, policy: str, sweep_label: str):
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6, 4.5))

    cmap = plt.get_cmap("tab10")
    mc_vals = sorted(set(r["mc"] for r in rows))
    color_map = {mc: cmap(i / max(len(mc_vals) - 1, 1))
                 for i, mc in enumerate(mc_vals)}

    seen_mc = set()
    for r in rows:
        if r["ttft_p99"] is None:
            continue
        mc = r["mc"]
        label = f"max_consec={mc}" if mc not in seen_mc else None
        seen_mc.add(mc)
        ax.scatter(r["throughput"], r["ttft_p99"] / 1000.0,
                   color=color_map[mc], s=80, zorder=3, label=label)
        ax.annotate(f"bs={r['bs']}", (r["throughput"], r["ttft_p99"] / 1000.0),
                    textcoords="offset points", xytext=(4, 3), fontsize=7)

    ax.set_xlabel("Throughput (req/s)", fontsize=11)
    ax.set_ylabel("TTFT p99 (s)", fontsize=11)
    ax.set_title(f"TTFT-p99 vs Throughput — {sweep_label}\npolicy={policy}",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[pareto]  {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-dir", required=True, type=Path,
                    help="Root of a dbatch_sweep run (contains <NN>_bs*_mc* dirs)")
    ap.add_argument("--policy", default="max_util_full",
                    help="Policy name used for the sweep (default: max_util_full)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Output directory (default: <sweep-dir>/plots)")
    args = ap.parse_args()

    sweep_dir = args.sweep_dir.resolve()
    if not sweep_dir.is_dir():
        sys.exit(f"[error] sweep-dir not found: {sweep_dir}")

    out_dir = args.out_dir or sweep_dir / "plots"
    sweep_label = sweep_dir.name

    print(f"[heatmap] scanning {sweep_dir}")
    rows = discover_runs(sweep_dir, args.policy)
    if not rows:
        sys.exit("[error] no bs/mc sub-runs found — check naming (expect *_bs<N>_mc<M>)")

    print(f"[heatmap] found {len(rows)} cells: "
          f"bs={sorted(set(r['bs'] for r in rows))}  "
          f"mc={sorted(set(r['mc'] for r in rows))}")

    make_heatmap(rows, out_dir / "decode_batch_heatmap.png",
                 args.policy, sweep_label)
    make_pareto_scatter(rows, out_dir / "decode_batch_pareto.png",
                        args.policy, sweep_label)

    print(f"[heatmap] done — plots in {out_dir}")


if __name__ == "__main__":
    main()
