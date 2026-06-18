#!/usr/bin/env python3
"""Serial vs pipeline executor comparison plots.

Two figures:
  figA_fc_vs_attention.{png,pdf}  -- cost-table explanation of why decode is
      feed-forward-bound (GPU FFN >> attention across context), Pi0 + OpenVLA.
  figB_serial_vs_pipeline.{png,pdf} -- empirical comparison across offered load
      (achieved throughput, E2E p99, makespan) for the cells in --sweep-dir,
      whose subdirs are named {serial,pipeline}_s<scale>.

Usage:
    python thesis_plotting/scripts/thesis_pipeline_compare.py \
        --sweep-dir cluster_outputs/online_serving_runs/pipe_compare_pi0_<TS> \
        --cost-dir cluster_outputs/cost_tables_energyfix_pi0_a6000
"""
import argparse
import csv
import glob
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
FIGDIR = REPO / "thesis_plotting" / "figures"


def fnum(r, k):
    try:
        return float(r[k])
    except Exception:
        return 0.0


def fc_attn(path):
    """dict[bs] -> sorted [(lin, fc_ms, attn_ms)] for decode rows."""
    rows = list(csv.DictReader(open(path)))
    out, seen = {}, set()
    for r in rows:
        lin = int(float(r["Lin"])); lout = int(float(r["Lout"])); bs = int(float(r["bs"]))
        if lout > 32:
            continue
        if (bs, lin) in seen:
            continue
        seen.add((bs, lin))
        fc = (fnum(r, "g_qkv_time") + fnum(r, "g_prj_time") + fnum(r, "g_ff_time")
              + fnum(r, "g_etc") + fnum(r, "g2g_comm"))
        if fc <= 0:
            fc = fnum(r, "g_fc") + fnum(r, "g_etc") + fnum(r, "g2g_comm")
        attn = fnum(r, "g_matmul") + fnum(r, "g_softmax")
        out.setdefault(bs, []).append((lin, fc, attn))
    for bs in out:
        out[bs].sort()
    return out


def fig_a(outdir):
    tabs = {
        "Pi0":     REPO / "cluster_outputs/cost_tables_energyfix_pi0_a6000/lpddr5_pim_bank.csv",
        "OpenVLA": REPO / "cluster_outputs/cost_tables_energyfix_openvla_a6000/lpddr5_pim_bank.csv",
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, (name, p) in zip(axes, tabs.items()):
        if not Path(p).exists():
            ax.set_title(f"{name}: cost table missing")
            continue
        d = fc_attn(p)
        bs = 4 if 4 in d else sorted(d)[0]
        lins = [x[0] for x in d[bs]]
        fc = [x[1] for x in d[bs]]
        attn = [x[2] for x in d[bs]]
        ax.plot(lins, fc, "o-", color="#406ea3", lw=2.5, ms=6,
                label="GPU feed-forward (FFN+proj)")
        ax.plot(lins, attn, "s-", color="#c0504d", lw=2.5, ms=6,
                label="attention (matmul+softmax)")
        ax.set_yscale("log")
        ax.set_xlabel("context length (tokens)", fontsize=12, fontweight="bold")
        ax.set_ylabel("per decode-step time (ms)", fontsize=12, fontweight="bold")
        ratio = fc[-1] / attn[-1] if attn[-1] else float("nan")
        ax.set_title(f"{name} (bs={bs}):  FC / attn = {ratio:.0f}x at ctx {lins[-1]}",
                     fontsize=12, fontweight="bold")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=10, loc="center right")
    fig.suptitle("Why decode is feed-forward-bound: attention (what PIM "
                 "accelerates) is a minority of the step",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"figA_fc_vs_attention.{ext}", dpi=130)
    plt.close(fig)
    print("wrote figA_fc_vs_attention.{png,pdf}")


def parse_summary(d):
    fs = glob.glob(str(Path(d) / "policy_compare_summary.txt"))
    if not fs:
        return None
    txt = open(fs[0]).read()

    def g(pat):
        m = re.search(pat + r"\s+([0-9.]+)", txt)
        return float(m.group(1)) if m else None
    return dict(thru=g(r"Throughput \(req/s\)"), e2e=g(r"E2E p99 \(ms\)"),
               ttft=g(r"TTFT p99 \(ms\)"), span=g(r"Span \(ms\)"))


def offered_rps(d):
    cs = glob.glob(str(Path(d) / "*" / "requests_out.csv"))
    if not cs:
        return None
    n, mx = 0, 0.0
    for r in csv.DictReader(open(cs[0])):
        a = float(r["arrival_ms"]); mx = max(mx, a); n += 1
    return n / (mx / 1000.0) if mx > 0 else None


def fig_b(sweep_dir, outdir):
    rows = {"serial": [], "pipeline": []}
    for d in sorted(glob.glob(str(Path(sweep_dir) / "*_s*"))):
        name = Path(d).name
        exec_ = "serial" if name.startswith("serial") else "pipeline" if name.startswith("pipeline") else None
        if exec_ is None:
            continue
        s = parse_summary(d); off = offered_rps(d)
        if not s or off is None:
            continue
        rows[exec_].append((off, s))
    if not rows["serial"] or not rows["pipeline"]:
        print("fig_b: insufficient sweep data (need serial_* and pipeline_* cells)")
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = [("thru", "achieved throughput (req/s)"),
               ("e2e", "E2E p99 (ms)"),
               ("span", "makespan / span (ms)")]
    style = {"serial": dict(color="#888", marker="o", ls="--"),
             "pipeline": dict(color="#406ea3", marker="s", ls="-")}
    for ax, (key, ylab) in zip(axes, metrics):
        for exec_ in ("serial", "pipeline"):
            pts = sorted(rows[exec_])
            xs = [p[0] for p in pts]; ys = [p[1][key] for p in pts]
            ax.plot(xs, ys, lw=2.5, ms=8, label=exec_, **style[exec_])
        ax.set_xlabel("offered load (req/s)", fontsize=12, fontweight="bold")
        ax.set_ylabel(ylab, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3); ax.legend(fontsize=11)
    mx = max(p[0] for p in rows["serial"])
    axes[0].plot([0, mx], [0, mx], ls=":", color="#bbb", lw=1)
    fig.suptitle("Serial vs pipeline executor across offered load "
                 "(max_util_full) — equivalent in the feed-forward-bound regime",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"figB_serial_vs_pipeline.{ext}", dpi=130)
    plt.close(fig)
    print("wrote figB_serial_vs_pipeline.{png,pdf}")
    print(f"\n{'exec':<9}{'offered':>9}{'achieved':>10}{'E2Ep99ms':>10}{'span_ms':>9}")
    for exec_ in ("serial", "pipeline"):
        for off, s in sorted(rows[exec_]):
            print(f"{exec_:<9}{off:>9.2f}{s['thru']:>10.2f}{s['e2e']:>10.0f}{s['span']:>9.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", type=Path, default=None,
                    help="pipe_compare_* sweep dir with {serial,pipeline}_s* cells")
    ap.add_argument("--cost-dir", type=Path, default=None,
                    help="(unused; kept for symmetry with other renderers)")
    ap.add_argument("--out-dir", type=Path, default=FIGDIR)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_a(args.out_dir)
    if args.sweep_dir:
        fig_b(args.sweep_dir, args.out_dir)


if __name__ == "__main__":
    main()
