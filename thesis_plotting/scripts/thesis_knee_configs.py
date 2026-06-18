#!/usr/bin/env python3
"""Generic RPS-knee renderer: E2E p99 (and achieved throughput) vs OFFERED load,
one curve per config. Cells are labelled <config>_s<scale>; the config is
everything before the trailing _s<digits>. Offered RPS is computed from the
arrival timestamps in <cell>/*/requests_out.csv (NOT achieved throughput).

Works for any config-labelled sweep, e.g.:
  - sched_2gpu : gpu_only / pim_only / hyb_2gpu
  - numgpu_knee: g1 / g2

Usage:
  python thesis_plotting/scripts/thesis_knee_configs.py \
      --dir cluster_outputs/online_serving_runs/sched_2gpu_bootstrap_pi0_<TS> \
      --out-name fig_sched_2gpu_knee \
      --title "Scheduler value (bootstrap): gpu_only vs pim_only vs 2-GPU hybrid"
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

# Display name + colour + marker per known config. Unknown configs fall back
# to an auto-assigned colour from the cycle.
STYLE = {
    "gpu_only": dict(label="GPU-only",            color="#c0504d", marker="o", ls="--"),
    "pim_only": dict(label="PIM-only",            color="#e8a86c", marker="s", ls="-."),
    "hyb_2gpu": dict(label="2-GPU hybrid (sched)", color="#406ea3", marker="D", ls="-"),
    "g1":       dict(label="1 GPU (hybrid)",       color="#e8a86c", marker="s", ls="--"),
    "g2":       dict(label="2 GPUs (hybrid+overflow)", color="#406ea3", marker="D", ls="-"),
}
FALLBACK_COLORS = ["#5a9b5a", "#8064a2", "#4bacc6", "#9b59b6"]

CELL_RE = re.compile(r"^(?P<cfg>.+)_s\d+$")


def _grep(path, pat):
    m = re.search(pat, open(path).read())
    return m if m else None


def parse_summary(cell):
    fs = glob.glob(str(Path(cell) / "policy_compare_summary.txt"))
    if not fs:
        return None
    txt = open(fs[0]).read()
    def g(pat):
        m = re.search(pat, txt)
        return float(m.group(1)) if m else None
    e2e = g(r"E2E p99 \(ms\)\s+([0-9.]+)")
    ttft = g(r"TTFT p99 \(ms\)\s+([0-9.]+)")
    thr = g(r"Throughput \(req/s\)\s+([0-9.]+)")
    rm = re.search(r"Routes \(PIM/GPU\)\s+(\d+)/(\d+)", txt)
    pim_gpu = (int(rm.group(1)), int(rm.group(2))) if rm else None
    return dict(e2e_p99=e2e, ttft_p99=ttft, throughput=thr, routes=pim_gpu)


def offered_rps(cell):
    cs = glob.glob(str(Path(cell) / "*" / "requests_out.csv"))
    if not cs:
        return None
    n, mx = 0, 0.0
    for r in csv.DictReader(open(cs[0])):
        a = float(r["arrival_ms"]); mx = max(mx, a); n += 1
    return n / (mx / 1000.0) if mx > 0 else None


def load(sweep_dir):
    """dict[cfg] -> sorted list of dict(offered, e2e_p99, ttft_p99, throughput, routes)."""
    out = {}
    for d in sorted(glob.glob(str(Path(sweep_dir) / "*_s*"))):
        m = CELL_RE.match(Path(d).name)
        if not m:
            continue
        cfg = m.group("cfg")
        off = offered_rps(d)
        s = parse_summary(d)
        if off is None or not s or s["e2e_p99"] is None:
            continue
        s["offered"] = off
        out.setdefault(cfg, []).append(s)
    for k in out:
        out[k].sort(key=lambda r: r["offered"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out-name", default="fig_knee_configs")
    ap.add_argument("--out-dir", type=Path, default=FIGDIR)
    ap.add_argument("--title", default="RPS knee: E2E p99 vs offered load")
    ap.add_argument("--order", nargs="+", default=None,
                    help="explicit config order (default: discovery order)")
    ap.add_argument("--logy", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    curves = load(args.dir)
    if not curves:
        raise SystemExit(f"[error] no cells in {args.dir}")
    order = args.order or list(curves.keys())

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15.0, 6.2))
    print(f"\n{'config':<10}{'offered':>9}{'E2Ep99ms':>10}{'TTFTp99':>9}"
          f"{'achRPS':>8}  routes(PIM/GPU)")
    fb = 0
    for cfg in order:
        pts = curves.get(cfg)
        if not pts:
            continue
        st = STYLE.get(cfg)
        if st is None:
            st = dict(label=cfg, color=FALLBACK_COLORS[fb % len(FALLBACK_COLORS)],
                      marker="^", ls="-"); fb += 1
        xs = [p["offered"] for p in pts]
        for ax, key in ((axL, "e2e_p99"), (axR, "throughput")):
            ys = [p[key] / (1000.0 if key == "e2e_p99" else 1.0) for p in pts]
            ax.plot(xs, ys, color=st["color"], lw=2.6, ms=9, marker=st["marker"],
                    ls=st["ls"], markeredgecolor="black", markeredgewidth=0.6,
                    label=st["label"], zorder=4)
        for p in pts:
            r = p["routes"]
            rs = f"{r[0]}/{r[1]}" if r else "n/a"
            print(f"{cfg:<10}{p['offered']:>9.2f}{p['e2e_p99']:>10.0f}"
                  f"{(p['ttft_p99'] or 0):>9.0f}{(p['throughput'] or 0):>8.2f}  {rs}")

    axL.set_xlabel("Offered load  (req/s)", fontsize=13, fontweight="bold")
    axL.set_ylabel("E2E p99 latency  (s)", fontsize=13, fontweight="bold")
    if args.logy:
        axL.set_yscale("log")
    axL.grid(True, which="both", alpha=0.35); axL.tick_params(labelsize=11)
    axL.set_title("E2E p99 knee", fontsize=13, fontweight="bold")
    axL.legend(fontsize=11, framealpha=0.95)

    axR.set_xlabel("Offered load  (req/s)", fontsize=13, fontweight="bold")
    axR.set_ylabel("Achieved throughput  (req/s)", fontsize=13, fontweight="bold")
    axR.grid(True, which="both", alpha=0.35); axR.tick_params(labelsize=11)
    axR.set_title("Achieved throughput (saturation)", fontsize=13, fontweight="bold")
    axR.legend(fontsize=11, framealpha=0.95)

    fig.suptitle(args.title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"{args.out_name}.{ext}", dpi=130)
    plt.close(fig)
    print(f"\nwrote {args.out_name}.{{png,pdf}} to {args.out_dir}")


if __name__ == "__main__":
    main()
