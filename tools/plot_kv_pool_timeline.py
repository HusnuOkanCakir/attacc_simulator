#!/usr/bin/env python3
"""
Render PIM KV-pool occupancy timelines by replaying a serving_online debug log.

Parses ARRIVE / ADMIT / KV_ALLOC / KV_GROW / TAIL_TRIM / PREEMPT / KV_FREE /
DONE / TASK_START / TASK_DONE events and emits a stacked area plot where

    y-axis = bytes currently reserved in the PIM KV pool
    stacked per request, colored by phase (prefill vs decode)
    hatched grey band on top = free capacity up to kv_pool_bytes

Usage
-----
    python tools/plot_kv_pool_timeline.py                          # latest run, all 4 policies
    python tools/plot_kv_pool_timeline.py --subdir max_util_tail   # one policy
    python tools/plot_kv_pool_timeline.py --t-max 2500             # zoom to [0, 2500] ms
    python tools/plot_kv_pool_timeline.py --combined               # one 4-panel figure too
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
RUN_ROOT = REPO / "cluster_outputs/online_serving_runs"

POLICIES = ["guaranteed_no_evict", "max_util_full", "max_util_tail", "max_util_tail_pred"]

PHASE_COLORS = {
    "prefill": "#1f77b4",
    "decode":  "#2ca02c",
    "waiting": "#dddddd",
}
FREE_HATCH_COLOR = "#f2f2f2"
FREE_EDGE_COLOR  = "#bbbbbb"

# ── Log parsing ───────────────────────────────────────────────────────────────

_T = r"\[\s*([\d.]+)ms\]"
RX = {
    "arrive":     re.compile(_T + r"\s*ARRIVE\s+req=\s*(\d+)"),
    "admit":      re.compile(_T + r"\s*ADMIT\s+req=\s*(\d+)\s+route=\s*(\S+)"),
    "kv_alloc":   re.compile(_T + r"\s*KV_ALLOC\s+req=\s*(\d+)\s+region=(Key|Value)\s+bytes=(\d+)"),
    "kv_grow":    re.compile(_T + r"\s*KV_GROW\s+req=\s*(\d+)\s+delta_bytes=\d+\s+new_reserved=(\d+)"),
    "tail_trim":  re.compile(_T + r"\s*TAIL_TRIM\s+req=\s*(\d+)\s+tokens=\d+\s+freed_bytes=(\d+)"),
    "preempt":    re.compile(_T + r"\s*PREEMPT\s+req=\s*(\d+)\s+freed_bytes=(\d+)"),
    "kv_free":    re.compile(_T + r"\s*KV_FREE\s+req=\s*(\d+)\s+region=(Key|Value)"),
    "done":       re.compile(_T + r"\s*DONE\s+req=\s*(\d+)"),
    "task_start": re.compile(_T + r"\s*TASK_START\s+phase=(\w+)\s+route=\s*\S+\s+reqs=\[([^\]]*)\]"),
    "task_done":  re.compile(_T + r"\s*TASK_DONE\s+phase=(\w+)\s+route=\s*\S+\s+reqs=\[([^\]]*)\]"),
}

def parse_log(log_path: Path):
    """Return list of (t_ms, event_name, fields_dict) in log order."""
    events = []
    with log_path.open() as f:
        for line in f:
            if not line.startswith("["):
                continue
            for name, rx in RX.items():
                m = rx.match(line)
                if not m:
                    continue
                g = m.groups()
                t = float(g[0])
                if name == "arrive":
                    events.append((t, name, {"rid": int(g[1])}))
                elif name == "admit":
                    events.append((t, name, {"rid": int(g[1]), "route": g[2]}))
                elif name == "kv_alloc":
                    events.append((t, name, {"rid": int(g[1]), "region": g[2], "bytes": int(g[3])}))
                elif name == "kv_grow":
                    events.append((t, name, {"rid": int(g[1]), "new_reserved": int(g[2])}))
                elif name == "tail_trim":
                    events.append((t, name, {"rid": int(g[1]), "freed_bytes": int(g[2])}))
                elif name == "preempt":
                    events.append((t, name, {"rid": int(g[1]), "freed_bytes": int(g[2])}))
                elif name == "kv_free":
                    events.append((t, name, {"rid": int(g[1]), "region": g[2]}))
                elif name == "done":
                    events.append((t, name, {"rid": int(g[1])}))
                elif name in ("task_start", "task_done"):
                    reqs = [int(x) for x in g[2].split(",") if x.strip()]
                    events.append((t, name, {"phase": g[1], "reqs": reqs}))
                break
    return events

# ── Replay to per-request piecewise-constant (bytes, phase) segments ──────────

def replay(events):
    state = {}       # rid -> {"bytes": int, "phase": str}
    segments = {}    # rid -> list of (t, bytes, phase)

    def emit(rid, t):
        s = state[rid]
        segments.setdefault(rid, []).append((t, s["bytes"], s["phase"]))

    for t, evt, f in events:
        if evt == "arrive":
            state.setdefault(f["rid"], {"bytes": 0, "phase": "waiting"})
            emit(f["rid"], t)
        elif evt == "admit":
            s = state.setdefault(f["rid"], {"bytes": 0, "phase": "prefill"})
            s["phase"] = "prefill"
            emit(f["rid"], t)
        elif evt == "kv_alloc":
            s = state.setdefault(f["rid"], {"bytes": 0, "phase": "prefill"})
            s["bytes"] += f["bytes"]
            emit(f["rid"], t)
        elif evt == "kv_grow":
            s = state.setdefault(f["rid"], {"bytes": 0, "phase": "decode"})
            s["bytes"] = f["new_reserved"]
            emit(f["rid"], t)
        elif evt == "tail_trim":
            s = state.setdefault(f["rid"], {"bytes": 0, "phase": "decode"})
            s["bytes"] = max(0, s["bytes"] - f["freed_bytes"])
            emit(f["rid"], t)
        elif evt == "preempt":
            s = state.setdefault(f["rid"], {"bytes": 0, "phase": "waiting"})
            s["bytes"] = 0
            s["phase"] = "waiting"
            emit(f["rid"], t)
        elif evt == "kv_free":
            s = state.setdefault(f["rid"], {"bytes": 0, "phase": "waiting"})
            s["bytes"] = 0
            emit(f["rid"], t)
        elif evt == "done":
            s = state.setdefault(f["rid"], {"bytes": 0, "phase": "waiting"})
            s["bytes"] = 0
            s["phase"] = "waiting"
            emit(f["rid"], t)
        elif evt == "task_start":
            for rid in f["reqs"]:
                if rid in state:
                    state[rid]["phase"] = f["phase"]
                    emit(rid, t)
        elif evt == "task_done":
            # prefill→decode transition fires once; other task_done lines are no-ops.
            if f["phase"] == "prefill":
                for rid in f["reqs"]:
                    if rid in state:
                        state[rid]["phase"] = "decode"
                        emit(rid, t)
    return segments

# ── Plotting ─────────────────────────────────────────────────────────────────

def build_series(segments, t_max):
    """Master time grid + per-rid (bytes, phase) signals."""
    times = sorted({seg[0] for segs in segments.values() for seg in segs})
    if times and times[0] > 0.0:
        times = [0.0] + times
    if t_max is not None:
        times = [t for t in times if t <= t_max]
        if not times or times[-1] < t_max:
            times.append(t_max)
    T = np.asarray(times, dtype=float)

    rids = sorted(segments.keys())
    byte_series = {}
    phase_series = {}
    for rid in rids:
        segs = sorted(segments[rid], key=lambda x: x[0])
        y  = np.zeros(len(T))
        ph = ["waiting"] * len(T)
        idx = 0
        cur_b, cur_p = 0, "waiting"
        for j, t in enumerate(T):
            while idx < len(segs) and segs[idx][0] <= t:
                _, cur_b, cur_p = segs[idx]
                idx += 1
            y[j]  = cur_b
            ph[j] = cur_p
        byte_series[rid]  = y
        phase_series[rid] = ph
    return T, rids, byte_series, phase_series

def plot_timeline(ax, segments, pool_bytes, title, t_max=None):
    T, rids, byte_series, phase_series = build_series(segments, t_max)
    if len(T) < 2:
        ax.set_title(f"{title} — (no events)")
        return

    # Stable stacking order: first admission time per request.
    def first_alloc_t(rid):
        for (t, b, _) in segments[rid]:
            if b > 0:
                return t
        return float("inf")
    rid_order = sorted(rids, key=first_alloc_t)

    bottom = np.zeros(len(T))
    for rid in rid_order:
        y  = byte_series[rid]
        ph = phase_series[rid]
        top = bottom + y

        # Group consecutive same-phase steps, emit one fill_between per group.
        j = 0
        while j < len(T):
            k = j
            while k + 1 < len(T) and ph[k + 1] == ph[j]:
                k += 1
            end = min(k + 1, len(T) - 1)
            ts = T[j:end + 1]
            bb = bottom[j:end + 1]
            tt = top[j:end + 1]
            color = PHASE_COLORS.get(ph[j], "#cccccc")
            ax.fill_between(ts, bb, tt, step="post",
                            color=color, alpha=0.85,
                            linewidth=0)
            j = k + 1

        # Thin white divider between requests.
        ax.plot(T, top, color="white", linewidth=0.3, drawstyle="steps-post")

        # Label wide-enough bands with request id.
        nz = np.where(y > pool_bytes * 0.025)[0]
        if len(nz) > 1:
            mid = nz[len(nz) // 2]
            ax.text(T[mid], bottom[mid] + y[mid] / 2, f"r{rid}",
                    ha="center", va="center",
                    color="white", fontsize=7, fontweight="bold")

        bottom = top

    # Free-capacity hatched band up to pool_bytes.
    ax.fill_between(T, bottom, np.full_like(T, pool_bytes),
                    step="post", facecolor=FREE_HATCH_COLOR,
                    edgecolor=FREE_EDGE_COLOR, linewidth=0.3,
                    hatch="//", alpha=0.6)

    ax.axhline(pool_bytes, color="black", lw=0.8)
    ax.set_xlim(T[0], T[-1])

    # Auto-zoom Y-axis when peak usage is much smaller than the pool ceiling
    # (added 2026-05-27 — previously hardcoded to (0, pool*1.02), which made
    # plots look "empty" when the pool was over-provisioned, e.g. Pi0 + 8 GiB
    # synthetic VLA with peak ~0.09 GiB). If peak < 25% of pool, zoom to
    # peak × 1.5; otherwise show the full pool. Either way, the pool ceiling
    # is marked by the horizontal black line (drawn above) so the reader can
    # still see the absolute capacity.
    peak = float(bottom.max()) if len(bottom) else 0.0
    if peak > 0 and peak < pool_bytes * 0.25:
        ax.set_ylim(0, max(peak * 1.5, 1.0))
    else:
        ax.set_ylim(0, pool_bytes * 1.02)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("KV bytes reserved")
    gb = pool_bytes / (1024 ** 3)
    peak_gb = peak / (1024 ** 3)
    util_pct = (100.0 * peak / pool_bytes) if pool_bytes > 0 else 0.0
    ax.set_title(f"{title}  (pool={gb:.2f} GB, peak={peak_gb:.2f} GB, util={util_pct:.1f}%)")
    ax.grid(axis="y", alpha=0.2)

    from matplotlib.patches import Patch
    handles = [
        Patch(color=PHASE_COLORS["prefill"], label="prefill"),
        Patch(color=PHASE_COLORS["decode"],  label="decode"),
        Patch(facecolor=FREE_HATCH_COLOR, edgecolor=FREE_EDGE_COLOR,
              hatch="//", label="free"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8)

def read_pool_bytes(subdir: Path) -> int:
    cfg = yaml.safe_load((subdir / "config.yaml").read_text())
    return int(cfg["Frontend"]["kv_pool_bytes"])

def latest_policy_compare() -> Path:
    dirs = sorted(RUN_ROOT.glob("*_policy_compare"))
    if not dirs:
        sys.exit(f"No *_policy_compare dirs under {RUN_ROOT}")
    return dirs[-1]

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="policy_compare run dir (default: latest)")
    ap.add_argument("--subdir", default=None,
                    help="single policy subdir to plot (default: all 4)")
    ap.add_argument("--t-max", type=float, default=None,
                    help="clip x-axis to [0, t_max] ms (default: full span)")
    ap.add_argument("--combined", action="store_true",
                    help="also emit a single 4-panel figure")
    args = ap.parse_args()

    run_dir = (args.run_dir or latest_policy_compare()).resolve()
    print(f"[run_dir] {run_dir}")

    subs = [args.subdir] if args.subdir else POLICIES

    per_policy = {}
    for sub in subs:
        subdir = run_dir / sub
        log = subdir / "debug.log"
        if not log.exists():
            print(f"[skip] {sub}: no debug.log")
            continue
        events = parse_log(log)
        segments = replay(events)
        pool = read_pool_bytes(subdir)
        per_policy[sub] = (subdir, segments, pool)

        plots_dir = subdir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        out = plots_dir / f"{sub}_kv_pool_timeline.png"
        fig, ax = plt.subplots(figsize=(16, 5))
        plot_timeline(ax, segments, pool, sub, t_max=args.t_max)
        fig.tight_layout()
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"[plot] {out.relative_to(REPO)}")

    if args.combined and per_policy:
        fig, axes = plt.subplots(len(per_policy), 1,
                                 figsize=(16, 4.0 * len(per_policy)),
                                 sharex=True)
        if len(per_policy) == 1:
            axes = [axes]
        for ax, (label, (_sd, segs, pool)) in zip(axes, per_policy.items()):
            plot_timeline(ax, segs, pool, label, t_max=args.t_max)
        fig.suptitle("PIM KV pool occupancy over time — per policy", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        combined_dir = run_dir / "plots"
        combined_dir.mkdir(parents=True, exist_ok=True)
        out = combined_dir / "kv_pool_timeline_compare.png"
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"[plot] {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
