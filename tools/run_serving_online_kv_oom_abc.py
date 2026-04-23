#!/usr/bin/env python3
"""
Run the A/B/C KV-OOM policy comparison for serving_online and plot the results.

  A: kv_oom_policy = hold              (wait until PIM KV frees)
  B: kv_oom_policy = fallback_gpu      (route to GPU immediately on OOM)
  C: kv_oom_policy = hold_then_fallback (wait up to --hold-limit-ms, then GPU)

Usage
-----
  python tools/run_serving_online_kv_oom_abc.py                  # uses default 200 ms for C
  python tools/run_serving_online_kv_oom_abc.py --hold-limit-ms 1000
  python tools/run_serving_online_kv_oom_abc.py --skip-run --run-dir <path>  # re-plot only
"""

from __future__ import annotations

import argparse
import datetime as _dt
import io
import math
import subprocess
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Repo layout ───────────────────────────────────────────────────────────────

REPO     = Path(__file__).resolve().parent.parent
RAMU_BIN = REPO / "ramulator2/build/ramulator2"
RUN_ROOT = REPO / "cluster_outputs/online_serving_runs"

REQUESTS_CSV    = "cluster_outputs/azure/AzureLLMInferenceTrace_conv_first50.csv"
COST_GPU_CSV    = "cluster_outputs/cost_tables_full_energy/gpu_only.csv"
COST_HYBRID_CSV = "cluster_outputs/cost_tables_full_energy/lpddr5_pim_bank.csv"

# Visual style matching the existing serving run plots
PREFILL_COLOR = "#aec7e8"
DECODE_COLOR  = "#ffbb78"
POLICY_COLORS = {"A": "#2ca02c", "B": "#d62728", "C": "#1f77b4"}

# ── YAML ──────────────────────────────────────────────────────────────────────

def yaml_text(policy: str, run_dir: Path, hold_limit_ms: float) -> str:
    """Render the YAML for one A/B/C run. run_dir is the per-policy subdir."""
    out_csv = run_dir.relative_to(REPO) / "requests_out.csv"
    extra = ""
    if policy == "hold_then_fallback":
        extra = f"  kv_oom_hold_limit_ms: {hold_limit_ms}\n"
    return f"""Frontend:
  impl: ServingOnlineFrontend
  clock_ratio: 1

  requests_csv: {REQUESTS_CSV}
  cost_gpu_csv: {COST_GPU_CSV}
  cost_hybrid_csv: {COST_HYBRID_CSV}

  requests_out_csv: {out_csv}

  gpu_route_name: gpu_only
  hybrid_route_name: lpddr5_pim_bank

  route_policy: min_finish
  energy_latency_guard_ms: 20.0

  max_decode_batch_size: 8
  prompt_priority: true
  max_consecutive_decode_batches: 4

  admission_max_wait_ms: 150.0
  admission_retry_interval_ms: 10.0
  kv_oom_policy: {policy}
{extra}
  slo_e2e_ms: 900.0
  slo_ttft_ms: -1.0

  arrival_time_scale: 0.1
  arrival_limit: -1

  kv_pool_bytes: 1503238553

  generator_num_layers: 40
  generator_num_heads: 8
  generator_dhead: 256
  generator_dtype_bytes: 2
  generator_channel_count: 16
  translation_pagesize_KB: 4

  Translation:
    impl: ServingOnlineTranslation
    max_addr: 2147483648

MemorySystem:
  impl: PIMDRAM
  clock_ratio: 1
  DRAM:
    impl: LPDDR5-PIM
    org:
      preset: LPDDR5_2Gb_x16
      channel: 16
    timing:
      preset: LPDDR5_6400

  Controller:
    impl: HBM3-PIM
    Scheduler:
      impl: PIM
    RefreshManager:
      impl: AllBank

  AddrMapper:
    impl: ChRaBaRoCo
"""

# ── Simulation runner ─────────────────────────────────────────────────────────

def run_one(letter: str, policy: str, run_dir: Path, hold_limit_ms: float) -> Path:
    """Write YAML, mkdir, exec ramulator2. Returns the requests_out.csv path."""
    run_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = run_dir / "config.yaml"
    run_log   = run_dir / "run.log"
    out_csv   = run_dir / "requests_out.csv"
    yaml_path.write_text(yaml_text(policy, run_dir, hold_limit_ms))

    print(f"[run] {letter} ({policy})"
          + (f" hold_limit={hold_limit_ms}ms" if policy == "hold_then_fallback" else "")
          + f" -> {run_dir.relative_to(REPO)}")
    result = subprocess.run(
        [str(RAMU_BIN), "-f", str(yaml_path)],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    run_log.write_text(result.stdout + "\n---STDERR---\n" + result.stderr)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"ramulator2 failed for {letter} ({policy})")
    (run_dir / "ramulator_summary.yaml").write_text(result.stdout)
    return out_csv

# ── Cost table (for predicted-vs-actual plot) ─────────────────────────────────

def _load_cost_table() -> dict[str, pd.DataFrame]:
    """Load both cost CSVs, return {route_name: df}."""
    tables = {}
    for route, path in [("gpu_only", REPO / COST_GPU_CSV),
                        ("lpddr5_pim_bank", REPO / COST_HYBRID_CSV)]:
        if path.exists():
            tables[route] = pd.read_csv(path)
    return tables

def _predict_e2e(row: pd.Series, cost_tables: dict[str, pd.DataFrame]) -> Optional[float]:
    """Nearest-neighbour lookup in the cost table for one request row."""
    route = str(row.get("route", ""))
    if route not in cost_tables:
        return None
    df = cost_tables[route]
    lin  = int(row["context_tokens"])
    lout = int(row["generated_tokens"])
    dist = (df["Lin"] - lin).abs() + (df["Lout"] - lout).abs()
    best = df.loc[dist.idxmin()]
    # s_time = prefill e2e; g_time (ms) = per-decode-step e2e
    prefill_ms = float(best["s_time"])
    decode_ms  = float(best["g_time (ms)"]) * max(0, lout - 1)
    return prefill_ms + decode_ms

# ── Summary stats ─────────────────────────────────────────────────────────────

def summarize(label: str, csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    d  = df[df["state"] == "done"]
    pim = int((d["route"] == "lpddr5_pim_bank").sum())
    gpu = int((d["route"] == "gpu_only").sum())
    def pct(col, q): return float(d[col].quantile(q))
    return {
        "label":    label,
        "csv":      csv_path,
        "done":     len(d),
        "pim":      pim,
        "gpu":      gpu,
        "ttft_p50": pct("ttft_ms", .5),   "ttft_p90": pct("ttft_ms", .9),
        "ttft_p99": pct("ttft_ms", .99),  "ttft_max": float(d["ttft_ms"].max()),
        "e2e_p50":  pct("e2e_ms",  .5),   "e2e_p90":  pct("e2e_ms",  .9),
        "e2e_p99":  pct("e2e_ms",  .99),  "e2e_max":  float(d["e2e_ms"].max()),
        "tbt_p50":  pct("tbt_mean_ms", .5), "tbt_p90": pct("tbt_mean_ms", .9),
        "tbt_max":  float(d["tbt_mean_ms"].max()),
        "adm_mean": float(d["admission_attempts"].mean()),
        "adm_max":  int(d["admission_attempts"].max()),
        "span_ms":  float(d["completion_ms"].max()),
    }

# ── Per-run plots ─────────────────────────────────────────────────────────────

def plot_execution_timeline(df: pd.DataFrame, label: str, out_path: Path) -> None:
    """Gantt-style prefill/decode bars per request, sorted by arrival time."""
    done = df[df["state"] == "done"].copy().sort_values(["arrival_ms", "request_id"])
    fig_h = min(40, max(4, 0.22 * len(done) + 1.5))
    fig, ax = plt.subplots(figsize=(12, fig_h))

    yticks, ylabels = [], []
    for y, (_, row) in enumerate(done.iterrows()):
        arrival     = float(row["arrival_ms"])
        first_token = arrival + float(row["ttft_ms"])
        finish      = arrival + float(row["e2e_ms"])
        yticks.append(y + 0.4)
        ylabels.append(f"r{int(row['request_id'])}")

        if first_token > arrival:
            ax.broken_barh([(arrival, first_token - arrival)], (y + 0.05, 0.7),
                           facecolors=PREFILL_COLOR, edgecolors="black", linewidth=0.4)
        if finish > first_token:
            ax.broken_barh([(first_token, finish - first_token)], (y + 0.05, 0.7),
                           facecolors=DECODE_COLOR, edgecolors="black", linewidth=0.4)
        if len(done) <= 200:
            ax.text(finish + max(1, finish * 0.005), y + 0.4,
                    str(row.get("route", "")), va="center", fontsize=7)

    handles = [plt.Rectangle((0,0),1,1, facecolor=PREFILL_COLOR, edgecolor="black", lw=0.4),
               plt.Rectangle((0,0),1,1, facecolor=DECODE_COLOR,  edgecolor="black", lw=0.4)]
    ax.legend(handles, ["prefill window", "decode window"], loc="lower right")
    tick_step = max(1, math.ceil(len(yticks) / 60))
    ax.set_yticks(yticks[::tick_step])
    ax.set_yticklabels(ylabels[::tick_step], fontsize=7)
    ax.set_xlabel("Simulation time (ms)")
    ax.set_ylabel("Request")
    ax.set_title(f"Request Execution Timeline — {label}")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  [plot] {out_path.name}")


def plot_route_mix(df: pd.DataFrame, label: str, out_path: Path) -> None:
    """Route counts bar + admission reject reason bar."""
    route_counts  = df["route"].fillna("none").value_counts().sort_index()
    reject_counts = df["admission_last_reject_reason"].replace("", "none") \
                      .fillna("none").value_counts().sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(label, fontsize=11)

    for ax, counts, title in [
        (axes[0], route_counts,  "Route Counts"),
        (axes[1], reject_counts, "Admission Reject Reasons"),
    ]:
        bars = ax.bar(counts.index.astype(str), counts.values, color="#1f77b4")
        ax.bar_label(bars, fmt="%.0f", padding=3, fontsize=9)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, axis="y", alpha=0.3)
        ymax = float(counts.max()) if len(counts) else 1
        ax.set_ylim(top=ymax * 1.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  [plot] {out_path.name}")


def plot_predicted_vs_actual(df: pd.DataFrame, label: str,
                             cost_tables: dict[str, pd.DataFrame],
                             out_path: Path) -> None:
    """Cost-model predicted finish vs actual finish, anchored at prefill_start_ms.

    The x-axis is  prefill_start_ms + cost_model_e2e  — "if the cost model is
    right about execution time, when should this request finish given when it
    actually started executing?"

    The y-axis is  completion_ms  — when it actually finished.

    Anchoring on prefill_start_ms (not arrival_ms) removes queuing / KV-OOM
    hold delay from the comparison so we are testing execution accuracy only,
    not the scheduler's ability to predict queue depth.
    """
    if not cost_tables:
        return
    done = df[df["state"] == "done"].copy()
    if "prefill_start_ms" not in done.columns or "completion_ms" not in done.columns:
        return
    done["predicted_e2e_ms"] = done.apply(
        lambda r: _predict_e2e(r, cost_tables), axis=1)
    done = done.dropna(subset=["predicted_e2e_ms", "prefill_start_ms", "completion_ms"])
    if done.empty:
        return

    # Anchor on actual start time so queuing delay does not appear as prediction error.
    done["pred_finish"] = done["prefill_start_ms"] + done["predicted_e2e_ms"]
    done["act_finish"]  = done["completion_ms"]

    lost_mask = done["is_lost"].astype(str).str.lower().isin({"1", "true"})

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(done.loc[~lost_mask, "pred_finish"],
               done.loc[~lost_mask, "act_finish"],
               s=18, alpha=0.55, label="non-lost", color="#5785c1")
    if lost_mask.any():
        ax.scatter(done.loc[lost_mask, "pred_finish"],
                   done.loc[lost_mask, "act_finish"],
                   s=24, alpha=0.8, label="lost (SLO miss)", color="tab:orange")

    lo = min(done["pred_finish"].min(), done["act_finish"].min()) * 0.95
    hi = max(done["pred_finish"].max(), done["act_finish"].max()) * 1.05
    ax.plot([lo, hi], [lo, hi], "--", color="black", linewidth=1.5, label="ideal")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Predicted Finish (ms)\n[prefill_start + cost_model_e2e]")
    ax.set_ylabel("Actual Finish = completion_ms")
    ax.set_title(f"Predicted vs Actual Finish — {label}")
    ax.ticklabel_format(style="plain", useOffset=False)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"  [plot] {out_path.name}")

# ── Combined ABC comparison plot ──────────────────────────────────────────────

def plot_abc_comparison(stats: list[dict], frames: list[pd.DataFrame],
                        hold_limit_ms: float, out_path: Path) -> None:
    labels     = [s["label"] for s in stats]
    bar_colors = [POLICY_COLORS[s["label"][0]] for s in stats]
    x = np.arange(len(stats))
    w = 0.25

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        f"serving_online KV-OOM policy comparison  "
        f"(A=hold, B=fallback_gpu, C=hold_then_fallback @ {hold_limit_ms:g} ms)",
        fontsize=13)

    # Route mix — stacked bar
    ax = axes[0, 0]
    pim = [s["pim"] for s in stats]
    gpu = [s["gpu"] for s in stats]
    ax.bar(labels, pim, label="PIM (lpddr5_pim_bank)", color="#1f77b4")
    ax.bar(labels, gpu, bottom=pim, label="GPU (gpu_only)", color="#ff7f0e")
    ax.set_ylabel("Requests"); ax.set_title("Route mix"); ax.legend()
    for i, (p, g) in enumerate(zip(pim, gpu)):
        ax.text(i, p + g + 0.4, f"{p}/{g}", ha="center", fontsize=9)

    # TTFT percentile bars
    ax = axes[0, 1]
    for shift, q, alpha, qlabel in [(-w, "ttft_p50", 0.50, "p50"),
                                     ( 0, "ttft_p90", 0.75, "p90"),
                                     ( w, "ttft_p99", 1.00, "p99")]:
        vals = [s[q] for s in stats]
        bars = ax.bar(x + shift, vals, w, color=bar_colors, alpha=alpha, label=qlabel)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("TTFT (ms, log)"); ax.set_yscale("log")
    ax.set_title("TTFT (lower is better)"); ax.legend()
    ax.grid(axis="y", which="both", alpha=0.25)

    # E2E percentile bars
    ax = axes[0, 2]
    for shift, q, alpha, qlabel in [(-w, "e2e_p50", 0.50, "p50"),
                                     ( 0, "e2e_p90", 0.75, "p90"),
                                     ( w, "e2e_p99", 1.00, "p99")]:
        ax.bar(x + shift, [s[q] for s in stats], w,
               color=bar_colors, alpha=alpha, label=qlabel)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("E2E (ms)"); ax.set_title("E2E (lower is better)")
    ax.legend(); ax.grid(axis="y", alpha=0.25)

    # TTFT CDF
    ax = axes[1, 0]
    for s, df, c in zip(stats, frames, bar_colors):
        v = np.sort(df[df["state"] == "done"]["ttft_ms"].values)
        ax.plot(v, np.linspace(0, 1, len(v)), label=s["label"], color=c, lw=2)
    ax.set_xlabel("TTFT (ms)"); ax.set_ylabel("CDF")
    ax.set_xscale("symlog", linthresh=10); ax.set_title("TTFT CDF")
    ax.legend(); ax.grid(alpha=0.25)

    # E2E CDF
    ax = axes[1, 1]
    for s, df, c in zip(stats, frames, bar_colors):
        v = np.sort(df[df["state"] == "done"]["e2e_ms"].values)
        ax.plot(v, np.linspace(0, 1, len(v)), label=s["label"], color=c, lw=2)
    ax.set_xlabel("E2E (ms)"); ax.set_ylabel("CDF")
    ax.set_title("E2E CDF"); ax.legend(); ax.grid(alpha=0.25)

    # Admission attempts (sorted)
    ax = axes[1, 2]
    for s, df, c in zip(stats, frames, bar_colors):
        v = np.sort(df[df["state"] == "done"]["admission_attempts"].values)
        ax.plot(np.arange(len(v)), v, label=s["label"], color=c, lw=2)
    ax.set_xlabel("Request rank (sorted)"); ax.set_ylabel("Admission attempts")
    ax.set_yscale("log"); ax.set_title("Admission attempts")
    ax.legend(); ax.grid(which="both", alpha=0.25)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"\n[plot] {out_path.name}")

# ── Text table ────────────────────────────────────────────────────────────────

def print_table(stats: list[dict]) -> None:
    cols = [
        ("Routes (PIM/GPU)",        lambda s: f"{s['pim']}/{s['gpu']}"),
        ("Adm attempts mean/max",   lambda s: f"{s['adm_mean']:.1f}/{s['adm_max']}"),
        ("TTFT p50/p90 (ms)",       lambda s: f"{s['ttft_p50']:.0f}/{s['ttft_p90']:.0f}"),
        ("TTFT p99 (ms)",           lambda s: f"{s['ttft_p99']:.0f}"),
        ("E2E p50/p90 (ms)",        lambda s: f"{s['e2e_p50']:.0f}/{s['e2e_p90']:.0f}"),
        ("TBT mean p50/p90 (ms)",   lambda s: f"{s['tbt_p50']:.0f}/{s['tbt_p90']:.0f}"),
        ("Span (ms)",               lambda s: f"{s['span_ms']:.0f}"),
    ]
    headers = [s["label"] for s in stats]
    w_name  = max(len(c[0]) for c in cols) + 2
    w_col   = max(14, max(len(h) for h in headers) + 2)

    print(f"\n{'Metric':<{w_name}}" + "".join(f"{h:>{w_col}}" for h in headers))
    print("-" * (w_name + w_col * len(headers)))
    for name, fn in cols:
        print(f"{name:<{w_name}}" + "".join(f"{fn(s):>{w_col}}" for s in stats))

# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hold-limit-ms", type=float, default=200.0,
                    help="kv_oom_hold_limit_ms for policy C (default: 200)")
    ap.add_argument("--run-label", default="kv_oom_abc",
                    help="Label suffix for the timestamped run dir (default: kv_oom_abc)")
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="Override run dir (default: auto-timestamped under "
                         "cluster_outputs/online_serving_runs/)")
    ap.add_argument("--skip-run", action="store_true",
                    help="Skip simulation; re-plot from existing CSVs under --run-dir")
    args = ap.parse_args()

    # ── Timestamped run dir (mirrors run_realtime_online_serving_case.sh) ──────
    if args.run_dir is None:
        ts    = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        label = f"{args.run_label}_hold{int(args.hold_limit_ms)}ms"
        run_dir = RUN_ROOT / f"{ts}_{label}"
    else:
        run_dir = args.run_dir.resolve()

    plots_dir = run_dir / "plots"
    run_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    runs = [
        ("A", "hold",               "A_hold"),
        ("B", "fallback_gpu",       "B_fallback_gpu"),
        ("C", "hold_then_fallback", f"C_hold_then_fallback_{int(args.hold_limit_ms)}ms"),
    ]

    # ── Provenance ────────────────────────────────────────────────────────────
    (run_dir / "run_info.txt").write_text(
        f"run_dir={run_dir}\n"
        f"timestamp={_dt.datetime.now().isoformat(timespec='seconds')}\n"
        f"requests_csv={REQUESTS_CSV}\n"
        f"cost_gpu_csv={COST_GPU_CSV}\n"
        f"cost_hybrid_csv={COST_HYBRID_CSV}\n"
        f"hold_limit_ms={args.hold_limit_ms}\n"
        + "".join(f"{L}.policy={p} subdir={sub}\n" for L, p, sub in runs)
    )

    # ── Run simulations ───────────────────────────────────────────────────────
    csvs: list[Path] = []
    for letter, policy, subdir in runs:
        sub_dir  = run_dir / subdir
        csv_path = sub_dir / "requests_out.csv"
        if not args.skip_run:
            csv_path = run_one(letter, policy, sub_dir, args.hold_limit_ms)
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing {csv_path} — re-run without --skip-run")
        csvs.append(csv_path)

    labels = ["A (hold)",
              "B (fallback_gpu)",
              f"C (hold_then_fallback, {args.hold_limit_ms:g} ms)"]
    stats  = [summarize(lbl, p) for lbl, p in zip(labels, csvs)]
    frames = [pd.read_csv(p) for p in csvs]

    # ── Text summary ──────────────────────────────────────────────────────────
    print_table(stats)
    buf = io.StringIO()
    import builtins
    _real_print = builtins.print
    builtins.print = lambda *a, **k: _real_print(*a, file=buf, **k)
    try:
        print_table(stats)
    finally:
        builtins.print = _real_print
    (run_dir / "abc_summary.txt").write_text(buf.getvalue())

    # ── Load cost tables for predicted-vs-actual plots ────────────────────────
    cost_tables = _load_cost_table()

    # ── Per-policy plots (timeline + route mix + predicted vs actual) ─────────
    for (letter, policy, subdir), label, df, csv_path in zip(runs, labels, frames, csvs):
        sub_plots = plots_dir / subdir
        sub_plots.mkdir(parents=True, exist_ok=True)
        slug = subdir.lower()
        print(f"\n[plots] {label}")
        plot_execution_timeline(
            df, label,
            sub_plots / f"{slug}_request_execution_timeline.png")
        plot_route_mix(
            df, label,
            sub_plots / f"{slug}_route_mix.png")
        plot_predicted_vs_actual(
            df, label, cost_tables,
            sub_plots / f"{slug}_predicted_finish_vs_actual_finish.png")

    # ── Combined ABC comparison plot ──────────────────────────────────────────
    plot_abc_comparison(stats, frames, args.hold_limit_ms,
                        plots_dir / "serving_online_kv_oom_abc.png")

    print(f"\n[run_dir] {run_dir}")
    print(f"[plots]   {plots_dir}")


if __name__ == "__main__":
    main()
