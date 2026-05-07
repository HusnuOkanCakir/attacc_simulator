#!/usr/bin/env python3
"""
Compare KV scheduler policies for serving_online and plot the results.

Four configs run against the same Azure first-50 trace and KV pool:

  guaranteed_no_evict  — reserve full worst-case KV at admission, never evict.
                         kv_oom_policy: hold. Regression anchor.
  max_util_full        — max_utilization + kv_evict_granularity=full. Whole-
                         request LIFO preemption (old behavior).
  max_util_tail        — max_utilization + kv_evict_granularity=tail. Free
                         only the victim's most recent decode pages; victim
                         stays in WaitingDecode and recomputes the trimmed tail.
  max_util_tail_pred   — max_util_tail + predictive_admission=true. Oracle
                         size-aware gate holds long requests under pressure.

Usage
-----
  python tools/run_serving_online_policy_compare.py
  python tools/run_serving_online_policy_compare.py --skip-run --run-dir <path>  # re-plot
"""

from __future__ import annotations

import argparse
import datetime as _dt
import io
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml as _yaml

# ── Repo layout ───────────────────────────────────────────────────────────────

REPO     = Path(__file__).resolve().parent.parent
RAMU_BIN = REPO / "ramulator2/build/ramulator2"
RUN_ROOT = REPO / "cluster_outputs/online_serving_runs"

REQUESTS_CSV_DEFAULT = "cluster_outputs/azure/AzureLLMInferenceTrace_conv_first50.csv"
REQUESTS_CSV_FULL    = "cluster_outputs/azure/AzureLLMInferenceTrace_conv.csv"

# These module-level vars are overridden by CLI args in main().
REQUESTS_CSV    = REQUESTS_CSV_DEFAULT
ARRIVAL_SCALE   = 0.1
COST_GPU_CSV    = "cluster_outputs/cost_tables_full_energy/gpu_only.csv"
COST_HYBRID_CSV = "cluster_outputs/cost_tables_full_energy/lpddr5_pim_bank.csv"
COST_GPU_MODEL    = ""  # non-empty → ML tree inference overrides nearest-neighbor
COST_HYBRID_MODEL = ""
DEBUG_LOG         = False  # write per-translation debug log (slow; off by default)
MAX_ACTIVE        = 0      # 0 = unlimited (legacy); >0 caps concurrent active requests
KV_POOL_BYTES     = 1503238553  # ~1.4 GiB; override via --kv-pool-gb
SLO_E2E_MS        = 900.0   # E2E latency SLO; -1 = disabled
SLO_TTFT_MS       = -1.0    # TTFT SLO; -1 = disabled
KV_OOM_POLICY     = "hold"  # hold | fallback_gpu | hold_then_fallback
KV_OOM_HOLD_LIMIT_MS = 50.0 # max hold time before GPU fallback (hold_then_fallback only)
PIM_COMMAND_MODE  = "realistic"  # realistic | simple — diagnostic A/B knob
PIM_CACHE_ENABLED = True         # cache PIM-decode drain cycles by shape (Lever 1)

# Model architecture (filled from MODEL_PRESETS in main()).
NUM_LAYERS           = 40
NUM_HEADS            = 8
D_HEAD               = 256
DTYPE_BYTES          = 2
VISION_PREFIX_TOKENS = 0

# Per-model defaults: architecture + cost-table paths. Selected via --model.
# CLI flags --n-layers / --n-heads / --d-head / --vision-prefix-tokens override
# the preset for ablation runs. CLI flags --ml / --dense pick model-specific
# alternatives derived from the preset paths.
MODEL_PRESETS: dict[str, dict] = {
    "pi0": dict(
        num_layers=40, num_heads=8, d_head=256, dtype_bytes=2,
        vision_prefix_tokens=0,
        cost_dir_csv="cluster_outputs/cost_tables_full_energy",
        cost_dir_models="cluster_outputs/cost_models_full_energy",
    ),
    "openvla": dict(
        num_layers=32, num_heads=32, d_head=128, dtype_bytes=2,
        vision_prefix_tokens=256,
        cost_dir_csv="cluster_outputs/cost_tables_full_energy_openvla",
        cost_dir_models="cluster_outputs/cost_models_full_energy_openvla",
    ),
}

POLICY_COLORS = {
    "guaranteed_no_evict": "#1f77b4",
    "max_util_full":       "#d62728",
    "max_util_tail":       "#2ca02c",
    "max_util_tail_pred":  "#9467bd",
    "max_util_tail_x4":    "#ff7f0e",
    "max_util_tail_x8":    "#8c564b",
}

# Per-label knobs passed into the YAML template.
POLICY_SETTINGS = {
    "guaranteed_no_evict": dict(
        kv_scheduler_policy="guaranteed_no_evict",
    ),
    "max_util_full": dict(
        kv_scheduler_policy="max_utilization",
        kv_decode_chunk_tokens=32,
        kv_evict_granularity="full",
    ),
    "max_util_tail": dict(
        kv_scheduler_policy="max_utilization",
        kv_decode_chunk_tokens=32,
        kv_evict_granularity="tail",
    ),
    "max_util_tail_pred": dict(
        kv_scheduler_policy="max_utilization",
        kv_decode_chunk_tokens=32,
        kv_evict_granularity="tail",
        predictive_admission=True,
        predictive_pressure_threshold=1.0,
        predictive_hold_ms=50.0,
    ),
    "max_util_tail_x4": dict(
        kv_scheduler_policy="max_utilization",
        kv_decode_chunk_tokens=32,
        kv_evict_granularity="tail",
        kv_tail_trim_multiplier=4,
    ),
    "max_util_tail_x8": dict(
        kv_scheduler_policy="max_utilization",
        kv_decode_chunk_tokens=32,
        kv_evict_granularity="tail",
        kv_tail_trim_multiplier=8,
    ),
}

# ── YAML ──────────────────────────────────────────────────────────────────────

def yaml_text(label: str, run_dir: Path) -> str:
    """Render the YAML for one run. label selects the scheduler policy."""
    out_csv   = run_dir.relative_to(REPO) / "requests_out.csv"
    debug_log = (run_dir.relative_to(REPO) / "debug.log") if DEBUG_LOG else ""

    settings = POLICY_SETTINGS[label]
    lines = []
    for key, value in settings.items():
        if isinstance(value, bool):
            lines.append(f"  {key}: {'true' if value else 'false'}")
        else:
            lines.append(f"  {key}: {value}")
    policy_lines = "\n".join(lines) + "\n"

    ml_lines = ""
    if COST_GPU_MODEL:
        ml_lines += f"  cost_gpu_model: {COST_GPU_MODEL}\n"
    if COST_HYBRID_MODEL:
        ml_lines += f"  cost_hybrid_model: {COST_HYBRID_MODEL}\n"

    # Translation max_addr must be >= kv_pool_bytes (the allocator hands out
    # physical addresses up to kv_pool_bytes). Round up to the next GiB.
    MAX_ADDR = max(KV_POOL_BYTES, 1 << 31)
    if MAX_ADDR & ((1 << 30) - 1):
        MAX_ADDR = ((MAX_ADDR >> 30) + 1) << 30

    PIM_CACHE_ENABLED_YAML = "true" if PIM_CACHE_ENABLED else "false"

    return f"""Frontend:
  impl: ServingOnlineFrontend
  clock_ratio: 1

  requests_csv: {REQUESTS_CSV}
  cost_gpu_csv: {COST_GPU_CSV}
  cost_hybrid_csv: {COST_HYBRID_CSV}
{ml_lines}

  requests_out_csv: {out_csv}
  debug_log_path: {debug_log}

  gpu_route_name: gpu_only
  hybrid_route_name: lpddr5_pim_bank

  route_policy: min_finish
  energy_latency_guard_ms: 20.0

  max_decode_batch_size: 8
  prompt_priority: true
  max_consecutive_decode_batches: 4
  max_active_requests: {MAX_ACTIVE}

  admission_max_wait_ms: 150.0
  admission_retry_interval_ms: 10.0
  kv_oom_policy: {KV_OOM_POLICY}
  kv_oom_hold_limit_ms: {KV_OOM_HOLD_LIMIT_MS}
{policy_lines}
  slo_e2e_ms: {SLO_E2E_MS}
  slo_ttft_ms: {SLO_TTFT_MS}

  arrival_time_scale: {ARRIVAL_SCALE}
  arrival_limit: -1

  kv_pool_bytes: {KV_POOL_BYTES}

  generator_num_layers: {NUM_LAYERS}
  generator_num_heads: {NUM_HEADS}
  generator_dhead: {D_HEAD}
  generator_dtype_bytes: {DTYPE_BYTES}
  vision_prefix_tokens: {VISION_PREFIX_TOKENS}
  pim_command_mode: {PIM_COMMAND_MODE}
  pim_cache_enabled: {PIM_CACHE_ENABLED_YAML}
  generator_channel_count: 16
  translation_pagesize_KB: 4

  Translation:
    impl: ServingOnlineTranslation
    max_addr: {MAX_ADDR}

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

def run_one(label: str, run_dir: Path) -> tuple[Path, dict]:
    """Write YAML, mkdir, exec ramulator2. Returns (csv_path, summary_dict)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = run_dir / "config.yaml"
    run_log   = run_dir / "run.log"
    out_csv   = run_dir / "requests_out.csv"
    yaml_path.write_text(yaml_text(label, run_dir))

    print(f"[run] {label} -> {run_dir.relative_to(REPO)}")
    result = subprocess.run(
        [str(RAMU_BIN), "-f", str(yaml_path)],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    run_log.write_text(result.stdout + "\n---STDERR---\n" + result.stderr)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"ramulator2 failed for {label}")
    summary_path = run_dir / "ramulator_summary.yaml"
    summary_path.write_text(result.stdout)
    summary = _load_summary(summary_path)
    return out_csv, summary


def _load_summary(path: Path) -> dict:
    try:
        data = _yaml.safe_load(path.read_text()) or {}
    except _yaml.YAMLError:
        return {}
    fe = data.get("Frontend", {}) or {}
    return {
        "admitted":          int(fe.get("admitted_requests", 0)),
        "completed":         int(fe.get("completed_requests", 0)),
        "dropped":           int(fe.get("dropped_requests", 0)),
        "kv_oom_holds":      int(fe.get("kv_oom_holds", 0)),
        "preempt_count":     int(fe.get("preempt_count", 0)),
        "tail_trim_count":   int(fe.get("tail_trim_count", 0)),
        "full_evict_count":  int(fe.get("full_evict_count", 0)),
        "tokens_recomputed": int(fe.get("tokens_recomputed", 0)),
        "predictive_holds":  int(fe.get("predictive_holds", 0)),
        "scheduler_policy":  str(fe.get("kv_scheduler_policy", "")),
    }

# ── Summary stats ─────────────────────────────────────────────────────────────

def summarize(label: str, csv_path: Path, summary: dict) -> dict:
    df = pd.read_csv(csv_path)
    d  = df[df["state"] == "done"]
    pim = int((d["route"] == "lpddr5_pim_bank").sum())
    gpu = int((d["route"] == "gpu_only").sum())
    def pct(col, q): return float(d[col].quantile(q)) if len(d) else float("nan")
    span_ms = float(d["completion_ms"].max()) if len(d) else float("nan")
    throughput_rps = len(d) / (span_ms / 1000.0) if span_ms and span_ms > 0 else float("nan")
    preempt_total = int(d["preempt_count"].sum()) if "preempt_count" in d else 0

    # Effective recomputation estimate: for tail mode the scheduler already
    # reports tokens_recomputed (it's summed across trim events). For the full-
    # evict path we have to impute: each full preempt recomputes prefill +
    # completed_decodes. We don't have that detail in the summary — use
    # per-request prefill_end_ms - admission_ms bookkeeping is overkill, so
    # approximate with generated_tokens mean × preempt_count as an upper bound.
    tokens_recomputed_reported = summary.get("tokens_recomputed", 0)
    if summary.get("full_evict_count", 0) > 0 and tokens_recomputed_reported == 0:
        # Legacy: old preempt_lru did not fill tokens_recomputed. Estimate
        # as full_evicts × mean generated_tokens.
        gen_mean = float(df["generated_tokens"].mean()) if len(df) else 0.0
        tokens_recomputed_est = int(summary["full_evict_count"] * gen_mean)
    else:
        tokens_recomputed_est = tokens_recomputed_reported

    return {
        "label":    label,
        "csv":      csv_path,
        "done":     len(d),
        "pim":      pim,
        "gpu":      gpu,
        "ttft_p50": pct("ttft_ms", .5),   "ttft_p90": pct("ttft_ms", .9),
        "ttft_p99": pct("ttft_ms", .99),  "ttft_max": float(d["ttft_ms"].max()) if len(d) else float("nan"),
        "e2e_p50":  pct("e2e_ms",  .5),   "e2e_p90":  pct("e2e_ms",  .9),
        "e2e_p99":  pct("e2e_ms",  .99),  "e2e_max":  float(d["e2e_ms"].max()) if len(d) else float("nan"),
        "span_ms":  span_ms,
        "throughput_rps": throughput_rps,
        "admitted":          summary.get("admitted", 0),
        "dropped":           summary.get("dropped", 0),
        "kv_oom_holds":      summary.get("kv_oom_holds", 0),
        "preempt_count":     summary.get("preempt_count", preempt_total),
        "tail_trim_count":   summary.get("tail_trim_count", 0),
        "full_evict_count":  summary.get("full_evict_count", 0),
        "tokens_recomputed": tokens_recomputed_est,
        "predictive_holds":  summary.get("predictive_holds", 0),
        "scheduler_policy":  summary.get("scheduler_policy", ""),
    }

# ── Combined comparison plot (2×3 grid) ───────────────────────────────────────

def plot_policy_comparison(stats: list[dict], frames: list[pd.DataFrame],
                           out_path: Path) -> None:
    labels     = [s["label"] for s in stats]
    bar_colors = [POLICY_COLORS[s["label"]] for s in stats]
    x = np.arange(len(stats))
    w = 0.25

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        "serving_online KV scheduler policy comparison "
        "(guaranteed_no_evict vs max_util_full vs max_util_tail vs max_util_tail_pred)",
        fontsize=11)

    # 1. Route mix — stacked bar
    ax = axes[0, 0]
    pim = [s["pim"] for s in stats]
    gpu = [s["gpu"] for s in stats]
    ax.bar(labels, pim, label="PIM (lpddr5_pim_bank)", color="#1f77b4")
    ax.bar(labels, gpu, bottom=pim, label="GPU (gpu_only)", color="#ff7f0e")
    ax.set_ylabel("Completed requests"); ax.set_title("Route mix"); ax.legend()
    for i, (p, g) in enumerate(zip(pim, gpu)):
        ax.text(i, p + g + 0.4, f"{p}/{g}", ha="center", fontsize=9)
    ax.tick_params(axis="x", rotation=20)

    # 2. TTFT percentile bars
    ax = axes[0, 1]
    for shift, q, alpha, qlabel in [(-w, "ttft_p50", 0.50, "p50"),
                                     ( 0, "ttft_p90", 0.75, "p90"),
                                     ( w, "ttft_p99", 1.00, "p99")]:
        ax.bar(x + shift, [s[q] for s in stats], w,
               color=bar_colors, alpha=alpha, label=qlabel)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("TTFT (ms, log)"); ax.set_yscale("log")
    ax.set_title("TTFT (lower is better)"); ax.legend()
    ax.grid(axis="y", which="both", alpha=0.25)

    # 3. E2E percentile bars
    ax = axes[0, 2]
    for shift, q, alpha, qlabel in [(-w, "e2e_p50", 0.50, "p50"),
                                     ( 0, "e2e_p90", 0.75, "p90"),
                                     ( w, "e2e_p99", 1.00, "p99")]:
        ax.bar(x + shift, [s[q] for s in stats], w,
               color=bar_colors, alpha=alpha, label=qlabel)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("E2E (ms)"); ax.set_title("E2E (lower is better)")
    ax.legend(); ax.grid(axis="y", alpha=0.25)

    # 4. TTFT CDF
    ax = axes[1, 0]
    for s, df, c in zip(stats, frames, bar_colors):
        v = np.sort(df[df["state"] == "done"]["ttft_ms"].values)
        if len(v) == 0: continue
        ax.plot(v, np.linspace(0, 1, len(v)), label=s["label"], color=c, lw=2)
    ax.set_xlabel("TTFT (ms)"); ax.set_ylabel("CDF")
    ax.set_xscale("symlog", linthresh=10); ax.set_title("TTFT CDF")
    ax.legend(); ax.grid(alpha=0.25)

    # 5. E2E CDF
    ax = axes[1, 1]
    for s, df, c in zip(stats, frames, bar_colors):
        v = np.sort(df[df["state"] == "done"]["e2e_ms"].values)
        if len(v) == 0: continue
        ax.plot(v, np.linspace(0, 1, len(v)), label=s["label"], color=c, lw=2)
    ax.set_xlabel("E2E (ms)"); ax.set_ylabel("CDF")
    ax.set_title("E2E CDF"); ax.legend(); ax.grid(alpha=0.25)

    # 6. Bars: preempts + tokens_recomputed + predictive_holds. Preempts and
    # predictive_holds on the left axis (count, symlog); tokens_recomputed on a
    # twin right axis (tokens, linear).
    ax = axes[1, 2]
    gw = 0.28
    offsets = [-gw, 0.0]
    # Left axis: preempts, predictive_holds.
    left_metrics = [
        ("preempt_count",    offsets[0]),
        ("predictive_holds", offsets[1]),
    ]
    for m, off in left_metrics:
        vals = [s[m] for s in stats]
        bars = ax.bar(x + off, vals, gw, label=m,
                      alpha=0.85,
                      color=[POLICY_COLORS[s["label"]] for s in stats],
                      hatch="//" if m == "predictive_holds" else None,
                      edgecolor="black", linewidth=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                    f"{v}", ha="center", va="bottom", fontsize=7)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylabel("Preempts / pred holds (symlog)"); ax.grid(axis="y", alpha=0.25)

    ax_r = ax.twinx()
    vals = [s["tokens_recomputed"] for s in stats]
    bars = ax_r.bar(x + gw, vals, gw, label="tokens_recomputed",
                    color="#555555", alpha=0.6,
                    edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, vals):
        ax_r.text(b.get_x() + b.get_width()/2, b.get_height(),
                  f"{v}", ha="center", va="bottom", fontsize=7)
    ax_r.set_ylabel("Tokens recomputed")

    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20)
    ax.set_title("Preempts / pred holds / tokens recomputed")

    # Combine legends (left + right axis) in one box.
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_r.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"\n[plot] {out_path.name}")

# ── Text summary table ────────────────────────────────────────────────────────

def print_table(stats: list[dict]) -> None:
    cols = [
        ("Routes (PIM/GPU)",        lambda s: f"{s['pim']}/{s['gpu']}"),
        ("Admitted/Dropped",        lambda s: f"{s['admitted']}/{s['dropped']}"),
        ("KV-OOM holds",            lambda s: f"{s['kv_oom_holds']}"),
        ("Preempt count (tail/full)", lambda s: f"{s['preempt_count']} ({s['tail_trim_count']}/{s['full_evict_count']})"),
        ("Tokens recomputed",       lambda s: f"{s['tokens_recomputed']}"),
        ("Predictive holds",        lambda s: f"{s['predictive_holds']}"),
        ("TTFT p50/p90 (ms)",       lambda s: f"{s['ttft_p50']:.0f}/{s['ttft_p90']:.0f}"),
        ("TTFT p99 (ms)",           lambda s: f"{s['ttft_p99']:.0f}"),
        ("E2E p50/p90 (ms)",        lambda s: f"{s['e2e_p50']:.0f}/{s['e2e_p90']:.0f}"),
        ("E2E p99 (ms)",            lambda s: f"{s['e2e_p99']:.0f}"),
        ("Span (ms)",               lambda s: f"{s['span_ms']:.0f}"),
        ("Throughput (req/s)",      lambda s: f"{s['throughput_rps']:.2f}"),
    ]
    headers = [s["label"] for s in stats]
    w_name  = max(len(c[0]) for c in cols) + 2
    w_col   = max(22, max(len(h) for h in headers) + 2)

    print(f"\n{'Metric':<{w_name}}" + "".join(f"{h:>{w_col}}" for h in headers))
    print("-" * (w_name + w_col * len(headers)))
    for name, fn in cols:
        print(f"{name:<{w_name}}" + "".join(f"{fn(s):>{w_col}}" for s in stats))

# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    global REQUESTS_CSV, ARRIVAL_SCALE

    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="Override run dir (default: auto-timestamped under "
                         "cluster_outputs/online_serving_runs/)")
    ap.add_argument("--skip-run", action="store_true",
                    help="Skip simulation; re-plot from existing CSVs under --run-dir")
    ap.add_argument("--n-requests", type=int, default=50,
                    help="Use first N requests from the full Azure conv trace. "
                         "A slice CSV is created automatically if needed. Default: 50.")
    ap.add_argument("--arrival-scale", type=float, default=0.1,
                    help="arrival_time_scale passed to the simulator. "
                         "< 1.0 speeds up arrivals. Default: 0.1.")
    ap.add_argument("--max-active", type=int, default=0,
                    help="Cap on concurrent active requests in the scheduler. "
                         "0 = unlimited (legacy). Recommended: 4× max_decode_batch_size = 32. "
                         "Bounds per-tick scan cost; required for large N to avoid O(n²) blowup.")
    ap.add_argument("--model", choices=sorted(MODEL_PRESETS.keys()), default="pi0",
                    help="Model preset selecting architecture (num_layers/num_heads/d_head) "
                         "and the matching cost tables / ML models. Default: pi0.")
    ap.add_argument("--n-layers", type=int, default=None,
                    help="Override generator_num_layers from the model preset.")
    ap.add_argument("--n-heads", type=int, default=None,
                    help="Override generator_num_heads from the model preset.")
    ap.add_argument("--d-head", type=int, default=None,
                    help="Override generator_dhead from the model preset.")
    ap.add_argument("--vision-prefix-tokens", type=int, default=None,
                    help="Override vision_prefix_tokens from the model preset (e.g. 0 to "
                         "treat OpenVLA as text-only for ablation).")
    ap.add_argument("--kv-pool-gb", type=float, default=None,
                    help="PIM KV pool size in GiB. Default: 1.4 GiB (matches Pi0 baseline). "
                         "OpenVLA needs ~2× more per request, so 4 GiB recommended for n>=20.")
    ap.add_argument("--policies", nargs="+",
                    choices=["guaranteed_no_evict", "max_util_full",
                             "max_util_tail", "max_util_tail_pred",
                             "max_util_tail_x4", "max_util_tail_x8"],
                    default=None,
                    help="Subset of policies to run/plot. Default: all four.")
    ap.add_argument("--dense", action="store_true",
                    help="Use dense ML-predicted cost tables (*_dense.csv) instead "
                         "of the sparse originals. Generate with tools/gen_dense_cost_table.py.")
    ap.add_argument("--ml", action="store_true",
                    help="Use ML tree-ensemble models (.bin) for cost estimation instead "
                         "of nearest-neighbor CSV lookup. Requires the .bin files generated "
                         "by tools/export_cost_model_trees.py.")
    ap.add_argument("--slo-e2e-ms", type=float, default=None,
                    help="E2E latency SLO in ms. Requests predicted to miss it may be rerouted "
                         "to GPU. -1 disables. Default: 900.")
    ap.add_argument("--slo-ttft-ms", type=float, default=None,
                    help="TTFT (time-to-first-token) SLO in ms. -1 disables. Default: -1.")
    ap.add_argument("--kv-oom-policy",
                    choices=["hold", "fallback_gpu", "hold_then_fallback"], default=None,
                    help="Behavior when PIM KV pool is full at admission. "
                         "'hold': wait until memory frees (default). "
                         "'fallback_gpu': immediately reroute to GPU. "
                         "'hold_then_fallback': hold up to --kv-oom-hold-limit-ms then reroute.")
    ap.add_argument("--kv-oom-hold-limit-ms", type=float, default=None,
                    help="Max hold time before GPU fallback under hold_then_fallback. Default: 50.")
    ap.add_argument("--debug-log", action="store_true",
                    help="Write per-translation debug.log for each run (very slow; off by default).")
    ap.add_argument("--pim-command-mode", choices=["realistic", "simple"], default=None,
                    help="PIM command stream complexity. 'realistic' (default) emits the full "
                         "per-block LPDDR5-PIM sequence (16K-86K cmds/decode-step). 'simple' emits "
                         "only num_layers*num_heads*2 PIM_MAC_AB commands. Diagnostic A/B knob: "
                         "compare wall-clock to bisect DRAM-tick vs frontend overhead.")
    ap.add_argument("--no-pim-cache", action="store_true",
                    help="Disable per-shape PIM-decode cycle cache. By default, repeated decode "
                         "tasks with the same (route, ctx_bucket=ceil(ctx/32), batch_size) shape "
                         "skip Ramulator and reuse the first inline-measured drain cycles. "
                         "Disable for verification: cached vs uncached should produce identical "
                         "TTFT/E2E (just much slower without).")
    args = ap.parse_args()

    # Resolve requests CSV — create a slice if needed.
    n = args.n_requests
    if n == 50:
        REQUESTS_CSV = REQUESTS_CSV_DEFAULT
    else:
        slice_csv = REPO / f"cluster_outputs/azure/AzureLLMInferenceTrace_conv_first{n}.csv"
        if not slice_csv.exists():
            full = REPO / REQUESTS_CSV_FULL
            with open(full) as fin, open(slice_csv, "w") as fout:
                for i, line in enumerate(fin):
                    fout.write(line)
                    if i >= n:  # header + n data rows
                        break
            print(f"[csv] created {slice_csv.relative_to(REPO)}")
        REQUESTS_CSV = str(slice_csv.relative_to(REPO))
    ARRIVAL_SCALE = args.arrival_scale

    # Apply model preset (architecture + cost-table directories), then let
    # CLI flags override individual fields.
    preset = MODEL_PRESETS[args.model]
    global NUM_LAYERS, NUM_HEADS, D_HEAD, DTYPE_BYTES, VISION_PREFIX_TOKENS
    NUM_LAYERS  = args.n_layers if args.n_layers is not None else preset["num_layers"]
    NUM_HEADS   = args.n_heads  if args.n_heads  is not None else preset["num_heads"]
    D_HEAD      = args.d_head   if args.d_head   is not None else preset["d_head"]
    DTYPE_BYTES = preset["dtype_bytes"]
    VISION_PREFIX_TOKENS = (args.vision_prefix_tokens
                            if args.vision_prefix_tokens is not None
                            else preset["vision_prefix_tokens"])

    global COST_GPU_CSV, COST_HYBRID_CSV
    csv_dir = preset["cost_dir_csv"]
    if args.dense:
        COST_GPU_CSV    = f"{csv_dir}/gpu_only_dense.csv"
        COST_HYBRID_CSV = f"{csv_dir}/lpddr5_pim_bank_dense.csv"
    else:
        COST_GPU_CSV    = f"{csv_dir}/gpu_only.csv"
        COST_HYBRID_CSV = f"{csv_dir}/lpddr5_pim_bank.csv"
    if args.ml:
        global COST_GPU_MODEL, COST_HYBRID_MODEL
        model_dir = preset["cost_dir_models"]
        COST_GPU_MODEL    = f"{model_dir}/gpu_only_trees.bin"
        COST_HYBRID_MODEL = f"{model_dir}/lpddr5_pim_bank_trees.bin"
    if args.debug_log:
        global DEBUG_LOG
        DEBUG_LOG = True
    if args.max_active > 0:
        global MAX_ACTIVE
        MAX_ACTIVE = args.max_active
    if args.kv_pool_gb is not None:
        global KV_POOL_BYTES
        KV_POOL_BYTES = int(args.kv_pool_gb * (1 << 30))
    if args.slo_e2e_ms is not None:
        global SLO_E2E_MS
        SLO_E2E_MS = args.slo_e2e_ms
    if args.slo_ttft_ms is not None:
        global SLO_TTFT_MS
        SLO_TTFT_MS = args.slo_ttft_ms
    if args.kv_oom_policy is not None:
        global KV_OOM_POLICY
        KV_OOM_POLICY = args.kv_oom_policy
    if args.kv_oom_hold_limit_ms is not None:
        global KV_OOM_HOLD_LIMIT_MS
        KV_OOM_HOLD_LIMIT_MS = args.kv_oom_hold_limit_ms
    if args.pim_command_mode is not None:
        global PIM_COMMAND_MODE
        PIM_COMMAND_MODE = args.pim_command_mode
    if args.no_pim_cache:
        global PIM_CACHE_ENABLED
        PIM_CACHE_ENABLED = False
    print(f"[csv]           {REQUESTS_CSV}  ({n} requests)")
    print(f"[model]         {args.model}  layers={NUM_LAYERS} heads={NUM_HEADS} "
          f"d_head={D_HEAD} dtype_bytes={DTYPE_BYTES} vision_prefix={VISION_PREFIX_TOKENS}")
    print(f"[max_active]    {MAX_ACTIVE} (0 = unlimited)")
    print(f"[kv_pool]       {KV_POOL_BYTES} bytes ({KV_POOL_BYTES / (1<<30):.2f} GiB)")
    print(f"[arrival_scale] {ARRIVAL_SCALE}")
    print(f"[slo]           e2e={SLO_E2E_MS}ms  ttft={SLO_TTFT_MS}ms")
    print(f"[kv_oom]        policy={KV_OOM_POLICY}  hold_limit={KV_OOM_HOLD_LIMIT_MS}ms")
    print(f"[pim_cmd_mode]  {PIM_COMMAND_MODE}")
    print(f"[pim_cache]     {'enabled' if PIM_CACHE_ENABLED else 'disabled'}")
    print(f"[cost_tables]   gpu={COST_GPU_CSV}  hybrid={COST_HYBRID_CSV}")
    if COST_GPU_MODEL:
        print(f"[cost_models]   gpu={COST_GPU_MODEL}  hybrid={COST_HYBRID_MODEL}")

    if args.run_dir is None:
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = RUN_ROOT / f"{ts}_n{n}_scale{ARRIVAL_SCALE}_policy_compare"
    else:
        run_dir = args.run_dir.resolve()

    plots_dir = run_dir / "plots"
    run_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    all_runs = [
        ("guaranteed_no_evict",  "guaranteed_no_evict"),
        ("max_util_full",        "max_util_full"),
        ("max_util_tail",        "max_util_tail"),
        ("max_util_tail_pred",   "max_util_tail_pred"),
        ("max_util_tail_x4",     "max_util_tail_x4"),
        ("max_util_tail_x8",     "max_util_tail_x8"),
    ]
    selected = set(args.policies) if args.policies else {lbl for lbl, _ in all_runs}
    runs = [(lbl, sub) for lbl, sub in all_runs if lbl in selected]

    (run_dir / "run_info.txt").write_text(
        f"run_dir={run_dir}\n"
        f"timestamp={_dt.datetime.now().isoformat(timespec='seconds')}\n"
        f"requests_csv={REQUESTS_CSV}\n"
        f"cost_gpu_csv={COST_GPU_CSV}\n"
        f"cost_hybrid_csv={COST_HYBRID_CSV}\n"
        f"model={args.model}\n"
        f"num_layers={NUM_LAYERS}\n"
        f"num_heads={NUM_HEADS}\n"
        f"d_head={D_HEAD}\n"
        f"dtype_bytes={DTYPE_BYTES}\n"
        f"vision_prefix_tokens={VISION_PREFIX_TOKENS}\n"
        f"n_requests={n}\n"
        f"arrival_scale={ARRIVAL_SCALE}\n"
        + "".join(f"{lbl} -> {sub}\n" for lbl, sub in runs)
    )

    # ── Run simulations ───────────────────────────────────────────────────────
    csvs: list[Path]  = []
    summaries: list[dict] = []
    for label, subdir in runs:
        sub_dir  = run_dir / subdir
        csv_path = sub_dir / "requests_out.csv"
        summary_path = sub_dir / "ramulator_summary.yaml"
        if not args.skip_run:
            csv_path, summary = run_one(label, sub_dir)
        else:
            if not csv_path.exists():
                raise FileNotFoundError(f"Missing {csv_path} — re-run without --skip-run")
            summary = _load_summary(summary_path) if summary_path.exists() else {}
        csvs.append(csv_path)
        summaries.append(summary)

    stats  = [summarize(lbl, p, smy)
              for (lbl, _), p, smy in zip(runs, csvs, summaries)]
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
    (run_dir / "policy_compare_summary.txt").write_text(buf.getvalue())

    # ── Combined comparison plot ──────────────────────────────────────────────
    combined_png = plots_dir / "serving_online_policy_compare.png"
    plot_policy_comparison(stats, frames, combined_png)

    # Also deposit the combined figure into docs/figures for convenience.
    canonical = REPO / "docs/figures/serving_online_policy_compare.png"
    canonical.parent.mkdir(parents=True, exist_ok=True)
    canonical.write_bytes(combined_png.read_bytes())
    print(f"[plot] {canonical.relative_to(REPO)}")

    # ── Predicted-vs-actual plots (cost-model sanity check) ───────────────────
    # Pulls the .pkl bundles next to the .bin files we already pointed at.
    sys.path.insert(0, str(REPO / "tools"))
    try:
        from plot_predicted_vs_actual import plot_run_dir as _plot_pred_vs_actual  # noqa: E402
    except ImportError:
        _plot_pred_vs_actual = None
    if _plot_pred_vs_actual is not None:
        gpu_pkl    = REPO / preset["cost_dir_models"] / "gpu_only.pkl"
        hybrid_pkl = REPO / preset["cost_dir_models"] / "lpddr5_pim_bank.pkl"
        if gpu_pkl.exists() or hybrid_pkl.exists():
            print("\n[predicted_vs_actual] generating per-policy plots ...")
            import warnings as _warnings
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                _plot_pred_vs_actual(run_dir, gpu_pkl, hybrid_pkl,
                                     vision_prefix_tokens=VISION_PREFIX_TOKENS)
        else:
            print("\n[predicted_vs_actual skip] no cost-model .pkl bundles found")

    # ── KV pool timeline plots ────────────────────────────────────────────────
    print("\n[timeline] generating KV pool occupancy plots ...")
    sys.path.insert(0, str(REPO / "tools"))
    from plot_kv_pool_timeline import (  # noqa: E402
        parse_log, replay, plot_timeline, read_pool_bytes,
    )
    import matplotlib.pyplot as _plt
    fig_combined, axes_combined = _plt.subplots(
        len(runs), 1, figsize=(16, 4.0 * len(runs)), sharex=False)
    if len(runs) == 1:
        axes_combined = [axes_combined]
    for ax, (label, subdir_name) in zip(axes_combined, runs):
        subdir = run_dir / subdir_name
        log = subdir / "debug.log"
        if not log.exists():
            print(f"[timeline skip] {label}: no debug.log")
            continue
        events  = parse_log(log)
        segs    = replay(events)
        pool    = read_pool_bytes(subdir)
        per_pol = subdir / "plots"
        per_pol.mkdir(parents=True, exist_ok=True)
        out_single = per_pol / f"{label}_kv_pool_timeline.png"
        fig_s, ax_s = _plt.subplots(figsize=(16, 5))
        plot_timeline(ax_s, segs, pool, label)
        fig_s.tight_layout()
        fig_s.savefig(out_single, dpi=140)
        _plt.close(fig_s)
        print(f"[timeline] {out_single.relative_to(REPO)}")
        plot_timeline(ax, segs, pool, label)
    fig_combined.suptitle("PIM KV pool occupancy over time — per policy", fontsize=12)
    fig_combined.tight_layout(rect=[0, 0, 1, 0.98])
    out_combined = plots_dir / "kv_pool_timeline_compare.png"
    fig_combined.savefig(out_combined, dpi=140)
    _plt.close(fig_combined)
    print(f"[timeline] {out_combined.relative_to(REPO)}")

    print(f"\n[run_dir] {run_dir}")
    print(f"[plots]   {plots_dir}")


if __name__ == "__main__":
    main()
