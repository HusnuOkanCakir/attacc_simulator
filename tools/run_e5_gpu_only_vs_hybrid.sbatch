#!/bin/bash
#SBATCH -J e5_gpu_vs_hybrid
#SBATCH -p high_latency
#SBATCH -c 4
#SBATCH --mem=32GB
#SBATCH -D /mnt/galactica/ocakir/attacc_simulator
#SBATCH -o /mnt/galactica/ocakir/logs/e5_gpu_vs_hybrid_%j.out
#SBATCH -e /mnt/galactica/ocakir/logs/e5_gpu_vs_hybrid_%j.err
#
# E5: GPU-only vs hybrid (GPU+LPDDR5-PIM) head-to-head comparison.
#
# Thesis main claim: hybrid PIM+GPU outperforms GPU-only for VLA serving.
# This sweep produces the explicit head-to-head evidence by running the
# same workload twice per model:
#   (a) hybrid (baseline) — current scheduler with both routes
#   (b) gpu_only — PIM route disabled via the new --gpu-only flag
#
# 2 models × 2 modes = 4 cells.
#
# Usage:
#   sbatch tools/run_e5_gpu_only_vs_hybrid.sbatch
#
# Env-var overrides (defaults updated 2026-05-27 per user: Azure trace, bs=4):
#   N_REQUESTS       (default 2000)
#   ARRIVAL_SCALE    (default 1.0)
#   KV_POOL_GB       (default 8)
#   MAX_ACTIVE       (default 32)
#   POLICY           (default max_util_full)         — the canonical KV policy
#   MAX_DECODE_BS    (default 4)                     — bs=8 was the H-sweep peak
#   MAX_CONSEC_DEC   (default 4)
#   SLO_E2E_MS       (default 500)
#   SLO_TTFT_MS      (default 200)
#   KV_OOM_POLICY    (default hold_then_fallback)
#   MODELS           (default "pi0 openvla")
#   GPU_TYPE         (default a6000)
#   TRACE_KIND       (default "azure")               — "azure" (conv first2000)
#                                                       or "synthetic" (per-model
#                                                       Lout=7/50 Poisson 6Hz)
#
# Output: cluster_outputs/online_serving_runs/e5_gpu_vs_hybrid_${TS}/
#           01_pi0_hybrid/<policy>/...
#           02_pi0_gpuonly/<policy>/...
#           03_openvla_hybrid/<policy>/...
#           04_openvla_gpuonly/<policy>/...
#         plots/sweep_compare.png

set -euo pipefail

ROOT_DIR="/mnt/galactica/ocakir/attacc_simulator"
LOG_DIR="/mnt/galactica/ocakir/logs"
mkdir -p "${LOG_DIR}"
cd "${ROOT_DIR}"

export TMPDIR=/mnt/galactica/ocakir/tmp
mkdir -p "$TMPDIR"

source /mnt/galactica/ocakir/miniforge3/etc/profile.d/conda.sh
conda activate lerobot

echo "hostname:  $(hostname)"
echo "python:    $(python -V 2>&1)"

N_REQUESTS="${N_REQUESTS:-2000}"
ARRIVAL_SCALE="${ARRIVAL_SCALE:-1.0}"
KV_POOL_GB="${KV_POOL_GB:-8}"
MAX_ACTIVE="${MAX_ACTIVE:-32}"
POLICY="${POLICY:-max_util_full}"
MAX_DECODE_BS="${MAX_DECODE_BS:-4}"
MAX_CONSEC_DEC="${MAX_CONSEC_DEC:-4}"
SLO_E2E_MS="${SLO_E2E_MS:-500}"
SLO_TTFT_MS="${SLO_TTFT_MS:-200}"
KV_OOM_POLICY="${KV_OOM_POLICY:-hold_then_fallback}"
KV_OOM_HOLD_LIMIT_MS="${KV_OOM_HOLD_LIMIT_MS:-50}"
MODELS="${MODELS:-pi0 openvla}"
GPU_TYPE="${GPU_TYPE:-a6000}"

PY=$(which python)
RUNNER="${ROOT_DIR}/tools/run_serving_online_policy_compare.py"
CACHE_DIR="${ROOT_DIR}/cluster_outputs/pim_cache"
TS=$(date +%Y%m%d_%H%M%S)
# TRACE_KIND switch: "azure" (default per user 2026-05-27) or "synthetic".
# Azure conv is the canonical workload for the thesis main-claim head-to-head.
TRACE_KIND="${TRACE_KIND:-azure}"
SWEEP_DIR="${ROOT_DIR}/cluster_outputs/online_serving_runs/e5_gpu_vs_hybrid_${TRACE_KIND}_${TS}"
LOG="${SWEEP_DIR}/sweep.log"

mkdir -p "${CACHE_DIR}" "${SWEEP_DIR}"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "E5: GPU-only vs hybrid head-to-head"
log "trace_kind=$TRACE_KIND"
log "models=($MODELS)  gpu=$GPU_TYPE"
log "n=$N_REQUESTS  scale=$ARRIVAL_SCALE  kv_pool=${KV_POOL_GB}GiB  max_active=$MAX_ACTIVE"
log "policy=$POLICY  bs=$MAX_DECODE_BS  mc=$MAX_CONSEC_DEC  prompt_priority=false"
log "slo_e2e=${SLO_E2E_MS}ms  slo_ttft=${SLO_TTFT_MS}ms  kv_oom=$KV_OOM_POLICY"
log "sweep_dir=$SWEEP_DIR"
log ""

# Optional synthetic-trace pre-generation (only when TRACE_KIND=synthetic).
if [ "$TRACE_KIND" = "synthetic" ]; then
  TRACE_GEN="${ROOT_DIR}/tools/gen_vla_synthetic_trace.py"
  SYNTH_TRACE_DIR="${ROOT_DIR}/cluster_outputs/synthetic_vla"
  mkdir -p "$SYNTH_TRACE_DIR"
  for model in $MODELS; do
    out_csv="${SYNTH_TRACE_DIR}/synthetic_vla_${model}_n${N_REQUESTS}_seed0.csv"
    if [ ! -s "$out_csv" ]; then
      log "[trace] generating synthetic VLA trace for $model n=$N_REQUESTS"
      "$PY" "$TRACE_GEN" --model "$model" --n-requests "$N_REQUESTS" \
          --rate-hz 6.0 --jitter 1.0 --seed 0 --out-csv "$out_csv" 2>&1 | tee -a "$LOG"
    else
      log "[trace] reusing $out_csv"
    fi
  done
fi

# args: <label> <model> <mode>   where mode = "hybrid" or "gpuonly"
run_one() {
  local label=$1
  local model=$2
  local mode=$3
  local run_dir="${SWEEP_DIR}/${label}"
  # Distinct PIM cache per (model, mode, trace). gpuonly cells won't use PIM,
  # but the runner still touches the cache file path — keep them isolated.
  local cache_file="${CACHE_DIR}/${model}_${GPU_TYPE}_e5_${TRACE_KIND}_${mode}.bin"

  # Pick the trace: azure default (no --csv-path) or per-model synthetic CSV.
  local csv_arg=""
  if [ "$TRACE_KIND" = "synthetic" ]; then
    local csv_path="${ROOT_DIR}/cluster_outputs/synthetic_vla/synthetic_vla_${model}_n${N_REQUESTS}_seed0.csv"
    csv_arg="--csv-path $csv_path"
  fi
  # For azure, omit --csv-path → runner uses cluster_outputs/azure/AzureLLMInferenceTrace_conv_first<N>.csv

  local gpu_only_flag=""
  if [ "$mode" = "gpuonly" ]; then
    gpu_only_flag="--gpu-only"
  fi

  log "── start: $label  model=$model  mode=$mode  trace=$TRACE_KIND ──"
  log "    cache: $cache_file"
  if [ -n "$csv_arg" ]; then
    log "    csv:   $csv_arg"
  fi

  local t0=$(date +%s)
  "$PY" "$RUNNER" \
      --model "$model" \
      --gpu "$GPU_TYPE" \
      --n-requests "$N_REQUESTS" \
      --policies "$POLICY" \
      --kv-pool-gb "$KV_POOL_GB" --ml \
      --max-active "$MAX_ACTIVE" \
      --arrival-scale "$ARRIVAL_SCALE" \
      --slo-e2e-ms "$SLO_E2E_MS" --slo-ttft-ms "$SLO_TTFT_MS" \
      --kv-oom-policy "$KV_OOM_POLICY" \
      --kv-oom-hold-limit-ms "$KV_OOM_HOLD_LIMIT_MS" \
      --max-decode-batch-size "$MAX_DECODE_BS" \
      --max-consecutive-decode-batches "$MAX_CONSEC_DEC" \
      --no-prompt-priority \
      $csv_arg \
      --pim-cache-file "$cache_file" \
      $gpu_only_flag \
      --debug-log \
      --run-dir "$run_dir" 2>&1 | tee -a "$LOG"
  local dt=$(( $(date +%s) - t0 ))
  log "── done:  $label  wall=${dt}s ──"
}

# ── Sweep: 2 models × 2 modes ────────────────────────────────────────────────
i=1
for model in $MODELS; do
  for mode in hybrid gpuonly; do
    label=$(printf "%02d_%s_%s" "$i" "$model" "$mode")
    run_one "$label" "$model" "$mode"
    i=$((i+1))
  done
done

# ── Summary + cross-cell plot ───────────────────────────────────────────────
log "=========================================================================="
log "E5 ALL DONE  sweep_dir: $SWEEP_DIR"
log "=========================================================================="

PLOT_COMPARE="${ROOT_DIR}/tools/plot_sweep_compare.py"
if [ -f "$PLOT_COMPARE" ]; then
  log "[plot] generating cross-cell comparison ..."
  "$PY" "$PLOT_COMPARE" --sweep-dir "$SWEEP_DIR" 2>&1 | tee -a "$LOG"
fi

log ""
log "Pass criterion for the thesis main claim:"
log "  For each model, hybrid throughput >= gpuonly throughput AND"
log "  hybrid TTFT/E2E p99 <= gpuonly p99 by a meaningful margin."
log ""
log "Expected: PIM is faster than GPU for decode steps (decode_e2e_ms is"
log "  44% lower for PIM at OpenVLA Lin=2048 bs=8). In hybrid mode the"
log "  scheduler routes 80-90%% to PIM for Pi0 / 15-50%% for OpenVLA, so"
log "  hybrid should win by the cumulative speedup of those routed-to-PIM"
log "  requests. Magnitude depends on workload Lout-distribution."
