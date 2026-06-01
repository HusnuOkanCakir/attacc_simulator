#!/bin/bash
#SBATCH -J f_full19k_azure
#SBATCH -p high_latency
#SBATCH -c 4
#SBATCH --mem=48GB
#SBATCH -D /mnt/galactica/ocakir/attacc_simulator
#SBATCH -o /mnt/galactica/ocakir/logs/f_full19k_azure_%j.out
#SBATCH -e /mnt/galactica/ocakir/logs/f_full19k_azure_%j.err
#
# F: full Azure conv trace (19,365 requests) — GPU-only vs hybrid for
# both models. Direct apples-to-apples scale-up of E5 v2 (which used the
# first 2000 of the same trace).
#
# Cache isolation: this script uses cache filenames with the `_f_full`
# suffix so it CAN run in parallel with E5 / E6 jobs without clobbering
# their cache files.
#
# Matrix: 2 models × 2 modes = 4 cells.
#
# Usage:
#   sbatch tools/run_f_full19k_azure.sbatch
#
# Env-var overrides (all defaults match E5 v2 for direct comparison):
#   N_REQUESTS        (default 19365 — full Azure conv)
#   ARRIVAL_SCALE     (default 1.0   — matches E5 v2)
#   KV_POOL_GB        (default 8)
#   MAX_ACTIVE        (default 32)
#   POLICY            (default max_util_full)
#   MAX_DECODE_BS     (default 4     — H-sweep Pareto best, matches E5 v2)
#   MAX_CONSEC_DEC    (default 4)
#   SLO_E2E_MS        (default 500)
#   SLO_TTFT_MS       (default 200)
#   KV_OOM_POLICY     (default hold_then_fallback)
#   MODELS            (default "pi0 openvla")
#   GPU_TYPE          (default a6000)
#
# Wall-time estimate: 2-5 hours per cell (10× E5's n=2000 scale + cache
# warmup on first run). 4 cells sequential → ~8-20 hours. Submit and walk
# away; high_latency partition has no time limit.

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

N_REQUESTS="${N_REQUESTS:-19365}"
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
SWEEP_DIR="${ROOT_DIR}/cluster_outputs/online_serving_runs/f_full19k_azure_${TS}"
LOG="${SWEEP_DIR}/sweep.log"

mkdir -p "${CACHE_DIR}" "${SWEEP_DIR}"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "F: full Azure conv (19k) head-to-head"
log "models=($MODELS)  gpu=$GPU_TYPE"
log "n=$N_REQUESTS  scale=$ARRIVAL_SCALE  kv_pool=${KV_POOL_GB}GiB  max_active=$MAX_ACTIVE"
log "policy=$POLICY  bs=$MAX_DECODE_BS  mc=$MAX_CONSEC_DEC  prompt_priority=false"
log "slo_e2e=${SLO_E2E_MS}ms  slo_ttft=${SLO_TTFT_MS}ms  kv_oom=$KV_OOM_POLICY"
log "sweep_dir=$SWEEP_DIR"
log ""

# args: <label> <model> <mode>   where mode = "hybrid" or "gpuonly"
run_one() {
  local label=$1
  local model=$2
  local mode=$3
  local run_dir="${SWEEP_DIR}/${label}"
  # Distinct from E5/E6 cache namespaces — guarantees no collision.
  local cache_file="${CACHE_DIR}/${model}_${GPU_TYPE}_f_full_${mode}.bin"

  local gpu_only_flag=""
  if [ "$mode" = "gpuonly" ]; then
    gpu_only_flag="--gpu-only"
  fi

  log "── start: $label  model=$model  mode=$mode ──"
  log "    cache: $cache_file"

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
      --pim-cache-file "$cache_file" \
      $gpu_only_flag \
      --debug-log \
      --no-kv-timeline \
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

log "=========================================================================="
log "F ALL DONE  sweep_dir: $SWEEP_DIR"
log "=========================================================================="

PLOT_COMPARE="${ROOT_DIR}/tools/plot_sweep_compare.py"
if [ -f "$PLOT_COMPARE" ]; then
  log "[plot] generating cross-cell comparison ..."
  "$PY" "$PLOT_COMPARE" --sweep-dir "$SWEEP_DIR" 2>&1 | tee -a "$LOG"
fi

log ""
log "Compare against E5 v2 (n=2000) at:"
log "  cluster_outputs/online_serving_runs/e5_gpu_vs_hybrid_azure_20260528_152128/"
log "Expected: same hybrid-wins-both-models pattern as E5 v2, just at 10x scale."
