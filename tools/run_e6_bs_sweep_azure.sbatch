#!/bin/bash
#SBATCH -J e6_bs_sweep_azure
#SBATCH -p high_latency
#SBATCH -c 4
#SBATCH --mem=32GB
#SBATCH -D /mnt/galactica/ocakir/attacc_simulator
#SBATCH -o /mnt/galactica/ocakir/logs/e6_bs_sweep_azure_%j.out
#SBATCH -e /mnt/galactica/ocakir/logs/e6_bs_sweep_azure_%j.err
#
# E6: max_decode_batch_size sweep on Azure conv trace, both models.
#
# This is the diagnostic sweep that pairs with the Phase E telemetry
# additions (decode_batching block in ramulator_summary.yaml and
# per-event queue_snapshots.csv/.txt). Goal: see whether the H-sweep's
# 1/B throughput regression reproduces in the current code AND whether
# the batches actually fill up to the configured max.
#
# Matrix: 2 models × 5 bs values = 10 cells.
#
# Defaults match the canonical "head-to-head" config (E5 v2): Azure conv,
# max_util_full, scale=1.0, kv_pool=8 GiB, max_active=32, SLO 500/200ms,
# prompt_priority=false, mc=4, A6000 cost models.
#
# Usage:
#   sbatch tools/run_e6_bs_sweep_azure.sbatch
#
# Env overrides:
#   N_REQUESTS        (default 2000)
#   ARRIVAL_SCALE     (default 1.0)
#   KV_POOL_GB        (default 8)
#   MAX_ACTIVE        (default 32)
#   POLICY            (default max_util_full)
#   MAX_CONSEC_DEC    (default 4)
#   SLO_E2E_MS        (default 500)
#   SLO_TTFT_MS       (default 200)
#   KV_OOM_POLICY     (default hold_then_fallback)
#   MODELS            (default "pi0 openvla")
#   BS_VALUES         (default "1 2 4 8 16")
#   GPU_TYPE          (default a6000)

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
MAX_CONSEC_DEC="${MAX_CONSEC_DEC:-4}"
SLO_E2E_MS="${SLO_E2E_MS:-500}"
SLO_TTFT_MS="${SLO_TTFT_MS:-200}"
KV_OOM_POLICY="${KV_OOM_POLICY:-hold_then_fallback}"
KV_OOM_HOLD_LIMIT_MS="${KV_OOM_HOLD_LIMIT_MS:-50}"
MODELS="${MODELS:-pi0 openvla}"
BS_VALUES="${BS_VALUES:-1 2 4 8 16}"
GPU_TYPE="${GPU_TYPE:-a6000}"

PY=$(which python)
RUNNER="${ROOT_DIR}/tools/run_serving_online_policy_compare.py"
CACHE_DIR="${ROOT_DIR}/cluster_outputs/pim_cache"
TS=$(date +%Y%m%d_%H%M%S)
SWEEP_DIR="${ROOT_DIR}/cluster_outputs/online_serving_runs/e6_bs_sweep_azure_${TS}"
LOG="${SWEEP_DIR}/sweep.log"

mkdir -p "${CACHE_DIR}" "${SWEEP_DIR}"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

n_models=$(echo "$MODELS" | wc -w)
n_bs=$(echo "$BS_VALUES" | wc -w)
n_total=$(( n_models * n_bs ))

log "E6: max_decode_batch_size sweep on Azure conv"
log "models=($MODELS)  bs_values=($BS_VALUES)  gpu=$GPU_TYPE"
log "n=$N_REQUESTS  scale=$ARRIVAL_SCALE  kv_pool=${KV_POOL_GB}GiB  max_active=$MAX_ACTIVE"
log "policy=$POLICY  mc=$MAX_CONSEC_DEC  prompt_priority=false  kv_oom=$KV_OOM_POLICY"
log "slo_e2e=${SLO_E2E_MS}ms  slo_ttft=${SLO_TTFT_MS}ms"
log "matrix: ${n_models} models × ${n_bs} bs values = ${n_total} cells"
log "sweep_dir=$SWEEP_DIR"
log ""

# args: <label> <model> <bs>
run_one() {
  local label=$1
  local model=$2
  local bs=$3
  local run_dir="${SWEEP_DIR}/${label}"
  local cache_file="${CACHE_DIR}/${model}_${GPU_TYPE}_e6_bs${bs}.bin"

  log "── start: $label  model=$model  bs=$bs ──"
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
      --max-decode-batch-size "$bs" \
      --max-consecutive-decode-batches "$MAX_CONSEC_DEC" \
      --no-prompt-priority \
      --pim-cache-file "$cache_file" \
      --debug-log \
      --run-dir "$run_dir" 2>&1 | tee -a "$LOG"
  local dt=$(( $(date +%s) - t0 ))
  log "── done:  $label  wall=${dt}s ──"
}

# ── Cross-product sweep ─────────────────────────────────────────────────────
i=1
for model in $MODELS; do
  for bs in $BS_VALUES; do
    label=$(printf "%02d_%s_bs%s" "$i" "$model" "$bs")
    run_one "$label" "$model" "$bs"
    i=$((i+1))
  done
done

log "=========================================================================="
log "E6 ALL DONE  sweep_dir: $SWEEP_DIR"
log "=========================================================================="

# Cross-cell 6-panel plot.
PLOT_COMPARE="${ROOT_DIR}/tools/plot_sweep_compare.py"
if [ -f "$PLOT_COMPARE" ]; then
  log "[plot] generating cross-cell comparison ..."
  "$PY" "$PLOT_COMPARE" --sweep-dir "$SWEEP_DIR" 2>&1 | tee -a "$LOG"
fi

# E6-specific: bs telemetry plots.
PLOT_BS="${ROOT_DIR}/tools/plot_decode_batching.py"
if [ -f "$PLOT_BS" ]; then
  log "[plot] generating decode-batching diagnostics ..."
  "$PY" "$PLOT_BS" --sweep-dir "$SWEEP_DIR" 2>&1 | tee -a "$LOG"
fi

log ""
log "What to look for:"
log "  bs_actual_vs_configured.png — does mean_bs approach the configured max?"
log "    If mean_bs << max  → batch isn't filling (queue too shallow);"
log "    If mean_bs ≈ max   → batch fills, but throughput still drops → per-step time issue."
log "  bs_throughput_vs_config.png — does throughput rise with bs (expected) or fall (regression)?"
log "  bs_histogram.png — distribution of actual batch sizes; fat tail at bs=1 means most steps are single-request."
log "  Per-cell queue_snapshots.{csv,txt} let you grep for hold reasons (kv_oom, predictive_pressure)."
