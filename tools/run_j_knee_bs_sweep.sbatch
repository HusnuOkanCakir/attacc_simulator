#!/bin/bash
#SBATCH -J j_knee_bs
#SBATCH -p high_latency
#SBATCH -c 4
#SBATCH --mem=32GB
#SBATCH -D /mnt/galactica/ocakir/attacc_simulator
#SBATCH -o /mnt/galactica/ocakir/logs/j_knee_bs_%j.out
#SBATCH -e /mnt/galactica/ocakir/logs/j_knee_bs_%j.err
#
# J: max_decode_batch_size sweep AT THE KNEE.
#
# E6 (2026-05-28) ran the bs sweep at arrival_scale=1.0 (deep
# saturation) and found:
#   - Throughput scales linearly bs=1→8
#   - bs=16 starvation cliff (mean_bs collapses to 1.30)
#
# Open question: is the bs=16 cliff a real effect of max_active=32 (the
# scheduler can't accumulate 16 ready-decode requests) OR a saturation
# artifact (long queue means most requests are in prefill, not decode)?
#
# This sweep runs the SAME bs values but at the knee — if the cliff
# survives knee scaling, it's a real scheduler limit. If it disappears,
# the cliff was a queue artifact.
#
# Matrix: 2 models × 5 bs values × 1 dataset = 10 cells.
#
# Cache isolation: `_j_knee_<dataset>_bs<N>` namespace.
#
# Usage:
#   sbatch tools/run_j_knee_bs_sweep.sbatch                    # azure_full19k
#   DATASET=azure_poisson sbatch tools/run_j_knee_bs_sweep.sbatch
#   DATASET=vla_poisson   sbatch tools/run_j_knee_bs_sweep.sbatch
#
# Env-var overrides:
#   DATASET            (default azure_full19k; also: azure_poisson, vla_poisson)
#   N_REQUESTS, ARRIVAL_SCALE — derived from DATASET if unset
#   BS_VALUES          (default "1 2 4 8 16")
#   MODELS             (default "pi0 openvla")
#   POLICY             (default max_util_full)
#   KV_POOL_GB         (default 8)
#   MAX_ACTIVE         (default 32)
#   MAX_CONSEC_DEC     (default 4)
#   SLO_E2E_MS         (default 500)
#   SLO_TTFT_MS        (default 200)
#   KV_OOM_POLICY      (default hold_then_fallback)
#   GPU_TYPE           (default a6000)

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

DATASET="${DATASET:-azure_full19k}"

case "$DATASET" in
  azure_full19k)
    DEFAULT_CSV="${ROOT_DIR}/cluster_outputs/azure/AzureLLMInferenceTrace_conv.csv"
    DEFAULT_N=19365
    DEFAULT_SCALE=4.24
    ;;
  azure_poisson)
    DEFAULT_CSV="${ROOT_DIR}/cluster_outputs/synthetic_v2/azure_poisson_n5000_seed0.csv"
    DEFAULT_N=5000
    DEFAULT_SCALE=3.75
    ;;
  vla_poisson)
    DEFAULT_CSV="${ROOT_DIR}/cluster_outputs/synthetic_v2/vla_poisson_n5000_seed0.csv"
    DEFAULT_N=5000
    DEFAULT_SCALE=2.44
    ;;
  *)
    echo "[error] unknown DATASET=$DATASET" >&2
    exit 1
    ;;
esac

CSV_PATH="${CSV_PATH:-$DEFAULT_CSV}"
N_REQUESTS="${N_REQUESTS:-$DEFAULT_N}"
ARRIVAL_SCALE="${ARRIVAL_SCALE:-$DEFAULT_SCALE}"
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
SWEEP_DIR="${ROOT_DIR}/cluster_outputs/online_serving_runs/j_knee_bs_${DATASET}_${TS}"
LOG="${SWEEP_DIR}/sweep.log"

mkdir -p "${CACHE_DIR}" "${SWEEP_DIR}"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

n_models=$(echo "$MODELS" | wc -w)
n_bs=$(echo "$BS_VALUES" | wc -w)
n_total=$(( n_models * n_bs ))

log "J: bs sweep AT THE KNEE"
log "dataset=$DATASET  csv=$CSV_PATH"
log "models=($MODELS)  bs_values=($BS_VALUES)  gpu=$GPU_TYPE"
log "n=$N_REQUESTS  arrival_scale=$ARRIVAL_SCALE  kv_pool=${KV_POOL_GB}GiB  max_active=$MAX_ACTIVE"
log "policy=$POLICY  mc=$MAX_CONSEC_DEC  kv_oom=$KV_OOM_POLICY"
log "slo_e2e=${SLO_E2E_MS}ms  slo_ttft=${SLO_TTFT_MS}ms"
log "matrix: ${n_models} models × ${n_bs} bs = ${n_total} cells"
log "sweep_dir=$SWEEP_DIR"
log ""

if [ ! -s "$CSV_PATH" ]; then
  log "[error] CSV not found at $CSV_PATH"; exit 1
fi

run_one() {
  local label=$1
  local model=$2
  local bs=$3
  local run_dir="${SWEEP_DIR}/${label}"
  local cache_file="${CACHE_DIR}/${model}_${GPU_TYPE}_j_knee_${DATASET}_bs${bs}.bin"

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
      --csv-path "$CSV_PATH" \
      --pim-cache-file "$cache_file" \
      --debug-log \
      --no-kv-timeline \
      --run-dir "$run_dir" 2>&1 | tee -a "$LOG"
  local dt=$(( $(date +%s) - t0 ))
  log "── done:  $label  wall=${dt}s ──"
}

# ── Sweep ────────────────────────────────────────────────────────────────────
i=1
for model in $MODELS; do
  for bs in $BS_VALUES; do
    label=$(printf "%02d_%s_bs%s" "$i" "$model" "$bs")
    run_one "$label" "$model" "$bs"
    i=$((i+1))
  done
done

log "=========================================================================="
log "J ALL DONE  sweep_dir: $SWEEP_DIR"
log "=========================================================================="

PLOT_COMPARE="${ROOT_DIR}/tools/plot_sweep_compare.py"
if [ -f "$PLOT_COMPARE" ]; then
  log "[plot] generating cross-cell comparison ..."
  "$PY" "$PLOT_COMPARE" --sweep-dir "$SWEEP_DIR" 2>&1 | tee -a "$LOG"
fi

THESIS_E6="${ROOT_DIR}/thesis_plotting/scripts/thesis_e6_bs_sweep.py"
if [ -f "$THESIS_E6" ]; then
  log "[plot] generating thesis bs figures ..."
  "$PY" "$THESIS_E6" --sweep-dir "$SWEEP_DIR" 2>&1 | tee -a "$LOG"
fi
