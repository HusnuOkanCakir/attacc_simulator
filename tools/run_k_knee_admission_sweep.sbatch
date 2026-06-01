#!/bin/bash
#SBATCH -J k_knee_admit
#SBATCH -p high_latency
#SBATCH -c 4
#SBATCH --mem=32GB
#SBATCH -D /mnt/galactica/ocakir/attacc_simulator
#SBATCH -o /mnt/galactica/ocakir/logs/k_knee_admit_%j.out
#SBATCH -e /mnt/galactica/ocakir/logs/k_knee_admit_%j.err
#
# K: admission-violation-budget X sweep AT THE KNEE.
#
# Phase C (2026-05-20–24) tested X ∈ {-1, 0, 1, 5} at arrival_scale=1.0
# and found all 4 variants wrong-direction or null: at saturation,
# holding requests just transfers their violation onto themselves.
#
# At the knee, holding may genuinely buffer short Poisson bursts because
# the system has slack to drain. This is the regime where admission
# control is supposed to work.
#
# Matrix: 1 model × 5 budgets × 1 dataset = 5 cells.
# (Pi0 only; OpenVLA admission control at knee is a follow-up.)
#
# Cache isolation: `_k_knee_<dataset>` namespace per model.
#
# Usage:
#   sbatch tools/run_k_knee_admission_sweep.sbatch                    # azure_full19k
#   DATASET=azure_poisson sbatch tools/run_k_knee_admission_sweep.sbatch
#   DATASET=vla_poisson   sbatch tools/run_k_knee_admission_sweep.sbatch
#
# Env-var overrides:
#   DATASET            (default azure_full19k)
#   N_REQUESTS, ARRIVAL_SCALE — derived from DATASET if unset
#   BUDGETS            (default "-1 0 1 5 20")
#   MODELS             (default "pi0")
#   POLICY             (default max_util_full)
#   KV_POOL_GB         (default 8)
#   MAX_ACTIVE         (default 32)
#   MAX_DECODE_BS      (default 4)
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
MAX_DECODE_BS="${MAX_DECODE_BS:-4}"
MAX_CONSEC_DEC="${MAX_CONSEC_DEC:-4}"
SLO_E2E_MS="${SLO_E2E_MS:-500}"
SLO_TTFT_MS="${SLO_TTFT_MS:-200}"
KV_OOM_POLICY="${KV_OOM_POLICY:-hold_then_fallback}"
KV_OOM_HOLD_LIMIT_MS="${KV_OOM_HOLD_LIMIT_MS:-50}"
MODELS="${MODELS:-pi0}"
BUDGETS="${BUDGETS:--1 0 1 5 20}"
GPU_TYPE="${GPU_TYPE:-a6000}"

PY=$(which python)
RUNNER="${ROOT_DIR}/tools/run_serving_online_policy_compare.py"
CACHE_DIR="${ROOT_DIR}/cluster_outputs/pim_cache"
TS=$(date +%Y%m%d_%H%M%S)
SWEEP_DIR="${ROOT_DIR}/cluster_outputs/online_serving_runs/k_knee_admit_${DATASET}_${TS}"
LOG="${SWEEP_DIR}/sweep.log"

mkdir -p "${CACHE_DIR}" "${SWEEP_DIR}"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

n_models=$(echo "$MODELS" | wc -w)
n_budgets=$(echo "$BUDGETS" | wc -w)
n_total=$(( n_models * n_budgets ))

log "K: admission-violation-budget X sweep AT THE KNEE"
log "dataset=$DATASET  csv=$CSV_PATH"
log "models=($MODELS)  budgets=($BUDGETS)  gpu=$GPU_TYPE"
log "n=$N_REQUESTS  arrival_scale=$ARRIVAL_SCALE  kv_pool=${KV_POOL_GB}GiB  max_active=$MAX_ACTIVE"
log "policy=$POLICY  bs=$MAX_DECODE_BS  mc=$MAX_CONSEC_DEC  kv_oom=$KV_OOM_POLICY"
log "slo_e2e=${SLO_E2E_MS}ms  slo_ttft=${SLO_TTFT_MS}ms"
log "matrix: ${n_models} models × ${n_budgets} budgets = ${n_total} cells"
log "sweep_dir=$SWEEP_DIR"
log ""

if [ ! -s "$CSV_PATH" ]; then
  log "[error] CSV not found at $CSV_PATH"; exit 1
fi

# Pretty label helper for budget value (-1 → m1, 0 → 0, 1 → 1).
budget_label() {
  local b=$1
  if [[ "$b" =~ ^- ]]; then echo "m${b#-}"; else echo "${b}"; fi
}

run_one() {
  local label=$1
  local model=$2
  local budget=$3
  local run_dir="${SWEEP_DIR}/${label}"
  local cache_file="${CACHE_DIR}/${model}_${GPU_TYPE}_k_knee_${DATASET}_x$(budget_label $budget).bin"

  log "── start: $label  model=$model  X=$budget ──"
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
      --admission-violation-budget "$budget" \
      --csv-path "$CSV_PATH" \
      --pim-cache-file "$cache_file" \
      --debug-log \
      --no-kv-timeline \
      --run-dir "$run_dir" 2>&1 | tee -a "$LOG"
  local dt=$(( $(date +%s) - t0 ))
  log "── done:  $label  wall=${dt}s ──"
}

i=1
for model in $MODELS; do
  for budget in $BUDGETS; do
    label=$(printf "%02d_%s_x%s" "$i" "$model" "$(budget_label $budget)")
    run_one "$label" "$model" "$budget"
    i=$((i+1))
  done
done

log "=========================================================================="
log "K ALL DONE  sweep_dir: $SWEEP_DIR"
log "=========================================================================="

PLOT_COMPARE="${ROOT_DIR}/tools/plot_sweep_compare.py"
if [ -f "$PLOT_COMPARE" ]; then
  log "[plot] generating cross-cell comparison ..."
  "$PY" "$PLOT_COMPARE" --sweep-dir "$SWEEP_DIR" 2>&1 | tee -a "$LOG"
fi
