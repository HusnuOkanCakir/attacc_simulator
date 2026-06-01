#!/bin/bash
#SBATCH -J g_knee_main_claim
#SBATCH -p high_latency
#SBATCH -c 4
#SBATCH --mem=48GB
#SBATCH -D /mnt/galactica/ocakir/attacc_simulator
#SBATCH -o /mnt/galactica/ocakir/logs/g_knee_main_claim_%j.out
#SBATCH -e /mnt/galactica/ocakir/logs/g_knee_main_claim_%j.err
#
# G: thesis MAIN CLAIM at the knee.
#
# Same 2 models × 2 modes matrix as F (full 19k Azure conv), but the
# arrival_scale is set so the trace's peak instantaneous RPS touches
# the system's measured `a` (the knee — sustainable peak RPS where p99
# starts climbing super-linearly).
#
# At scale=1.0 (F) the system is in DEEP saturation: p99 in thousands
# of seconds, SLO compliance ≈ 0%. The hybrid-vs-gpu-only comparison is
# RELATIVE only because absolute compliance was meaningless.
#
# At the knee (this sweep), p99 should drop into the SLO range or just
# above — the absolute hybrid-vs-gpu-only gap becomes a real claim
# instead of a "saturation-dominated ratio" footnote.
#
# Matrix: 2 models (pi0, openvla) × 2 modes (hybrid, gpuonly) = 4 cells.
#
# Cache isolation: `_g_knee_` namespace so it doesn't collide with F.
#
# Usage:
#   sbatch tools/run_g_knee_main_claim.sbatch
#
# Env-var overrides:
#   DATASET            (default azure_full19k; also: azure_poisson, vla_poisson)
#                      Selects the input trace + arrival_scale. If
#                      ARRIVAL_SCALE / N_REQUESTS are NOT set explicitly
#                      they're derived from DATASET (see helper below).
#   N_REQUESTS         (defaults per dataset)
#   ARRIVAL_SCALE      (defaults per dataset = peak / a)
#   KV_POOL_GB         (default 8)
#   MAX_ACTIVE         (default 32)
#   POLICY             (default max_util_full)
#   MAX_DECODE_BS      (default 4)
#   MAX_CONSEC_DEC     (default 4)
#   SLO_E2E_MS         (default 500)
#   SLO_TTFT_MS        (default 200)
#   KV_OOM_POLICY      (default hold_then_fallback)
#   MODELS             (default "pi0 openvla")
#   GPU_TYPE           (default a6000)
#
# Dataset → (csv, default n_requests, default arrival_scale):
#   azure_full19k  → AzureLLMInferenceTrace_conv.csv, n=19365, scale=4.24
#   azure_poisson  → synthetic_v2/azure_poisson_n5000_seed0.csv, n=5000, scale=3.75
#   vla_poisson    → synthetic_v2/vla_poisson_n5000_seed0.csv,   n=5000, scale=2.44
#
# Wall time estimate: at the knee the system is NOT saturated, so each
# cell processes 19k requests at ~2.5 req/s sustained = ~2-4 h cell.
# 4 cells sequential → ~10-16 h. Much faster than F's 9h per cell at
# scale=1.0 because there's no queue blowup.

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

# Per-dataset defaults
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
  azure_poisson_wide)
    DEFAULT_CSV="${ROOT_DIR}/cluster_outputs/synthetic_v2/azure_poisson_wide_n5000_seed0.csv"
    DEFAULT_N=5000
    DEFAULT_SCALE=6.5
    ;;
  vla_poisson)
    DEFAULT_CSV="${ROOT_DIR}/cluster_outputs/synthetic_v2/vla_poisson_n5000_seed0.csv"
    DEFAULT_N=5000
    DEFAULT_SCALE=2.44
    ;;
  azure_empirical)
    DEFAULT_CSV="${ROOT_DIR}/cluster_outputs/synthetic_v2/azure_empirical_n5000_seed0.csv"
    DEFAULT_N=5000
    DEFAULT_SCALE=3.77
    ;;
  *)
    echo "[error] unknown DATASET=$DATASET; choose azure_full19k | azure_poisson | azure_poisson_wide | vla_poisson | azure_empirical" >&2
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
MODELS="${MODELS:-pi0 openvla}"
GPU_TYPE="${GPU_TYPE:-a6000}"

PY=$(which python)
RUNNER="${ROOT_DIR}/tools/run_serving_online_policy_compare.py"
CACHE_DIR="${ROOT_DIR}/cluster_outputs/pim_cache"
TS=$(date +%Y%m%d_%H%M%S)
SWEEP_DIR="${ROOT_DIR}/cluster_outputs/online_serving_runs/g_knee_${DATASET}_${TS}"
LOG="${SWEEP_DIR}/sweep.log"

mkdir -p "${CACHE_DIR}" "${SWEEP_DIR}"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "G: thesis main claim AT THE KNEE"
log "dataset=$DATASET  csv=$CSV_PATH"
log "models=($MODELS)  gpu=$GPU_TYPE"
log "n=$N_REQUESTS  arrival_scale=$ARRIVAL_SCALE  (peak/a = knee-scaled)"
log "kv_pool=${KV_POOL_GB}GiB  max_active=$MAX_ACTIVE"
log "policy=$POLICY  bs=$MAX_DECODE_BS  mc=$MAX_CONSEC_DEC"
log "slo_e2e=${SLO_E2E_MS}ms  slo_ttft=${SLO_TTFT_MS}ms  kv_oom=$KV_OOM_POLICY"
log "sweep_dir=$SWEEP_DIR"
log ""

if [ ! -s "$CSV_PATH" ]; then
  log "[error] CSV not found at $CSV_PATH"
  exit 1
fi

# args: <label> <model> <mode>   where mode = "hybrid" or "gpuonly"
run_one() {
  local label=$1
  local model=$2
  local mode=$3
  local run_dir="${SWEEP_DIR}/${label}"
  # Per-dataset namespace so azure_full19k and azure_poisson don't collide.
  local cache_file="${CACHE_DIR}/${model}_${GPU_TYPE}_g_knee_${DATASET}_${mode}.bin"

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
      --csv-path "$CSV_PATH" \
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
log "G ALL DONE  sweep_dir: $SWEEP_DIR"
log "=========================================================================="

# Cross-cell standard plot.
PLOT_COMPARE="${ROOT_DIR}/tools/plot_sweep_compare.py"
if [ -f "$PLOT_COMPARE" ]; then
  log "[plot] generating cross-cell comparison ..."
  "$PY" "$PLOT_COMPARE" --sweep-dir "$SWEEP_DIR" 2>&1 | tee -a "$LOG"
fi

# Thesis main-claim figure (F1-style).
THESIS_E5="${ROOT_DIR}/thesis_plotting/scripts/thesis_e5_gpu_vs_hybrid.py"
if [ -f "$THESIS_E5" ]; then
  log "[plot] generating thesis main-claim figure ..."
  "$PY" "$THESIS_E5" \
      --sweep-dir "$SWEEP_DIR" \
      --out-name "fig_g_knee_main_claim_${DATASET}" 2>&1 | tee -a "$LOG"
fi

log ""
log "Compare against F (full 19k @ scale=1.0) at:"
log "  cluster_outputs/online_serving_runs/f_full19k_azure_20260530_110944/"
log "Expected: similar hybrid-wins-both-models pattern, but with"
log "absolute p99 values in the SLO range instead of saturation territory."
