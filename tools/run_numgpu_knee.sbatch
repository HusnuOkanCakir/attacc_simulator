#!/bin/bash
#SBATCH -J numgpu_knee
#SBATCH -p high_latency
#SBATCH -c 4
#SBATCH --mem=64GB
#SBATCH --exclude=safari-nexus1
#SBATCH -D /mnt/galactica/ocakir/attacc_simulator
#SBATCH -o /mnt/galactica/ocakir/logs/numgpu_knee_%j.out
#SBATCH -e /mnt/galactica/ocakir/logs/numgpu_knee_%j.err
#
# RPS-knee comparison: 1 GPU vs 2 GPUs under the pipelined hybrid executor.
#
# Sweeps arrival_scale (offered load) under exec_model=pipeline + min_finish
# routing for BOTH num_gpus=1 and num_gpus=2 on one dataset, so the
# achieved-throughput / E2E-p99 knee can be compared. With num_gpus=2 a
# dedicated gpu_only lane on a 2nd GPU absorbs overflow once the hybrid
# (GPU1+PIM) queue's predicted finish exceeds GPU2's idle path, so the knee
# is expected to shift RIGHT (higher sustainable RPS, lower tail latency).
# num_gpus=1 is the single-GPU hybrid baseline (all-PIM at bs=1 by the
# FFN-bound finding, so the GPU route never wins on speed alone).
#
# Dataset: real-shape bootstrap trace (joint (Lin,Lout) resampling of the
# real Azure 19k conv trace).
#   azure_bootstrap_n5000_seed0.csv
#
# A shared per-(model) PIM-shape cache is reused across all cells (num_gpus
# does not change PIM attention drains and is not in the cache fingerprint),
# so the first cell warms it and the rest are fast.
#
# Usage:
#   sbatch tools/run_numgpu_knee.sbatch
#   MODEL=openvla SCALES="27 13.6 10.9 9.1 7.8 6.4 5.44 4.5 3.6" \
#     COST_DIR=cluster_outputs/cost_tables_energyfix_openvla_a6000 \
#     sbatch tools/run_numgpu_knee.sbatch
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

DATASET="${DATASET:-bootstrap}"
case "$DATASET" in
  bootstrap) CSV_NAME="azure_bootstrap_n5000_seed0.csv" ;;
  wide)      CSV_NAME="azure_poisson_wide_n5000_seed0.csv" ;;
  *) echo "[error] DATASET must be 'bootstrap' or 'wide' (got '$DATASET')"; exit 1 ;;
esac

MODEL="${MODEL:-pi0}"
GPU_TYPE="${GPU_TYPE:-a6000}"
POLICY=max_util_full
# Chapter-6 knee config: max_util_full, 64 GiB PIM KV pool (remove KV
# pressure so the knee shift is purely the 2nd-GPU queueing relief), decode
# batch cap 4, max_active=5000 (bypass the cap; 32 distorts the knee),
# no-prompt-priority, hold_then_fallback, min_finish routing.
KV_POOL_GB="${KV_POOL_GB:-64}"
MAX_ACTIVE="${MAX_ACTIVE:-5000}"
MAX_DECODE_BS="${MAX_DECODE_BS:-4}"
MAX_CONSEC_DEC=4
SLO_E2E_MS="${SLO_E2E_MS:-500}"
SLO_TTFT_MS="${SLO_TTFT_MS:-200}"
KV_OOM_POLICY="${KV_OOM_POLICY:-hold_then_fallback}"
KV_OOM_HOLD_LIMIT_MS="${KV_OOM_HOLD_LIMIT_MS:-50}"
COST_DIR="${COST_DIR:-cluster_outputs/cost_tables_energyfix_${MODEL}_${GPU_TYPE}}"

# Offered RPS = k/arrival_scale (k~5.44 for pi0). The 2-GPU knee shifts
# right, so the default range extends to higher RPS (lower scale) than the
# single-GPU knee sweep to capture both knees.
if [ "$MODEL" = "openvla" ]; then
  DEFAULT_N=2000
  DEFAULT_SCALES="27 13.6 10.9 9.1 7.8 6.4 5.44 4.5 3.6 3.0"   # offered ~0.2 -> 1.5 RPS
else
  DEFAULT_N=5000
  DEFAULT_SCALES="6.8 4.5 3.4 2.85 2.6 2.36 2.18 2.0 1.8 1.6 1.4 1.2 1.0"  # offered ~0.8 -> 5.4 RPS
fi
N_REQUESTS="${N_REQUESTS:-$DEFAULT_N}"
read -r -a SCALES    <<< "${SCALES:-$DEFAULT_SCALES}"
read -r -a NUM_GPUS_LIST <<< "${NUM_GPUS_LIST:-1 2}"

CSV_PATH="${ROOT_DIR}/cluster_outputs/synthetic_v2/${CSV_NAME}"
CACHE_DIR="${ROOT_DIR}/cluster_outputs/pim_cache"
TS=$(date +%Y%m%d_%H%M%S)
SWEEP_DIR="${ROOT_DIR}/cluster_outputs/online_serving_runs/numgpu_knee_${DATASET}_${MODEL}_${TS}"
LOG="${SWEEP_DIR}/sweep.log"
mkdir -p "${CACHE_DIR}" "${SWEEP_DIR}"
SHARED_CACHE="${CACHE_DIR}/${MODEL}_${GPU_TYPE}_numgpuknee_${DATASET}.bin"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

[ -s "$CSV_PATH" ] || { log "[error] CSV missing: $CSV_PATH"; exit 1; }
[ -s "${ROOT_DIR}/${COST_DIR}/gpu_only.csv" ] || { log "[error] cost tables missing: $COST_DIR"; exit 1; }
log "numgpu-knee — dataset=$DATASET  model=$MODEL  pool=$KV_POOL_GB GiB  max_active=$MAX_ACTIVE"
log "scales=(${SCALES[*]})  num_gpus=(${NUM_GPUS_LIST[*]})  exec=pipeline  route=min_finish"
log "csv=$CSV_PATH  cost=$COST_DIR  shared_cache=$SHARED_CACHE"
log "sweep_dir=$SWEEP_DIR"

PY=$(which python)
RUNNER="${ROOT_DIR}/tools/run_serving_online_policy_compare.py"

# run_cell <num_gpus> <scale>
run_cell() {
  local ngpu=$1; local scale=$2
  local s_safe=$(echo "$scale" | tr -d '.')
  local label="g${ngpu}_s${s_safe}"
  local run_dir="${SWEEP_DIR}/${label}"
  log "── start: ${label} ──"
  local t0=$(date +%s)
  "$PY" "$RUNNER" \
      --model "$MODEL" \
      --gpu "$GPU_TYPE" \
      --cost-dir "$COST_DIR" \
      --n-requests "$N_REQUESTS" \
      --policies "$POLICY" \
      --kv-pool-gb "$KV_POOL_GB" \
      --max-active "$MAX_ACTIVE" \
      --arrival-scale "$scale" \
      --exec-model pipeline \
      --num-gpus "$ngpu" \
      --route-policy min_finish \
      --slo-e2e-ms "$SLO_E2E_MS" --slo-ttft-ms "$SLO_TTFT_MS" \
      --kv-oom-policy "$KV_OOM_POLICY" --kv-oom-hold-limit-ms "$KV_OOM_HOLD_LIMIT_MS" \
      --max-decode-batch-size "$MAX_DECODE_BS" \
      --max-consecutive-decode-batches "$MAX_CONSEC_DEC" \
      --admission-violation-budget -1 \
      --no-prompt-priority \
      --csv-path "$CSV_PATH" \
      --pim-cache-file "$SHARED_CACHE" \
      --no-kv-timeline \
      --run-dir "$run_dir" 2>&1 | tee -a "$LOG"
  local dt=$(( $(date +%s) - t0 ))
  log "── done:  ${label}  wall=${dt}s ──"
}

for ngpu in "${NUM_GPUS_LIST[@]}"; do
  for s in "${SCALES[@]}"; do
    run_cell "$ngpu" "$s"
  done
done

log
log "=== summary ==="
for cell in "${SWEEP_DIR}"/*/; do
  s="${cell}policy_compare_summary.txt"
  if [ -f "$s" ]; then
    log "--- $(basename "$cell") ---"
    grep -E "Throughput|Routes|TTFT p99|E2E p99|Span" "$s" 2>&1 | tee -a "$LOG"
  fi
done
log
log "Render locally with thesis_plotting/scripts/thesis_pipeline_knee.py"
log "(cells are labelled g<ngpu>_s<scale>; one curve per num_gpus)."
