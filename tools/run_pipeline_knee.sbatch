#!/bin/bash
#SBATCH -J pipe_knee
#SBATCH -p high_latency
#SBATCH -c 4
#SBATCH --mem=64GB
#SBATCH --exclude=safari-nexus1
#SBATCH -D /mnt/galactica/ocakir/attacc_simulator
#SBATCH -o /mnt/galactica/ocakir/logs/pipe_knee_%j.out
#SBATCH -e /mnt/galactica/ocakir/logs/pipe_knee_%j.err
#
# RPS-knee comparison: serial vs pipeline executor.
#
# Sweeps arrival_scale (offered load) under BOTH exec_model=serial and
# exec_model=pipeline on one dataset, so the achieved-throughput / E2E-p99
# knee can be compared between the two execution models. The pipeline (1 GPU
# + 1 PIM, concurrent) is expected to sustain a higher knee and lower tail
# latency than the serial single-task executor; the routing ordering
# (pim_only-optimal) is unchanged because decode is feed-forward-bound.
#
# Two datasets (submit once each):
#   DATASET=bootstrap  -> azure_bootstrap_n5000_seed0.csv  (real Azure 19k
#                         conv shapes, joint-resampled)
#   DATASET=wide       -> azure_poisson_wide_n5000_seed0.csv (wide-Poisson)
#
# A shared per-(model,dataset) PIM-shape cache is reused across all cells
# (exec_model is not in the cache fingerprint and the drains are identical),
# so the first cell warms it and the rest are fast.
#
# Usage:
#   DATASET=bootstrap sbatch tools/run_pipeline_knee.sbatch
#   DATASET=wide      sbatch tools/run_pipeline_knee.sbatch
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
# Chapter-6 knee config (Table "Configuration of the primary Chapter 6
# evidence" + the active-window note): max_util_full, 64 GiB PIM KV pool,
# decode batch cap 4, max_active=5000 (bypass the cap; 32 collapses the knee),
# raw E2E p99 vs offered RPS, no-prompt-priority, hold_then_fallback.
KV_POOL_GB="${KV_POOL_GB:-64}"         # chapter uses 64 GiB to remove KV pressure
MAX_ACTIVE="${MAX_ACTIVE:-5000}"       # large: do NOT cap concurrency (32 distorts the knee + pipeline overlap)
MAX_DECODE_BS="${MAX_DECODE_BS:-4}"
MAX_CONSEC_DEC=4
SLO_E2E_MS="${SLO_E2E_MS:-500}"
SLO_TTFT_MS="${SLO_TTFT_MS:-200}"
KV_OOM_POLICY="${KV_OOM_POLICY:-hold_then_fallback}"
KV_OOM_HOLD_LIMIT_MS="${KV_OOM_HOLD_LIMIT_MS:-50}"
COST_DIR="${COST_DIR:-cluster_outputs/cost_tables_energyfix_${MODEL}_${GPU_TYPE}}"

# Per-model defaults. Offered RPS = k/arrival_scale with the SAME k for both
# datasets (Poisson arrivals at the same mean) and BOTH models (offered load
# is model-independent); the Pi0 wide knee is ~2.1 RPS (ref op 1.91 at
# scale 2.85, so k~5.44), the OpenVLA knee is far lower (~0.6 RPS), hence the
# different scale ranges. OpenVLA decode is slow (cache-miss command drains),
# so N is smaller; high_latency partition has no time limit.
if [ "$MODEL" = "openvla" ]; then
  DEFAULT_N=2000
  DEFAULT_SCALES="27 13.6 10.9 9.1 7.8 6.4 5.44 4.5"   # offered ~0.2 -> 1.2 RPS
else
  DEFAULT_N=5000
  DEFAULT_SCALES="6.8 4.5 3.4 2.85 2.6 2.36 2.18 2.0 1.8 1.6"  # offered ~0.8 -> 3.4 RPS
fi
N_REQUESTS="${N_REQUESTS:-$DEFAULT_N}"
read -r -a SCALES     <<< "${SCALES:-$DEFAULT_SCALES}"
read -r -a EXEC_MODES <<< "${EXEC_MODES:-serial pipeline}"

CSV_PATH="${ROOT_DIR}/cluster_outputs/synthetic_v2/${CSV_NAME}"
CACHE_DIR="${ROOT_DIR}/cluster_outputs/pim_cache"
TS=$(date +%Y%m%d_%H%M%S)
SWEEP_DIR="${ROOT_DIR}/cluster_outputs/online_serving_runs/pipe_knee_${DATASET}_${MODEL}_${TS}"
LOG="${SWEEP_DIR}/sweep.log"
mkdir -p "${CACHE_DIR}" "${SWEEP_DIR}"
SHARED_CACHE="${CACHE_DIR}/${MODEL}_${GPU_TYPE}_pipeknee_${DATASET}.bin"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

[ -s "$CSV_PATH" ] || { log "[error] CSV missing: $CSV_PATH"; exit 1; }
[ -s "${ROOT_DIR}/${COST_DIR}/gpu_only.csv" ] || { log "[error] cost tables missing: $COST_DIR"; exit 1; }
log "pipeline-knee — dataset=$DATASET  model=$MODEL  pool=$KV_POOL_GB GiB  max_active=$MAX_ACTIVE"
log "scales=(${SCALES[*]})  exec=(${EXEC_MODES[*]})"
log "csv=$CSV_PATH  cost=$COST_DIR  shared_cache=$SHARED_CACHE"
log "sweep_dir=$SWEEP_DIR"

PY=$(which python)
RUNNER="${ROOT_DIR}/tools/run_serving_online_policy_compare.py"

# run_cell <exec> <scale>
run_cell() {
  local exec=$1; local scale=$2
  local s_safe=$(echo "$scale" | tr -d '.')
  local label="${exec}_s${s_safe}"
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
      --exec-model "$exec" \
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

for exec in "${EXEC_MODES[@]}"; do
  for s in "${SCALES[@]}"; do
    run_cell "$exec" "$s"
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
log "Render (this dataset):"
log "  python thesis_plotting/scripts/thesis_pipeline_knee.py --dirs $SWEEP_DIR --labels $DATASET"
log "Render (overlay both datasets): --dirs <bootstrap_dir> <wide_dir> --labels bootstrap wide"
