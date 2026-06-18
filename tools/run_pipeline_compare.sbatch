#!/bin/bash
#SBATCH -J pipe_cmp
#SBATCH -p high_latency
#SBATCH -c 4
#SBATCH --mem=64GB
#SBATCH --exclude=safari-nexus1
#SBATCH -D /mnt/galactica/ocakir/attacc_simulator
#SBATCH -o /mnt/galactica/ocakir/logs/pipe_cmp_%j.out
#SBATCH -e /mnt/galactica/ocakir/logs/pipe_cmp_%j.err
#
# Serial vs pipeline executor comparison across offered load.
#
# Runs the SAME routing policy (max_util_full) under exec_model=serial and
# exec_model=pipeline at a sweep of arrival scales, on the real-shape bootstrap
# trace. Expectation: in this feed-forward-bound regime the two executors are
# ~equivalent (both bound by the GPU FFN stage); the pipeline is the more
# faithful model (sequential FC->attention within a step, concurrent GPU/PIM
# across requests) and only diverges in an attention-bound regime.
#
# A shared per-model PIM-shape cache is reused across ALL cells: exec_model is
# not part of the cache fingerprint and the PIM command drains are identical
# under both executors, so the first cell warms the cache and the rest are fast.
#
# Defaults target the Pi0 knee (~2.1 RPS). For OpenVLA (~0.6 RPS) override:
#   MODEL=openvla COST_DIR=cluster_outputs/cost_tables_energyfix_openvla_a6000 \
#   SCALES="14 12 10 8.5 7.5" KV_POOL_GB=16 N_REQUESTS=2000 TAG=openvla \
#   sbatch tools/run_pipeline_compare.sbatch
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

MODEL="${MODEL:-pi0}"
GPU_TYPE=a6000
POLICY=max_util_full
N_REQUESTS="${N_REQUESTS:-2000}"
KV_POOL_GB="${KV_POOL_GB:-8}"      # large enough to avoid KV pressure (isolate the executor effect)
MAX_ACTIVE=5000
MAX_DECODE_BS="${MAX_DECODE_BS:-4}"
MAX_CONSEC_DEC=4
SLO_E2E_MS=500
SLO_TTFT_MS=200
TAG="${TAG:-pi0}"
COST_DIR="${COST_DIR:-cluster_outputs/cost_tables_energyfix_pi0_a6000}"
# Arrival scales spanning below -> above the knee (higher scale = slower = lower RPS).
read -r -a SCALES   <<< "${SCALES:-4.0 3.0 2.5 2.2 2.0}"
read -r -a EXEC_MODES <<< "${EXEC_MODES:-serial pipeline}"

CSV_PATH="${ROOT_DIR}/cluster_outputs/synthetic_v2/azure_bootstrap_n5000_seed0.csv"
CACHE_DIR="${ROOT_DIR}/cluster_outputs/pim_cache"
TS=$(date +%Y%m%d_%H%M%S)
SWEEP_DIR="${ROOT_DIR}/cluster_outputs/online_serving_runs/pipe_compare_${TAG}_${TS}"
LOG="${SWEEP_DIR}/sweep.log"
mkdir -p "${CACHE_DIR}" "${SWEEP_DIR}"

# One shared cache for the whole sweep (warm once, reuse for every cell).
SHARED_CACHE="${CACHE_DIR}/${MODEL}_${GPU_TYPE}_pipecmp_shared.bin"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

[ -s "$CSV_PATH" ] || { log "[error] CSV missing: $CSV_PATH"; exit 1; }
[ -s "${ROOT_DIR}/${COST_DIR}/gpu_only.csv" ] || { log "[error] cost tables missing: $COST_DIR"; exit 1; }
log "serial-vs-pipeline comparison — model=$MODEL  kv_pool=$KV_POOL_GB GiB  scales=(${SCALES[*]})  exec=(${EXEC_MODES[*]})"
log "cost tables: $COST_DIR   shared cache: $SHARED_CACHE"
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

# Warm the shared cache once with the lightest serial cell, then sweep both
# executors. (Serial warms PIM shapes the same as pipeline; exec_model is not
# in the cache fingerprint.)
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
log "Render with:"
log "  python thesis_plotting/scripts/thesis_pipeline_compare.py --sweep-dir $SWEEP_DIR --cost-dir $COST_DIR"
