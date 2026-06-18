#!/bin/bash
#SBATCH -J sched_2gpu
#SBATCH -p high_latency
#SBATCH -c 4
#SBATCH --mem=64GB
#SBATCH --exclude=safari-nexus1
#SBATCH -D /mnt/galactica/ocakir/attacc_simulator
#SBATCH -o /mnt/galactica/ocakir/logs/sched_2gpu_%j.out
#SBATCH -e /mnt/galactica/ocakir/logs/sched_2gpu_%j.err
#
# Scheduler-value experiment: does an intelligent scheduler with BOTH GPU and
# PIM beat single-resource execution?  Knee sweep (offered RPS) over three
# configs on the real-shape bootstrap trace:
#
#   gpu_only  : --gpu-only           (single GPU does prefill+decode; no PIM)
#   pim_only  : --pim-only           (pipeline GPU1-FC ∥ PIM-attention, all PIM,
#                                      no routing choice = "scheduler disabled")
#   hyb_2gpu  : --num-gpus 2 --exec-model pipeline --route-policy min_finish
#                                     (hybrid GPU1+PIM lane + dedicated gpu_only
#                                      overflow lane on GPU2; the scheduler
#                                      routes per request to the faster lane)
#
# Expected ordering (worst -> best): gpu_only < pim_only < hyb_2gpu.
# gpu_only has no FC∥attention overlap; pim_only overlaps but cannot shed
# load when its single GPU+PIM pipeline saturates; hyb_2gpu spills overflow
# to the idle 2nd GPU once the hybrid queue is deeper than GPU2's path, so it
# sustains the highest RPS / lowest tail.
#
# Pool = 64 GiB (no KV pressure) so the win is the scheduler's queueing-aware
# routing across heterogeneous resources, NOT KV-capacity fallback.
#
# Usage:
#   sbatch tools/run_scheduler_value_2gpu.sbatch
#   MODEL=openvla SCALES="13.6 9.1 6.4 5.44 4.5" KV_POOL_GB=8 \
#     COST_DIR=cluster_outputs/cost_tables_energyfix_openvla_a6000 \
#     sbatch tools/run_scheduler_value_2gpu.sbatch
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
GPU_TYPE="${GPU_TYPE:-a6000}"
POLICY=max_util_full
KV_POOL_GB="${KV_POOL_GB:-64}"
MAX_ACTIVE=5000
MAX_DECODE_BS="${MAX_DECODE_BS:-4}"
MAX_CONSEC_DEC=4
SLO_E2E_MS=500
SLO_TTFT_MS=200
TAG="${TAG:-bootstrap}"
COST_DIR="${COST_DIR:-cluster_outputs/cost_tables_energyfix_${MODEL}_${GPU_TYPE}}"

# Knee scales (offered RPS = k/scale, k~5.44 for pi0). Default spans the
# pim_only knee (~2.1 RPS) up through where hyb_2gpu still holds (the 2nd
# GPU lifts the ceiling), so all three curves' knees are visible.
if [ "$MODEL" = "openvla" ]; then
  DEFAULT_N=2000
  DEFAULT_SCALES="13.6 10.9 9.1 7.8 6.4 5.44 4.5 3.6"
else
  DEFAULT_N=5000
  DEFAULT_SCALES="4.5 3.4 2.85 2.6 2.36 2.18 2.0 1.8 1.6 1.4 1.2"
fi
N_REQUESTS="${N_REQUESTS:-$DEFAULT_N}"
read -r -a SCALES <<< "${SCALES:-$DEFAULT_SCALES}"
read -r -a CONFIGS <<< "${CONFIGS:-gpu_only pim_only hyb_2gpu}"

CSV_PATH="${ROOT_DIR}/cluster_outputs/synthetic_v2/azure_bootstrap_n5000_seed0.csv"
CACHE_DIR="${ROOT_DIR}/cluster_outputs/pim_cache"
TS=$(date +%Y%m%d_%H%M%S)
SWEEP_DIR="${ROOT_DIR}/cluster_outputs/online_serving_runs/sched_2gpu_${TAG}_${MODEL}_${TS}"
LOG="${SWEEP_DIR}/sweep.log"
mkdir -p "${CACHE_DIR}" "${SWEEP_DIR}"
# pim_only and hyb_2gpu produce identical PIM attention drains -> shared cache;
# gpu_only never touches PIM so the cache is simply unused for it.
SHARED_CACHE="${CACHE_DIR}/${MODEL}_${GPU_TYPE}_sched2gpu_${TAG}.bin"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

[ -s "$CSV_PATH" ] || { log "[error] CSV missing: $CSV_PATH"; exit 1; }
[ -s "${ROOT_DIR}/${COST_DIR}/gpu_only.csv" ] || { log "[error] cost tables missing: $COST_DIR"; exit 1; }
log "scheduler-value (2-GPU) — model=$MODEL  pool=$KV_POOL_GB GiB  max_active=$MAX_ACTIVE"
log "configs=(${CONFIGS[*]})  scales=(${SCALES[*]})"
log "csv=$CSV_PATH  cost=$COST_DIR  shared_cache=$SHARED_CACHE"
log "sweep_dir=$SWEEP_DIR"

PY=$(which python)
RUNNER="${ROOT_DIR}/tools/run_serving_online_policy_compare.py"

# run_cell <config> <scale>
run_cell() {
  local cfg=$1; local scale=$2
  local s_safe=$(echo "$scale" | tr -d '.')
  local label="${cfg}_s${s_safe}"
  local run_dir="${SWEEP_DIR}/${label}"

  # Per-config flags.
  local -a cfg_flags=()
  case "$cfg" in
    gpu_only) cfg_flags=( --gpu-only --exec-model pipeline --num-gpus 1 ) ;;
    pim_only) cfg_flags=( --pim-only --exec-model pipeline --num-gpus 1 ) ;;
    hyb_2gpu) cfg_flags=( --exec-model pipeline --num-gpus 2 --route-policy min_finish
                          --kv-oom-policy hold_then_fallback --kv-oom-hold-limit-ms 50
                          --kv-max-preempt-tries 3 ) ;;
    *) log "[error] unknown config: $cfg"; return 1 ;;
  esac

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
      --slo-e2e-ms "$SLO_E2E_MS" --slo-ttft-ms "$SLO_TTFT_MS" \
      --max-decode-batch-size "$MAX_DECODE_BS" \
      --max-consecutive-decode-batches "$MAX_CONSEC_DEC" \
      --admission-violation-budget -1 \
      --no-prompt-priority \
      --csv-path "$CSV_PATH" \
      --pim-cache-file "$SHARED_CACHE" \
      --no-kv-timeline \
      --run-dir "$run_dir" \
      "${cfg_flags[@]}" 2>&1 | tee -a "$LOG"
  local dt=$(( $(date +%s) - t0 ))
  log "── done:  ${label}  wall=${dt}s ──"
}

for cfg in "${CONFIGS[@]}"; do
  for s in "${SCALES[@]}"; do
    run_cell "$cfg" "$s"
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
log "Cells are labelled <config>_s<scale>; one knee curve per config"
log "(gpu_only / pim_only / hyb_2gpu). Render locally; post-process energy"
log "with tools/compute_serving_energy.py if an energy panel is wanted."
