#!/bin/bash
#SBATCH -J bs4_lowkv
#SBATCH -p high_latency
#SBATCH -c 4
#SBATCH --mem=64GB
#SBATCH --exclude=safari-nexus1
#SBATCH -D /mnt/galactica/ocakir/attacc_simulator
#SBATCH -o /mnt/galactica/ocakir/logs/bs4_lowkv_%j.out
#SBATCH -e /mnt/galactica/ocakir/logs/bs4_lowkv_%j.err
#
# Companion to the existing bs=4 droop probe (which used 64 GiB KV pool).
# Hypothesis: at a deliberately constrained KV pool, the hybrid
# scheduler's GPU-fallback path is load-bearing and PIM-only collapses.
# This flips the Fig 6.8 narrative from "scheduler hurts due to
# over-routing" to "scheduler is essential at the binding capacity
# constraint."
#
# Runs 14 cells: 7 hybrid + 7 PIM-only at KV pool = 0.25 GiB, same
# seven scales as the existing droop probe (10, 5, 3, 2.85, 2.5, 2,
# 1.5). bs=4, max_active=5000, n=5000 requests.
#
# Wall estimate: heavy KV thrashing at deep saturation will dominate;
# expect ~12-20 h total. Hybrid cells should be faster than PIM-only
# at high load because hybrid escapes to GPU; PIM-only cells will
# block on KV-OOM holds.

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

N_REQUESTS=5000
MODEL=pi0
GPU_TYPE=a6000
POLICY=max_util_full
KV_POOL_GB=0.25
MAX_ACTIVE=5000
MAX_DECODE_BS=4
MAX_CONSEC_DEC=4
SLO_E2E_MS=500
SLO_TTFT_MS=200
KV_OOM_POLICY_HYBRID=hold_then_fallback
KV_OOM_HOLD_LIMIT_MS=50

SCALES=(10 5 3 2.85 2.5 2 1.5)

CSV_PATH="${ROOT_DIR}/cluster_outputs/synthetic_v2/azure_poisson_wide_n5000_seed0.csv"
CACHE_DIR="${ROOT_DIR}/cluster_outputs/pim_cache"
TS=$(date +%Y%m%d_%H%M%S)
SWEEP_DIR="${ROOT_DIR}/cluster_outputs/online_serving_runs/bs4_lowkv_hybrid_vs_pim_${TS}"
LOG="${SWEEP_DIR}/sweep.log"
mkdir -p "${CACHE_DIR}" "${SWEEP_DIR}"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

[ -s "$CSV_PATH" ] || { log "[error] CSV missing"; exit 1; }
log "bs=4 low-KV probe — kv_pool=$KV_POOL_GB GiB, max_active=$MAX_ACTIVE, scales=(${SCALES[*]})"
log "sweep_dir=$SWEEP_DIR"

PY=$(which python)
RUNNER="${ROOT_DIR}/tools/run_serving_online_policy_compare.py"

run_hybrid_cell() {
  local idx=$1
  local scale=$2
  local s_safe=$(echo "$scale" | tr -d '.')
  local label=$(printf "%02d_hybrid_bs_4_s%03d" "$idx" "$s_safe")
  local run_dir="${SWEEP_DIR}/${label}"
  local cache_file="${CACHE_DIR}/${MODEL}_${GPU_TYPE}_bs4_lowkv_hybrid_s${s_safe}.bin"
  log "── start hybrid: ${label}  scale=${scale} ──"
  local t0=$(date +%s)
  "$PY" "$RUNNER" \
      --model "$MODEL" \
      --gpu "$GPU_TYPE" \
      --n-requests "$N_REQUESTS" \
      --policies "$POLICY" \
      --kv-pool-gb "$KV_POOL_GB" --ml \
      --max-active "$MAX_ACTIVE" \
      --arrival-scale "$scale" \
      --slo-e2e-ms "$SLO_E2E_MS" --slo-ttft-ms "$SLO_TTFT_MS" \
      --kv-oom-policy "$KV_OOM_POLICY_HYBRID" \
      --kv-oom-hold-limit-ms "$KV_OOM_HOLD_LIMIT_MS" \
      --max-decode-batch-size "$MAX_DECODE_BS" \
      --max-consecutive-decode-batches "$MAX_CONSEC_DEC" \
      --admission-violation-budget -1 \
      --no-prompt-priority \
      --csv-path "$CSV_PATH" \
      --pim-cache-file "$cache_file" \
      --no-kv-timeline \
      --run-dir "$run_dir" 2>&1 | tee -a "$LOG"
  local dt=$(( $(date +%s) - t0 ))
  log "── done hybrid:  ${label}  wall=${dt}s ──"
}

run_pim_cell() {
  local idx=$1
  local scale=$2
  local s_safe=$(echo "$scale" | tr -d '.')
  local label=$(printf "%02d_pim_bs_4_s%03d" "$idx" "$s_safe")
  local run_dir="${SWEEP_DIR}/${label}"
  local cache_file="${CACHE_DIR}/${MODEL}_${GPU_TYPE}_bs4_lowkv_pim_s${s_safe}.bin"
  log "── start pim_only: ${label}  scale=${scale} ──"
  local t0=$(date +%s)
  "$PY" "$RUNNER" \
      --model "$MODEL" \
      --gpu "$GPU_TYPE" \
      --n-requests "$N_REQUESTS" \
      --policies "$POLICY" \
      --kv-pool-gb "$KV_POOL_GB" --ml \
      --max-active "$MAX_ACTIVE" \
      --arrival-scale "$scale" \
      --slo-e2e-ms "$SLO_E2E_MS" --slo-ttft-ms "$SLO_TTFT_MS" \
      --max-decode-batch-size "$MAX_DECODE_BS" \
      --max-consecutive-decode-batches "$MAX_CONSEC_DEC" \
      --admission-violation-budget -1 \
      --no-prompt-priority \
      --pim-only \
      --csv-path "$CSV_PATH" \
      --pim-cache-file "$cache_file" \
      --no-kv-timeline \
      --run-dir "$run_dir" 2>&1 | tee -a "$LOG"
  local dt=$(( $(date +%s) - t0 ))
  log "── done pim_only:  ${label}  wall=${dt}s ──"
}

# Run hybrid first (faster — GPU fallback drains pressure), then PIM-only.
idx=1
for s in "${SCALES[@]}"; do
  run_hybrid_cell "$idx" "$s"
  idx=$((idx + 1))
done
for s in "${SCALES[@]}"; do
  run_pim_cell "$idx" "$s"
  idx=$((idx + 1))
done

log
log "=== summary ==="
for cell in "${SWEEP_DIR}"/*/; do
  s="${cell}policy_compare_summary.txt"
  if [ -f "$s" ]; then
    log "--- $(basename "$cell") ---"
    grep -E "Throughput|Routes|KV-OOM holds|Decode batch mean|TTFT p99|E2E p99|Span" "$s" 2>&1 | tee -a "$LOG"
  fi
done
log
log "Compare against bs4_pim_only_droop_probe_20260608_161357 (which used 64 GiB pool)."
log "Render with thesis_bs4_pim_only_vs_hybrid.py pointing at this sweep dir."
