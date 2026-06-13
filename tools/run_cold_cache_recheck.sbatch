#!/bin/bash
#SBATCH -J cold_cache
#SBATCH -p high_latency
#SBATCH -c 4
#SBATCH --mem=32GB
#SBATCH -t 12:00:00
#SBATCH -D /mnt/galactica/ocakir/attacc_simulator
#SBATCH -o /mnt/galactica/ocakir/logs/cold_cache_%j.out
#SBATCH -e /mnt/galactica/ocakir/logs/cold_cache_%j.err
#
# Cold/warm sensitivity recheck for the chapter-6 main comparison.
#
# The current §6.7 absolute TTFT/E2E numbers (GPU-only 952 s TTFT p99 /
# 1013 s E2E p99 vs hybrid 1.13 s / 38.85 s) come from cells with the
# PIM shape cache fully warm (526 351 hits, 5 misses). Chapter 5 §"PIM
# shape cache" notes the cache is not timing-neutral, so absolute
# latency from cached runs needs a no-cache sensitivity check before
# being defensible as final evidence.
#
# This sbatch reproduces *both* legs of the 1.09 RPS comparison with
# --no-pim-cache and compares the resulting absolute numbers to the
# cached baseline. If they agree within ~5%, the cache caveat can be
# dropped. If they diverge, the chapter should report the GPU/hybrid
# *ratio* as headline and demote absolute numbers to "indicative".
#
# Config mirrors the existing main-comparison cells in
# knee_vs_route_20260601_151039 / 03_route_hybrid_s050 and
# /09_route_gpu_only_s050:
#   model=pi0, gpu=a6000, policy=max_util_full, KV_POOL_GB=8,
#   max_active=32, bs=4, max_consec_decode=4, SLO=500/200,
#   kv_oom=hold_then_fallback, hold=50, scale=5.0, n=5000,
#   route=hybrid (cell A) and gpu_only (cell B).

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

# ── Config (mirrors the main §6.7 comparison cell) ────────────────────
N_REQUESTS=5000
MODEL=pi0
GPU_TYPE=a6000
POLICY=max_util_full
KV_POOL_GB=8
MAX_ACTIVE=32
MAX_DECODE_BS=4
MAX_CONSEC_DEC=4
ARRIVAL_SCALE=5.0
SLO_E2E_MS=500
SLO_TTFT_MS=200
KV_OOM_POLICY=hold_then_fallback
KV_OOM_HOLD_LIMIT_MS=50

CSV_PATH="${ROOT_DIR}/cluster_outputs/synthetic_v2/azure_poisson_wide_n5000_seed0.csv"
TS=$(date +%Y%m%d_%H%M%S)
SWEEP_DIR="${ROOT_DIR}/cluster_outputs/online_serving_runs/cold_cache_recheck_${TS}"
LOG="${SWEEP_DIR}/sweep.log"
mkdir -p "${SWEEP_DIR}"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

[ -s "$CSV_PATH" ] || { log "[error] CSV missing"; exit 1; }
log "Cold-cache recheck (Pi0 + azure_poisson_wide, scale=5, n=5000)"
log "sweep_dir=$SWEEP_DIR"

PY=$(which python)
RUNNER="${ROOT_DIR}/tools/run_serving_online_policy_compare.py"

run_cell() {
  local label=$1; shift
  local run_dir="${SWEEP_DIR}/${label}"
  log "── start: ${label}  extra=$* ──"
  local t0=$(date +%s)
  "$PY" "$RUNNER" \
      --model "$MODEL" \
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
      --admission-violation-budget -1 \
      --no-prompt-priority \
      --no-pim-cache \
      --csv-path "$CSV_PATH" \
      --no-kv-timeline \
      --run-dir "$run_dir" \
      "$@" 2>&1 | tee -a "$LOG"
  local dt=$(( $(date +%s) - t0 ))
  log "── done:  ${label}  wall=${dt}s ──"
}

# Cell A — hybrid (matches knee_vs_route .../03_route_hybrid_s050)
run_cell "A_hybrid_nocache"

# Cell B — gpu-only (matches knee_vs_route .../09_route_gpu_only_s050)
run_cell "B_gpu_only_nocache" --gpu-only

log
log "Summary of cold-cache results:"
for cell in A_hybrid_nocache B_gpu_only_nocache; do
  s="${SWEEP_DIR}/${cell}/policy_compare_summary.txt"
  if [ -f "$s" ]; then
    log "--- $cell ---"
    grep -E "Throughput|TTFT p99|E2E p99|Routes|Span" "$s" 2>&1 | tee -a "$LOG"
  fi
done
log
log "Compare against cached baseline:"
log "  Hybrid (cached): 1.09 RPS, TTFT p99 1.13 s, E2E p99 38.85 s"
log "  GPU-only (cached): 0.90 RPS, TTFT p99 952 s, E2E p99 1013 s"
log "  Source: knee_vs_route_20260601_151039 / {03_route_hybrid_s050, 09_route_gpu_only_s050}"
