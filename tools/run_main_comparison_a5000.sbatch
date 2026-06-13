#!/bin/bash
#SBATCH -J main_a5k
#SBATCH -p high_latency
#SBATCH -c 4
#SBATCH --mem=64GB
#SBATCH --exclude=safari-nexus1
#SBATCH -D /mnt/galactica/ocakir/attacc_simulator
#SBATCH -o /mnt/galactica/ocakir/logs/main_a5k_%j.out
#SBATCH -e /mnt/galactica/ocakir/logs/main_a5k_%j.err
#
# Re-run of the chapter-6 main GPU-only vs hybrid comparison cell with the
# new max_active=5000 default, at the new safe operating point
# arrival_scale=2.85 (mean offered ≈ 1.91 RPS, 0.85× the measured ≈ 2.10
# RPS service ceiling).
#
# Drives:
#   - Table 6.3 (route-availability comparison absolute numbers)
#   - Figure 6.4 (ch6_pi0_wide_safe_gpu_vs_hybrid)
#   - The "21% throughput / TTFT / E2E" prose in §6.6.
#
# Two cells:
#   A_hybrid_a5000     — both routes available, min_finish scheduler
#   B_gpu_only_a5000   — PIM route disabled via sentinel hybrid_route_name
#
# Single job, one cell at a time, exactly like run_cold_cache_recheck.sbatch.
# kv_pool=64 GiB to handle max_active=5000 worst-case KV demand.

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

# ── Config (matches new chapter default; arrival_scale shifted) ─────
N_REQUESTS=5000
MODEL=pi0
GPU_TYPE=a6000
POLICY=max_util_full
KV_POOL_GB=64
MAX_ACTIVE=5000
MAX_DECODE_BS=4
MAX_CONSEC_DEC=4
ARRIVAL_SCALE=2.85
SLO_E2E_MS=500
SLO_TTFT_MS=200
KV_OOM_POLICY=hold_then_fallback
KV_OOM_HOLD_LIMIT_MS=50

CSV_PATH="${ROOT_DIR}/cluster_outputs/synthetic_v2/azure_poisson_wide_n5000_seed0.csv"
CACHE_DIR="${ROOT_DIR}/cluster_outputs/pim_cache"
TS=$(date +%Y%m%d_%H%M%S)
SWEEP_DIR="${ROOT_DIR}/cluster_outputs/online_serving_runs/main_comparison_a5000_${TS}"
LOG="${SWEEP_DIR}/sweep.log"
mkdir -p "${CACHE_DIR}" "${SWEEP_DIR}"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

[ -s "$CSV_PATH" ] || { log "[error] CSV missing"; exit 1; }
log "Main comparison rerun at max_active=$MAX_ACTIVE, arrival_scale=$ARRIVAL_SCALE"
log "sweep_dir=$SWEEP_DIR"

PY=$(which python)
RUNNER="${ROOT_DIR}/tools/run_serving_online_policy_compare.py"

run_cell() {
  local label=$1; shift
  local run_dir="${SWEEP_DIR}/${label}"
  local cache_file="${CACHE_DIR}/${MODEL}_${GPU_TYPE}_main_comparison_a5000_${label}.bin"
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
      --csv-path "$CSV_PATH" \
      --pim-cache-file "$cache_file" \
      --no-kv-timeline \
      --run-dir "$run_dir" \
      "$@" 2>&1 | tee -a "$LOG"
  local dt=$(( $(date +%s) - t0 ))
  log "── done:  ${label}  wall=${dt}s ──"
}

run_cell "A_hybrid_a5000"
run_cell "B_gpu_only_a5000" --gpu-only

log
log "=== summary ==="
for cell in A_hybrid_a5000 B_gpu_only_a5000; do
  s="${SWEEP_DIR}/${cell}/policy_compare_summary.txt"
  if [ -f "$s" ]; then
    log "--- $cell ---"
    grep -E "Throughput|Routes|TTFT p99|E2E p99|Span" "$s" 2>&1 | tee -a "$LOG"
  fi
done
log
log "Update Table 6.3 + Figure 6.4 + \\S6.6 prose with the numbers above."
