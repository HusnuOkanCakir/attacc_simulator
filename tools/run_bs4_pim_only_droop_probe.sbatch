#!/bin/bash
#SBATCH -J bs4_pim
#SBATCH -p high_latency
#SBATCH -c 4
#SBATCH --mem=64GB
#SBATCH --exclude=safari-nexus1
#SBATCH -D /mnt/galactica/ocakir/attacc_simulator
#SBATCH -o /mnt/galactica/ocakir/logs/bs4_pim_%j.out
#SBATCH -e /mnt/galactica/ocakir/logs/bs4_pim_%j.err
#
# Control experiment for the bs=4 deep-saturation droop seen in
# knee_vs_bs_20260606_114716:
#   s=2.5  -> 2.10 RPS
#   s=2.0  -> 1.99 RPS  (droop starts; PIM share 94% -> 72%)
#   s=1.5  -> 1.90 RPS  (droop deepens; 18.5M KV-OOM holds)
#
# Hypothesis: the droop is caused by the min_finish scheduler migrating
# load to GPU under saturation, plus KV-OOM-forced GPU fallback at deep
# saturation. To test this, we disable the GPU route entirely
# (--pim-only) and switch KV-OOM to pure hold (no fallback path).
# If the droop disappears, the scheduler is the cause.
# If the droop persists, the cause is queue dynamics / KV churn.
#
# Same 7-scale row as the original bs=4 cells, single job, one cell at
# a time, kv_pool=64 GiB.

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
KV_POOL_GB=64
MAX_ACTIVE=5000
MAX_DECODE_BS=4
MAX_CONSEC_DEC=4
SLO_E2E_MS=500
SLO_TTFT_MS=200
# --pim-only auto-forces these, but set them explicitly anyway so the run
# log makes the experimental setup obvious:
KV_OOM_POLICY=hold
KV_OOM_HOLD_LIMIT_MS=-1   # effectively unbounded

# 7 scales matching the original knee_vs_bs bs=4 row exactly.
SCALES=(10 5 3 2.85 2.5 2 1.5)

CSV_PATH="${ROOT_DIR}/cluster_outputs/synthetic_v2/azure_poisson_wide_n5000_seed0.csv"
CACHE_DIR="${ROOT_DIR}/cluster_outputs/pim_cache"
TS=$(date +%Y%m%d_%H%M%S)
SWEEP_DIR="${ROOT_DIR}/cluster_outputs/online_serving_runs/bs4_pim_only_droop_probe_${TS}"
LOG="${SWEEP_DIR}/sweep.log"
mkdir -p "${CACHE_DIR}" "${SWEEP_DIR}"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

[ -s "$CSV_PATH" ] || { log "[error] CSV missing"; exit 1; }
log "bs=4 PIM-only droop probe — max_active=$MAX_ACTIVE, scales=(${SCALES[*]})"
log "sweep_dir=$SWEEP_DIR"

PY=$(which python)
RUNNER="${ROOT_DIR}/tools/run_serving_online_policy_compare.py"

run_cell() {
  local idx=$1
  local scale=$2
  local s_safe=$(echo "$scale" | tr -d '.')
  local label=$(printf "%02d_bs_4_pim_s%03d" "$idx" "$s_safe")
  local run_dir="${SWEEP_DIR}/${label}"
  local cache_file="${CACHE_DIR}/${MODEL}_${GPU_TYPE}_bs4_pim_only_s${s_safe}.bin"
  log "── start: ${label}  scale=${scale} ──"
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
  log "── done:  ${label}  wall=${dt}s ──"
}

idx=1
for s in "${SCALES[@]}"; do
  run_cell "$idx" "$s"
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
log "Compare against knee_vs_bs_20260606_114716/{15..21}_bs_4_s* cells."
