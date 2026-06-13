#!/bin/bash
#SBATCH -J knee_finen5
#SBATCH -p high_latency
#SBATCH -c 4
#SBATCH --mem=32GB
#SBATCH -D /mnt/galactica/ocakir/attacc_simulator
#SBATCH -o /mnt/galactica/ocakir/logs/knee_finen5_%j.out
#SBATCH -e /mnt/galactica/ocakir/logs/knee_finen5_%j.err
#
# Fine-granularity RPS knee sweep at n=5000.
#
# Companion to the n=10000 fine-cliff sweep
# (run_rps_knee_pi0_wide_fine_cliff.sbatch). Same scale set, same
# canonical config — only the trace length differs. Lets us see how
# the cliff position + bistability boundaries change between n=5k and
# n=10k traces.
#
# Scales (23 cells): cover pre-cliff + cliff onset (fine 0.01-step at
# s=2.65-2.70) + overflow + partial-recovery transition (fine 0.01-step
# at s=2.50-2.55) + deep saturation.

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

# ── Config ─────────────────────────────────────────────────────────────
N_REQUESTS=5000
MODEL=pi0
GPU_TYPE=a6000
POLICY=max_util_full
KV_POOL_GB=8
MAX_ACTIVE=32
MAX_DECODE_BS=4
MAX_CONSEC_DEC=4
SLO_E2E_MS=500
SLO_TTFT_MS=200
KV_OOM_POLICY=hold_then_fallback
KV_OOM_HOLD_LIMIT_MS=50

# Same 23-scale set as the n=10000 fine-cliff sweep
ARRIVAL_SCALES="10 7 5 4 3.5 3 2.85 2.75 2.70 2.69 2.68 2.67 2.66 2.65 2.60 2.55 2.54 2.53 2.52 2.51 2.50 2.4 2 1.5"

CSV_PATH="${ROOT_DIR}/cluster_outputs/synthetic_v2/azure_poisson_wide_n5000_seed0.csv"
CACHE_DIR="${ROOT_DIR}/cluster_outputs/pim_cache"
TS=$(date +%Y%m%d_%H%M%S)
SWEEP_DIR="${ROOT_DIR}/cluster_outputs/online_serving_runs/rps_knee_pi0_wide_fine_n5k_${TS}"
LOG="${SWEEP_DIR}/sweep.log"

mkdir -p "${CACHE_DIR}" "${SWEEP_DIR}"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

if [ ! -s "$CSV_PATH" ]; then
  log "[error] CSV not found at $CSV_PATH"
  exit 1
fi
log "Reuse existing trace at $CSV_PATH ($(wc -l < "$CSV_PATH") lines)"

# ── Sweep ──────────────────────────────────────────────────────────────
n_scales=$(echo "$ARRIVAL_SCALES" | wc -w)
log ""
log "Fine-cliff knee sweep  (n=5000 companion)"
log "model=$MODEL  gpu=$GPU_TYPE  policy=$POLICY"
log "n=$N_REQUESTS  kv_pool=${KV_POOL_GB}GiB  max_active=$MAX_ACTIVE"
log "bs=$MAX_DECODE_BS  mc=$MAX_CONSEC_DEC"
log "SLO  e2e=${SLO_E2E_MS}ms  ttft=${SLO_TTFT_MS}ms"
log "kv_oom=$KV_OOM_POLICY  hold_limit=${KV_OOM_HOLD_LIMIT_MS}ms"
log "scales=($ARRIVAL_SCALES)  total cells = ${n_scales}"
log "sweep_dir=$SWEEP_DIR"
log ""

PY=$(which python)
RUNNER="${ROOT_DIR}/tools/run_serving_online_policy_compare.py"

i=1
for scale in $ARRIVAL_SCALES; do
  scale_label=$(awk -v x=$scale 'BEGIN{printf "s%04d", x*100}')
  label=$(printf "%02d_azure_poisson_wide_%s" "$i" "$scale_label")
  run_dir="${SWEEP_DIR}/${label}"
  cache_file="${CACHE_DIR}/${MODEL}_${GPU_TYPE}_knee_pi0_wide_fine_n5k_${scale_label}.bin"

  log "── start: ${label}  scale=${scale} ──"
  log "    cache: $cache_file"
  t0=$(date +%s)
  "$PY" "$RUNNER" \
      --model "$MODEL" \
      --gpu "$GPU_TYPE" \
      --n-requests "$N_REQUESTS" \
      --policies "$POLICY" \
      --kv-pool-gb "$KV_POOL_GB" --ml \
      --max-active "$MAX_ACTIVE" \
      --arrival-scale "$scale" \
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
      --run-dir "$run_dir" 2>&1 | tee -a "$LOG"
  dt=$(( $(date +%s) - t0 ))
  log "── done:  ${label}  wall=${dt}s ──"
  i=$((i+1))
done

log ""
log "Sweep complete. Sweep dir: $SWEEP_DIR"
