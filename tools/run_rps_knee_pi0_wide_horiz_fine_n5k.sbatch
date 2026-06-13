#!/bin/bash
#SBATCH -J knee_horiz
#SBATCH -p high_latency
#SBATCH -c 4
#SBATCH --mem=32GB
#SBATCH --exclude=safari-nexus1
#SBATCH -D /mnt/galactica/ocakir/attacc_simulator
#SBATCH -o /mnt/galactica/ocakir/logs/knee_horiz_%j.out
#SBATCH -e /mnt/galactica/ocakir/logs/knee_horiz_%j.err
#
# Knee sweep with horizontal-axis fine resolution.
#
# Companion to run_rps_knee_pi0_wide_fine_cliff_n5k.sbatch: that sweep
# spent its 0.01-step resolution in the vertical saturation tower at
# rps≈2.13. This sweep instead spreads its samples along the climbing
# region of the knee curve, mean offered load ≈ 1.2 .. 2.2 RPS, which
# is what the reader actually wants to read off the plot.
#
# Scale design (mean offered RPS = 5.5 / arrival_scale):
#   - 4 reference cells at lower RPS:  s=12, 7, 5  (≈ 0.46, 0.79, 1.10 RPS)
#   - 22 fine cells at 0.10 step in s ∈ [2.50, 4.60], which covers
#     offered RPS ∈ [1.20, 2.20]:
#       s=4.6 → 1.20 RPS    s=3.5 → 1.57 RPS
#       s=4.5 → 1.22 RPS    s=3.4 → 1.62 RPS
#       s=4.4 → 1.25 RPS    s=3.3 → 1.67 RPS
#       s=4.3 → 1.28 RPS    s=3.2 → 1.72 RPS
#       s=4.2 → 1.31 RPS    s=3.1 → 1.77 RPS
#       s=4.1 → 1.34 RPS    s=3.0 → 1.83 RPS
#       s=4.0 → 1.38 RPS    s=2.9 → 1.90 RPS  (star is near here)
#       s=3.9 → 1.41 RPS    s=2.8 → 1.96 RPS
#       s=3.8 → 1.45 RPS    s=2.7 → 2.04 RPS
#       s=3.7 → 1.49 RPS    s=2.6 → 2.12 RPS
#       s=3.6 → 1.53 RPS    s=2.5 → 2.20 RPS
#
# Achieved RPS for cells in s ∈ [2.65, 2.85] may collapse to ~0.8 RPS
# (chaotic overflow) — these will show up as the outliers we already
# know about. The renderer's deficit classifier handles them.

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

# ── Config (matches the existing fine-cliff sweep so the two figures
#    can be overlaid if needed) ────────────────────────────────────────
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

# 4 reference cells + 22 fine cells along the climbing region.
ARRIVAL_SCALES="12 7 5 4.6 4.5 4.4 4.3 4.2 4.1 4.0 3.9 3.8 3.7 3.6 3.5 3.4 3.3 3.2 3.1 3.0 2.9 2.8 2.7 2.6 2.5"

CSV_PATH="${ROOT_DIR}/cluster_outputs/synthetic_v2/azure_poisson_wide_n5000_seed0.csv"
CACHE_DIR="${ROOT_DIR}/cluster_outputs/pim_cache"
TS=$(date +%Y%m%d_%H%M%S)
SWEEP_DIR="${ROOT_DIR}/cluster_outputs/online_serving_runs/rps_knee_pi0_wide_horiz_fine_n5k_${TS}"
LOG="${SWEEP_DIR}/sweep.log"

mkdir -p "${CACHE_DIR}" "${SWEEP_DIR}"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

if [ ! -s "$CSV_PATH" ]; then
  log "[error] CSV not found at $CSV_PATH"
  exit 1
fi
log "Reuse existing trace at $CSV_PATH ($(wc -l < "$CSV_PATH") lines)"

n_scales=$(echo "$ARRIVAL_SCALES" | wc -w)
log ""
log "Horizontal-axis fine knee sweep  (n=5000, azure_poisson_wide, Pi0)"
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
  cache_file="${CACHE_DIR}/${MODEL}_${GPU_TYPE}_knee_pi0_wide_horiz_fine_n5k_${scale_label}.bin"

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
log ""
log "After rsync, render the figure with:"
log "  edit thesis_plotting/scripts/thesis_rps_knee_normalized_fine_clean.py"
log "  → set sweep_dir to this run (or add a new SWEEPS entry)"
log "  → python thesis_plotting/scripts/thesis_rps_knee_normalized_fine_clean.py"
