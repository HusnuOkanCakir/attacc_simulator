#!/bin/bash
#SBATCH -J knee_fine
#SBATCH -p high_latency
#SBATCH -c 4
#SBATCH --mem=32GB
#SBATCH -D /mnt/galactica/ocakir/attacc_simulator
#SBATCH -o /mnt/galactica/ocakir/logs/knee_fine_%j.out
#SBATCH -e /mnt/galactica/ocakir/logs/knee_fine_%j.err
#
# Fine-granularity RPS knee sweep — Pi0 + azure_poisson_wide v2 at n=10000.
#
# Refines the existing dense knee + metastability sweeps with much tighter
# sampling at the two critical transition points:
#
#   1. Cliff onset (s ~ 2.65–2.70):
#      where the system flips from DRAIN (rps ~ 2.0) to OVERFLOW (rps ~ 1.25).
#      Currently sampled at s=2.65, 2.70 only — we add s=2.66, 2.67, 2.68, 2.69.
#
#   2. Stable-saturation recovery (s ~ 2.50–2.55):
#      where the system recovers from PARTIAL (rps ~ 1.53) back to DRAIN
#      (rps ~ 2.13). Currently sampled at s=2.50, 2.55 only — we add s=2.51,
#      2.52, 2.53, 2.54.
#
# Plus full coverage of the surrounding regime (pre-cliff, mid-cliff, deep
# saturation) so the new sweep can stand on its own for re-rendering the
# normalized-latency knee figure.
#
# n=10000 (long trace, vLLM §6 1-hour-equivalent at our chosen a). All other
# config matches the existing rps_knee_pi0_20260601_175358 dense sweep.
#
# Output dir: cluster_outputs/online_serving_runs/rps_knee_pi0_wide_fine_<TS>/

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
N_REQUESTS=10000
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

# Scale set — 23 cells covering pre-cliff, cliff onset (fine 0.01-step),
# overflow region, partial-recovery transition (fine 0.01-step), deep
# saturation. Ordered roughly low-load → high-load so the cache warms up
# along the curve.
ARRIVAL_SCALES="10 7 5 4 3.5 3 2.85 2.75 2.70 2.69 2.68 2.67 2.66 2.65 2.60 2.55 2.54 2.53 2.52 2.51 2.50 2.4 2 1.5"

CSV_PATH="${ROOT_DIR}/cluster_outputs/synthetic_v2/azure_poisson_wide_n10000_seed0.csv"
CACHE_DIR="${ROOT_DIR}/cluster_outputs/pim_cache"
TS=$(date +%Y%m%d_%H%M%S)
SWEEP_DIR="${ROOT_DIR}/cluster_outputs/online_serving_runs/rps_knee_pi0_wide_fine_${TS}"
LOG="${SWEEP_DIR}/sweep.log"

mkdir -p "${CACHE_DIR}" "${SWEEP_DIR}"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# ── Step 0: generate n=10000 trace if missing ──────────────────────────
if [ ! -s "$CSV_PATH" ]; then
  log "Generating n=10000 azure_poisson_wide v2 trace ..."
  python tools/gen_synthetic_trace_v2.py \
      --preset conversational \
      --shape-distribution mixture_poisson \
      --n-requests "$N_REQUESTS" \
      --rate-hz 5.5 \
      --seed 0 \
      --out-csv "$CSV_PATH" 2>&1 | tee -a "$LOG"
  log "Generated trace at $CSV_PATH ($(wc -l < "$CSV_PATH") lines)"
else
  log "Reuse existing trace at $CSV_PATH ($(wc -l < "$CSV_PATH") lines)"
fi

log "[profile] azure_poisson_wide n=10000"
python tools/profile_trace_rps.py --csv "$CSV_PATH" --bucket-ms 1000 2>&1 \
  | tee -a "$LOG" | head -30 || true

# ── Step 1: sweep ──────────────────────────────────────────────────────
n_scales=$(echo "$ARRIVAL_SCALES" | wc -w)
log ""
log "Fine-cliff knee sweep"
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
  # 4-digit scale encoding (scale*100) so fractional scales don't collide
  scale_label=$(awk -v x=$scale 'BEGIN{printf "s%04d", x*100}')
  label=$(printf "%02d_azure_poisson_wide_%s" "$i" "$scale_label")
  run_dir="${SWEEP_DIR}/${label}"
  cache_file="${CACHE_DIR}/${MODEL}_${GPU_TYPE}_knee_pi0_wide_fine_${scale_label}.bin"

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
log "Next: rsync to laptop and re-render normalized-latency knee figure."
log "  rsync -av ocakir@safari-proxy:${SWEEP_DIR} ~/attacc_simulator/cluster_outputs/online_serving_runs/"
log "  python thesis_plotting/scripts/thesis_rps_knee_normalized.py"
log "  (edit DENSE/META paths in the script to point at this new sweep dir)"
