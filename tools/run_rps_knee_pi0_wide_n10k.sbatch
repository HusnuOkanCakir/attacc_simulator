#!/bin/bash
#SBATCH -J knee_n10k
#SBATCH -p high_latency
#SBATCH -c 4
#SBATCH --mem=32GB
#SBATCH -D /mnt/galactica/ocakir/attacc_simulator
#SBATCH -o /mnt/galactica/ocakir/logs/knee_n10k_%j.out
#SBATCH -e /mnt/galactica/ocakir/logs/knee_n10k_%j.err
#
# Long-trace RPS knee sweep — Pi0 + azure_poisson_wide v2 at n=10000.
#
# Motivation: vLLM (Kwon et al. SOSP'23) uses 1-hour Poisson traces for the
# normalized-latency knee. Our existing n=5000 sweep produces ~38 min of
# arrival span at scale=2.5 (the cliff regime), which under-samples steady
# state. n=10000 doubles the arrival span to ~76 min — matching vLLM's
# methodology while still tractable on the cluster.
#
# What this sbatch does:
#   1. Generate the n=10000 azure_poisson_wide v2 CSV if missing.
#   2. Sweep arrival_scale across a dense scale set covering the cliff +
#      metastable bistability region.
#
# Config (matches the existing rps_knee_pi0_20260601_175358 dense sweep
# except for the trace size):
#   model=pi0, gpu=a6000, policy=max_util_full
#   bs=4, max_active=32, kv_pool=8 GiB
#   SLO 500/200ms, kv_oom=hold_then_fallback (--kv-max-preempt-tries 3 default)
#
# Output dir: cluster_outputs/online_serving_runs/rps_knee_pi0_wide_n10k_<TS>/
# Cache files: ${MODEL}_${GPU}_knee_pi0_wide_n10k_s${scale*10}.bin
#
# After completion the runner auto-renders per-cell summary; the headline
# normalized-latency knee figure should be re-rendered manually with:
#   python thesis_plotting/scripts/thesis_rps_knee_normalized.py \
#       --sweep-dir <new-sweep-dir>

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

# Dense scale set covering pre-cliff + cliff + metastable + saturation
ARRIVAL_SCALES="10 7 5 4 3.5 3 2.75 2.7 2.65 2.6 2.55 2.5 2.45 2.4 2 1.5"

CSV_PATH="${ROOT_DIR}/cluster_outputs/synthetic_v2/azure_poisson_wide_n10000_seed0.csv"
CACHE_DIR="${ROOT_DIR}/cluster_outputs/pim_cache"
TS=$(date +%Y%m%d_%H%M%S)
SWEEP_DIR="${ROOT_DIR}/cluster_outputs/online_serving_runs/rps_knee_pi0_wide_n10k_${TS}"
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

# Quick profile dump so the log records what was actually fed to the sweep.
log "[profile] azure_poisson_wide n=10000"
python tools/profile_trace_rps.py --csv "$CSV_PATH" --bucket-ms 1000 2>&1 \
  | tee -a "$LOG" | head -30 || true

# ── Step 1: sweep ──────────────────────────────────────────────────────
n_scales=$(echo "$ARRIVAL_SCALES" | wc -w)
log ""
log "Long-trace knee sweep"
log "model=$MODEL  gpu=$GPU_TYPE  policy=$POLICY"
log "n=$N_REQUESTS  kv_pool=${KV_POOL_GB}GiB  max_active=$MAX_ACTIVE"
log "bs=$MAX_DECODE_BS  mc=$MAX_CONSEC_DEC"
log "SLO  e2e=${SLO_E2E_MS}ms  ttft=${SLO_TTFT_MS}ms"
log "kv_oom=$KV_OOM_POLICY  hold_limit=${KV_OOM_HOLD_LIMIT_MS}ms"
log "scales=($ARRIVAL_SCALES)  matrix: ${n_scales} cells"
log "sweep_dir=$SWEEP_DIR"
log ""

PY=$(which python)
RUNNER="${ROOT_DIR}/tools/run_serving_online_policy_compare.py"

i=1
for scale in $ARRIVAL_SCALES; do
  scale_label=$(awk -v x=$scale 'BEGIN{printf "s%04d", x*100}')
  label=$(printf "%02d_azure_poisson_wide_%s" "$i" "$scale_label")
  run_dir="${SWEEP_DIR}/${label}"
  cache_file="${CACHE_DIR}/${MODEL}_${GPU_TYPE}_knee_pi0_wide_n10k_${scale_label}.bin"

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
log "After rsyncing to laptop, render normalized-latency knee figure:"
log "  python thesis_plotting/scripts/thesis_rps_knee_normalized.py"
log "(edit DENSE/META paths in the script to point at this new sweep dir)"
