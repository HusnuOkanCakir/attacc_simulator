#!/bin/bash
#SBATCH -J knee_fcfs
#SBATCH -p high_latency
#SBATCH -c 4
#SBATCH --mem=32GB
#SBATCH --exclude=safari-nexus1
#SBATCH -D /mnt/galactica/ocakir/attacc_simulator
#SBATCH -o /mnt/galactica/ocakir/logs/knee_fcfs_%j.out
#SBATCH -e /mnt/galactica/ocakir/logs/knee_fcfs_%j.err
#
# Fine-cliff knee sweep with a "bypassed" scheduler: strict FCFS on a
# fixed PIM route, no concurrency, no batching, no route choice.
#
# Configuration that removes all scheduler decisions:
#   --max-active 1                     → one request in flight at a time
#   --max-decode-batch-size 1          → no cross-request decode batching
#   --max-consecutive-decode-batches 1 → no prefill/decode reordering
#   --pim-only                         → fixed route, no min_finish call;
#                                         also forces kv_oom_policy=hold and
#                                         kv_max_preempt_tries=-1
#
# Hypothesis being tested:
#   The non-monotonic switching region observed in
#   rps_knee_pi0_wide_fine_n5k_20260602_183013 is a queue/batching
#   phenomenon. With serial-FCFS execution there is no queue depth, no
#   bs=4 cohort fill, and no min_finish route shuffling, so the chaos
#   zone should disappear and the achieved-RPS curve should be
#   monotonic in offered RPS. The achieved ceiling will be lower than
#   2.13 RPS because there is no batching speedup.
#
# Same 24 arrival_scales as the existing fine-cliff sweep so the
# resulting figure is directly comparable to
# fig_rps_knee_pi0_wide_normalized_fine_n5k_clean.png.

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

# ── Config ────────────────────────────────────────────────────────────
N_REQUESTS=5000
MODEL=pi0
GPU_TYPE=a6000
POLICY=max_util_full        # eviction policy; irrelevant at max_active=1
KV_POOL_GB=8
MAX_ACTIVE=1                # ← FCFS: one in flight at a time
MAX_DECODE_BS=1             # ← no decode batching
MAX_CONSEC_DEC=1            # ← no consecutive-decode reordering
SLO_E2E_MS=500
SLO_TTFT_MS=200
# kv_oom_policy and kv_max_preempt_tries are forced by --pim-only;
# values here are just for log clarity.
KV_OOM_POLICY=hold_then_fallback
KV_OOM_HOLD_LIMIT_MS=50

# Same 24-scale list as the existing fine-cliff sweep.
ARRIVAL_SCALES="10 7 5 4 3.5 3 2.85 2.75 2.70 2.69 2.68 2.67 2.66 2.65 2.60 2.55 2.54 2.53 2.52 2.51 2.50 2.4 2 1.5"

CSV_PATH="${ROOT_DIR}/cluster_outputs/synthetic_v2/azure_poisson_wide_n5000_seed0.csv"
CACHE_DIR="${ROOT_DIR}/cluster_outputs/pim_cache"
TS=$(date +%Y%m%d_%H%M%S)
SWEEP_DIR="${ROOT_DIR}/cluster_outputs/online_serving_runs/rps_knee_pi0_wide_fine_n5k_fcfs_${TS}"
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
log "FCFS fine-cliff knee sweep  (n=5000, azure_poisson_wide, Pi0)"
log "model=$MODEL  gpu=$GPU_TYPE  policy=$POLICY  route=PIM-ONLY"
log "n=$N_REQUESTS  kv_pool=${KV_POOL_GB}GiB  max_active=${MAX_ACTIVE}  bs=${MAX_DECODE_BS}"
log "max_consec_decode=$MAX_CONSEC_DEC  (all set to 1 to disable scheduler reordering)"
log "SLO  e2e=${SLO_E2E_MS}ms  ttft=${SLO_TTFT_MS}ms"
log "kv_oom=(forced hold by --pim-only)"
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
  cache_file="${CACHE_DIR}/${MODEL}_${GPU_TYPE}_knee_pi0_wide_fine_n5k_fcfs_${scale_label}.bin"

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
      --pim-only \
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
log "Per-cell summary (offered ≈ 5.5 / scale, achieved = throughput):"
for d in "$SWEEP_DIR"/*; do
  [ -d "$d" ] || continue
  s="${d}/policy_compare_summary.txt"
  [ -f "$s" ] || continue
  rps=$(grep -E "^Throughput" "$s" | awk '{print $NF}')
  routes=$(grep -E "Routes" "$s" | awk -F'PIM/GPU\\)' '{print $2}')
  log "$(basename "$d"):  rps=$rps  routes=$routes"
done
log ""
log "Next steps after rsync to laptop:"
log "  Edit the SWEEPS list in"
log "    thesis_plotting/scripts/thesis_rps_knee_normalized_fine_clean.py"
log "  to point at this sweep dir (and choose an output filename that"
log "  preserves the existing figure, e.g. _fcfs)."
log "  Then run the script. The chaos-zone V-dive should be gone."
