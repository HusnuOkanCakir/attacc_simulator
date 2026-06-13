#!/bin/bash
#SBATCH -J knee_a5k
#SBATCH -p high_latency
#SBATCH -c 4
#SBATCH --mem=64GB
#SBATCH --exclude=safari-nexus1
#SBATCH -D /mnt/galactica/ocakir/attacc_simulator
#SBATCH -o /mnt/galactica/ocakir/logs/knee_a5k_%j.out
#SBATCH -e /mnt/galactica/ocakir/logs/knee_a5k_%j.err
#
# Re-run of the fine-cliff knee sweep with max_active=5000 (effectively
# unbounded for this trace) — the direct follow-up to the
# max_active_chaos_probe result, which showed that increasing
# max_active from 32 → 100+ rescues s=2.70 from the collapsed 0.82 RPS
# attractor back to 2.01 RPS.
#
# The hypothesis being tested here is broader: does the entire
# non-monotonic switching region s ∈ [2.65, 2.85] disappear once the
# active-set FIFO bottleneck is removed? If yes, the chaos-zone
# bistability is fully accounted for by the (max_active, queue depth,
# min_finish) interaction in the chapter. If some scales still
# collapse despite max_active=5000, there is a second mechanism.
#
# Cells (same 24-scale list as the existing fine-cliff sweep behind
# fig_rps_knee_pi0_wide_normalized_fine_n5k_clean.png):
#   reference  : 10 7 5
#   safe / approach : 4 3.5 3 2.85 2.75
#   chaos zone : 2.70 2.69 2.68 2.67 2.66 2.65
#   boundary   : 2.60 2.55 2.54 2.53 2.52 2.51 2.50
#   saturation : 2.4 2.0 1.5
#
# Memory note: at max_active=5000 the worst-case active set's KV
# footprint can exceed 32 GiB at the stable-saturation scales (s ≤
# 2.55, where offered ≫ 2.13 ceiling, queue grows without bound). The
# pool is set to 64 GiB to keep the chaos-zone cells clean; the
# stable-saturation cells may still trigger kv_oom_holds and
# admission-time GPU fallback. That is acceptable for the central
# question of "does the chaos zone disappear?", which only depends on
# the cells in s ∈ [2.50, 3.0].

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

# ── Config (mirrors run_rps_knee_pi0_wide_fine_cliff_n5k.sbatch except
#    for max_active and kv_pool) ────────────────────────────────────
N_REQUESTS=5000
MODEL=pi0
GPU_TYPE=a6000
POLICY=max_util_full
KV_POOL_GB=64                       # was 8; bumped for max_active=5000
MAX_ACTIVE=5000                     # was 32; the variable under test
MAX_DECODE_BS=4
MAX_CONSEC_DEC=4
SLO_E2E_MS=500
SLO_TTFT_MS=200
KV_OOM_POLICY=hold_then_fallback
KV_OOM_HOLD_LIMIT_MS=50

# Same 24-scale list as the existing fine-cliff sweep.
ARRIVAL_SCALES="10 7 5 4 3.5 3 2.85 2.75 2.70 2.69 2.68 2.67 2.66 2.65 2.60 2.55 2.54 2.53 2.52 2.51 2.50 2.4 2 1.5"

CSV_PATH="${ROOT_DIR}/cluster_outputs/synthetic_v2/azure_poisson_wide_n5000_seed0.csv"
CACHE_DIR="${ROOT_DIR}/cluster_outputs/pim_cache"
TS=$(date +%Y%m%d_%H%M%S)
SWEEP_DIR="${ROOT_DIR}/cluster_outputs/online_serving_runs/rps_knee_pi0_wide_fine_n5k_a5000_${TS}"
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
log "Fine-cliff knee sweep at max_active=5000  (n=5000, azure_poisson_wide, Pi0)"
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
  cache_file="${CACHE_DIR}/${MODEL}_${GPU_TYPE}_knee_pi0_wide_fine_n5k_a5000_${scale_label}.bin"

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
log "Side-by-side throughput summary (baseline max_active=32 vs this max_active=5000 run):"
for d in "$SWEEP_DIR"/*; do
  [ -d "$d" ] || continue
  s="${d}/policy_compare_summary.txt"
  [ -f "$s" ] || continue
  log "$(basename "$d"):  $(grep -E 'Throughput|Routes|KV-OOM holds' "$s" | tr '\n' ' | ')"
done
log ""
log "Next steps after rsync:"
log "  1. Compare against existing rps_knee_pi0_wide_fine_n5k_20260602_183013."
log "  2. Edit the SWEEPS list in"
log "     thesis_plotting/scripts/thesis_rps_knee_normalized_fine_clean.py"
log "     to point at this sweep dir."
log "  3. Run the script. The chaos-zone (red overflow markers) should be"
log "     gone or dramatically reduced if max_active=32 was the cause."
