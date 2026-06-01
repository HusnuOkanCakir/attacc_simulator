#!/bin/bash
#SBATCH -J pim_batching_sweep
#SBATCH -p high_latency
#SBATCH -c 4
#SBATCH --mem=32GB
#SBATCH -D /mnt/galactica/ocakir/attacc_simulator
#SBATCH -o /mnt/galactica/ocakir/logs/pim_batching_sweep_%j.out
#SBATCH -e /mnt/galactica/ocakir/logs/pim_batching_sweep_%j.err
#
# PIM-attention batching mode sweep — measures throughput / TTFT under each
# pim_command_mode at a few decode batch sizes.
#
# Modes tested:
#   realistic         — current default (per-request commands; linear in B)
#   attacc_qbroadcast — AttAcc-faithful Q/V broadcast amortization
#   bank_stripe       — experimental shared K/V MACs (ideal bank parallelism)
#
# Each (mode, bs) sub-run uses its own PIM shape cache file
# (cluster_outputs/pim_cache/<model>_<mode>.bin) so prior runs in other
# modes don't contaminate the new mode's measurements.
#
# Usage:
#   sbatch tools/run_pim_batching_sweep.sbatch
#
#   # Explicit override (e.g. only AttAcc vs bank_stripe at bs=4):
#   sbatch --export=ALL,MODES='attacc_qbroadcast bank_stripe',\
#       DECODE_BATCH_SIZES='4' tools/run_pim_batching_sweep.sbatch
#
# Env-var overrides:
#   MODEL                       (default openvla)
#   N_REQUESTS                  (default 2000)
#   KV_POOL_GB                  (default 4)
#   MAX_ACTIVE                  (default 32)
#   ARRIVAL_SCALE               (default 1.0) — saturated; where batching matters
#   KV_OOM_POLICY               (default hold_then_fallback)
#   SLO_E2E_MS                  (default 500)
#   SLO_TTFT_MS                 (default 200)
#   POLICY                      (default max_util_full)
#   MODES                       (default "realistic attacc_qbroadcast bank_stripe")
#   DECODE_BATCH_SIZES          (default "1 4 8 16")
#   MAX_CONSEC_DECODE_BATCHES   (default 4)
#   SKIP_PRIMING                (default 0)
#
# Output: cluster_outputs/online_serving_runs/pim_batching_sweep_${MODEL}_n${N}_${TS}/
#         <NN>_<mode>_bs<bs>/<policy>/{policy_compare_summary.txt, requests_out.csv, ...}
#
# Monitor:
#   squeue -u $USER
#   tail -f /mnt/galactica/ocakir/logs/pim_batching_sweep_<jobid>.out

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
echo "conda env: ${CONDA_DEFAULT_ENV:-unknown}"

MODEL="${MODEL:-openvla}"
N_REQUESTS="${N_REQUESTS:-2000}"
KV_POOL_GB="${KV_POOL_GB:-4}"
MAX_ACTIVE="${MAX_ACTIVE:-32}"
ARRIVAL_SCALE="${ARRIVAL_SCALE:-1.0}"
KV_OOM_POLICY="${KV_OOM_POLICY:-hold_then_fallback}"
KV_OOM_HOLD_LIMIT_MS="${KV_OOM_HOLD_LIMIT_MS:-50}"
SLO_E2E_MS="${SLO_E2E_MS:-500}"
SLO_TTFT_MS="${SLO_TTFT_MS:-200}"
POLICY="${POLICY:-max_util_full}"
MODES="${MODES:-realistic attacc_qbroadcast bank_stripe}"
DECODE_BATCH_SIZES="${DECODE_BATCH_SIZES:-1 4 8 16}"
MAX_CONSEC_DECODE_BATCHES="${MAX_CONSEC_DECODE_BATCHES:-4}"
SKIP_PRIMING="${SKIP_PRIMING:-0}"

PY=$(which python)
RUNNER="${ROOT_DIR}/tools/run_serving_online_policy_compare.py"
CACHE_DIR="${ROOT_DIR}/cluster_outputs/pim_cache"
TS=$(date +%Y%m%d_%H%M%S)
SWEEP_DIR="${ROOT_DIR}/cluster_outputs/online_serving_runs/pim_batching_sweep_${MODEL}_n${N_REQUESTS}_${TS}"
LOG="${SWEEP_DIR}/sweep.log"

mkdir -p "${CACHE_DIR}" "${SWEEP_DIR}"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

n_modes=$(echo "$MODES" | wc -w)
n_bs=$(echo "$DECODE_BATCH_SIZES" | wc -w)
n_total=$(( n_modes * n_bs ))

log "model=$MODEL  n=$N_REQUESTS  kv_pool=${KV_POOL_GB}GiB  max_active=$MAX_ACTIVE"
log "arrival_scale=$ARRIVAL_SCALE  slo_e2e=${SLO_E2E_MS}ms  slo_ttft=${SLO_TTFT_MS}ms"
log "policy=$POLICY  max_consec=$MAX_CONSEC_DECODE_BATCHES"
log "modes=($MODES)  decode_batch_sizes=($DECODE_BATCH_SIZES)"
log "matrix: ${n_modes} modes × ${n_bs} batch sizes = ${n_total} runs"
log "sweep_dir=$SWEEP_DIR"

# args: <label> <mode> <bs> <cache_file>
run_one() {
  local label=$1
  local mode=$2
  local bs=$3
  local cache_file=$4
  local run_dir="${SWEEP_DIR}/${label}"
  log "── start: $label  mode=$mode  bs=$bs ──"
  log "    cache:       $cache_file"
  log "    watch live:  watch -n 2 cat $run_dir/$POLICY/status.txt"
  local t0=$(date +%s)
  "$PY" "$RUNNER" \
      --model "$MODEL" \
      --n-requests "$N_REQUESTS" \
      --policies "$POLICY" \
      --kv-pool-gb "$KV_POOL_GB" --ml \
      --max-active "$MAX_ACTIVE" \
      --arrival-scale "$ARRIVAL_SCALE" \
      --slo-e2e-ms "$SLO_E2E_MS" --slo-ttft-ms "$SLO_TTFT_MS" \
      --kv-oom-policy "$KV_OOM_POLICY" \
      --kv-oom-hold-limit-ms "$KV_OOM_HOLD_LIMIT_MS" \
      --max-decode-batch-size "$bs" \
      --max-consecutive-decode-batches "$MAX_CONSEC_DECODE_BATCHES" \
      --pim-command-mode "$mode" \
      --pim-cache-file "$cache_file" \
      --debug-log \
      --run-dir "$run_dir" 2>&1 | tee -a "$LOG"
  local dt=$(( $(date +%s) - t0 ))
  log "── done:  $label  wall=${dt}s ──"
}

# ── Stage 1: prime per-mode caches (cold first run for each mode at bs=1) ───
i=1
for mode in $MODES; do
  cache_file="${CACHE_DIR}/${MODEL}_${mode}.bin"
  if [ -s "$cache_file" ] && [ "$SKIP_PRIMING" = "1" ]; then
    log "[prime] SKIPPED — cache exists at $cache_file ($(grep -cv '^#' "$cache_file" 2>/dev/null || echo 0) entries)"
  else
    log "[prime] starting cold prime for mode=$mode at bs=8"
    label=$(printf "00_prime_%s" "$mode")
    run_one "$label" "$mode" 8 "$cache_file"
    log "[prime] cache populated for $mode; entries=$(grep -cv '^#' "$cache_file" 2>/dev/null || echo 0)"
  fi
done

# ── Stage 2: cross-product sweep ────────────────────────────────────────────
log "== Stage 2: ${n_total}-cell sweep =="
i=1
for mode in $MODES; do
  cache_file="${CACHE_DIR}/${MODEL}_${mode}.bin"
  for bs in $DECODE_BATCH_SIZES; do
    label=$(printf "%02d_%s_bs%s" "$i" "$mode" "$bs")
    run_one "$label" "$mode" "$bs" "$cache_file"
    i=$((i+1))
  done
done

# ── Summary ──────────────────────────────────────────────────────────────────
log "=========================================================================="
log "ALL DONE"
log "  sweep_dir: $SWEEP_DIR"
log "  log:       $LOG"
log "  matrix:    ${n_modes} modes × ${n_bs} batch sizes = ${n_total} runs"
log "  caches:"
for mode in $MODES; do
  cf="${CACHE_DIR}/${MODEL}_${mode}.bin"
  entries=$(grep -cv '^#' "$cf" 2>/dev/null || echo 0)
  log "    $cf  ($entries entries)"
done
log "=========================================================================="

# ── Optional plot: sweep_compare per cell + a cross-mode heatmap ────────────
PLOT_COMPARE="${ROOT_DIR}/tools/plot_sweep_compare.py"
if [ -f "$PLOT_COMPARE" ]; then
  log "[plot] generating cross-cell comparison ..."
  "$PY" "$PLOT_COMPARE" --sweep-dir "$SWEEP_DIR" 2>&1 | tee -a "$LOG"
fi

log ""
log "Hypotheses to test from this sweep:"
log "  1. realistic throughput drops as 1/B (baseline; bs=4 > bs=8 > bs=16)."
log "  2. attacc_qbroadcast also drops with B but slower (smaller coefficient)."
log "  3. bank_stripe throughput is approximately FLAT in B (commands shared)."
log "  4. TTFT p99 at bs=16: bank_stripe << attacc_qbroadcast << realistic."
log "  5. At bs=1, all three modes should give very similar throughput."
