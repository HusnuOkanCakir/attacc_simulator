#!/bin/bash
#SBATCH -J dbatch_sweep
#SBATCH -p high_latency
#SBATCH -c 4
#SBATCH --mem=32GB
#SBATCH -D /mnt/galactica/ocakir/attacc_simulator
#SBATCH -o /mnt/galactica/ocakir/logs/dbatch_sweep_%j.out
#SBATCH -e /mnt/galactica/ocakir/logs/dbatch_sweep_%j.err
#
# Experiment H / H2 — Decode batch size × max-consec-decode joint sweep.
#
# `--max-decode-batch-size` caps how many decode requests merge into one
# step. Larger batches amortize the GPU FC cost per token (throughput up)
# but increase tail latency (slowest request in the batch holds the others)
# and KV pressure.
#
# `--max-consecutive-decode-batches` forces a prefill step every N decode
# batches to keep TTFT bounded under sustained decode load. Larger values
# let decode hog the system; smaller values guarantee prefill fairness.
#
# This sbatch iterates the full DECODE_BATCH_SIZES × MAX_CONSEC_DECODE_BATCHES
# matrix in one job (instead of 12 slurm submissions). Use lists like
# DECODE_BATCH_SIZES="4 8 16" and MAX_CONSEC_DECODE_BATCHES="1 2 4 8".
#
# Cache-warmup is identical to run_pim_cache_warmup_sweep.sbatch.
#
# Usage:
#   # Single-axis sweep (default — batch sizes only, fixed max_consec=4):
#   sbatch tools/run_decode_batch_sweep.sbatch
#
#   # Full 2D sweep (Experiment H2 — 3×4 = 12 runs + 1 prime):
#   # IMPORTANT: use single quotes for space-separated lists (not \") to avoid
#   #            slurm splitting the value into separate positional arguments.
#   sbatch --export=ALL,DECODE_BATCH_SIZES='4 8 16',\
#       MAX_CONSEC_DECODE_BATCHES='1 2 4 8',SKIP_PRIMING=1 \
#       tools/run_decode_batch_sweep.sbatch
#
# Env-var overrides:
#   MODEL                       (default openvla)
#   N_REQUESTS                  (default 2000)
#   KV_POOL_GB                  (default 4)
#   MAX_ACTIVE                  (default 32)
#   ARRIVAL_SCALE               (default 1.0)   — saturated load is where
#                                                 batching matters most
#   KV_OOM_POLICY               (default hold_then_fallback)
#   KV_OOM_HOLD_LIMIT_MS        (default 50)
#   SLO_E2E_MS                  (default 500)
#   SLO_TTFT_MS                 (default 200)
#   POLICY                      (default max_util_full)
#   DECODE_BATCH_SIZES          (default "4 8 16")    — outer sweep axis
#   MAX_CONSEC_DECODE_BATCHES   (default "4")         — inner sweep axis;
#                                                       pass a list for 2D sweep
#   SKIP_PRIMING                (default 0)
#
# Monitor:
#   squeue -u $USER
#   tail -f /mnt/galactica/ocakir/logs/dbatch_sweep_<jobid>.out

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
DECODE_BATCH_SIZES="${DECODE_BATCH_SIZES:-4 8 16}"
MAX_CONSEC_DECODE_BATCHES="${MAX_CONSEC_DECODE_BATCHES:-4}"
SKIP_PRIMING="${SKIP_PRIMING:-0}"

PY=$(which python)
RUNNER="${ROOT_DIR}/tools/run_serving_online_policy_compare.py"
CACHE_DIR="${ROOT_DIR}/cluster_outputs/pim_cache"
CACHE_FILE="${CACHE_DIR}/${MODEL}_lpddr5.bin"
TS=$(date +%Y%m%d_%H%M%S)
SWEEP_DIR="${ROOT_DIR}/cluster_outputs/online_serving_runs/dbatch_sweep_${MODEL}_n${N_REQUESTS}_${TS}"
LOG="${SWEEP_DIR}/sweep.log"

mkdir -p "${CACHE_DIR}" "${SWEEP_DIR}"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# Count matrix size for progress display
n_bs=$(echo "$DECODE_BATCH_SIZES" | wc -w)
n_mc=$(echo "$MAX_CONSEC_DECODE_BATCHES" | wc -w)
n_total=$(( n_bs * n_mc ))

log "model=$MODEL  n=$N_REQUESTS  kv_pool=${KV_POOL_GB}GiB  max_active=$MAX_ACTIVE"
log "arrival_scale=$ARRIVAL_SCALE  slo_e2e=${SLO_E2E_MS}ms  slo_ttft=${SLO_TTFT_MS}ms"
log "policy=$POLICY"
log "decode_batch_sizes=($DECODE_BATCH_SIZES)  ×  max_consec=($MAX_CONSEC_DECODE_BATCHES)"
log "matrix size: ${n_bs} × ${n_mc} = ${n_total} runs"
log "cache_file=$CACHE_FILE"
log "sweep_dir=$SWEEP_DIR"

# args: <label> <batch_size> <max_consec>
run_one() {
  local label=$1
  local batch_size=$2
  local max_consec=$3
  local run_dir="${SWEEP_DIR}/${label}"
  log "── start: $label  max_decode_batch_size=$batch_size  max_consec_decode=$max_consec ──"
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
      --max-decode-batch-size "$batch_size" \
      --max-consecutive-decode-batches "$max_consec" \
      --debug-log \
      --pim-cache-file "$CACHE_FILE" \
      --run-dir "$run_dir" 2>&1 | tee -a "$LOG"
  local dt=$(( $(date +%s) - t0 ))
  log "── done:  $label  wall=${dt}s ──"
}

# ── Stage 1: prime ───────────────────────────────────────────────────────────
if [ -s "$CACHE_FILE" ] && [ "$SKIP_PRIMING" = "1" ]; then
  log "[prime] SKIPPED — cache exists at $CACHE_FILE"
  log "[prime] cache file size: $(wc -l < "$CACHE_FILE") lines"
else
  log "[prime] starting cold priming run (bs=8, max_consec=4)"
  run_one "00_prime" 8 4
  log "[prime] cache populated; entries=$(grep -cv '^#' "$CACHE_FILE" || echo 0)"
fi

# ── Stage 2: 2D sweep ────────────────────────────────────────────────────────
i=1
for bs in $DECODE_BATCH_SIZES; do
  for mc in $MAX_CONSEC_DECODE_BATCHES; do
    label=$(printf "%02d_bs%s_mc%s" "$i" "$bs" "$mc")
    run_one "$label" "$bs" "$mc"
    i=$((i+1))
  done
done

# ── Summary ──────────────────────────────────────────────────────────────────
log "=========================================================================="
log "ALL DONE"
log "  cache:     $CACHE_FILE  ($(grep -cv '^#' "$CACHE_FILE" 2>/dev/null || echo 0) entries)"
log "  sweep_dir: $SWEEP_DIR"
log "  log:       $LOG"
log "  matrix:    ${n_bs} batch sizes × ${n_mc} max_consec values = ${n_total} runs"
log "=========================================================================="
log ""
log "Hypotheses to test from the matrix (sub-runs labeled bs<X>_mc<Y>):"
log "  1. Iso-product hypothesis: (bs=4, mc=8) ≈ (bs=8, mc=4) ≈ (bs=16, mc=2) in throughput"
log "     — if the binding metric is total decode time per prefill window."
log "  2. TTFT-throughput Pareto frontier: chart TTFT-p99 vs throughput across (bs, mc) cells."
log "  3. Optimal (bs, mc) given a TTFT SLO: smallest mc that meets SLO at each bs."

# ── Heatmap / Pareto plots ───────────────────────────────────────────────────
PLOT_SCRIPT="${ROOT_DIR}/tools/plot_decode_batch_heatmap.py"
if [ -f "$PLOT_SCRIPT" ] && [ "$n_mc" -gt 1 ]; then
  log "[plot] generating decode-batch heatmap ..."
  "$PY" "$PLOT_SCRIPT" \
      --sweep-dir "$SWEEP_DIR" \
      --policy    "$POLICY" \
      --out-dir   "${SWEEP_DIR}/plots" 2>&1 | tee -a "$LOG"
  log "[plot] plots written to ${SWEEP_DIR}/plots/"
  log "[plot] scp to local:  scp -J ocakir@safari-proxy ocakir@<node>:${SWEEP_DIR}/plots/*.png ."
elif [ "$n_mc" -le 1 ]; then
  log "[plot] skipped — only 1 max_consec value; need ≥2 for a 2-D heatmap"
fi
