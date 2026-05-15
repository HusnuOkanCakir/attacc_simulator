#!/bin/bash
#SBATCH -J pprio_sweep
#SBATCH -p high_latency
#SBATCH -c 4
#SBATCH --mem=32GB
#SBATCH -D /mnt/galactica/ocakir/attacc_simulator
#SBATCH -o /mnt/galactica/ocakir/logs/pprio_sweep_%j.out
#SBATCH -e /mnt/galactica/ocakir/logs/pprio_sweep_%j.err
#
# Experiment H3 — prompt_priority sweep.
#
# H2 showed `max_consecutive_decode_batches` has zero effect under the YAML
# default `prompt_priority: true` — because pick_task() short-circuits the
# mc check whenever a prefill is waiting. To make mc meaningful we have to
# disable prompt_priority. This sweep tests:
#
#   (prompt_priority=true , mc=4)  — baseline (matches every prior run)
#   (prompt_priority=false, mc=1)  — strictest decode-yield-to-prefill
#   (prompt_priority=false, mc=4)  — moderate
#   (prompt_priority=false, mc=8)  — loosest (decode hogs the system longer)
#
# Same MODEL / N_REQUESTS / KV_POOL_GB / ARRIVAL_SCALE / SLO as run_decode_batch_sweep.
#
# Usage:
#   sbatch tools/run_prompt_priority_sweep.sbatch
#
#   # Override mc values:
#   sbatch --export=ALL,MC_VALUES='1 2 4 8',SKIP_PRIMING=1 \
#       tools/run_prompt_priority_sweep.sbatch
#
# Env-var overrides:
#   MODEL              (default openvla)
#   N_REQUESTS         (default 2000)
#   KV_POOL_GB         (default 4)
#   MAX_ACTIVE         (default 32)
#   ARRIVAL_SCALE      (default 1.0)
#   KV_OOM_POLICY      (default hold_then_fallback)
#   SLO_E2E_MS         (default 500)
#   SLO_TTFT_MS        (default 200)
#   POLICY             (default max_util_full)
#   BATCH_SIZE         (default 8)
#   MC_VALUES          (default "1 4 8")  — mc values to sweep when prompt_priority=false
#   SKIP_PRIMING       (default 0)
#
# Monitor:
#   squeue -u $USER
#   tail -f /mnt/galactica/ocakir/logs/pprio_sweep_<jobid>.out

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
BATCH_SIZE="${BATCH_SIZE:-8}"
MC_VALUES="${MC_VALUES:-1 4 8}"
SKIP_PRIMING="${SKIP_PRIMING:-0}"

PY=$(which python)
RUNNER="${ROOT_DIR}/tools/run_serving_online_policy_compare.py"
CACHE_DIR="${ROOT_DIR}/cluster_outputs/pim_cache"
CACHE_FILE="${CACHE_DIR}/${MODEL}_lpddr5.bin"
TS=$(date +%Y%m%d_%H%M%S)
SWEEP_DIR="${ROOT_DIR}/cluster_outputs/online_serving_runs/pprio_sweep_${MODEL}_n${N_REQUESTS}_${TS}"
LOG="${SWEEP_DIR}/sweep.log"

mkdir -p "${CACHE_DIR}" "${SWEEP_DIR}"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

n_mc=$(echo "$MC_VALUES" | wc -w)
n_total=$(( 1 + n_mc ))   # 1 baseline + N mc cells with prompt_priority=false

log "model=$MODEL  n=$N_REQUESTS  kv_pool=${KV_POOL_GB}GiB  max_active=$MAX_ACTIVE"
log "arrival_scale=$ARRIVAL_SCALE  slo_e2e=${SLO_E2E_MS}ms  slo_ttft=${SLO_TTFT_MS}ms"
log "policy=$POLICY  bs=$BATCH_SIZE"
log "mc_values=($MC_VALUES)  n_total=$n_total runs (1 baseline + $n_mc disabled)"
log "cache_file=$CACHE_FILE"
log "sweep_dir=$SWEEP_DIR"

# args: <label> <mc> <prompt_priority_flag>
run_one() {
  local label=$1
  local mc=$2
  local pp_flag=$3
  local run_dir="${SWEEP_DIR}/${label}"
  log "── start: $label  bs=$BATCH_SIZE  mc=$mc  prompt_priority_flag=$pp_flag ──"
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
      --max-decode-batch-size "$BATCH_SIZE" \
      --max-consecutive-decode-batches "$mc" \
      $pp_flag \
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
  log "[prime] starting cold priming run (bs=$BATCH_SIZE, mc=4, prompt_priority=true)"
  run_one "00_prime" 4 "--prompt-priority"
  log "[prime] cache populated; entries=$(grep -cv '^#' "$CACHE_FILE" || echo 0)"
fi

# ── Stage 2: baseline (prompt_priority=true) — matches every prior run ──────
log "== Stage 2: baseline with prompt_priority=true =="
run_one "01_pprio_true_mc4" 4 "--prompt-priority"

# ── Stage 3: prompt_priority=false sweep over mc values ─────────────────────
log "== Stage 3: prompt_priority=false sweep =="
i=2
for mc in $MC_VALUES; do
  label=$(printf "%02d_pprio_false_mc%s" "$i" "$mc")
  run_one "$label" "$mc" "--no-prompt-priority"
  i=$((i+1))
done

# ── Summary ──────────────────────────────────────────────────────────────────
log "=========================================================================="
log "ALL DONE"
log "  cache:     $CACHE_FILE  ($(grep -cv '^#' "$CACHE_FILE" 2>/dev/null || echo 0) entries)"
log "  sweep_dir: $SWEEP_DIR"
log "  log:       $LOG"
log "  total:     $n_total runs"
log "=========================================================================="
log ""
log "Compare:"
log "  01_pprio_true_mc4   — baseline (prefill always wins)"
log "  02_pprio_false_mc1  — strictest fairness (1 decode → forced prefill)"
log "  03_pprio_false_mc4  — moderate"
log "  04_pprio_false_mc8  — loosest (8 decodes before forced prefill)"
log ""
log "Hypothesis: with prompt_priority=false, TTFT will rise monotonically with"
log "mc (more decode hogging = higher first-token latency), while throughput"
log "will rise then plateau (more batching amortization until KV pressure caps it)."
log "The (prompt_priority=true, mc=4) baseline should match prior dbatch_sweep results."
