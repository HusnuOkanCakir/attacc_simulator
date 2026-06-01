#!/bin/bash
#SBATCH -J e2_pi0_tail
#SBATCH -p high_latency
#SBATCH -c 4
#SBATCH --mem=32GB
#SBATCH -D /mnt/galactica/ocakir/attacc_simulator
#SBATCH -o /mnt/galactica/ocakir/logs/e2_pi0_tail_%j.out
#SBATCH -e /mnt/galactica/ocakir/logs/e2_pi0_tail_%j.err
#
# E2: Pi0 synthetic tail-latency mitigation
#
# D2 Pi0 synthetic VLA showed TTFT p99 = 95 ms but E2E p99 = 4746 ms (50×
# tail gap). Max_active=32 was NOT saturating (avg concurrent ≈ 28-29).
# This sweep tests which knob recovers the tail.
#
# 6 cells, Pi0 + synthetic VLA only:
#   1: baseline (bs=8, kv_oom=hold_then_fallback, kv_pool=8GiB) — reproduces D2.
#   2: smaller bs (bs=4) — does shrinking the batch help?
#   3: even smaller bs (bs=2)
#   4: larger bs (bs=16) — confirms direction
#   5: GPU bias under load (kv_oom=fallback_gpu)
#   6: smaller PIM pool (kv_pool=4GiB) — alternative GPU-bias mechanism
#
# Output: cluster_outputs/online_serving_runs/e2_pi0_tail_${TS}/
#           01_baseline_bs8/<policy>/...
#           02_bs4/<policy>/...
#           ...
#           06_kvpool4/<policy>/...
#
# Usage:
#   sbatch tools/run_e2_pi0_tail_mitigation.sbatch
#
# Env-var overrides:
#   N_REQUESTS         (default 2000)
#   ARRIVAL_SCALE      (default 1.0)
#   MAX_ACTIVE         (default 32)
#   POLICY             (default max_util_full)
#   MAX_CONSEC_DEC     (default 4)
#   SLO_E2E_MS         (default 500)
#   SLO_TTFT_MS        (default 200)
#   GPU_TYPE           (default a6000)

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

N_REQUESTS="${N_REQUESTS:-2000}"
ARRIVAL_SCALE="${ARRIVAL_SCALE:-1.0}"
MAX_ACTIVE="${MAX_ACTIVE:-32}"
POLICY="${POLICY:-max_util_full}"
MAX_CONSEC_DEC="${MAX_CONSEC_DEC:-4}"
SLO_E2E_MS="${SLO_E2E_MS:-500}"
SLO_TTFT_MS="${SLO_TTFT_MS:-200}"
KV_OOM_HOLD_LIMIT_MS="${KV_OOM_HOLD_LIMIT_MS:-50}"
GPU_TYPE="${GPU_TYPE:-a6000}"
MODEL="pi0"

PY=$(which python)
RUNNER="${ROOT_DIR}/tools/run_serving_online_policy_compare.py"
TRACE_GEN="${ROOT_DIR}/tools/gen_vla_synthetic_trace.py"
CACHE_DIR="${ROOT_DIR}/cluster_outputs/pim_cache"
SYNTH_TRACE_DIR="${ROOT_DIR}/cluster_outputs/synthetic_vla"
TS=$(date +%Y%m%d_%H%M%S)
SWEEP_DIR="${ROOT_DIR}/cluster_outputs/online_serving_runs/e2_pi0_tail_${TS}"
LOG="${SWEEP_DIR}/sweep.log"

mkdir -p "${CACHE_DIR}" "${SWEEP_DIR}" "${SYNTH_TRACE_DIR}"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "E2: Pi0 synthetic tail-latency mitigation"
log "n=$N_REQUESTS  scale=$ARRIVAL_SCALE  max_active=$MAX_ACTIVE  gpu=$GPU_TYPE"
log "policy=$POLICY  mc=$MAX_CONSEC_DEC  slo_e2e=${SLO_E2E_MS}ms  slo_ttft=${SLO_TTFT_MS}ms"
log "sweep_dir=$SWEEP_DIR"
log ""

# Pre-generate Pi0 synthetic trace.
TRACE_CSV="${SYNTH_TRACE_DIR}/synthetic_vla_${MODEL}_n${N_REQUESTS}_seed0.csv"
if [ ! -s "$TRACE_CSV" ]; then
  log "[trace] generating $TRACE_CSV"
  "$PY" "$TRACE_GEN" --model "$MODEL" --n-requests "$N_REQUESTS" \
      --rate-hz 6.0 --jitter 1.0 --seed 0 --out-csv "$TRACE_CSV" 2>&1 | tee -a "$LOG"
else
  log "[trace] reusing $TRACE_CSV"
fi

# args: <label> <bs> <kv_pool_gb> <kv_oom_policy>
run_one() {
  local label=$1
  local bs=$2
  local pool=$3
  local oom=$4
  local run_dir="${SWEEP_DIR}/${label}"
  local cache_file="${CACHE_DIR}/${MODEL}_${GPU_TYPE}_e2_${label}.bin"

  log "── start: $label  bs=$bs  pool=${pool}GiB  oom=$oom ──"
  log "    cache: $cache_file"

  local t0=$(date +%s)
  "$PY" "$RUNNER" \
      --model "$MODEL" \
      --gpu "$GPU_TYPE" \
      --n-requests "$N_REQUESTS" \
      --policies "$POLICY" \
      --kv-pool-gb "$pool" --ml \
      --max-active "$MAX_ACTIVE" \
      --arrival-scale "$ARRIVAL_SCALE" \
      --slo-e2e-ms "$SLO_E2E_MS" --slo-ttft-ms "$SLO_TTFT_MS" \
      --kv-oom-policy "$oom" \
      --kv-oom-hold-limit-ms "$KV_OOM_HOLD_LIMIT_MS" \
      --max-decode-batch-size "$bs" \
      --max-consecutive-decode-batches "$MAX_CONSEC_DEC" \
      --no-prompt-priority \
      --csv-path "$TRACE_CSV" \
      --pim-cache-file "$cache_file" \
      --debug-log \
      --run-dir "$run_dir" 2>&1 | tee -a "$LOG"
  local dt=$(( $(date +%s) - t0 ))
  log "── done:  $label  wall=${dt}s ──"
}

# args: label, bs, kv_pool_gb, kv_oom_policy
run_one "01_baseline_bs8"  "8"  "8"  "hold_then_fallback"
run_one "02_bs4"           "4"  "8"  "hold_then_fallback"
run_one "03_bs2"           "2"  "8"  "hold_then_fallback"
run_one "04_bs16"          "16" "8"  "hold_then_fallback"
run_one "05_oom_fallback"  "8"  "8"  "fallback_gpu"
run_one "06_kvpool4"       "8"  "4"  "hold_then_fallback"

log "=========================================================================="
log "E2 ALL DONE  sweep_dir: $SWEEP_DIR"
log "=========================================================================="

PLOT_COMPARE="${ROOT_DIR}/tools/plot_sweep_compare.py"
if [ -f "$PLOT_COMPARE" ]; then
  log "[plot] generating cross-cell comparison ..."
  "$PY" "$PLOT_COMPARE" --sweep-dir "$SWEEP_DIR" 2>&1 | tee -a "$LOG"
fi

log ""
log "Goal: find a cell with E2E p99 < 4746 ms (D2 baseline) and reasonable"
log "throughput/TTFT. If all cells produce similar E2E p99, the tail is"
log "fundamental to Pi0 + synthetic VLA at this load."
