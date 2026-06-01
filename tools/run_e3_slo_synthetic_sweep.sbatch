#!/bin/bash
#SBATCH -J e3_slo_sweep
#SBATCH -p high_latency
#SBATCH -c 4
#SBATCH --mem=32GB
#SBATCH -D /mnt/galactica/ocakir/attacc_simulator
#SBATCH -o /mnt/galactica/ocakir/logs/e3_slo_sweep_%j.out
#SBATCH -e /mnt/galactica/ocakir/logs/e3_slo_sweep_%j.err
#
# E3: SLO sweep under synthetic VLA (G2 — "the SLO knob that actually works")
#
# G (2026-05-14) found that SLO 100→2000 ms produced bit-identical results
# under the Azure trace because predicted_finish at admission was ~20 ms
# (Lout~10) while observed p99 was ~500 ms — the eligibility filter at
# scheduler.cpp:309 never tripped. Under synthetic VLA (Pi0 Lout=50,
# OpenVLA Lout=7), predicted_finish is honest and close to observed.
# So the SLO knob should finally bind here.
#
# 2 models × 6 SLO values × synthetic VLA = 12 cells.
#
# Output: cluster_outputs/online_serving_runs/e3_slo_sweep_${TS}/
#           01_pi0_slo0100/<policy>/...
#           02_pi0_slo0200/...
#           ...
#           12_openvla_slo3200/...
#
# Usage:
#   sbatch tools/run_e3_slo_synthetic_sweep.sbatch
#
# Env-var overrides:
#   N_REQUESTS         (default 2000)
#   ARRIVAL_SCALE      (default 1.0)
#   KV_POOL_GB         (default 8)
#   MAX_ACTIVE         (default 32)
#   POLICY             (default max_util_full)
#   MAX_DECODE_BS      (default 8)
#   MAX_CONSEC_DEC     (default 4)
#   KV_OOM_POLICY      (default hold_then_fallback)
#   MODELS             (default "pi0 openvla")
#   SLO_VALUES         (default "100 200 400 800 1600 3200")
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
KV_POOL_GB="${KV_POOL_GB:-8}"
MAX_ACTIVE="${MAX_ACTIVE:-32}"
POLICY="${POLICY:-max_util_full}"
MAX_DECODE_BS="${MAX_DECODE_BS:-8}"
MAX_CONSEC_DEC="${MAX_CONSEC_DEC:-4}"
KV_OOM_POLICY="${KV_OOM_POLICY:-hold_then_fallback}"
KV_OOM_HOLD_LIMIT_MS="${KV_OOM_HOLD_LIMIT_MS:-50}"
MODELS="${MODELS:-pi0 openvla}"
SLO_VALUES="${SLO_VALUES:-100 200 400 800 1600 3200}"
GPU_TYPE="${GPU_TYPE:-a6000}"
# slo_ttft is intentionally disabled (-1) to isolate the E2E filter.
SLO_TTFT_MS="-1"

PY=$(which python)
RUNNER="${ROOT_DIR}/tools/run_serving_online_policy_compare.py"
TRACE_GEN="${ROOT_DIR}/tools/gen_vla_synthetic_trace.py"
CACHE_DIR="${ROOT_DIR}/cluster_outputs/pim_cache"
SYNTH_TRACE_DIR="${ROOT_DIR}/cluster_outputs/synthetic_vla"
TS=$(date +%Y%m%d_%H%M%S)
SWEEP_DIR="${ROOT_DIR}/cluster_outputs/online_serving_runs/e3_slo_sweep_${TS}"
LOG="${SWEEP_DIR}/sweep.log"

mkdir -p "${CACHE_DIR}" "${SWEEP_DIR}" "${SYNTH_TRACE_DIR}"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

n_models=$(echo "$MODELS" | wc -w)
n_slos=$(echo "$SLO_VALUES" | wc -w)
n_total=$(( n_models * n_slos ))

log "E3: SLO sweep under synthetic VLA"
log "models=($MODELS)  slo_values=($SLO_VALUES) ms  gpu=$GPU_TYPE"
log "n=$N_REQUESTS  scale=$ARRIVAL_SCALE  kv_pool=${KV_POOL_GB}GiB  max_active=$MAX_ACTIVE"
log "policy=$POLICY  bs=$MAX_DECODE_BS  mc=$MAX_CONSEC_DEC"
log "slo_ttft=${SLO_TTFT_MS} (DISABLED to isolate E2E filter)"
log "matrix: ${n_models} models × ${n_slos} SLO values = ${n_total} cells"
log "sweep_dir=$SWEEP_DIR"
log ""

# Pre-generate synthetic VLA traces if missing.
for model in $MODELS; do
  out_csv="${SYNTH_TRACE_DIR}/synthetic_vla_${model}_n${N_REQUESTS}_seed0.csv"
  if [ ! -s "$out_csv" ]; then
    log "[trace] generating synthetic VLA trace for $model n=$N_REQUESTS"
    "$PY" "$TRACE_GEN" --model "$model" --n-requests "$N_REQUESTS" \
        --rate-hz 6.0 --jitter 1.0 --seed 0 --out-csv "$out_csv" 2>&1 | tee -a "$LOG"
  else
    log "[trace] reusing $out_csv"
  fi
done

# args: <label> <model> <slo_e2e_ms>
run_one() {
  local label=$1
  local model=$2
  local slo=$3
  local run_dir="${SWEEP_DIR}/${label}"
  # Cache is per-model only — same shape across SLO values, so cache hits
  # are valid across cells. The SLO only affects routing decisions, not
  # the per-shape drain_clk values.
  local cache_file="${CACHE_DIR}/${model}_${GPU_TYPE}_e3.bin"
  local csv_path="${SYNTH_TRACE_DIR}/synthetic_vla_${model}_n${N_REQUESTS}_seed0.csv"

  log "── start: $label  model=$model  slo_e2e=${slo}ms ──"
  log "    cache: $cache_file (shared per-model)"
  log "    csv:   $csv_path"

  local t0=$(date +%s)
  "$PY" "$RUNNER" \
      --model "$model" \
      --gpu "$GPU_TYPE" \
      --n-requests "$N_REQUESTS" \
      --policies "$POLICY" \
      --kv-pool-gb "$KV_POOL_GB" --ml \
      --max-active "$MAX_ACTIVE" \
      --arrival-scale "$ARRIVAL_SCALE" \
      --slo-e2e-ms "$slo" --slo-ttft-ms "$SLO_TTFT_MS" \
      --kv-oom-policy "$KV_OOM_POLICY" \
      --kv-oom-hold-limit-ms "$KV_OOM_HOLD_LIMIT_MS" \
      --max-decode-batch-size "$MAX_DECODE_BS" \
      --max-consecutive-decode-batches "$MAX_CONSEC_DEC" \
      --no-prompt-priority \
      --csv-path "$csv_path" \
      --pim-cache-file "$cache_file" \
      --debug-log \
      --run-dir "$run_dir" 2>&1 | tee -a "$LOG"
  local dt=$(( $(date +%s) - t0 ))
  log "── done:  $label  wall=${dt}s ──"
}

# ── Cross-product sweep ─────────────────────────────────────────────────────
i=1
for model in $MODELS; do
  for slo in $SLO_VALUES; do
    label=$(printf "%02d_%s_slo%04d" "$i" "$model" "$slo")
    run_one "$label" "$model" "$slo"
    i=$((i+1))
  done
done

log "=========================================================================="
log "E3 ALL DONE  sweep_dir: $SWEEP_DIR"
log "=========================================================================="

PLOT_COMPARE="${ROOT_DIR}/tools/plot_sweep_compare.py"
if [ -f "$PLOT_COMPARE" ]; then
  log "[plot] generating cross-cell comparison ..."
  "$PY" "$PLOT_COMPARE" --sweep-dir "$SWEEP_DIR" 2>&1 | tee -a "$LOG"
fi

PLOT_SLO="${ROOT_DIR}/tools/plot_slo_route_mix.py"
if [ -f "$PLOT_SLO" ]; then
  log "[plot] generating E3 SLO route-mix curve ..."
  "$PY" "$PLOT_SLO" --sweep-dir "$SWEEP_DIR" 2>&1 | tee -a "$LOG"
fi

log ""
log "Expected: at tight SLO (≤200ms for OpenVLA, ≤400ms for Pi0), PIM share"
log "drops sharply because PIM predicted_finish exceeds the deadline → only"
log "GPU is eligible → 100% GPU. At loose SLO (≥1600ms), both eligible →"
log "min_finish picks (Pi0→PIM, OpenVLA→GPU). The cliff is the figure."
