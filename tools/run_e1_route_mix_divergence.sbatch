#!/bin/bash
#SBATCH -J e1_route_mix
#SBATCH -p high_latency
#SBATCH -c 4
#SBATCH --mem=32GB
#SBATCH -D /mnt/galactica/ocakir/attacc_simulator
#SBATCH -o /mnt/galactica/ocakir/logs/e1_route_mix_%j.out
#SBATCH -e /mnt/galactica/ocakir/logs/e1_route_mix_%j.err
#
# E1: Route-mix divergence study (architecture-driven routing)
#
# D2 showed Pi0 routes 91% to PIM while OpenVLA routes 85% to GPU under
# synthetic VLA at scale=1.0. This sweep tests whether the divergence holds
# across load points (arrival_scale ∈ {0.5, 1.0, 2.0, 5.0}).
#
# 2 models × 4 arrival_scales × synthetic VLA = 8 cells.
#
# Output: cluster_outputs/online_serving_runs/e1_route_mix_${TS}/
#           01_pi0_s05/<policy>/...
#           02_pi0_s10/<policy>/...
#           ...
#           08_openvla_s50/<policy>/...
#
# Usage:
#   sbatch tools/run_e1_route_mix_divergence.sbatch
#
# Env-var overrides:
#   N_REQUESTS         (default 2000)
#   KV_POOL_GB         (default 8)
#   MAX_ACTIVE         (default 32)
#   POLICY             (default max_util_full)
#   MAX_DECODE_BS      (default 8)
#   MAX_CONSEC_DEC     (default 4)
#   SLO_E2E_MS         (default 500)
#   SLO_TTFT_MS        (default 200)
#   KV_OOM_POLICY      (default hold_then_fallback)
#   MODELS             (default "pi0 openvla")
#   ARRIVAL_SCALES     (default "0.5 1.0 2.0 5.0")
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
echo "conda env: ${CONDA_DEFAULT_ENV:-unknown}"

N_REQUESTS="${N_REQUESTS:-2000}"
KV_POOL_GB="${KV_POOL_GB:-8}"
MAX_ACTIVE="${MAX_ACTIVE:-32}"
POLICY="${POLICY:-max_util_full}"
MAX_DECODE_BS="${MAX_DECODE_BS:-8}"
MAX_CONSEC_DEC="${MAX_CONSEC_DEC:-4}"
SLO_E2E_MS="${SLO_E2E_MS:-500}"
SLO_TTFT_MS="${SLO_TTFT_MS:-200}"
KV_OOM_POLICY="${KV_OOM_POLICY:-hold_then_fallback}"
KV_OOM_HOLD_LIMIT_MS="${KV_OOM_HOLD_LIMIT_MS:-50}"
MODELS="${MODELS:-pi0 openvla}"
ARRIVAL_SCALES="${ARRIVAL_SCALES:-0.5 1.0 2.0 5.0}"
GPU_TYPE="${GPU_TYPE:-a6000}"

PY=$(which python)
RUNNER="${ROOT_DIR}/tools/run_serving_online_policy_compare.py"
TRACE_GEN="${ROOT_DIR}/tools/gen_vla_synthetic_trace.py"
CACHE_DIR="${ROOT_DIR}/cluster_outputs/pim_cache"
SYNTH_TRACE_DIR="${ROOT_DIR}/cluster_outputs/synthetic_vla"
TS=$(date +%Y%m%d_%H%M%S)
SWEEP_DIR="${ROOT_DIR}/cluster_outputs/online_serving_runs/e1_route_mix_${TS}"
LOG="${SWEEP_DIR}/sweep.log"

mkdir -p "${CACHE_DIR}" "${SWEEP_DIR}" "${SYNTH_TRACE_DIR}"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

n_models=$(echo "$MODELS" | wc -w)
n_scales=$(echo "$ARRIVAL_SCALES" | wc -w)
n_total=$(( n_models * n_scales ))

log "E1: route-mix divergence study"
log "models=($MODELS)  arrival_scales=($ARRIVAL_SCALES)  gpu=$GPU_TYPE"
log "n=$N_REQUESTS  kv_pool=${KV_POOL_GB}GiB  max_active=$MAX_ACTIVE"
log "policy=$POLICY  bs=$MAX_DECODE_BS  mc=$MAX_CONSEC_DEC"
log "slo_e2e=${SLO_E2E_MS}ms  slo_ttft=${SLO_TTFT_MS}ms (loose; off the binding regime)"
log "matrix: ${n_models} models × ${n_scales} scales = ${n_total} cells"
log "sweep_dir=$SWEEP_DIR"
log ""

# Pre-generate synthetic VLA traces if missing (one per model).
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

# Pretty label helper: scale_str(0.5) -> s05; scale_str(5.0) -> s50.
scale_label() {
  local s=$1
  echo "s$(awk -v x=$s 'BEGIN{printf "%02d", x*10}')"
}

# args: <label> <model> <arrival_scale>
run_one() {
  local label=$1
  local model=$2
  local scale=$3
  local run_dir="${SWEEP_DIR}/${label}"
  local cache_file="${CACHE_DIR}/${model}_${GPU_TYPE}_e1_$(scale_label $scale).bin"
  local csv_path="${SYNTH_TRACE_DIR}/synthetic_vla_${model}_n${N_REQUESTS}_seed0.csv"

  log "── start: $label  model=$model  scale=$scale ──"
  log "    cache: $cache_file"
  log "    csv:   $csv_path"

  local t0=$(date +%s)
  "$PY" "$RUNNER" \
      --model "$model" \
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
  for scale in $ARRIVAL_SCALES; do
    label=$(printf "%02d_%s_%s" "$i" "$model" "$(scale_label $scale)")
    run_one "$label" "$model" "$scale"
    i=$((i+1))
  done
done

# ── Summary ──────────────────────────────────────────────────────────────────
log "=========================================================================="
log "E1 ALL DONE  sweep_dir: $SWEEP_DIR"
log "=========================================================================="

# Cross-cell 6-panel comparison plot.
PLOT_COMPARE="${ROOT_DIR}/tools/plot_sweep_compare.py"
if [ -f "$PLOT_COMPARE" ]; then
  log "[plot] generating cross-cell comparison ..."
  "$PY" "$PLOT_COMPARE" --sweep-dir "$SWEEP_DIR" 2>&1 | tee -a "$LOG"
fi

# E1-specific: PIM-share vs arrival_scale curve per model.
PLOT_E1="${ROOT_DIR}/tools/plot_e1_route_share_vs_scale.py"
if [ -f "$PLOT_E1" ]; then
  log "[plot] generating E1 route-share-vs-scale ..."
  "$PY" "$PLOT_E1" --sweep-dir "$SWEEP_DIR" 2>&1 | tee -a "$LOG"
fi

log ""
log "What to look for:"
log "  Pi0 PIM share should stay >50% across all scales (architecture preference)."
log "  OpenVLA PIM share should stay <50% across all scales."
log "  At scale=5.0 (light), both may converge to whichever route the cost"
log "  table predicts faster at idle — that's a legitimate finding too."
