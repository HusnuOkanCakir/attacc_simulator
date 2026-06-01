#!/bin/bash
#SBATCH -J rps_knee
#SBATCH -p high_latency
#SBATCH -c 4
#SBATCH --mem=32GB
#SBATCH -D /mnt/galactica/ocakir/attacc_simulator
#SBATCH -o /mnt/galactica/ocakir/logs/rps_knee_%j.out
#SBATCH -e /mnt/galactica/ocakir/logs/rps_knee_%j.err
#
# RPS knee-finder sweep.
#
# Goal: locate the peak RPS `a` our system can sustain at p99-SLO.
# Method: fix model + policy + KV pool, vary arrival_scale ∈
#   {10, 5, 2.5, 1.5, 1.0, 0.7, 0.5, 0.3, 0.2, 0.1} on a base
# synthetic trace generated at high rate_hz (so even arrival_scale=10
# gives meaningful offered load). E2E p99 vs effective RPS reveals the
# knee.
#
# Four named datasets get the same sweep so we can compare:
#   - vla_poisson      — Poisson(λ=306/7) shape, 30 Hz base rate
#   - azure_poisson    — Poisson(λ=1155/211) shape, 5.5 Hz base rate
#                        (Azure-mean shape but narrow Poisson std)
#   - azure_empirical  — empirical CDF lifted from the full 19k Azure
#                        conv (heavy-tail preserved), 5.5 Hz base rate
#   - azure_real       — the REAL Azure conv CSV (no generation).
#                        Ground truth — verifies whether
#                        azure_empirical matches the real knee.
#
# Matrix: 4 datasets × 10 scales = 40 cells.
#
# Cache namespace `_knee_` keeps this isolated from E5 / E6 / F runs.
#
# Usage:
#   sbatch tools/run_rps_knee_sweep.sbatch
#
# Env-var overrides:
#   MODEL              (default pi0; openvla also works)
#   N_REQUESTS         (default 5000 — needs enough for p99 stability)
#   KV_POOL_GB         (default 8)
#   MAX_ACTIVE         (default 32)
#   POLICY             (default max_util_full)
#   MAX_DECODE_BS      (default 4)
#   MAX_CONSEC_DEC     (default 4)
#   SLO_E2E_MS         (default 500)
#   SLO_TTFT_MS        (default 200)
#   KV_OOM_POLICY      (default hold_then_fallback)
#   DATASETS           (default "vla_poisson azure_poisson azure_empirical")
#   ARRIVAL_SCALES     (default "10.0 5.0 2.5 1.5 1.0 0.7 0.5 0.3 0.2 0.1")
#   GPU_TYPE           (default a6000)
#   ROBOTIC_RATE_HZ    (default 30 — base rate for robotic preset)
#   CONV_RATE_HZ       (default 5.5 — base rate matching Azure mean)
#   LOUT_MODE          (default openvla — for robotic preset only)

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

MODEL="${MODEL:-pi0}"
N_REQUESTS="${N_REQUESTS:-5000}"
KV_POOL_GB="${KV_POOL_GB:-8}"
MAX_ACTIVE="${MAX_ACTIVE:-32}"
POLICY="${POLICY:-max_util_full}"
MAX_DECODE_BS="${MAX_DECODE_BS:-4}"
MAX_CONSEC_DEC="${MAX_CONSEC_DEC:-4}"
SLO_E2E_MS="${SLO_E2E_MS:-500}"
SLO_TTFT_MS="${SLO_TTFT_MS:-200}"
KV_OOM_POLICY="${KV_OOM_POLICY:-hold_then_fallback}"
KV_OOM_HOLD_LIMIT_MS="${KV_OOM_HOLD_LIMIT_MS:-50}"
# Three named workloads, each a (preset, shape, rate_hz) triple:
#   vla_poisson      — robotic_vla shape, Poisson(λ=306/7), rate 30 Hz
#   azure_poisson    — Azure-mean shape, Poisson(λ=1155/211), rate 5.5 Hz
#   azure_empirical  — Azure-CDF shape (heavy tail), rate 5.5 Hz
DATASETS="${DATASETS:-vla_poisson azure_poisson azure_empirical azure_real}"
ARRIVAL_SCALES="${ARRIVAL_SCALES:-10.0 5.0 2.5 1.5 1.0 0.7 0.5 0.3 0.2 0.1}"
GPU_TYPE="${GPU_TYPE:-a6000}"
ROBOTIC_RATE_HZ="${ROBOTIC_RATE_HZ:-30}"
CONV_RATE_HZ="${CONV_RATE_HZ:-5.5}"
LOUT_MODE="${LOUT_MODE:-openvla}"

PY=$(which python)
RUNNER="${ROOT_DIR}/tools/run_serving_online_policy_compare.py"
GEN_V2="${ROOT_DIR}/tools/gen_synthetic_trace_v2.py"
PROFILER="${ROOT_DIR}/tools/profile_trace_rps.py"
CACHE_DIR="${ROOT_DIR}/cluster_outputs/pim_cache"
SYNTH_DIR="${ROOT_DIR}/cluster_outputs/synthetic_v2"
TS=$(date +%Y%m%d_%H%M%S)
SWEEP_DIR="${ROOT_DIR}/cluster_outputs/online_serving_runs/rps_knee_${MODEL}_${TS}"
LOG="${SWEEP_DIR}/sweep.log"

mkdir -p "${CACHE_DIR}" "${SWEEP_DIR}" "${SYNTH_DIR}"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

n_datasets=$(echo "$DATASETS" | wc -w)
n_scales=$(echo "$ARRIVAL_SCALES" | wc -w)
n_total=$(( n_datasets * n_scales ))

log "RPS knee-finder sweep"
log "model=$MODEL  gpu=$GPU_TYPE"
log "datasets=($DATASETS)  arrival_scales=($ARRIVAL_SCALES)"
log "n=$N_REQUESTS  kv_pool=${KV_POOL_GB}GiB  max_active=$MAX_ACTIVE"
log "policy=$POLICY  bs=$MAX_DECODE_BS  mc=$MAX_CONSEC_DEC"
log "slo_e2e=${SLO_E2E_MS}ms  slo_ttft=${SLO_TTFT_MS}ms"
log "kv_oom=$KV_OOM_POLICY"
log "matrix: ${n_datasets} datasets × ${n_scales} scales = ${n_total} cells"
log "sweep_dir=$SWEEP_DIR"
log ""

# ── Step 1: generate (or locate) base traces, one per dataset ───────────────
# Each synthetic dataset name → (preset, shape_distribution, rate_hz) triple.
# azure_real is special — points at the real Azure CSV, no generation.
preset_for() {
  case "$1" in
    vla_poisson)         echo "robotic_vla" ;;
    azure_poisson)       echo "conversational" ;;
    azure_empirical)     echo "conversational" ;;
    azure_poisson_wide)  echo "conversational" ;;
    azure_real|azure_full19k) echo "REAL" ;;
    *)                   echo "robotic_vla" ;;
  esac
}
shape_for() {
  case "$1" in
    azure_empirical)     echo "empirical" ;;
    azure_poisson_wide)  echo "mixture_poisson" ;;
    azure_real|azure_full19k) echo "real" ;;
    *)                   echo "poisson" ;;
  esac
}
get_rate_hz() {
  case "$1" in
    vla_poisson)         echo "$ROBOTIC_RATE_HZ" ;;
    azure_real|azure_full19k) echo "n/a" ;;
    azure_*)             echo "$CONV_RATE_HZ" ;;
    *)                   echo "5.0" ;;
  esac
}
# Per-dataset n_requests cap. azure_full19k uses the full trace
# (19,365 rows); azure_real and synthetics use the sweep's N_REQUESTS.
n_requests_for() {
  case "$1" in
    azure_full19k)      echo "19365" ;;
    *)                  echo "$N_REQUESTS" ;;
  esac
}
# Resolve the on-disk CSV for a dataset. Synthetic datasets land in
# SYNTH_DIR; azure_real / azure_full19k point at the canonical
# Azure CSV in the repo (the runner's --n-requests handles truncation).
csv_path_for() {
  local dataset=$1
  case "$dataset" in
    azure_real|azure_full19k)
      echo "${ROOT_DIR}/cluster_outputs/azure/AzureLLMInferenceTrace_conv.csv" ;;
    *)
      echo "${SYNTH_DIR}/${dataset}_n${N_REQUESTS}_seed0.csv" ;;
  esac
}

for dataset in $DATASETS; do
  out_csv=$(csv_path_for $dataset)
  ds_n=$(n_requests_for $dataset)
  if [ "$dataset" = "azure_real" ] || [ "$dataset" = "azure_full19k" ]; then
    if [ ! -s "$out_csv" ]; then
      log "[error] $dataset CSV not found at $out_csv"
      exit 1
    fi
    log "[trace] $dataset: real Azure conv $out_csv  (n=$ds_n)"
  else
    preset=$(preset_for $dataset)
    shape=$(shape_for $dataset)
    rate=$(get_rate_hz $dataset)
    if [ ! -s "$out_csv" ]; then
      log "[trace] gen $dataset  preset=$preset  shape=$shape  rate=${rate}Hz  n=$N_REQUESTS"
      "$PY" "$GEN_V2" --preset "$preset" --n-requests "$N_REQUESTS" \
          --rate-hz "$rate" --lout-mode "$LOUT_MODE" \
          --shape-distribution "$shape" \
          --seed 0 --out-csv "$out_csv" 2>&1 | tee -a "$LOG"
    else
      log "[trace] reuse $out_csv"
    fi
  fi
  log "[profile] $dataset"
  "$PY" "$PROFILER" --csv "$out_csv" 2>&1 | tee -a "$LOG"
  log ""
done

# scale label helper: 10.0 -> s100; 0.1 -> s001
scale_label() {
  awk -v x=$1 'BEGIN{printf "s%03d", x*10}'
}

# args: <label> <dataset> <arrival_scale>
run_one() {
  local label=$1
  local dataset=$2
  local scale=$3
  local csv_path=$(csv_path_for $dataset)
  local ds_n=$(n_requests_for $dataset)
  local run_dir="${SWEEP_DIR}/${label}"
  local cache_file="${CACHE_DIR}/${MODEL}_${GPU_TYPE}_knee_${dataset}_$(scale_label $scale).bin"

  log "── start: $label  dataset=$dataset  scale=$scale  n=$ds_n ──"
  log "    cache: $cache_file"
  log "    csv:   $csv_path"

  local t0=$(date +%s)
  "$PY" "$RUNNER" \
      --model "$MODEL" \
      --gpu "$GPU_TYPE" \
      --n-requests "$ds_n" \
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
      --run-dir "$run_dir" 2>&1 | tee -a "$LOG"
  local dt=$(( $(date +%s) - t0 ))
  log "── done:  $label  wall=${dt}s ──"
}

# ── Sweep ────────────────────────────────────────────────────────────────────
i=1
for dataset in $DATASETS; do
  for scale in $ARRIVAL_SCALES; do
    label=$(printf "%02d_%s_%s" "$i" "$dataset" "$(scale_label $scale)")
    run_one "$label" "$dataset" "$scale"
    i=$((i+1))
  done
done

log "=========================================================================="
log "RPS knee sweep DONE  sweep_dir: $SWEEP_DIR"
log "=========================================================================="

# Cross-cell standard plot.
PLOT_COMPARE="${ROOT_DIR}/tools/plot_sweep_compare.py"
if [ -f "$PLOT_COMPARE" ]; then
  log "[plot] generating cross-cell comparison ..."
  "$PY" "$PLOT_COMPARE" --sweep-dir "$SWEEP_DIR" 2>&1 | tee -a "$LOG"
fi

# Thesis knee figure (this sweep's headline output).
# Use a model-suffixed filename so Pi0 and OpenVLA knee figures coexist.
KNEE_PLOTTER="${ROOT_DIR}/thesis_plotting/scripts/thesis_rps_knee.py"
if [ -f "$KNEE_PLOTTER" ]; then
  log "[plot] generating thesis knee figure ..."
  "$PY" "$KNEE_PLOTTER" --sweep-dir "$SWEEP_DIR" \
      --out-name "fig_rps_knee_${MODEL}" 2>&1 | tee -a "$LOG"
fi
