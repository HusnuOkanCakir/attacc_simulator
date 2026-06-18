#!/bin/bash
#SBATCH -J knee_knob
#SBATCH -p high_latency
#SBATCH -c 4
#SBATCH --mem=32GB
#SBATCH -D /mnt/galactica/ocakir/attacc_simulator
#SBATCH -o /mnt/galactica/ocakir/logs/knee_knob_%j.out
#SBATCH -e /mnt/galactica/ocakir/logs/knee_knob_%j.err
#
# Generic "knee curve under one varied knob" sweep.
#
# Plan: docs/plan/knee_vs_knob_sweeps.md
#
# Fixed dataset (azure_poisson_wide v2), fixed model (Pi0), fixed
# every-other-knob (canonical config). Vary ONE knob across N values
# AND arrival_scale across M scales. Output: N×M cells laid out so
# the plotter can group by knob value and overlay N knee curves on
# one figure.
#
# Usage (must specify these three):
#
#   KNOB_NAME=bs \
#   KNOB_FLAG='--max-decode-batch-size' \
#   KNOB_VALUES='1 2 4 8 16' \
#     sbatch tools/run_knee_vs_knob_sweep.sbatch
#
# For knobs that interact with another (SLO + admission, see plan),
# pin the partner via EXTRA_FLAGS:
#
#   KNOB_NAME=slo KNOB_FLAG='--slo-e2e-ms' \
#   KNOB_VALUES='2000 10000 30000 100000 300000' \
#   EXTRA_FLAGS='--admission-violation-budget 0' \
#     sbatch tools/run_knee_vs_knob_sweep.sbatch
#
#   KNOB_NAME=admbudget KNOB_FLAG='--admission-violation-budget' \
#   KNOB_VALUES='-1 0 1 5 20' \
#   EXTRA_FLAGS='--slo-e2e-ms 10000' \
#     sbatch tools/run_knee_vs_knob_sweep.sbatch
#
# Special knob: route mode. The runner uses a boolean --gpu-only flag,
# not a value. Use the KNOB_NAME=route convention:
#
#   KNOB_NAME=route KNOB_FLAG='--gpu-only-toggle' \
#   KNOB_VALUES='hybrid gpu_only' \
#     sbatch tools/run_knee_vs_knob_sweep.sbatch
#
# Env-var overrides (all optional):
#   DATASET           default azure_poisson_wide
#   ARRIVAL_SCALES    default "30 20 15 10 7 5 3 2 1.5"
#   N_REQUESTS        default 5000
#   MODEL             default pi0
#   POLICY            default max_util_full
#   KV_POOL_GB        default 8
#   MAX_ACTIVE        default 32
#   MAX_DECODE_BS     default 4
#   MAX_CONSEC_DEC    default 4
#   SLO_E2E_MS        default 500     (overridden if KNOB_NAME=slo, see below)
#   SLO_TTFT_MS       default 200
#   KV_OOM_POLICY     default hold_then_fallback
#   KV_OOM_HOLD_LIMIT_MS  default 50
#   ADMISSION_BUDGET  default -1      (off)  (overridden if KNOB_NAME=admbudget)
#   PROMPT_PRIORITY   default off ("--no-prompt-priority")
#   ENERGY_GUARD_MS   default unset (no flag passed)
#   GPU_TYPE          default a6000
#   EXTRA_FLAGS       default empty — pinned flags for interacting knobs
#
# Output dir: cluster_outputs/online_serving_runs/knee_vs_${KNOB_NAME}_${TS}/
# Cell label: 01_<knob>_<safe_value>_s<scale*10>     e.g. 01_bs_4_s065
# After the sweep, runs the thesis_knee_vs_knob.py plotter to produce
# fig_knee_vs_<knob>.png + fig_knee_vs_<knob>_linear.png.

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

# Required.
: "${KNOB_NAME:?must set KNOB_NAME (e.g. bs)}"
: "${KNOB_FLAG:?must set KNOB_FLAG (e.g. --max-decode-batch-size)}"
: "${KNOB_VALUES:?must set KNOB_VALUES (space-sep list)}"

# Optional.
DATASET="${DATASET:-azure_poisson_wide}"
ARRIVAL_SCALES="${ARRIVAL_SCALES:-30 20 15 10 7 5 3 2 1.5}"
# If OFFERED_RPS is set (space-separated offered RPS, NOT achieved), it
# overrides ARRIVAL_SCALES: each offered value is auto-translated to an
# arrival_scale via K = N/native-span (offered ~ K/scale). To keep the very
# low-offered cells tractable, per-cell N is reduced so the simulated span
# stays <= SPAN_CAP_S (offered RPS is N-independent, so the x-value is kept).
OFFERED_RPS="${OFFERED_RPS:-}"
SPAN_CAP_S="${SPAN_CAP_S:-9000}"
N_MIN="${N_MIN:-200}"
N_REQUESTS="${N_REQUESTS:-5000}"
MODEL="${MODEL:-pi0}"
POLICY="${POLICY:-max_util_full}"
KV_POOL_GB="${KV_POOL_GB:-8}"
MAX_ACTIVE="${MAX_ACTIVE:-32}"
MAX_DECODE_BS="${MAX_DECODE_BS:-4}"
MAX_CONSEC_DEC="${MAX_CONSEC_DEC:-4}"
SLO_E2E_MS="${SLO_E2E_MS:-500}"
SLO_TTFT_MS="${SLO_TTFT_MS:-200}"
KV_OOM_POLICY="${KV_OOM_POLICY:-hold_then_fallback}"
KV_OOM_HOLD_LIMIT_MS="${KV_OOM_HOLD_LIMIT_MS:-50}"
ADMISSION_BUDGET="${ADMISSION_BUDGET:--1}"
PROMPT_PRIORITY="${PROMPT_PRIORITY:-off}"   # off | on
ENERGY_GUARD_MS="${ENERGY_GUARD_MS:-}"
GPU_TYPE="${GPU_TYPE:-a6000}"
EXTRA_FLAGS="${EXTRA_FLAGS:-}"

# Dataset → CSV path resolution.
case "$DATASET" in
  azure_full19k)
    CSV_PATH="${ROOT_DIR}/cluster_outputs/azure/AzureLLMInferenceTrace_conv.csv"
    DEFAULT_N=19365
    ;;
  azure_poisson)
    CSV_PATH="${ROOT_DIR}/cluster_outputs/synthetic_v2/azure_poisson_n5000_seed0.csv"
    DEFAULT_N=5000
    ;;
  azure_poisson_wide)
    CSV_PATH="${ROOT_DIR}/cluster_outputs/synthetic_v2/azure_poisson_wide_n5000_seed0.csv"
    DEFAULT_N=5000
    ;;
  bootstrap)
    CSV_PATH="${ROOT_DIR}/cluster_outputs/synthetic_v2/azure_bootstrap_n5000_seed0.csv"
    DEFAULT_N=5000
    ;;
  azure_empirical)
    CSV_PATH="${ROOT_DIR}/cluster_outputs/synthetic_v2/azure_empirical_n5000_seed0.csv"
    DEFAULT_N=5000
    ;;
  vla_poisson)
    CSV_PATH="${ROOT_DIR}/cluster_outputs/synthetic_v2/vla_poisson_n5000_seed0.csv"
    DEFAULT_N=5000
    ;;
  *)
    echo "[error] unknown DATASET=$DATASET" >&2
    exit 1
    ;;
esac
N_REQUESTS="${N_REQUESTS:-$DEFAULT_N}"

PY=$(which python)
RUNNER="${ROOT_DIR}/tools/run_serving_online_policy_compare.py"
CACHE_DIR="${ROOT_DIR}/cluster_outputs/pim_cache"
TS=$(date +%Y%m%d_%H%M%S)
SWEEP_DIR="${ROOT_DIR}/cluster_outputs/online_serving_runs/knee_vs_${KNOB_NAME}_${TS}"
LOG="${SWEEP_DIR}/sweep.log"

mkdir -p "${CACHE_DIR}" "${SWEEP_DIR}"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# Safe-value label: replace special chars so we can embed in filenames.
# e.g. "-1" -> "m1", "guaranteed_no_evict" -> "gne", "max_util_full" -> "muf"
safe_label() {
  local v=$1
  case "$v" in
    -*) echo "m${v#-}" ;;
    guaranteed_no_evict) echo "gne" ;;
    max_util_full) echo "muf" ;;
    max_util_tail) echo "mut" ;;
    hold_then_fallback) echo "htf" ;;
    fallback_gpu) echo "fbg" ;;
    *) echo "${v//[^a-zA-Z0-9]/_}" ;;
  esac
}

scale_label() {
  awk -v x=$1 'BEGIN{printf "s%03d", x*10}'
}

n_values=$(echo "$KNOB_VALUES" | wc -w)
n_scales=$(echo "$ARRIVAL_SCALES" | wc -w)
n_total=$(( n_values * n_scales ))

log "Knee-vs-knob sweep"
log "knob=$KNOB_NAME  flag=$KNOB_FLAG  values=($KNOB_VALUES)"
log "dataset=$DATASET (csv=$CSV_PATH, n=$N_REQUESTS)"
log "scales=($ARRIVAL_SCALES)  matrix: ${n_values} × ${n_scales} = ${n_total} cells"
log "baseline: model=$MODEL  policy=$POLICY  bs=$MAX_DECODE_BS  max_active=$MAX_ACTIVE"
log "          SLO=${SLO_E2E_MS}/${SLO_TTFT_MS}ms  kv_oom=$KV_OOM_POLICY  hold=${KV_OOM_HOLD_LIMIT_MS}ms"
log "          admission_budget=$ADMISSION_BUDGET  prompt_priority=$PROMPT_PRIORITY"
log "extra_flags=$EXTRA_FLAGS"
log "sweep_dir=$SWEEP_DIR"
log ""

if [ ! -s "$CSV_PATH" ]; then
  log "[error] CSV not found at $CSV_PATH"; exit 1
fi

# Build the per-cell (arrival_scale, n_requests) plan.
#   - Default: ARRIVAL_SCALES at the fixed N_REQUESTS.
#   - OFFERED_RPS set: translate each offered RPS to arrival_scale via
#     K = N/native-span (offered ~ K/scale), and cap per-cell N so the
#     simulated span stays <= SPAN_CAP_S.
CELL_SCALES=(); CELL_NS=(); CELL_OFF=()
if [ -n "$OFFERED_RPS" ]; then
  PLAN=$("$PY" - "$CSV_PATH" "$N_REQUESTS" "$SPAN_CAP_S" "$N_MIN" $OFFERED_RPS <<'PYEOF'
import sys, pandas as pd
csv, n_target, span_cap, n_min = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
offereds = [float(x) for x in sys.argv[5:]]
df = pd.read_csv(csv, nrows=n_target)
t = pd.to_datetime(df["TIMESTAMP"])
span_s = (t.max() - t.min()).total_seconds()
K = len(df) / span_s            # offered RPS at arrival_scale = 1
print(f"K {K:.6f}")
for off in offereds:
    scale = K / off
    ncell = min(n_target, int(span_cap * off))
    ncell = max(ncell, n_min)
    print(f"CELL {off:.3f} {scale:.5f} {ncell}")
PYEOF
)
  while read -r tag a b c; do
    case "$tag" in
      K)    log "offered->scale: K=$a (offered RPS at arrival_scale=1)" ;;
      CELL) CELL_OFF+=("$a"); CELL_SCALES+=("$b"); CELL_NS+=("$c") ;;
    esac
  done <<< "$PLAN"
  log "OFFERED_RPS mode: ${#CELL_SCALES[@]} points (span_cap=${SPAN_CAP_S}s, n_min=$N_MIN)"
  for k in "${!CELL_SCALES[@]}"; do
    log "  offered=${CELL_OFF[$k]}  scale=${CELL_SCALES[$k]}  n=${CELL_NS[$k]}"
  done
else
  for s in $ARRIVAL_SCALES; do
    CELL_SCALES+=("$s"); CELL_NS+=("$N_REQUESTS"); CELL_OFF+=("-")
  done
fi

# Build per-cell command. The varied knob's flag/value is inlined.
# Special-case toggles (flags that take no value):
#   --gpu-only-toggle      with values {hybrid, gpu_only}
#                          → injects "--gpu-only" or nothing
#   --prompt-priority-toggle  with values {off, on}
#                          → injects "--no-prompt-priority" or "--prompt-priority"
run_one() {
  local label=$1
  local knob_value=$2
  local scale=$3
  local n_cell=${4:-$N_REQUESTS}
  local run_dir="${SWEEP_DIR}/${label}"
  local cache_file="${CACHE_DIR}/${MODEL}_${GPU_TYPE}_knee_vs_${KNOB_NAME}_$(safe_label $knob_value)_${DATASET}.bin"

  # Resolve the knob-specific portion of the command line.
  local knob_arg=""
  local suppress_base_prompt_flag=0
  if [ "$KNOB_FLAG" = "--gpu-only-toggle" ]; then
    [ "$knob_value" = "gpu_only" ] && knob_arg="--gpu-only"
  elif [ "$KNOB_FLAG" = "--prompt-priority-toggle" ]; then
    # Suppress base prompt_flag and inject the toggle directly.
    suppress_base_prompt_flag=1
    if [ "$knob_value" = "on" ]; then
      knob_arg="--prompt-priority"
    else
      knob_arg="--no-prompt-priority"
    fi
  else
    knob_arg="${KNOB_FLAG} ${knob_value}"
  fi

  log "── start: $label  knob_value=$knob_value  scale=$scale  n=$n_cell ──"
  log "    cache: $cache_file"

  # Prompt priority flag — skipped when the toggle is the swept knob.
  local prompt_flag=""
  if [ "$suppress_base_prompt_flag" = "0" ]; then
    prompt_flag="--no-prompt-priority"
    [ "$PROMPT_PRIORITY" = "on" ] && prompt_flag="--prompt-priority"
  fi

  # Optional energy guard.
  local energy_arg=""
  [ -n "$ENERGY_GUARD_MS" ] && energy_arg="--energy-latency-guard-ms $ENERGY_GUARD_MS"

  local t0=$(date +%s)
  "$PY" "$RUNNER" \
      --model "$MODEL" \
      --gpu "$GPU_TYPE" \
      --n-requests "$n_cell" \
      --policies "$POLICY" \
      --kv-pool-gb "$KV_POOL_GB" --ml \
      --max-active "$MAX_ACTIVE" \
      --arrival-scale "$scale" \
      --slo-e2e-ms "$SLO_E2E_MS" --slo-ttft-ms "$SLO_TTFT_MS" \
      --kv-oom-policy "$KV_OOM_POLICY" \
      --kv-oom-hold-limit-ms "$KV_OOM_HOLD_LIMIT_MS" \
      --max-decode-batch-size "$MAX_DECODE_BS" \
      --max-consecutive-decode-batches "$MAX_CONSEC_DEC" \
      --admission-violation-budget "$ADMISSION_BUDGET" \
      $prompt_flag \
      $energy_arg \
      $knob_arg \
      $EXTRA_FLAGS \
      --csv-path "$CSV_PATH" \
      --pim-cache-file "$cache_file" \
      --no-kv-timeline \
      --run-dir "$run_dir" 2>&1 | tee -a "$LOG"
  local dt=$(( $(date +%s) - t0 ))
  log "── done:  $label  wall=${dt}s ──"
}

# ── Sweep ────────────────────────────────────────────────────────────────────
i=1
for knob_value in $KNOB_VALUES; do
  for idx in "${!CELL_SCALES[@]}"; do
    scale="${CELL_SCALES[$idx]}"; n_cell="${CELL_NS[$idx]}"
    label=$(printf "%02d_%s_%s_%s" "$i" "$KNOB_NAME" "$(safe_label $knob_value)" "$(scale_label $scale)")
    run_one "$label" "$knob_value" "$scale" "$n_cell"
    i=$((i+1))
  done
done

log "=========================================================================="
log "knee_vs_${KNOB_NAME} ALL DONE  sweep_dir: $SWEEP_DIR"
log "=========================================================================="

# Auto-plot.
PLOT="${ROOT_DIR}/thesis_plotting/scripts/thesis_knee_vs_knob.py"
if [ -f "$PLOT" ]; then
  log "[plot] generating knee-vs-knob figure ..."
  "$PY" "$PLOT" --sweep-dir "$SWEEP_DIR" \
      --knob "$KNOB_NAME" 2>&1 | tee -a "$LOG"
  "$PY" "$PLOT" --sweep-dir "$SWEEP_DIR" \
      --knob "$KNOB_NAME" --linear 2>&1 | tee -a "$LOG"
fi
