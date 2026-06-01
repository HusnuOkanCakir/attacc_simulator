#!/bin/bash
#SBATCH -J e4_viol_budget
#SBATCH -p high_latency
#SBATCH -c 4
#SBATCH --mem=32GB
#SBATCH -D /mnt/galactica/ocakir/attacc_simulator
#SBATCH -o /mnt/galactica/ocakir/logs/e4_viol_budget_%j.out
#SBATCH -e /mnt/galactica/ocakir/logs/e4_viol_budget_%j.err
#
# E4: Admission-control violation-budget sweep (Phase C validation)
#
# E3 showed SLO is structurally redundant under min_finish — the SLO knob
# can never change the route chosen by min_finish. Phase A added per-cell
# SLO violator counts; the compliance plot showed clear cliffs at
# SLO ≈ 200-400 ms on synthetic VLA.
#
# Phase C introduces an admission-control policy that USES the SLO to hold
# requests before admission, when admitting them would newly-violate
# in-flight requests' SLOs. CLI knob: --admission-violation-budget X.
#   X = -1   (default, disabled) — preserves prior behavior.
#   X = 0    strict: hold D if ANY in-flight request would be newly-violated.
#   X >= 1   loose: tolerate up to X newly-violated per admission.
#
# 2 models × 5 budget values × synthetic VLA = 10 cells. Fixed at SLO=400 ms
# (where E3 showed 15.8% Pi0 violators and 6.2% OpenVLA violators — tight
# enough to trigger the budget check).
#
# Output: cluster_outputs/online_serving_runs/e4_viol_budget_${TS}/
#           01_pi0_xN1/<policy>/...      (xN1 = X=-1 disabled baseline)
#           02_pi0_x00/...               (x00 = X=0 strict)
#           ...
#           10_openvla_x05/...           (x05 = X=5 loose)
#
# Usage:
#   sbatch tools/run_e4_violation_budget_sweep.sbatch
#
# Env-var overrides:
#   N_REQUESTS         (default 2000)
#   ARRIVAL_SCALE      (default 1.0)
#   KV_POOL_GB         (default 8)
#   MAX_ACTIVE         (default 32)
#   POLICY             (default max_util_full)
#   MAX_DECODE_BS      (default 8)
#   MAX_CONSEC_DEC     (default 4)
#   SLO_E2E_MS         (default 400)
#   KV_OOM_POLICY      (default hold_then_fallback)
#   MODELS             (default "pi0 openvla")
#   BUDGETS            (default "-1 0 1 2 5")
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
SLO_E2E_MS="${SLO_E2E_MS:-400}"
SLO_TTFT_MS="-1"
KV_OOM_POLICY="${KV_OOM_POLICY:-hold_then_fallback}"
KV_OOM_HOLD_LIMIT_MS="${KV_OOM_HOLD_LIMIT_MS:-50}"
MODELS="${MODELS:-pi0 openvla}"
BUDGETS="${BUDGETS:--1 0 1 2 5}"
GPU_TYPE="${GPU_TYPE:-a6000}"

PY=$(which python)
RUNNER="${ROOT_DIR}/tools/run_serving_online_policy_compare.py"
TRACE_GEN="${ROOT_DIR}/tools/gen_vla_synthetic_trace.py"
CACHE_DIR="${ROOT_DIR}/cluster_outputs/pim_cache"
SYNTH_TRACE_DIR="${ROOT_DIR}/cluster_outputs/synthetic_vla"
TS=$(date +%Y%m%d_%H%M%S)
SWEEP_DIR="${ROOT_DIR}/cluster_outputs/online_serving_runs/e4_viol_budget_${TS}"
LOG="${SWEEP_DIR}/sweep.log"

mkdir -p "${CACHE_DIR}" "${SWEEP_DIR}" "${SYNTH_TRACE_DIR}"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

n_models=$(echo "$MODELS" | wc -w)
n_budgets=$(echo "$BUDGETS" | wc -w)
n_total=$(( n_models * n_budgets ))

log "E4: admission-control violation-budget sweep"
log "models=($MODELS)  budgets=($BUDGETS)  gpu=$GPU_TYPE"
log "n=$N_REQUESTS  scale=$ARRIVAL_SCALE  kv_pool=${KV_POOL_GB}GiB  max_active=$MAX_ACTIVE"
log "policy=$POLICY  bs=$MAX_DECODE_BS  mc=$MAX_CONSEC_DEC"
log "slo_e2e=${SLO_E2E_MS}ms (tight enough to bind based on E3 data)"
log "matrix: ${n_models} models × ${n_budgets} budgets = ${n_total} cells"
log "sweep_dir=$SWEEP_DIR"
log ""

# Pre-generate synthetic traces if missing.
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

# Budget label: negative → "N1", non-neg → "0N" zero-padded to width 2.
budget_label() {
  local x=$1
  if [ "$x" = "-1" ]; then echo "xN1"
  else echo "x$(printf "%02d" "$x")"
  fi
}

# args: <label> <model> <budget>
run_one() {
  local label=$1
  local model=$2
  local budget=$3
  local run_dir="${SWEEP_DIR}/${label}"
  # Cache shared per-model — budget doesn't affect per-shape drain.
  local cache_file="${CACHE_DIR}/${model}_${GPU_TYPE}_e4.bin"
  local csv_path="${SYNTH_TRACE_DIR}/synthetic_vla_${model}_n${N_REQUESTS}_seed0.csv"

  log "── start: $label  model=$model  budget=$budget ──"
  log "    cache: $cache_file (shared per-model)"

  local t0=$(date +%s)
  "$PY" "$RUNNER" \
      --model "$model" \
      --gpu "$GPU_TYPE" \
      --n-requests "$N_REQUESTS" \
      --policies "$POLICY" \
      --kv-pool-gb "$KV_POOL_GB" --ml \
      --max-active "$MAX_ACTIVE" \
      --arrival-scale "$ARRIVAL_SCALE" \
      --slo-e2e-ms "$SLO_E2E_MS" --slo-ttft-ms "$SLO_TTFT_MS" \
      --kv-oom-policy "$KV_OOM_POLICY" \
      --kv-oom-hold-limit-ms "$KV_OOM_HOLD_LIMIT_MS" \
      --max-decode-batch-size "$MAX_DECODE_BS" \
      --max-consecutive-decode-batches "$MAX_CONSEC_DEC" \
      --no-prompt-priority \
      --admission-violation-budget "$budget" \
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
  for budget in $BUDGETS; do
    label=$(printf "%02d_%s_%s" "$i" "$model" "$(budget_label $budget)")
    run_one "$label" "$model" "$budget"
    i=$((i+1))
  done
done

log "=========================================================================="
log "E4 ALL DONE  sweep_dir: $SWEEP_DIR"
log "=========================================================================="

PLOT_COMPARE="${ROOT_DIR}/tools/plot_sweep_compare.py"
if [ -f "$PLOT_COMPARE" ]; then
  log "[plot] generating cross-cell comparison ..."
  "$PY" "$PLOT_COMPARE" --sweep-dir "$SWEEP_DIR" 2>&1 | tee -a "$LOG"
fi

# Retroactive SLO violator plot (works regardless of budget knob).
PLOT_SLO="${ROOT_DIR}/tools/plot_slo_violators.py"
if [ -f "$PLOT_SLO" ]; then
  log "[plot] generating SLO violator counts (Phase A) ..."
  "$PY" "$PLOT_SLO" --sweep-dir "$SWEEP_DIR" 2>&1 | tee -a "$LOG"
fi

log ""
log "Expected:"
log "  - At X=-1 (disabled): results match prior E1/E3 baseline (regression test)."
log "  - As X decreases from 5 → 0: admission_violation_holds increases."
log "  - As X decreases: SLO violators decrease (compliance improves)."
log "  - Trade-off: throughput may drop slightly at X=0 (more holds → longer queue)."
