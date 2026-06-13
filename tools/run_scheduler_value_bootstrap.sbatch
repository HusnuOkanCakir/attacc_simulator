#!/bin/bash
#SBATCH -J sched_val
#SBATCH -p high_latency
#SBATCH -c 4
#SBATCH --mem=64GB
#SBATCH --exclude=safari-nexus1
#SBATCH -D /mnt/galactica/ocakir/attacc_simulator
#SBATCH -o /mnt/galactica/ocakir/logs/sched_val_%j.out
#SBATCH -e /mnt/galactica/ocakir/logs/sched_val_%j.err
#
# Scheduler-value experiment on the real-shape bootstrap trace.
#
# Goal: locate the regime where request-level routing adds value, i.e.
# hybrid > PIM-only > GPU-only on latency, throughput, and energy.
# Reasoning: the GPU route adds genuine capacity only under PIM KV-pool
# pressure (GPU KV is untracked); steady-state speed arbitrage cannot
# win because PIM-route tasks share the GPU for FC. The energy-fixed
# cost tables make latency_guarded_energy a PIM-first policy with GPU
# spill only under predicted deep-queue delay.
#
# Matrix: 4 configs x 3 scales at KV pool 0.5 GiB + a 2-cell 64 GiB
# control pair = 14 cells. Workload: azure_bootstrap_n5000_seed0
# (rows resampled jointly from the real Azure 19k conv trace).
#
#   gpu_only : --gpu-only
#   pim_only : --pim-only (forces kv_oom=hold, preempt_tries=-1)
#   hyb_mf   : min_finish + kv_max_preempt_tries=3
#   hyb_lge  : latency_guarded_energy guard=200ms + kv_max_preempt_tries=3
#
# Cost tables: energy-fixed A6000 tables (CSV nearest-neighbor; NOT
# --ml, so routing energy and timing come from the same source).
# Requires the BATCH_VALUES=1,2,4,8 regen of
# cluster_outputs/cost_tables_energyfix_pi0_a6000.

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

N_REQUESTS=5000
MODEL="${MODEL:-pi0}"
GPU_TYPE=a6000
POLICY=max_util_full
KV_POOL_GB="${KV_POOL_GB:-0.5}"
KV_POOL_GB_CTRL="${KV_POOL_GB_CTRL:-64}"   # pressure-free control pool
MAX_ACTIVE=5000
MAX_DECODE_BS=4
MAX_CONSEC_DEC=4
SLO_E2E_MS=500
SLO_TTFT_MS=200
LGE_GUARD_MS="${LGE_GUARD_MS:-200}"
TAG="${TAG:-bootstrap}"
# Default cost-dir + scales target the pi0 knee (~2.1 RPS ceiling).
# Override SCALES / CTRL_SCALE for OpenVLA (~0.6 RPS ceiling), e.g.
#   MODEL=openvla COST_DIR=cluster_outputs/cost_tables_energyfix_openvla_a6000 \
#   SCALES="12 9.5 7.7" CTRL_SCALE=9.5 KV_POOL_GB=8 TAG=openvla sbatch ...
COST_DIR="${COST_DIR:-cluster_outputs/cost_tables_energyfix_pi0_a6000}"
read -r -a SCALES <<< "${SCALES:-3.5 2.85 2.4}"
CTRL_SCALE="${CTRL_SCALE:-2.85}"

CSV_PATH="${ROOT_DIR}/cluster_outputs/synthetic_v2/azure_bootstrap_n5000_seed0.csv"
CACHE_DIR="${ROOT_DIR}/cluster_outputs/pim_cache"
TS=$(date +%Y%m%d_%H%M%S)
SWEEP_DIR="${ROOT_DIR}/cluster_outputs/online_serving_runs/sched_value_${TAG}_${TS}"
LOG="${SWEEP_DIR}/sweep.log"
mkdir -p "${CACHE_DIR}" "${SWEEP_DIR}"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

[ -s "$CSV_PATH" ] || { log "[error] CSV missing: $CSV_PATH"; exit 1; }
[ -s "${ROOT_DIR}/${COST_DIR}/gpu_only.csv" ] || { log "[error] cost tables missing: $COST_DIR"; exit 1; }
log "scheduler-value experiment — kv_pool=$KV_POOL_GB GiB, guard=${LGE_GUARD_MS}ms, scales=(${SCALES[*]})"
log "cost tables: $COST_DIR"
log "sweep_dir=$SWEEP_DIR"

PY=$(which python)
RUNNER="${ROOT_DIR}/tools/run_serving_online_policy_compare.py"

# run_cell <idx> <config_name> <scale> <pool_gb> <extra flags...>
run_cell() {
  local idx=$1; local cfg=$2; local scale=$3; local pool=$4
  shift 4
  local s_safe=$(echo "$scale" | tr -d '.')
  local p_safe=$(echo "$pool" | tr -d '.')
  local label=$(printf "%02d_%s_p%s_s%s" "$idx" "$cfg" "$p_safe" "$s_safe")
  local run_dir="${SWEEP_DIR}/${label}"
  local cache_file="${CACHE_DIR}/${MODEL}_${GPU_TYPE}_schedval_${cfg}_p${p_safe}_s${s_safe}.bin"
  log "── start: ${label} ──"
  local t0=$(date +%s)
  "$PY" "$RUNNER" \
      --model "$MODEL" \
      --gpu "$GPU_TYPE" \
      --cost-dir "$COST_DIR" \
      --n-requests "$N_REQUESTS" \
      --policies "$POLICY" \
      --kv-pool-gb "$pool" \
      --max-active "$MAX_ACTIVE" \
      --arrival-scale "$scale" \
      --slo-e2e-ms "$SLO_E2E_MS" --slo-ttft-ms "$SLO_TTFT_MS" \
      --max-decode-batch-size "$MAX_DECODE_BS" \
      --max-consecutive-decode-batches "$MAX_CONSEC_DEC" \
      --admission-violation-budget -1 \
      --no-prompt-priority \
      --csv-path "$CSV_PATH" \
      --pim-cache-file "$cache_file" \
      --no-kv-timeline \
      --run-dir "$run_dir" \
      "$@" 2>&1 | tee -a "$LOG"
  local dt=$(( $(date +%s) - t0 ))
  log "── done:  ${label}  wall=${dt}s ──"
}

idx=1
for s in "${SCALES[@]}"; do
  # Hybrid-LGE first (fastest expected), then hybrid-MF, PIM-only, GPU-only.
  run_cell "$idx" "hyb_lge" "$s" "$KV_POOL_GB" \
      --route-policy latency_guarded_energy \
      --energy-latency-guard-ms "$LGE_GUARD_MS" \
      --kv-max-preempt-tries 3 \
      --kv-oom-policy hold_then_fallback --kv-oom-hold-limit-ms 50
  idx=$((idx + 1))
  run_cell "$idx" "hyb_mf" "$s" "$KV_POOL_GB" \
      --route-policy min_finish \
      --kv-max-preempt-tries 3 \
      --kv-oom-policy hold_then_fallback --kv-oom-hold-limit-ms 50
  idx=$((idx + 1))
  run_cell "$idx" "pim_only" "$s" "$KV_POOL_GB" \
      --pim-only
  idx=$((idx + 1))
  run_cell "$idx" "gpu_only" "$s" "$KV_POOL_GB" \
      --gpu-only
  idx=$((idx + 1))
done

# Pressure-free control pair at the middle scale: hybrid-LGE should
# converge to PIM-only when the pool is not the binding constraint.
run_cell "$idx" "hyb_lge" "$CTRL_SCALE" "$KV_POOL_GB_CTRL" \
    --route-policy latency_guarded_energy \
    --energy-latency-guard-ms "$LGE_GUARD_MS" \
    --kv-max-preempt-tries 3 \
    --kv-oom-policy hold_then_fallback --kv-oom-hold-limit-ms 50
idx=$((idx + 1))
run_cell "$idx" "pim_only" "$CTRL_SCALE" "$KV_POOL_GB_CTRL" \
    --pim-only

log
log "=== summary ==="
for cell in "${SWEEP_DIR}"/*/; do
  s="${cell}policy_compare_summary.txt"
  if [ -f "$s" ]; then
    log "--- $(basename "$cell") ---"
    grep -E "Throughput|Routes|KV-OOM holds|Preempt|Tokens recomputed|Eviction|TTFT p99|E2E p99|Span" "$s" 2>&1 | tee -a "$LOG"
  fi
done
log
log "Post-process energy with tools/compute_serving_energy.py; render with"
log "thesis_plotting/scripts/thesis_scheduler_value_bootstrap.py."
