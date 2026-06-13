#!/bin/bash
#SBATCH -J route_pol
#SBATCH -p high_latency
#SBATCH -c 4
#SBATCH --mem=32GB
#SBATCH --exclude=safari-nexus1
#SBATCH -D /mnt/galactica/ocakir/attacc_simulator
#SBATCH -o /mnt/galactica/ocakir/logs/route_pol_%j.out
#SBATCH -e /mnt/galactica/ocakir/logs/route_pol_%j.err
#
# Compare min_finish vs latency_guarded_energy routing on the same
# Pi0 + azure_poisson_wide cell at mean offered load = 1.09 RPS
# (arrival_scale = 5).
#
# Background:
#   §6.6 tail anatomy shows the worst-50 E2E requests are 100% on the
#   GPU lane under min_finish. The chapter now states the cost model
#   correctly identifies PIM as 1.5-2x faster per long-context
#   request, but min_finish minimizes queue-aware predicted finish,
#   not per-request exec time. When PIM is queued enough that
#   gpu_finish < pim_finish for a long request, min_finish ships it
#   to GPU; subsequent long requests then pile up on GPU because the
#   previous one is still executing slowly.
#
#   latency_guarded_energy keeps the same min_finish core but adds a
#   guard band: routes within `energy_latency_guard_ms` of the
#   fastest finish are reconsidered, and the lower-energy route wins
#   the tie. Per the cost data, PIM has higher per-token decode
#   energy at all three reference shapes, so this guard typically
#   pushes work back toward GPU rather than PIM. The empirical
#   question is whether the resulting top-50 tail still concentrates
#   on the GPU lane.
#
# Cells:
#   A_min_finish               — baseline (matches the existing hybrid
#                                cell at scale=5 in the chapter).
#   B_lge_guard5                — latency_guarded_energy, 5 ms guard
#                                (a tight guard, mostly the same as
#                                min_finish but breaks ties on energy).
#   C_lge_guard20              — 20 ms guard (canonical value used by
#                                the existing knee_vs_energyguard sweep).
#   D_lge_guard100             — 100 ms guard (loose; pulls many
#                                requests away from min_finish's choice).
#
# After rsync, run tools/analyze_request_tail.py on each cell to
# compute the top-50 worst-e2e GPU share and arrival-decile profile.

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

# ── Config: matches the chapter's main 1.09 RPS comparison cell ───────
N_REQUESTS=5000
MODEL=pi0
GPU_TYPE=a6000
POLICY=max_util_full
KV_POOL_GB=8
MAX_ACTIVE=32
MAX_DECODE_BS=4
MAX_CONSEC_DEC=4
ARRIVAL_SCALE=5.0
SLO_E2E_MS=500
SLO_TTFT_MS=200
KV_OOM_POLICY=hold_then_fallback
KV_OOM_HOLD_LIMIT_MS=50

CSV_PATH="${ROOT_DIR}/cluster_outputs/synthetic_v2/azure_poisson_wide_n5000_seed0.csv"
CACHE_DIR="${ROOT_DIR}/cluster_outputs/pim_cache"
TS=$(date +%Y%m%d_%H%M%S)
SWEEP_DIR="${ROOT_DIR}/cluster_outputs/online_serving_runs/route_policy_compare_s5_${TS}"
LOG="${SWEEP_DIR}/sweep.log"
mkdir -p "${CACHE_DIR}" "${SWEEP_DIR}"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

[ -s "$CSV_PATH" ] || { log "[error] CSV missing"; exit 1; }

PY=$(which python)
RUNNER="${ROOT_DIR}/tools/run_serving_online_policy_compare.py"

run_cell() {
  local label=$1; shift
  local run_dir="${SWEEP_DIR}/${label}"
  local cache_file="${CACHE_DIR}/${MODEL}_${GPU_TYPE}_route_policy_compare_${label}.bin"
  log "── start: ${label}  extra=$* ──"
  log "    cache: $cache_file"
  local t0=$(date +%s)
  "$PY" "$RUNNER" \
      --model "$MODEL" \
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

log "Route-policy comparison at scale=${ARRIVAL_SCALE} (mean offered = 1.09 RPS)"
log "sweep_dir=$SWEEP_DIR"

# A: baseline min_finish (matches the chapter's main comparison cell).
run_cell "A_min_finish"            --route-policy min_finish

# B-D: latency_guarded_energy with increasing guard widths.
run_cell "B_lge_guard5"            --route-policy latency_guarded_energy --energy-latency-guard-ms 5
run_cell "C_lge_guard20"           --route-policy latency_guarded_energy --energy-latency-guard-ms 20
run_cell "D_lge_guard100"          --route-policy latency_guarded_energy --energy-latency-guard-ms 100

log
log "=== route-share + tail-share summary (line-by-line) ==="
for cell in A_min_finish B_lge_guard5 C_lge_guard20 D_lge_guard100; do
  s="${SWEEP_DIR}/${cell}/policy_compare_summary.txt"
  if [ -f "$s" ]; then
    log "--- $cell ---"
    grep -E "Throughput|TTFT p99|E2E p99|Routes" "$s" 2>&1 | tee -a "$LOG"
  fi
done

log
log "After rsync, run tools/analyze_request_tail.py on each cell to"
log "compute the top-50 worst-e2e GPU share."
