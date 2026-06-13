#!/bin/bash
#SBATCH -J chaos_abl
#SBATCH -p high_latency
#SBATCH -c 4
#SBATCH --mem=32GB
#SBATCH -D /mnt/galactica/ocakir/attacc_simulator
#SBATCH -o /mnt/galactica/ocakir/logs/chaos_abl_%j.out
#SBATCH -e /mnt/galactica/ocakir/logs/chaos_abl_%j.err
#
# Phase 3 ablation: do GPU-fallback / routing-policy knobs explain the
# chaotic switching at s=2.70?
#
# Phase 2 found that the e2e tail in EVERY regime is GPU-route-dominated
# (GPU share in top-50 worst e2e is 4-100× higher than overall). The
# question now: if we change the GPU-routing paths, does the chaos at
# s=2.70 (which collapsed to OVERFLOW under the canonical config) shift?
#
# Cells (all at n=5000, scale=2.70, Pi0, max_util_full, KV=8 GiB,
# matching the original fine-cliff s=2.70 cell):
#
#   01_baseline:
#       canonical config — reproduces the original collapsed cell.
#       Acts as control / verification of determinism.
#
#   02_no_admit_fallback:
#       --kv-oom-policy hold  (was hold_then_fallback)
#       Disables the 50-ms-hold-then-GPU admission fallback.
#       If chaos = admission-fallback driven, behavior should clearly
#       diverge from baseline. Otherwise should match (since the
#       canonical kv_oom_holds was 0 at s=2.70).
#
#   03_no_preempt_reroute:
#       --kv-max-preempt-tries -1  (was 3)
#       Disables the per-request preempt-budget GPU reroute (the new
#       vLLM-style livelock fix). At s=2.70 the canonical preempt
#       count was 0, so this is expected to be a no-op IF preempts
#       aren't firing — confirms or refutes that.
#
#   04_alt_route_policy:
#       --route-policy latency_guarded_energy
#       Replaces min_finish (the policy that biases tails onto GPU
#       per the Phase 2 finding) with the energy-guarded alternative.
#       Most likely to actually move the behavior since min_finish
#       is implicated in the diagnostic.
#
#   05_no_fallback_no_reroute:
#       --kv-oom-policy hold AND --kv-max-preempt-tries -1
#       Combined disable of both GPU-fallback paths. If chaos disappears
#       only under this combination, both paths are jointly implicated.
#
# After the sweep, run:
#   python tools/analyze_request_tail.py --run-dir <cell> ...
# on each new cell to compare to the original baseline's tail anatomy.

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

# ── Common config (matches original s=2.70 fine-cliff cell) ────────────
N_REQUESTS=5000
MODEL=pi0
GPU_TYPE=a6000
POLICY=max_util_full
KV_POOL_GB=8
MAX_ACTIVE=32
ARRIVAL_SCALE=2.70
MAX_DECODE_BS=4
MAX_CONSEC_DEC=4
SLO_E2E_MS=500
SLO_TTFT_MS=200

CSV_PATH="${ROOT_DIR}/cluster_outputs/synthetic_v2/azure_poisson_wide_n5000_seed0.csv"
CACHE_DIR="${ROOT_DIR}/cluster_outputs/pim_cache"
TS=$(date +%Y%m%d_%H%M%S)
SWEEP_DIR="${ROOT_DIR}/cluster_outputs/online_serving_runs/chaos_abl_n5k_s270_${TS}"
LOG="${SWEEP_DIR}/sweep.log"

mkdir -p "${CACHE_DIR}" "${SWEEP_DIR}"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

if [ ! -s "$CSV_PATH" ]; then
  log "[error] CSV not found at $CSV_PATH"
  exit 1
fi

PY=$(which python)
RUNNER="${ROOT_DIR}/tools/run_serving_online_policy_compare.py"

# Per-cell wrapper. Accepts label + EXTRA_FLAGS.
run_cell() {
  local label=$1
  shift
  local extra=("$@")
  local run_dir="${SWEEP_DIR}/${label}"
  local cache_file="${CACHE_DIR}/${MODEL}_${GPU_TYPE}_chaos_abl_${label}.bin"

  log "── start: ${label}  extra=[${extra[*]}] ──"
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
      --max-decode-batch-size "$MAX_DECODE_BS" \
      --max-consecutive-decode-batches "$MAX_CONSEC_DEC" \
      --admission-violation-budget -1 \
      --no-prompt-priority \
      --csv-path "$CSV_PATH" \
      --pim-cache-file "$cache_file" \
      --no-kv-timeline \
      --run-dir "$run_dir" \
      "${extra[@]}" 2>&1 | tee -a "$LOG"
  local dt=$(( $(date +%s) - t0 ))
  log "── done:  ${label}  wall=${dt}s ──"
}

log ""
log "Chaos-mechanism ablation at scale=${ARRIVAL_SCALE} (n=${N_REQUESTS})"
log "sweep_dir=$SWEEP_DIR"
log ""

# Cell 01: canonical baseline
run_cell "01_baseline" \
  --kv-oom-policy hold_then_fallback \
  --kv-oom-hold-limit-ms 50 \
  --kv-max-preempt-tries 3 \
  --route-policy min_finish

# Cell 02: no admit-time fallback
run_cell "02_no_admit_fallback" \
  --kv-oom-policy hold \
  --kv-max-preempt-tries 3 \
  --route-policy min_finish

# Cell 03: no preempt-budget reroute
run_cell "03_no_preempt_reroute" \
  --kv-oom-policy hold_then_fallback \
  --kv-oom-hold-limit-ms 50 \
  --kv-max-preempt-tries -1 \
  --route-policy min_finish

# Cell 04: alternative route policy
run_cell "04_alt_route_policy" \
  --kv-oom-policy hold_then_fallback \
  --kv-oom-hold-limit-ms 50 \
  --kv-max-preempt-tries 3 \
  --route-policy latency_guarded_energy

# Cell 05: combined disable both GPU fallback paths
run_cell "05_no_fallback_no_reroute" \
  --kv-oom-policy hold \
  --kv-max-preempt-tries -1 \
  --route-policy min_finish

log ""
log "Ablation sweep complete. Sweep dir: $SWEEP_DIR"
log ""
log "Next: rsync to laptop and run per-cell tail-anatomy analysis."
log "  for c in 01_baseline 02_no_admit_fallback 03_no_preempt_reroute 04_alt_route_policy 05_no_fallback_no_reroute; do"
log "    python tools/analyze_request_tail.py --run-dir ${SWEEP_DIR}/\$c"
log "  done"
