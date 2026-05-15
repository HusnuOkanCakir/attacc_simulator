#!/bin/bash
#SBATCH -J route_sweep
#SBATCH -p high_latency
#SBATCH -c 4
#SBATCH --mem=32GB
#SBATCH -D /mnt/galactica/ocakir/attacc_simulator
#SBATCH -o /mnt/galactica/ocakir/logs/route_sweep_%j.out
#SBATCH -e /mnt/galactica/ocakir/logs/route_sweep_%j.err
#
# Experiment E — Route-policy / energy-aware routing sweep.
#
# Compares `min_finish` (latency-only) against `latency_guarded_energy` with
# different latency guards. Under latency_guarded_energy, routes within
# GUARD_MS of the fastest predicted finish are considered latency-equivalent
# and the lowest-energy route wins. Larger guard -> more PIM (lower DRAM
# energy) at the cost of some tail latency.
#
# Cache-warmup mechanics are identical to run_pim_cache_warmup_sweep.sbatch:
# the first run primes cluster_outputs/pim_cache/<model>_lpddr5.bin; every
# subsequent run reuses it.
#
# Usage:
#   sbatch tools/run_route_policy_sweep.sbatch
#
# Env-var overrides:
#   MODEL              (default openvla)
#   N_REQUESTS         (default 2000)
#   KV_POOL_GB         (default 4)
#   MAX_ACTIVE         (default 32)
#   ARRIVAL_SCALE      (default 10.0)   — light load; energy decisions only
#                                         matter when SLO eligibility passes
#   KV_OOM_POLICY      (default hold_then_fallback)
#   KV_OOM_HOLD_LIMIT_MS (default 50)
#   SLO_E2E_MS         (default 500)    — single SLO (use _value sweep G for SLO axis)
#   SLO_TTFT_MS        (default 200)
#   POLICY             (default max_util_full)   — single KV policy
#   GUARDS             (default "20 50 100")     — guard_ms values for
#                                                  latency_guarded_energy
#   INCLUDE_MIN_FINISH (default 1)    — also run baseline route_policy=min_finish
#   SKIP_PRIMING       (default 0)
#
# Monitor:
#   squeue -u $USER
#   tail -f /mnt/galactica/ocakir/logs/route_sweep_<jobid>.out

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
ARRIVAL_SCALE="${ARRIVAL_SCALE:-10.0}"
KV_OOM_POLICY="${KV_OOM_POLICY:-hold_then_fallback}"
KV_OOM_HOLD_LIMIT_MS="${KV_OOM_HOLD_LIMIT_MS:-50}"
SLO_E2E_MS="${SLO_E2E_MS:-500}"
SLO_TTFT_MS="${SLO_TTFT_MS:-200}"
POLICY="${POLICY:-max_util_full}"
GUARDS="${GUARDS:-20 50 100}"
INCLUDE_MIN_FINISH="${INCLUDE_MIN_FINISH:-1}"
SKIP_PRIMING="${SKIP_PRIMING:-0}"

PY=$(which python)
RUNNER="${ROOT_DIR}/tools/run_serving_online_policy_compare.py"
CACHE_DIR="${ROOT_DIR}/cluster_outputs/pim_cache"
CACHE_FILE="${CACHE_DIR}/${MODEL}_lpddr5.bin"
TS=$(date +%Y%m%d_%H%M%S)
SWEEP_DIR="${ROOT_DIR}/cluster_outputs/online_serving_runs/route_sweep_${MODEL}_n${N_REQUESTS}_${TS}"
LOG="${SWEEP_DIR}/sweep.log"

mkdir -p "${CACHE_DIR}" "${SWEEP_DIR}"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "model=$MODEL  n=$N_REQUESTS  kv_pool=${KV_POOL_GB}GiB  max_active=$MAX_ACTIVE"
log "arrival_scale=$ARRIVAL_SCALE  slo_e2e=${SLO_E2E_MS}ms  slo_ttft=${SLO_TTFT_MS}ms"
log "policy=$POLICY  guards=($GUARDS)  include_min_finish=$INCLUDE_MIN_FINISH"
log "cache_file=$CACHE_FILE"
log "sweep_dir=$SWEEP_DIR"

run_one() {
  local label=$1
  local route_policy=$2
  local guard_ms=$3
  local run_dir="${SWEEP_DIR}/${label}"
  log "── start: $label  route_policy=$route_policy  guard=${guard_ms}ms ──"
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
      --route-policy "$route_policy" \
      --energy-latency-guard-ms "$guard_ms" \
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
  log "[prime] starting cold priming run under guaranteed_no_evict + SLO=1000ms"
  run_one "00_prime" min_finish 20
  log "[prime] cache populated; entries=$(grep -cv '^#' "$CACHE_FILE" || echo 0)"
fi

# ── Stage 2: sweep ───────────────────────────────────────────────────────────
i=1
if [ "$INCLUDE_MIN_FINISH" = "1" ]; then
  label=$(printf "%02d_min_finish" "$i")
  run_one "$label" min_finish 0
  i=$((i+1))
fi
for guard in $GUARDS; do
  label=$(printf "%02d_energy_guard%s" "$i" "$guard")
  run_one "$label" latency_guarded_energy "$guard"
  i=$((i+1))
done

# ── Summary ──────────────────────────────────────────────────────────────────
log "=========================================================================="
log "ALL DONE"
log "  cache:     $CACHE_FILE  ($(grep -cv '^#' "$CACHE_FILE" 2>/dev/null || echo 0) entries)"
log "  sweep_dir: $SWEEP_DIR"
log "  log:       $LOG"
log "=========================================================================="
log ""
log "Expected: as guard widens (20 -> 50 -> 100 ms), PIM route share rises"
log "and total energy drops; tail latency may rise slightly."
