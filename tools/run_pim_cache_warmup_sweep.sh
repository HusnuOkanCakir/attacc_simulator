#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Prime the per-shape PIM cycle cache once, then sweep across SLO × KV-policy
# variants reusing the warm cache.
#
# Usage:
#   tools/run_pim_cache_warmup_sweep.sh
#
# Override any of these via env, e.g.:
#   MODEL=pi0 N_REQUESTS=200 ./tools/run_pim_cache_warmup_sweep.sh
#   SKIP_PRIMING=1 ./tools/run_pim_cache_warmup_sweep.sh         # cache file exists
#   SLOS="500 1000" POLICIES="max_util_tail" ./tools/run_pim_cache_warmup_sweep.sh
#
# While it runs you can watch progress via the per-run status.txt:
#   watch -n 2 cat <sweep_dir>/<sub_run>/<policy>/status.txt
# Path is printed below as soon as each sub-run starts.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Config (override via env) ────────────────────────────────────────────────
MODEL=${MODEL:-openvla}
N_REQUESTS=${N_REQUESTS:-2000}
KV_POOL_GB=${KV_POOL_GB:-4}
MAX_ACTIVE=${MAX_ACTIVE:-32}
ARRIVAL_SCALE=${ARRIVAL_SCALE:-1.0}
KV_OOM_POLICY=${KV_OOM_POLICY:-hold_then_fallback}
KV_OOM_HOLD_LIMIT_MS=${KV_OOM_HOLD_LIMIT_MS:-50}

# Sweep matrix (space-separated). Edit freely.
SLOS=${SLOS:-"500 1000 2000"}
POLICIES=${POLICIES:-"guaranteed_no_evict max_util_full max_util_tail"}

# Skip the priming step if the cache file already has entries.
SKIP_PRIMING=${SKIP_PRIMING:-0}

# Paths default to the local dev workstation. Override REPO and PY in the
# environment (e.g. from a SLURM wrapper) to run this same script on the
# cluster against /mnt/galactica/$USER/attacc_simulator + the conda lerobot
# env's python.
REPO=${REPO:-/home/okan/attacc_simulator}
PY=${PY:-/home/okan/anaconda3/envs/lerobot/bin/python}
RUNNER=$REPO/tools/run_serving_online_policy_compare.py
CACHE_DIR=$REPO/cluster_outputs/pim_cache
CACHE_FILE=$CACHE_DIR/${MODEL}_lpddr5.bin
TS=$(date +%Y%m%d_%H%M%S)
SWEEP_DIR=$REPO/cluster_outputs/online_serving_runs/sweep_${MODEL}_n${N_REQUESTS}_${TS}
LOG=$SWEEP_DIR/sweep.log

mkdir -p "$CACHE_DIR" "$SWEEP_DIR"
cd "$REPO"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "model=$MODEL  n=$N_REQUESTS  kv_pool=${KV_POOL_GB}GiB  max_active=$MAX_ACTIVE  arrival_scale=$ARRIVAL_SCALE"
log "cache_file=$CACHE_FILE"
log "sweep_dir=$SWEEP_DIR"
log "slos=($SLOS)  policies=($POLICIES)"

# ── Helper: run one configuration ────────────────────────────────────────────
# args: <label> <slo_e2e_ms> <policy> <run_dir>
run_one() {
  local label=$1
  local slo=$2
  local policy=$3
  local run_dir=$4
  log "── start: $label  ($run_dir) ──"
  log "    watch live:  watch -n 2 cat $run_dir/$policy/status.txt"
  local t0=$(date +%s)
  "$PY" "$RUNNER" \
      --model "$MODEL" \
      --n-requests "$N_REQUESTS" \
      --policies "$policy" \
      --kv-pool-gb "$KV_POOL_GB" --ml \
      --max-active "$MAX_ACTIVE" \
      --arrival-scale "$ARRIVAL_SCALE" \
      --slo-e2e-ms "$slo" --slo-ttft-ms 200 \
      --kv-oom-policy "$KV_OOM_POLICY" \
      --kv-oom-hold-limit-ms "$KV_OOM_HOLD_LIMIT_MS" \
      --debug-log \
      --pim-cache-file "$CACHE_FILE" \
      --run-dir "$run_dir" 2>&1 | tee -a "$LOG"
  local dt=$(( $(date +%s) - t0 ))
  log "── done:  $label  wall=${dt}s ──"
}

# ── Stage 1: prime ───────────────────────────────────────────────────────────
if [ -s "$CACHE_FILE" ] && [ "$SKIP_PRIMING" = "1" ]; then
  log "[prime] SKIPPED — cache exists at $CACHE_FILE (set SKIP_PRIMING=0 to re-prime)"
  log "[prime] cache file size: $(wc -l < "$CACHE_FILE") lines"
else
  log "[prime] starting cold priming run under guaranteed_no_evict + SLO=1000ms"
  log "[prime] this is the slow run; everything after reuses its cache"
  run_one "00_prime" 1000 guaranteed_no_evict "$SWEEP_DIR/00_prime"
  log "[prime] cache populated; entries=$(grep -cv '^#' "$CACHE_FILE" || echo 0)"
fi

# ── Stage 2: sweep with warm cache ───────────────────────────────────────────
i=1
for slo in $SLOS; do
  for policy in $POLICIES; do
    label=$(printf "%02d_slo%s_%s" "$i" "$slo" "$policy")
    run_one "$label" "$slo" "$policy" "$SWEEP_DIR/$label"
    i=$((i+1))
  done
done

# ── Summary ──────────────────────────────────────────────────────────────────
log "=========================================================================="
log "ALL DONE"
log "  cache:      $CACHE_FILE  ($(grep -cv '^#' "$CACHE_FILE" 2>/dev/null || echo 0) entries)"
log "  sweep_dir:  $SWEEP_DIR"
log "  log:        $LOG"
log "  artifacts per sub-run: <sub_run>/<policy>/{requests_out.csv,debug.log,status.txt,ramulator_summary.yaml}"
log "=========================================================================="
log ""
log "Quick cache-correctness check (optional, on one variant):"
log "  $PY $RUNNER --model $MODEL --n-requests $N_REQUESTS \\"
log "      --policies guaranteed_no_evict --kv-pool-gb $KV_POOL_GB --ml \\"
log "      --max-active $MAX_ACTIVE --arrival-scale $ARRIVAL_SCALE \\"
log "      --slo-e2e-ms 1000 --slo-ttft-ms 200 \\"
log "      --no-pim-cache --debug-log \\"
log "      --run-dir $SWEEP_DIR/zz_paranoia_uncached"
log "  diff <(cut -d, -f1-12 $SWEEP_DIR/01_slo1000_guaranteed_no_evict/guaranteed_no_evict/requests_out.csv) \\"
log "       <(cut -d, -f1-12 $SWEEP_DIR/zz_paranoia_uncached/guaranteed_no_evict/requests_out.csv)"
