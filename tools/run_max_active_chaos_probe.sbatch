#!/bin/bash
#SBATCH -J max_act
#SBATCH -p high_latency
#SBATCH -c 4
#SBATCH --mem=48GB
#SBATCH --exclude=safari-nexus1
#SBATCH -D /mnt/galactica/ocakir/attacc_simulator
#SBATCH -o /mnt/galactica/ocakir/logs/max_act_%j.out
#SBATCH -e /mnt/galactica/ocakir/logs/max_act_%j.err
#
# Probe: is the non-monotonic chaos-zone bistability driven by the
# max_active=32 constraint?
#
# Hypothesis (§6.6 in the thesis): once the queue at the cliff edge
# exceeds max_active=32, the active set is filled FIFO by long-context
# arrivals from the early burst, the bs=4 decode cohort processes
# long-context requests, per-request throughput drops below the offered
# rate, and the queue grows linearly thereafter — locking the system
# into the ≈ 0.8 RPS overflow attractor.
#
# Direct test: re-run the drain cell (s=2.69) and the overflow cell
# (s=2.70) at max_active ∈ {32, 100, 200, 500}. KV pool sized so it
# doesn't fill (Pi0 wide-Poisson avg ≈ 27 MiB/req, so 500 active ≈ 14
# GiB; we use 16 GiB to leave headroom). Plus one extreme cell at
# max_active=2000 with a 32 GiB pool — likely fills, allows GPU
# fallback, but tells us whether recovery is possible in principle.
#
# Expected outcomes:
#   (1) If max_active=32 is the bistability cause → at max_active ≥ 100
#       both s=2.69 and s=2.70 produce the same outcome (likely drain).
#   (2) If the lock-in is intrinsic → bistability persists across all
#       max_active values: s=2.69 keeps achieving ≈ 2.0 RPS and s=2.70
#       keeps achieving ≈ 0.8 RPS regardless of max_active.
#   (3) If a specific max_active value flips s=2.70 → that's the
#       threshold worth identifying.
#
# Each cell is n=5000, Pi0 + azure_poisson_wide. Identical to the
# fine-cliff sweep cells except for max_active and kv_pool.

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

# ── Common config ──────────────────────────────────────────────────────
N_REQUESTS=5000
MODEL=pi0
GPU_TYPE=a6000
POLICY=max_util_full
MAX_DECODE_BS=4
MAX_CONSEC_DEC=4
SLO_E2E_MS=500
SLO_TTFT_MS=200
KV_OOM_POLICY=hold_then_fallback
KV_OOM_HOLD_LIMIT_MS=50

CSV_PATH="${ROOT_DIR}/cluster_outputs/synthetic_v2/azure_poisson_wide_n5000_seed0.csv"
CACHE_DIR="${ROOT_DIR}/cluster_outputs/pim_cache"
TS=$(date +%Y%m%d_%H%M%S)
SWEEP_DIR="${ROOT_DIR}/cluster_outputs/online_serving_runs/max_active_chaos_probe_${TS}"
LOG="${SWEEP_DIR}/sweep.log"
mkdir -p "${CACHE_DIR}" "${SWEEP_DIR}"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

[ -s "$CSV_PATH" ] || { log "[error] CSV missing"; exit 1; }

PY=$(which python)
RUNNER="${ROOT_DIR}/tools/run_serving_online_policy_compare.py"

# scale, max_active, kv_pool_gb, label_suffix
CELLS=(
  "2.69  32   8   drain_baseline32"
  "2.69  100  16  drain_a100"
  "2.69  200  16  drain_a200"
  "2.69  500  16  drain_a500"
  "2.70  32   8   overflow_baseline32"
  "2.70  100  16  overflow_a100"
  "2.70  200  16  overflow_a200"
  "2.70  500  16  overflow_a500"
  "2.70  2000 32  overflow_a2000"
)

run_cell() {
  local scale=$1; local max_active=$2; local kv_pool_gb=$3; local suffix=$4
  local label="$(awk -v x=$scale 'BEGIN{printf "s%04d", x*100}')_${suffix}"
  local run_dir="${SWEEP_DIR}/${label}"
  local cache_file="${CACHE_DIR}/${MODEL}_${GPU_TYPE}_max_active_probe_${label}.bin"
  log "── start: ${label}  scale=${scale}  max_active=${max_active}  kv_pool=${kv_pool_gb}GiB ──"
  local t0=$(date +%s)
  "$PY" "$RUNNER" \
      --model "$MODEL" \
      --gpu "$GPU_TYPE" \
      --n-requests "$N_REQUESTS" \
      --policies "$POLICY" \
      --kv-pool-gb "$kv_pool_gb" --ml \
      --max-active "$max_active" \
      --arrival-scale "$scale" \
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
      --run-dir "$run_dir" 2>&1 | tee -a "$LOG"
  local dt=$(( $(date +%s) - t0 ))
  log "── done:  ${label}  wall=${dt}s ──"
}

log "max_active chaos probe: $(echo "${CELLS[@]}" | wc -w | awk '{print int($1/4)}') cells"
log "sweep_dir=$SWEEP_DIR"

for spec in "${CELLS[@]}"; do
  # shellcheck disable=SC2086
  run_cell $spec
done

log
log "=== summary (achieved RPS + routes + KV-OOM holds) ==="
for d in "$SWEEP_DIR"/s*; do
  s="${d}/policy_compare_summary.txt"
  if [ -f "$s" ]; then
    log "--- $(basename "$d") ---"
    grep -E "Throughput|Routes|KV-OOM holds|Span" "$s" 2>&1 | tee -a "$LOG"
  fi
done
log
log "After rsync, plot achieved RPS vs max_active per scale for the side-by-side comparison."
