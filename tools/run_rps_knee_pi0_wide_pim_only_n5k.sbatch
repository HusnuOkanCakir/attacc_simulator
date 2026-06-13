#!/bin/bash
#SBATCH -J knee_pim
#SBATCH -p high_latency
#SBATCH -c 4
#SBATCH --mem=32GB
#SBATCH -D /mnt/galactica/ocakir/attacc_simulator
#SBATCH -o /mnt/galactica/ocakir/logs/knee_pim_%j.out
#SBATCH -e /mnt/galactica/ocakir/logs/knee_pim_%j.err
#
# PIM-only RPS knee sweep at n=5000 on azure_poisson_wide.
#
# Mirror of run_rps_knee_pi0_wide_fine_cliff_n5k.sbatch but with
# --pim-only set on the runner. The flag (a) makes choose_route()
# skip the GPU lane → 100% PIM routing at admission, and (b) auto-
# forces kv_oom_policy=hold and kv_max_preempt_tries=-1 so the
# admission / preempt fallback paths cannot route to the disabled
# GPU lane either. End result: zero GPU traffic for the entire trace.
#
# What this sweep is for:
#  - Hybrid (existing) tail-anatomy shows the worst-50 are 100% on the
#    GPU lane at sub-cliff scales (a route-policy artifact of
#    min_finish). The chaos zone at s∈[2.65,2.85] persists under
#    every scheduler-policy ablation we tested, but min_finish still
#    routed ~34/5000 to GPU at s=2.70.
#  - PIM-only run answers: (a) does the chaos zone still exist when
#    no requests escape to GPU?  (b) does the new tail look the same
#    shape, just shifted to PIM, or does it collapse?
#
# Expected differences vs hybrid baseline:
#  - Lower saturation ceiling (hybrid ≈ 2.13 req/s → PIM-only likely
#    ~1.5–1.8 req/s because the ~25% GPU share is now PIM load).
#  - KV pool may bind earlier; if so the chaos zone shifts down.
#  - Some scales that were stable DRAIN under hybrid may now be
#    OVERFLOW because more load sits on PIM.
#
# Scale set: same 24 scales as the n=5k hybrid fine-cliff sweep
# (cover s=10 → 1.5 with fine 0.01-step at 2.65–2.70 and 2.50–2.55)
# PLUS three extra coarse scales (12, 8, 6) above the expected lower
# PIM-only ceiling so the knee detector can find the new cliff.

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

# ── Config ─────────────────────────────────────────────────────────────
N_REQUESTS=5000
MODEL=pi0
GPU_TYPE=a6000
POLICY=max_util_full
KV_POOL_GB=8
MAX_ACTIVE=32
MAX_DECODE_BS=4
MAX_CONSEC_DEC=4
SLO_E2E_MS=500
SLO_TTFT_MS=200
# kv_oom_policy and kv_max_preempt_tries are forced by --pim-only;
# leaving them at the canonical hold_then_fallback / 3 just to keep
# the cell labels comparable in logs — the runner will override and
# print "[pim_only] forcing ..." lines.
KV_OOM_POLICY=hold_then_fallback
KV_OOM_HOLD_LIMIT_MS=50

# 24 hybrid-mirror scales + 3 wider scales for PIM-only cliff search.
ARRIVAL_SCALES="12 10 8 7 6 5 4 3.5 3 2.85 2.75 2.70 2.69 2.68 2.67 2.66 2.65 2.60 2.55 2.54 2.53 2.52 2.51 2.50 2.4 2 1.5"

CSV_PATH="${ROOT_DIR}/cluster_outputs/synthetic_v2/azure_poisson_wide_n5000_seed0.csv"
CACHE_DIR="${ROOT_DIR}/cluster_outputs/pim_cache"
TS=$(date +%Y%m%d_%H%M%S)
SWEEP_DIR="${ROOT_DIR}/cluster_outputs/online_serving_runs/rps_knee_pi0_wide_pim_only_n5k_${TS}"
LOG="${SWEEP_DIR}/sweep.log"

mkdir -p "${CACHE_DIR}" "${SWEEP_DIR}"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

if [ ! -s "$CSV_PATH" ]; then
  log "[error] CSV not found at $CSV_PATH"
  exit 1
fi
log "Reuse existing trace at $CSV_PATH ($(wc -l < "$CSV_PATH") lines)"

# ── Sweep ──────────────────────────────────────────────────────────────
n_scales=$(echo "$ARRIVAL_SCALES" | wc -w)
log ""
log "PIM-only knee sweep  (n=5000, azure_poisson_wide, Pi0)"
log "model=$MODEL  gpu=$GPU_TYPE  policy=$POLICY  route=PIM-ONLY"
log "n=$N_REQUESTS  kv_pool=${KV_POOL_GB}GiB  max_active=$MAX_ACTIVE"
log "bs=$MAX_DECODE_BS  mc=$MAX_CONSEC_DEC"
log "SLO  e2e=${SLO_E2E_MS}ms  ttft=${SLO_TTFT_MS}ms"
log "kv_oom=(forced hold)  kv_max_preempt_tries=(forced -1)"
log "scales=($ARRIVAL_SCALES)  total cells = ${n_scales}"
log "sweep_dir=$SWEEP_DIR"
log ""

PY=$(which python)
RUNNER="${ROOT_DIR}/tools/run_serving_online_policy_compare.py"

i=1
for scale in $ARRIVAL_SCALES; do
  scale_label=$(awk -v x=$scale 'BEGIN{printf "s%04d", x*100}')
  label=$(printf "%02d_azure_poisson_wide_%s" "$i" "$scale_label")
  run_dir="${SWEEP_DIR}/${label}"
  cache_file="${CACHE_DIR}/${MODEL}_${GPU_TYPE}_knee_pi0_wide_pim_only_n5k_${scale_label}.bin"

  log "── start: ${label}  scale=${scale} ──"
  log "    cache: $cache_file"
  t0=$(date +%s)
  "$PY" "$RUNNER" \
      --model "$MODEL" \
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
      --admission-violation-budget -1 \
      --no-prompt-priority \
      --pim-only \
      --csv-path "$CSV_PATH" \
      --pim-cache-file "$cache_file" \
      --no-kv-timeline \
      --run-dir "$run_dir" 2>&1 | tee -a "$LOG"
  dt=$(( $(date +%s) - t0 ))
  log "── done:  ${label}  wall=${dt}s ──"
  i=$((i+1))
done

log ""
log "Sweep complete. Sweep dir: $SWEEP_DIR"
log ""
log "Next steps:"
log "  1) rsync the sweep dir back to the laptop"
log "  2) compare against the hybrid baseline:"
log "     rps_knee_pi0_wide_fine_n5k_20260602_183013  (existing hybrid)"
log "  3) per-cell tail-anatomy comparison:"
log "     python tools/analyze_request_tail.py --run-dir <pim-only-cell>"
log "     vs the same scales from the hybrid sweep."
