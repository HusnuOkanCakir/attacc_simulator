#!/bin/bash
#SBATCH -J smoke_pim
#SBATCH -p low_latency
#SBATCH -c 4
#SBATCH --mem=16GB
#SBATCH -t 02:00:00
#SBATCH -D /mnt/galactica/ocakir/attacc_simulator
#SBATCH -o /mnt/galactica/ocakir/logs/smoke_pim_%j.out
#SBATCH -e /mnt/galactica/ocakir/logs/smoke_pim_%j.err
#
# Smoke test: run 2 quick Pi0 cells on the cluster to verify the runner +
# libramulator build is healthy *before* re-launching the big sweeps that
# have been hanging.
#
# Cell A — hybrid baseline (default config). If this hangs, the cluster
#          build is broken — not specific to the new --pim-only flag.
# Cell B — --pim-only. If A finishes but B hangs, the --pim-only flag is
#          the cause and we need to debug that specifically.
#
# Each cell uses:
#   - n=1000 (5× smaller than the big sweep), scale=10 (deep sub-cliff,
#     fast to drain), --no-kv-timeline (avoid extra plotting overhead),
#     --no-prompt-priority (canonical).
#   - 1-hour SLURM time limit per cell × 2 cells ≤ 2 h total. Pi0 + n=1000
#     locally finishes in well under a minute, so anything over ~10 min
#     wall is already evidence of a hang.

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
echo "runner SHA / mtime:"
ls -l "${ROOT_DIR}/tools/run_serving_online_policy_compare.py"
sha256sum "${ROOT_DIR}/tools/run_serving_online_policy_compare.py"

TS=$(date +%Y%m%d_%H%M%S)
SMOKE_DIR="${ROOT_DIR}/cluster_outputs/online_serving_runs/smoke_pim_${TS}"
mkdir -p "${SMOKE_DIR}"

CSV_PATH="${ROOT_DIR}/cluster_outputs/synthetic_v2/azure_poisson_wide_n5000_seed0.csv"
[ -s "$CSV_PATH" ] || { echo "[error] CSV not found at $CSV_PATH"; exit 1; }

PY=$(which python)
RUNNER="${ROOT_DIR}/tools/run_serving_online_policy_compare.py"

run_cell() {
  local label=$1; shift
  local run_dir="${SMOKE_DIR}/${label}"
  echo
  echo "============================================================"
  echo "[$(date +%H:%M:%S)] ── start: ${label} ──"
  echo "extra flags: $*"
  echo "============================================================"
  local t0=$(date +%s)
  timeout 3600 "$PY" "$RUNNER" \
      --model pi0 \
      --gpu a6000 \
      --n-requests 1000 \
      --policies max_util_full \
      --kv-pool-gb 8 --ml \
      --max-active 32 \
      --arrival-scale 10 \
      --slo-e2e-ms 500 --slo-ttft-ms 200 \
      --max-decode-batch-size 4 \
      --max-consecutive-decode-batches 4 \
      --admission-violation-budget -1 \
      --no-prompt-priority \
      --csv-path "$CSV_PATH" \
      --no-kv-timeline \
      --run-dir "$run_dir" \
      "$@"
  local rc=$?
  local dt=$(( $(date +%s) - t0 ))
  echo "[$(date +%H:%M:%S)] ── done:  ${label}  rc=${rc}  wall=${dt}s ──"
  return $rc
}

echo
echo "###################################################################"
echo "# Cell A: hybrid baseline (no --pim-only)"
echo "###################################################################"
run_cell "A_hybrid_baseline" || { echo "[FAIL] cell A hung or errored — runner/build is broken; cell B skipped"; exit 1; }

echo
echo "###################################################################"
echo "# Cell B: --pim-only (the new flag we added yesterday)"
echo "###################################################################"
run_cell "B_pim_only" --pim-only || { echo "[FAIL] cell B hung or errored — --pim-only specifically is broken"; exit 1; }

echo
echo "###################################################################"
echo "# Smoke test PASSED — both cells finished."
echo "# Safe to relaunch the big sweeps."
echo "###################################################################"
echo "smoke_dir=$SMOKE_DIR"
echo
echo "Quick route-distribution check:"
for cell in "${SMOKE_DIR}"/A_hybrid_baseline "${SMOKE_DIR}"/B_pim_only; do
  csv="${cell}/max_util_full/requests_out.csv"
  [ -f "$csv" ] || { echo "  $(basename $cell): NO requests_out.csv"; continue; }
  awk -F, 'NR==1{for(i=1;i<=NF;i++) if($i=="route") col=i} NR>1{cnt[$col]++} END{for(k in cnt) print "  " k ": " cnt[k]}' "$csv" | sed "s|^|$(basename $cell): |"
done
