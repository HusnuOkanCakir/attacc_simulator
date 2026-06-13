#!/bin/bash
#SBATCH -J kv_timeline
#SBATCH -p high_latency
#SBATCH -c 4
#SBATCH --mem=32GB
#SBATCH -D /mnt/galactica/ocakir/attacc_simulator
#SBATCH -o /mnt/galactica/ocakir/logs/kv_timeline_%j.out
#SBATCH -e /mnt/galactica/ocakir/logs/kv_timeline_%j.err
#
# Re-runs four hand-picked cells from the kvpool sweep with --debug-log
# enabled so the kv_pool_timeline plotter has trace data to render.
#
# Cells (Pi0 + azure_poisson_wide v2, max_util_full, canonical config):
#   0.5 GiB pool, scale 1.5   (cliff, 5 evicts, 276 preempts)
#   0.5 GiB pool, scale 2.0   (under-cliff, 1 evict, 342 preempts)
#   8   GiB pool, scale 1.5   (over-provisioned baseline at cliff)
#   8   GiB pool, scale 2.0   (over-provisioned baseline under-cliff)
#
# Run:  sbatch tools/run_kv_pool_timeline_cells.sbatch
# After completion, rsync the run-dirs to local:
#   rsync -av \
#     'ocakir@safari-proxy:/mnt/galactica/ocakir/.../kv_timeline_*' \
#     /home/okan/attacc_simulator/cluster_outputs/online_serving_runs/

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

CSV="${ROOT_DIR}/cluster_outputs/synthetic_v2/azure_poisson_wide_n5000_seed0.csv"
CACHE_DIR="${ROOT_DIR}/cluster_outputs/pim_cache"
RUN_ROOT="${ROOT_DIR}/cluster_outputs/online_serving_runs"

# (label, pool_gb, arrival_scale)
CELLS=(
  "kvpool_0_5_s015  0.5  1.5"
  "kvpool_0_5_s020  0.5  2"
  "kvpool_8_s015    8    1.5"
  "kvpool_8_s020    8    2"
)

run_one() {
  local label=$1 pool=$2 scale=$3
  local out="${RUN_ROOT}/kv_timeline_${label}"
  local cache="${CACHE_DIR}/pi0_a6000_knee_vs_kvpool_${pool}_azure_poisson_wide.bin"

  echo "[$(date +%H:%M:%S)] ── start: ${label}  pool=${pool}GB  scale=${scale} ──"
  local t0=$(date +%s)
  python tools/run_serving_online_policy_compare.py \
      --model pi0 --gpu a6000 \
      --n-requests 5000 \
      --policies max_util_full \
      --kv-pool-gb "${pool}" --ml \
      --max-active 32 \
      --arrival-scale "${scale}" \
      --slo-e2e-ms 500 --slo-ttft-ms 200 \
      --kv-oom-policy hold_then_fallback \
      --max-decode-batch-size 4 \
      --max-consecutive-decode-batches 4 \
      --admission-violation-budget -1 \
      --no-prompt-priority \
      --debug-log \
      --csv-path "${CSV}" \
      --pim-cache-file "${cache}" \
      --no-kv-timeline \
      --run-dir "${out}"
  local dt=$(( $(date +%s) - t0 ))
  echo "[$(date +%H:%M:%S)] ── done:  ${label}  wall=${dt}s ──"
}

for spec in "${CELLS[@]}"; do
  # shellcheck disable=SC2086
  run_one $spec
done

echo "all done"
