#!/bin/bash
#SBATCH -J policy_compare
#SBATCH -p high_latency
#SBATCH -c 4
#SBATCH --mem=32GB
#SBATCH -D /mnt/galactica/ocakir/attacc_simulator
#SBATCH -o /mnt/galactica/ocakir/logs/policy_compare_%j.out
#SBATCH -e /mnt/galactica/ocakir/logs/policy_compare_%j.err

set -euo pipefail

ROOT_DIR="/mnt/galactica/ocakir/attacc_simulator"
LOG_DIR="/mnt/galactica/ocakir/logs"

mkdir -p "${LOG_DIR}"
mkdir -p "${ROOT_DIR}/cluster_outputs/online_serving_runs"

cd "${ROOT_DIR}"

source /mnt/galactica/ocakir/miniforge3/etc/profile.d/conda.sh
conda activate lerobot

echo "hostname: $(hostname)"
echo "python: $(python -V 2>&1)"
echo "conda env: ${CONDA_DEFAULT_ENV:-unknown}"

N_REQUESTS="${N_REQUESTS:-2000}"
ARRIVAL_SCALE="${ARRIVAL_SCALE:-1.0}"
POLICIES="${POLICIES:-guaranteed_no_evict max_util_full max_util_tail}"

echo "n_requests: ${N_REQUESTS}"
echo "arrival_scale: ${ARRIVAL_SCALE}"
echo "policies: ${POLICIES}"

python tools/run_serving_online_policy_compare.py \
  --n-requests "${N_REQUESTS}" \
  --arrival-scale "${ARRIVAL_SCALE}" \
  --policies ${POLICIES}
