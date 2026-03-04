#!/bin/bash
#SBATCH -J v31_tune
#SBATCH -p high_latency
#SBATCH -c 8
#SBATCH --mem=16GB
#SBATCH -t 12:00:00
#SBATCH -D /mnt/galactica/ocakir/attacc_simulator
#SBATCH -o /mnt/galactica/ocakir/logs/%x_%j.out
#SBATCH -e /mnt/galactica/ocakir/logs/%x_%j.err

set -euo pipefail

mkdir -p /mnt/galactica/ocakir/logs
mkdir -p /mnt/galactica/ocakir/attacc_simulator/cluster_outputs/v31_tuning_sweep
mkdir -p /mnt/galactica/ocakir/attacc_simulator/cluster_outputs/trace_cache

cd /mnt/galactica/ocakir/attacc_simulator

source /mnt/galactica/ocakir/miniforge3/etc/profile.d/conda.sh
conda activate attacc

echo "Active conda env: ${CONDA_DEFAULT_ENV:-unknown}"
hostname
python -V

TRACE_CACHE="/mnt/galactica/ocakir/attacc_simulator/cluster_outputs/trace_cache/AzureLLMInferenceTrace_code.csv"
if [[ -f "${TRACE_CACHE}" ]]; then
  export AZURE_TRACE="${TRACE_CACHE}"
  echo "Using cached Azure trace: ${AZURE_TRACE}"
fi

bash tools/run_v31_tuning_sweep.sh
