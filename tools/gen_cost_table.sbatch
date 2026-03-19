#!/bin/bash
#SBATCH -J gen_cost_small
#SBATCH -p high_latency
#SBATCH -c 8
#SBATCH --mem=16GB
#SBATCH -t 12:00:00
#SBATCH -D /mnt/galactica/ocakir/attacc_simulator
#SBATCH -o /mnt/galactica/ocakir/logs/%x_%j.out
#SBATCH -e /mnt/galactica/ocakir/logs/%x_%j.err

set -euo pipefail

mkdir -p /mnt/galactica/ocakir/logs
mkdir -p /mnt/galactica/ocakir/attacc_simulator/cluster_outputs/cost_tables_full_energy

cd /mnt/galactica/ocakir/attacc_simulator

source /mnt/galactica/ocakir/miniforge3/etc/profile.d/conda.sh
conda activate attacc

echo "Active conda env: ${CONDA_DEFAULT_ENV:-unknown}"
hostname
python -V

# python tools/gen_cost_table.py \
#   --preset azure-wide-energy \
#   --route gpu_only \
#   --route lpddr5_pim_bank \
#   --resume \
#   --sort


python tools/gen_cost_table.py \
  --route gpu_only \
  --route lpddr5_pim_bank \
  --lin-values 2,512:6144:512 \
  --lout-values 2,16:208:16 \
  --batch-values 1,8,32 \
  --resume \
  --sort \
  --out-dir /mnt/galactica/ocakir/attacc_simulator/cluster_outputs/cost_tables_full_energy
