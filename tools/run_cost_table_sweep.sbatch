#!/bin/bash
#SBATCH -J cost_table_sweep
#SBATCH -p high_latency
#SBATCH -c 4
#SBATCH --mem=32GB
#SBATCH -t 2-00:00:00
#SBATCH -D /mnt/galactica/ocakir/attacc_simulator
#SBATCH -o /mnt/galactica/ocakir/logs/cost_table_sweep_%j.out
#SBATCH -e /mnt/galactica/ocakir/logs/cost_table_sweep_%j.err
#
# Generate per-route cost tables (s_time, g_time, energy) by sweeping main.py
# over (Lin, Lout, bs). Output goes to cluster_outputs/cost_tables_full_energy_${MODEL,,}/.
#
# Usage on the cluster:
#   sbatch tools/run_cost_table_sweep.sbatch                 # OPENVLA, full Pi0-equivalent sweep
#   MODEL=PI0 sbatch tools/run_cost_table_sweep.sbatch       # regenerate Pi0
#   LIN_VALUES=2,512:6144:1024 sbatch tools/...              # custom sweep
#
# Env-var overrides:
#   MODEL          (default OPENVLA)         — passed to main.py via gen_cost_table.py
#   OUT_DIR        (default cluster_outputs/cost_tables_full_energy_<model>)
#   LIN_VALUES     (default 2,512:6144:512)
#   LOUT_VALUES    (default 2,16:400:16)     — extended past 208 to cover long generations
#   BATCH_VALUES   (default 1,2,4,8,32)      — densified at low bs (was 1,8,32) so the
#                                              cost model can interpolate the bs∈[1,8]
#                                              range that the scheduler actually uses.
#                                              HistGBR is piecewise-constant in bs;
#                                              with only 3 trained points it can't
#                                              interpolate bs=2/3/5/6 properly.
#   ROUTES         (default "gpu_only lpddr5_pim_bank")
#   GPU_TYPE       (default A100a)
#   GMEMCAP        (default 40)              — per-GPU memory cap, GB

set -euo pipefail

ROOT_DIR="/mnt/galactica/ocakir/attacc_simulator"
LOG_DIR="/mnt/galactica/ocakir/logs"

mkdir -p "${LOG_DIR}"
cd "${ROOT_DIR}"

source /mnt/galactica/ocakir/miniforge3/etc/profile.d/conda.sh
conda activate lerobot

echo "hostname:    $(hostname)"
echo "python:      $(python -V 2>&1)"
echo "conda env:   ${CONDA_DEFAULT_ENV:-unknown}"

MODEL="${MODEL:-OPENVLA}"
LIN_VALUES="${LIN_VALUES:-2,512:6144:512}"
LOUT_VALUES="${LOUT_VALUES:-2,16:400:16}"
BATCH_VALUES="${BATCH_VALUES:-1,2,4,8,16,32}"
ROUTES="${ROUTES:-gpu_only lpddr5_pim_bank}"
GPU_TYPE="${GPU_TYPE:-A100a}"
GMEMCAP="${GMEMCAP:-40}"

# Default output dir is model-suffixed lowercase: cost_tables_full_energy_openvla.
MODEL_LC=$(echo "${MODEL}" | tr '[:upper:]' '[:lower:]')
OUT_DIR="${OUT_DIR:-cluster_outputs/cost_tables_full_energy_${MODEL_LC}}"

mkdir -p "${OUT_DIR}"

echo "model:       ${MODEL}"
echo "out_dir:     ${OUT_DIR}"
echo "lin:         ${LIN_VALUES}"
echo "lout:        ${LOUT_VALUES}"
echo "batch:       ${BATCH_VALUES}"
echo "routes:      ${ROUTES}"

# Build --route args; one per route.
ROUTE_ARGS=()
for r in ${ROUTES}; do
  ROUTE_ARGS+=("--route" "${r}")
done

# --resume lets us safely re-run a partial sweep; --sort tidies the CSVs at the end.
python tools/gen_cost_table.py \
  --model "${MODEL}" \
  --gpu "${GPU_TYPE}" \
  --gmemcap "${GMEMCAP}" \
  --lin-values "${LIN_VALUES}" \
  --lout-values "${LOUT_VALUES}" \
  --batch-values "${BATCH_VALUES}" \
  --out-dir "${OUT_DIR}" \
  "${ROUTE_ARGS[@]}" \
  --resume --sort

echo "[done] cost-table sweep finished"
ls -la "${OUT_DIR}"/
