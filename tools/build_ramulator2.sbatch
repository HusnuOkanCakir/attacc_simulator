#!/bin/bash
#SBATCH -J ramulator2_build
#SBATCH -p high_latency
#SBATCH -c 16
#SBATCH --mem=16GB
#SBATCH --time=00:45:00
#SBATCH -D /mnt/galactica/ocakir/attacc_simulator
#SBATCH -o /mnt/galactica/ocakir/logs/ramulator2_build_%j.out
#SBATCH -e /mnt/galactica/ocakir/logs/ramulator2_build_%j.err
#
# Build ramulator2 on a SAFARI cluster compute node.
#
# Do not run cmake --build on safari-proxy — the cluster guide says not to
# run compute-intensive tasks on login nodes.
#
# Usage:
#   sbatch tools/build_ramulator2.sbatch
#
# Monitor:
#   squeue -u $USER
#   tail -f /mnt/galactica/ocakir/logs/ramulator2_build_<jobid>.out

set -euo pipefail

ROOT_DIR="/mnt/galactica/ocakir/attacc_simulator"
LOG_DIR="/mnt/galactica/ocakir/logs"

mkdir -p "${LOG_DIR}"

# Redirect GCC temp files away from /tmp (frequently full on cluster nodes).
export TMPDIR=/mnt/galactica/ocakir/tmp
mkdir -p "$TMPDIR"

echo "hostname: $(hostname)"
echo "date:     $(date)"
echo "cwd:      $(pwd)"
echo "cpus:     ${SLURM_CPUS_PER_TASK:-?}"
echo "TMPDIR:   ${TMPDIR}  (free: $(df -h "${TMPDIR}" | awk 'NR==2{print $4}'))"

cd "${ROOT_DIR}/ramulator2"

# Configure only if not already done.
if [ ! -d build ] || [ ! -f build/CMakeCache.txt ]; then
  echo "[build] configuring with CMake (Release)"
  cmake -B build -DCMAKE_BUILD_TYPE=Release .
fi

echo "[build] starting incremental build with -j${SLURM_CPUS_PER_TASK:-8}"
cmake --build build -j"${SLURM_CPUS_PER_TASK:-8}"

echo "[build] artifacts:"
ls -la libramulator.so ramulator2 2>/dev/null || true

echo "[build] done at $(date)"
