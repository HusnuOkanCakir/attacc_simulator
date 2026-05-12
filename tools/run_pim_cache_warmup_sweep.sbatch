#!/bin/bash
#SBATCH -J pim_sweep
#SBATCH -p high_latency
#SBATCH -c 4
#SBATCH --mem=32GB
#SBATCH -D /mnt/galactica/ocakir/attacc_simulator
#SBATCH -o /mnt/galactica/ocakir/logs/pim_sweep_%j.out
#SBATCH -e /mnt/galactica/ocakir/logs/pim_sweep_%j.err
#
# Prime the per-shape PIM cycle cache once, then sweep SLO x KV-policy
# variants reusing the warm cache.
#
# Stage 1 (prime): one run under guaranteed_no_evict + SLO=1000ms.
#   Every distinct (route, ctx_bucket, batch_size) shape hits Ramulator once
#   and the result is saved to cluster_outputs/pim_cache/<model>_lpddr5.bin.
#
# Stage 2 (sweep): SLOS x POLICIES runs, all reusing the warm cache.
#   Cache hits skip Ramulator entirely -> minutes per run instead of hours.
#
# Usage:
#   sbatch tools/run_pim_cache_warmup_sweep.sbatch
#
# Env-var overrides (pass before sbatch or via --export=ALL,KEY=val):
#   MODEL            (default openvla)   — openvla or pi0
#   N_REQUESTS       (default 2000)      — requests per run
#   KV_POOL_GB       (default 4)         — PIM KV pool in GiB
#   MAX_ACTIVE       (default 32)        — max concurrent requests
#   ARRIVAL_SCALE    (default 1.0)       — 1.0 = real Azure timing
#   KV_OOM_POLICY    (default hold_then_fallback)
#   KV_OOM_HOLD_LIMIT_MS (default 50)
#   SLOS             (default "500 1000 2000")   — space-separated
#   POLICIES         (default "guaranteed_no_evict max_util_full max_util_tail")
#   SKIP_PRIMING     (default 0)         — set 1 if cache file already exists
#
# Monitor:
#   squeue -u $USER
#   tail -f /mnt/galactica/ocakir/logs/pim_sweep_<jobid>.out
#   watch -n 2 cat cluster_outputs/online_serving_runs/sweep_<...>/<sub_run>/<policy>/status.txt

# NOTE: no --time on high_latency — this partition has no time limit, which is
# intentional so the priming + full sweep can run to completion without being
# killed by a 1-day low_latency cutoff.

set -euo pipefail

ROOT_DIR="/mnt/galactica/ocakir/attacc_simulator"
LOG_DIR="/mnt/galactica/ocakir/logs"

mkdir -p "${LOG_DIR}"
cd "${ROOT_DIR}"

# Redirect GCC/compiler temp files away from /tmp (frequently full on nodes).
export TMPDIR=/mnt/galactica/ocakir/tmp
mkdir -p "$TMPDIR"

source /mnt/galactica/ocakir/miniforge3/etc/profile.d/conda.sh
conda activate lerobot

echo "hostname:  $(hostname)"
echo "python:    $(python -V 2>&1)"
echo "conda env: ${CONDA_DEFAULT_ENV:-unknown}"

export REPO="${ROOT_DIR}"
export PY=$(which python)

export MODEL="${MODEL:-openvla}"
export N_REQUESTS="${N_REQUESTS:-2000}"
export KV_POOL_GB="${KV_POOL_GB:-4}"
export MAX_ACTIVE="${MAX_ACTIVE:-32}"
export ARRIVAL_SCALE="${ARRIVAL_SCALE:-1.0}"
export KV_OOM_POLICY="${KV_OOM_POLICY:-hold_then_fallback}"
export KV_OOM_HOLD_LIMIT_MS="${KV_OOM_HOLD_LIMIT_MS:-50}"
export SLOS="${SLOS:-500 1000 2000}"
export POLICIES="${POLICIES:-guaranteed_no_evict max_util_full max_util_tail}"
export SKIP_PRIMING="${SKIP_PRIMING:-0}"

echo "model:         ${MODEL}"
echo "n_requests:    ${N_REQUESTS}"
echo "kv_pool_gb:    ${KV_POOL_GB}"
echo "max_active:    ${MAX_ACTIVE}"
echo "arrival_scale: ${ARRIVAL_SCALE}"
echo "kv_oom_policy: ${KV_OOM_POLICY}"
echo "slos:          ${SLOS}"
echo "policies:      ${POLICIES}"
echo "skip_priming:  ${SKIP_PRIMING}"

exec bash tools/run_pim_cache_warmup_sweep.sh
