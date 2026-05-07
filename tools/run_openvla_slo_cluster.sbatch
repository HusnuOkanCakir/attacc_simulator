#!/bin/bash
#SBATCH -J openvla_slo
#SBATCH -p high_latency
#SBATCH -c 4
#SBATCH --mem=16GB
#SBATCH -D /mnt/galactica/ocakir/attacc_simulator
#SBATCH -o /mnt/galactica/ocakir/logs/openvla_slo_%j.out
#SBATCH -e /mnt/galactica/ocakir/logs/openvla_slo_%j.err
#
# Single-policy OpenVLA+PIM serving run with SLO constraints.
#
# Policy: max_util_full  (vLLM-style max utilization, full preemption)
# Model:  OpenVLA-7B (32 layers / 32 heads / d_head=128 / vision_prefix=256)
# KV pool: 2 GiB — intentionally tight to generate KV pressure and force
#          some requests to fall back to GPU via hold_then_fallback.
# SLO: 500ms E2E / 200ms TTFT (realistic robot-arm serving targets).
#      Requests that can't meet the deadline on PIM due to queue wait are
#      rerouted to GPU, producing mixed GPU+PIM routing.
#
# Env-var overrides:
#   N_REQUESTS      (default 50)    — number of Azure trace requests to replay
#   ARRIVAL_SCALE   (default 1.0)   — 1.0 = real-time Azure timing
#   KV_POOL_GB      (default 2)     — PIM KV pool in GiB; smaller = more GPU fallback
#   SLO_E2E_MS      (default 500)   — E2E SLO in ms
#   SLO_TTFT_MS     (default 200)   — TTFT SLO in ms (-1 to disable)
#   POLICY          (default max_util_full)
#
# Usage:
#   sbatch tools/run_openvla_slo_cluster.sbatch
#   N_REQUESTS=200 sbatch tools/run_openvla_slo_cluster.sbatch
#   KV_POOL_GB=1 SLO_E2E_MS=300 sbatch tools/run_openvla_slo_cluster.sbatch

set -euo pipefail

ROOT_DIR="/mnt/galactica/ocakir/attacc_simulator"
LOG_DIR="/mnt/galactica/ocakir/logs"

mkdir -p "${LOG_DIR}"
cd "${ROOT_DIR}"

source /mnt/galactica/ocakir/miniforge3/etc/profile.d/conda.sh
conda activate lerobot

echo "hostname:      $(hostname)"
echo "python:        $(python -V 2>&1)"
echo "conda env:     ${CONDA_DEFAULT_ENV:-unknown}"

N_REQUESTS="${N_REQUESTS:-50}"
ARRIVAL_SCALE="${ARRIVAL_SCALE:-1.0}"
KV_POOL_GB="${KV_POOL_GB:-2}"
SLO_E2E_MS="${SLO_E2E_MS:-500}"
SLO_TTFT_MS="${SLO_TTFT_MS:-200}"
POLICY="${POLICY:-max_util_full}"

echo "n_requests:    ${N_REQUESTS}"
echo "arrival_scale: ${ARRIVAL_SCALE}"
echo "kv_pool_gb:    ${KV_POOL_GB}"
echo "slo_e2e_ms:    ${SLO_E2E_MS}"
echo "slo_ttft_ms:   ${SLO_TTFT_MS}"
echo "policy:        ${POLICY}"

python tools/run_serving_online_policy_compare.py \
  --model openvla \
  --n-requests     "${N_REQUESTS}" \
  --arrival-scale  "${ARRIVAL_SCALE}" \
  --kv-pool-gb     "${KV_POOL_GB}" \
  --slo-e2e-ms     "${SLO_E2E_MS}" \
  --slo-ttft-ms    "${SLO_TTFT_MS}" \
  --kv-oom-policy  fallback_gpu \
  --max-active     32 \
  --ml \
  --policies       "${POLICY}"

echo "[done]"
