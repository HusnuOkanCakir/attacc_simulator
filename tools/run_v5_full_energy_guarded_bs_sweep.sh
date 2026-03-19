#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"

OUT_DIR="${OUT_DIR:-cluster_outputs/v5_full_energy_guarded_bs_sweep}"
PLOTS_DIR="${PLOTS_DIR:-cluster_outputs/v5_full_energy_guarded_bs_sweep_plots}"
TRACE_CACHE_DIR="${TRACE_CACHE_DIR:-cluster_outputs/trace_cache}"

LIMIT="${LIMIT:-300}"
ARRIVAL_SCALE="${ARRIVAL_SCALE:-1.0}"
SLO_E2E_MS="${SLO_E2E_MS:-5000}"
SLO_TBT_MS="${SLO_TBT_MS:-200}"
ENERGY_LATENCY_GUARD_MS="${ENERGY_LATENCY_GUARD_MS:-5}"

ADMISSION_MAX_HARMED_REQUESTS="${ADMISSION_MAX_HARMED_REQUESTS:-1}"
ADMISSION_MAX_TOTAL_HARM_MS="${ADMISSION_MAX_TOTAL_HARM_MS:-25}"
ADMISSION_MAX_SINGLE_HARM_MS="${ADMISSION_MAX_SINGLE_HARM_MS:-10}"
ADMISSION_MAX_OWN_MISS_MS="${ADMISSION_MAX_OWN_MISS_MS:-25}"
ADMISSION_MAX_BYPASS_COUNT="${ADMISSION_MAX_BYPASS_COUNT:-2}"
ADMISSION_MAX_WAIT_MS="${ADMISSION_MAX_WAIT_MS:-100}"
ADMISSION_AGING_HARM_MS_PER_MS="${ADMISSION_AGING_HARM_MS_PER_MS:-0.02}"
ADMISSION_AGING_HARMED_REQUESTS_PER_MS="${ADMISSION_AGING_HARMED_REQUESTS_PER_MS:-0.001}"
ADMISSION_RETRY_INTERVAL_MS="${ADMISSION_RETRY_INTERVAL_MS:-25}"

AZURE_TRACE="${AZURE_TRACE:-https://raw.githubusercontent.com/Azure/AzurePublicDataset/refs/heads/master/data/AzureLLMInferenceTrace_code.csv}"
GPU_MODEL="${GPU_MODEL:-cluster_outputs/cost_models_full_energy/gpu_only.pkl}"
LPDDR_MODEL="${LPDDR_MODEL:-cluster_outputs/cost_models_full_energy/lpddr5_pim_bank.pkl}"

mkdir -p "${OUT_DIR}" "${PLOTS_DIR}" "${TRACE_CACHE_DIR}"

if [[ "${AZURE_TRACE}" =~ ^https?:// ]]; then
  TRACE_CACHE_PATH="${TRACE_CACHE_DIR}/AzureLLMInferenceTrace_code.csv"
  if [[ ! -f "${TRACE_CACHE_PATH}" ]]; then
    echo "Caching Azure trace to ${TRACE_CACHE_PATH}"
    "${PYTHON_BIN}" - <<'PY' "${AZURE_TRACE}" "${TRACE_CACHE_PATH}"
import sys
import urllib.request

src, dst = sys.argv[1], sys.argv[2]
with urllib.request.urlopen(src) as resp, open(dst, "wb") as out:
    out.write(resp.read())
PY
  fi
  AZURE_TRACE="${TRACE_CACHE_PATH}"
fi

if (($# > 0)); then
  BS_VALUES=("$@")
else
  mapfile -t BS_VALUES < <(seq 1 32)
fi

for bs in "${BS_VALUES[@]}"; do
  summary_path="${OUT_DIR}/replay_summary_v5_energy_bs${bs}.json"
  requests_path="${OUT_DIR}/replay_requests_v5_energy_bs${bs}.csv"
  if [[ -f "${summary_path}" ]]; then
    echo "Skipping existing full-energy guarded BS sweep case: max_decode_batch_size=${bs}"
    continue
  fi

  echo "Running full-energy guarded BS sweep case: max_decode_batch_size=${bs}"

  "${PYTHON_BIN}" tools/run_trace_replay.py \
    --azure-trace "${AZURE_TRACE}" \
    --cost-model ml \
    --route-policy latency_guarded_energy \
    --energy-latency-guard-ms "${ENERGY_LATENCY_GUARD_MS}" \
    --slo-e2e-ms "${SLO_E2E_MS}" \
    --slo-tbt-ms "${SLO_TBT_MS}" \
    --limit "${LIMIT}" \
    --arrival-time-scale "${ARRIVAL_SCALE}" \
    --ml-profile "gpu_only=${GPU_MODEL}" \
    --ml-profile "lpddr5_pim_bank=${LPDDR_MODEL}" \
    --routes gpu_only lpddr5_pim_bank \
    --unsupported-policy clip \
    --local-scheduling-policy prefill_priority_fcfs_decode \
    --fcfs-decode-prediction-mode incremental \
    --enable-decode-batching \
    --max-decode-batch-size "${bs}" \
    --enable-slo-guarded-admission \
    --admission-max-harmed-requests "${ADMISSION_MAX_HARMED_REQUESTS}" \
    --admission-max-total-harm-ms "${ADMISSION_MAX_TOTAL_HARM_MS}" \
    --admission-max-single-harm-ms "${ADMISSION_MAX_SINGLE_HARM_MS}" \
    --admission-max-own-miss-ms "${ADMISSION_MAX_OWN_MISS_MS}" \
    --admission-max-bypass-count "${ADMISSION_MAX_BYPASS_COUNT}" \
    --admission-max-wait-ms "${ADMISSION_MAX_WAIT_MS}" \
    --admission-aging-harm-ms-per-ms "${ADMISSION_AGING_HARM_MS_PER_MS}" \
    --admission-aging-harmed-requests-per-ms "${ADMISSION_AGING_HARMED_REQUESTS_PER_MS}" \
    --admission-retry-interval-ms "${ADMISSION_RETRY_INTERVAL_MS}" \
    --requests-csv "${requests_path}" \
    --summary-json "${summary_path}"
done

"${PYTHON_BIN}" tools/plot_v5_energy_bs_sweep_results.py \
  --summary-dir "${OUT_DIR}" \
  --out-dir "${PLOTS_DIR}" \
  --prefix v5_full_energy_guarded_bs_sweep

OUT_DIR_FOR_PY="${OUT_DIR}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

print("max_bs,throughput_tokps,ttft_p95,e2e_p95,tbt_p95,energy_total_nj,energy_per_request_nj,decode_energy_total_nj,decode_energy_per_decode_token,route_counts")
for path in sorted(Path(os.environ["OUT_DIR_FOR_PY"]).glob("replay_summary_v5_energy_bs*.json")):
    stem = path.stem
    bs = stem.split("bs")[-1]
    with path.open() as f:
        d = json.load(f)
    print(
        f"{bs},"
        f"{d['throughput_tokps']},"
        f"{d['ttft_ms']['p95']},"
        f"{d['e2e_ms']['p95']},"
        f"{d['tbt_ms']['p95']},"
        f"{d.get('energy_nj', {}).get('total')},"
        f"{d.get('energy_nj_per_request')},"
        f"{d.get('decode_energy_nj', {}).get('total')},"
        f"{d.get('decode_energy_nj_per_decode_token')},"
        f"{d.get('route_counts')}"
    )
PY

echo "Wrote summaries to ${OUT_DIR}"
echo "Wrote plots to ${PLOTS_DIR}"
