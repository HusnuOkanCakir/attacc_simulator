#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

OUT_DIR="${OUT_DIR:-cluster_outputs/v5_energy_bs_sweep}"
PLOTS_DIR="${PLOTS_DIR:-cluster_outputs/v5_energy_bs_sweep_plots}"
TRACE_CACHE_DIR="${TRACE_CACHE_DIR:-cluster_outputs/trace_cache}"
LIMIT="${LIMIT:-300}"
ARRIVAL_SCALE="${ARRIVAL_SCALE:-0.25}"
SLO_E2E_MS="${SLO_E2E_MS:-5000}"
ENERGY_LATENCY_GUARD_MS="${ENERGY_LATENCY_GUARD_MS:-5}"
AZURE_TRACE="${AZURE_TRACE:-https://raw.githubusercontent.com/Azure/AzurePublicDataset/refs/heads/master/data/AzureLLMInferenceTrace_code.csv}"
GPU_MODEL="${GPU_MODEL:-cluster_outputs/cost_models/gpu_only.pkl}"
LPDDR_MODEL="${LPDDR_MODEL:-cluster_outputs/cost_models/lpddr5_pim_bank.pkl}"

mkdir -p "${OUT_DIR}" "${PLOTS_DIR}" "${TRACE_CACHE_DIR}"

if [[ "${AZURE_TRACE}" =~ ^https?:// ]]; then
  TRACE_CACHE_PATH="${TRACE_CACHE_DIR}/AzureLLMInferenceTrace_code.csv"
  if [[ ! -f "${TRACE_CACHE_PATH}" ]]; then
    echo "Caching Azure trace to ${TRACE_CACHE_PATH}"
    python - <<'PY' "${AZURE_TRACE}" "${TRACE_CACHE_PATH}"
import sys
import urllib.request

src, dst = sys.argv[1], sys.argv[2]
with urllib.request.urlopen(src) as resp, open(dst, "wb") as out:
    out.write(resp.read())
PY
  fi
  AZURE_TRACE="${TRACE_CACHE_PATH}"
fi

DEFAULT_BS_VALUES=(1 2 4 8 16 32)
if (($# > 0)); then
  BS_VALUES=("$@")
else
  BS_VALUES=("${DEFAULT_BS_VALUES[@]}")
fi

for bs in "${BS_VALUES[@]}"; do
  summary_path="${OUT_DIR}/replay_summary_v5_energy_bs${bs}.json"
  requests_path="${OUT_DIR}/replay_requests_v5_energy_bs${bs}.csv"
  if [[ -f "${summary_path}" ]]; then
    echo "Skipping existing v5 energy BS sweep case: max_decode_batch_size=${bs}"
    continue
  fi
  echo "Running v5 energy BS sweep case: max_decode_batch_size=${bs}"

  python tools/run_trace_replay.py \
    --azure-trace "${AZURE_TRACE}" \
    --cost-model ml \
    --route-policy latency_guarded_energy \
    --energy-latency-guard-ms "${ENERGY_LATENCY_GUARD_MS}" \
    --slo-e2e-ms "${SLO_E2E_MS}" \
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
    --requests-csv "${requests_path}" \
    --summary-json "${summary_path}"
done

python tools/plot_v5_energy_bs_sweep_results.py \
  --summary-dir "${OUT_DIR}" \
  --out-dir "${PLOTS_DIR}" \
  --prefix v5_energy_bs_sweep

OUT_DIR_FOR_PY="${OUT_DIR}" python - <<'PY'
import json
import os
from pathlib import Path

print("max_bs,throughput_tokps,ttft_p95,e2e_p95,tbt_p95,energy_total_nj,decode_energy_total_nj,decode_energy_per_decode_token,route_counts")
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
        f"{d.get('energy_nj', {}).get('total', d.get('decode_energy_nj', {}).get('total'))},"
        f"{d.get('decode_energy_nj', {}).get('total')},"
        f"{d.get('decode_energy_nj_per_decode_token')},"
        f"{d.get('route_counts')}"
    )
PY
