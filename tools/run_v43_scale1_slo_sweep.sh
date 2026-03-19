#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

OUT_DIR="${OUT_DIR:-cluster_outputs/v43_scale1_slo_sweep}"
PLOTS_DIR="${PLOTS_DIR:-cluster_outputs/v43_scale1_slo_sweep_plots}"
TRACE_CACHE_DIR="${TRACE_CACHE_DIR:-cluster_outputs/trace_cache}"
LIMIT="${LIMIT:-2000}"
ARRIVAL_SCALE="${ARRIVAL_SCALE:-1.0}"
SLO_TBT_MS="${SLO_TBT_MS:-200}"
AZURE_TRACE="${AZURE_TRACE:-https://raw.githubusercontent.com/Azure/AzurePublicDataset/refs/heads/master/data/AzureLLMInferenceTrace_code.csv}"

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

DEFAULT_SLOS=(3000 4000 5000 6000 8000 10000 15000)
if (($# > 0)); then
  SLOS=("$@")
else
  SLOS=("${DEFAULT_SLOS[@]}")
fi

for slo in "${SLOS[@]}"; do
  summary_path="${OUT_DIR}/replay_summary_ml_slack_slo${slo}.json"
  requests_path="${OUT_DIR}/replay_requests_ml_slack_slo${slo}.csv"
  if [[ -f "${summary_path}" ]]; then
    echo "Skipping existing E2E sweep case: slo_e2e_ms=${slo}"
    continue
  fi
  echo "Running v4.3 scale=1 sweep for slo_e2e_ms=${slo}"

  python tools/run_trace_replay.py \
    --azure-trace "${AZURE_TRACE}" \
    --cost-model ml \
    --route-policy slack_then_finish \
    --slo-e2e-ms "${slo}" \
    --slo-tbt-ms "${SLO_TBT_MS}" \
    --limit "${LIMIT}" \
    --arrival-time-scale "${ARRIVAL_SCALE}" \
    --ml-profile gpu_only=cluster_outputs/cost_models/gpu_only.pkl \
    --ml-profile lpddr5_pim_bank=cluster_outputs/cost_models/lpddr5_pim_bank.pkl \
    --routes gpu_only lpddr5_pim_bank \
    --unsupported-policy clip \
    --local-scheduling-policy prefill_priority_fcfs_decode \
    --fcfs-decode-prediction-mode incremental \
    --enable-decode-batching \
    --max-decode-batch-size 4 \
    --enable-slo-guarded-admission \
    --admission-max-harmed-requests 1 \
    --admission-max-total-harm-ms 25 \
    --admission-max-single-harm-ms 10 \
    --admission-max-own-miss-ms 25 \
    --admission-max-bypass-count 2 \
    --admission-max-wait-ms 100 \
    --admission-aging-harm-ms-per-ms 0.02 \
    --admission-aging-harmed-requests-per-ms 0.001 \
    --admission-retry-interval-ms 25 \
    --requests-csv "${requests_path}" \
    --summary-json "${summary_path}"
done

python tools/plot_slo_sweep_results.py \
  --summary-dir "${OUT_DIR}" \
  --out-dir "${PLOTS_DIR}" \
  --prefix v43_scale1_slo_sweep

OUT_DIR_FOR_PY="${OUT_DIR}" python - <<'PY'
import json
import os
from pathlib import Path

print("slo_e2e_ms,throughput_tokps,slo_e2e_miss_rate,ttft_p95,e2e_p95,tbt_p95,route_counts")
for path in sorted(Path(os.environ["OUT_DIR_FOR_PY"]).glob("replay_summary_ml_slack_slo*.json")):
    slo = path.stem.split("slo")[-1]
    with path.open() as f:
        d = json.load(f)
    print(
        f"{slo},"
        f"{d['throughput_tokps']},"
        f"{d['slo_e2e_miss_rate']},"
        f"{d['ttft_ms']['p95']},"
        f"{d['e2e_ms']['p95']},"
        f"{d['tbt_ms']['p95']},"
        f"{d['route_counts']}"
    )
PY
