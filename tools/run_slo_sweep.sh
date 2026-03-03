#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

OUT_DIR="cluster_outputs/slo_sweep"
mkdir -p "${OUT_DIR}"

SLOS=(500 1000 2000 5000 10000 20000 50000 100000)

for slo in "${SLOS[@]}"; do
  echo "Running SLO sweep for slo_e2e_ms=${slo}"

  python tools/run_trace_replay.py \
    --cost-model ml \
    --route-policy slack_then_finish \
    --limit 2000 \
    --arrival-time-scale 0.05 \
    --ml-profile gpu_only=cluster_outputs/cost_models/gpu_only.pkl \
    --ml-profile lpddr5_pim_bank=cluster_outputs/cost_models/lpddr5_pim_bank.pkl \
    --routes gpu_only lpddr5_pim_bank \
    --unsupported-policy clip \
    --pim-wait-threshold-ms 1 \
    --slo-e2e-ms "${slo}" \
    --requests-csv "${OUT_DIR}/replay_requests_ml_slack_slo${slo}.csv" \
    --summary-json "${OUT_DIR}/replay_summary_ml_slack_slo${slo}.json"
done

python - <<'PY'
import json
from pathlib import Path

print("slo_e2e_ms,throughput_tokps,slo_e2e_miss_rate,ttft_p95,e2e_p95,route_counts")
for path in sorted(Path("cluster_outputs/slo_sweep").glob("replay_summary_ml_slack_slo*.json")):
    slo = path.stem.split("slo")[-1]
    with path.open() as f:
        d = json.load(f)
    print(
        f"{slo},"
        f"{d['throughput_tokps']},"
        f"{d['slo_e2e_miss_rate']},"
        f"{d['ttft_ms']['p95']},"
        f"{d['e2e_ms']['p95']},"
        f"{d['route_counts']}"
    )
PY
