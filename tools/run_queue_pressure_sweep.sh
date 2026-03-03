#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

OUT_DIR="cluster_outputs/queue_pressure_sweep"
mkdir -p "${OUT_DIR}"

GPU_ALPHAS=(0.0 0.02 0.05 0.1)
PIM_ALPHAS=(0.0 0.02 0.05)
DECODE_ALPHAS=(0.0 0.0005 0.001 0.002)

for gpu_alpha in "${GPU_ALPHAS[@]}"; do
  for pim_alpha in "${PIM_ALPHAS[@]}"; do
    for decode_alpha in "${DECODE_ALPHAS[@]}"; do
      tag="g${gpu_alpha}_p${pim_alpha}_d${decode_alpha}"
      tag="${tag//./p}"
      echo "Running queue-pressure sweep: gpu=${gpu_alpha}, pim=${pim_alpha}, decode=${decode_alpha}"

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
        --slo-e2e-ms 500 \
        --gpu-queue-alpha "${gpu_alpha}" \
        --pim-queue-alpha "${pim_alpha}" \
        --decode-token-alpha "${decode_alpha}" \
        --requests-csv "${OUT_DIR}/replay_requests_${tag}.csv" \
        --summary-json "${OUT_DIR}/replay_summary_${tag}.json"
    done
  done
done

python - <<'PY'
import json
import re
from pathlib import Path

pat = re.compile(r"replay_summary_g([0-9p]+)_p([0-9p]+)_d([0-9p]+)\.json")
print("gpu_alpha,pim_alpha,decode_alpha,throughput_tokps,slo_e2e_miss_rate,gpu_util,pim_util,route_counts")
for path in sorted(Path("cluster_outputs/queue_pressure_sweep").glob("replay_summary_*.json")):
    m = pat.match(path.name)
    if not m:
        continue
    def dec(s: str) -> str:
        return s.replace("p", ".")
    with path.open() as f:
        d = json.load(f)
    print(
        f"{dec(m.group(1))},"
        f"{dec(m.group(2))},"
        f"{dec(m.group(3))},"
        f"{d['throughput_tokps']},"
        f"{d['slo_e2e_miss_rate']},"
        f"{d['gpu_util']},"
        f"{d['pim_util']},"
        f"{d['route_counts']}"
    )
PY
