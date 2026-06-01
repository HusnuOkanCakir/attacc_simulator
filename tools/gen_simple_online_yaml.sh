#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

REQUESTS_CSV="${REQUESTS_CSV:-cluster_outputs/simple_online_2x512_16gen.csv}"
TEMPLATE_DIR="${TEMPLATE_DIR:-ramulator2/trace_gen/templates_lpddr5_bank}"
YAML_FILE="${YAML_FILE:-ramulator2/simple_online_2x512_16gen.yaml}"
TRACE_LOG_PATH="${TRACE_LOG_PATH:-./log/simple_online/cmd.log}"

mkdir -p "$(dirname "$YAML_FILE")"
mkdir -p "$(dirname "$TRACE_LOG_PATH")"

cat > "$YAML_FILE" <<EOF
Frontend:
  impl: SimpleOnlineServingFrontend
  requests_csv: ${REQUESTS_CSV}
  template_dir: ${TEMPLATE_DIR}
  clock_ratio: 1

MemorySystem:
  impl: PIMDRAM
  clock_ratio: 1

  DRAM:
    impl: LPDDR5-PIM
    org:
      preset: LPDDR5_2Gb_x16
      channel: 16
    timing:
      preset: LPDDR5_6400

  Controller:
    impl: HBM3-PIM
    Scheduler:
      impl: PIM
    RefreshManager:
      impl: AllBank
    plugins:
    - ControllerPlugin:
        impl: TraceRecorder
        path: ${TRACE_LOG_PATH}

  AddrMapper:
    impl: ChRaBaRoCo
EOF

echo "wrote ${YAML_FILE}"
