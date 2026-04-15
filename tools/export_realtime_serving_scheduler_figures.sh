#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

FIG_DIR="${FIG_DIR:-docs/figures/trace_replay_scheduler_state_machine/realtime_serving}"
MERMAID_CLI_VERSION="${MERMAID_CLI_VERSION:-8.13.10}"

shopt -s nullglob
for mmd in "${FIG_DIR}"/*.mmd; do
  base="${mmd%.mmd}"
  svg="${base}.svg"
  png="${base}.png"

  npx -y "@mermaid-js/mermaid-cli@${MERMAID_CLI_VERSION}" \
    -i "${mmd}" \
    -o "${svg}"

  npx -y "@mermaid-js/mermaid-cli@${MERMAID_CLI_VERSION}" \
    -i "${mmd}" \
    -o "${png}" \
    -w 2400 \
    -b transparent

  echo "Wrote ${svg}"
  echo "Wrote ${png}"
done
