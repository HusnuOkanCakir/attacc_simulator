#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MMD_PATH="${MMD_PATH:-docs/figures/trace_replay_scheduler_state_machine_presentation.mmd}"
SVG_PATH="${SVG_PATH:-docs/figures/trace_replay_scheduler_state_machine_presentation.svg}"
PNG_PATH="${PNG_PATH:-docs/figures/trace_replay_scheduler_state_machine_presentation.png}"
MERMAID_CLI_VERSION="${MERMAID_CLI_VERSION:-8.13.10}"

npx -y "@mermaid-js/mermaid-cli@${MERMAID_CLI_VERSION}" \
  -i "${MMD_PATH}" \
  -o "${SVG_PATH}"

npx -y "@mermaid-js/mermaid-cli@${MERMAID_CLI_VERSION}" \
  -i "${MMD_PATH}" \
  -o "${PNG_PATH}" \
  -w 2400 \
  -b transparent

echo "Wrote ${SVG_PATH}"
echo "Wrote ${PNG_PATH}"
