#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

SRC_MD="${SRC_MD:-docs/trace_replay_scheduler_state_machine.md}"
OUT_DIR="${OUT_DIR:-docs/figures/trace_replay_scheduler_state_machine}"
MERMAID_CLI_VERSION="${MERMAID_CLI_VERSION:-8.13.10}"

mkdir -p "${OUT_DIR}"

python - <<'PY' "${SRC_MD}" "${OUT_DIR}"
from pathlib import Path
import re
import sys

src = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
text = src.read_text()

heading = None
blocks = []
inside = False
buf = []

for line in text.splitlines():
    if line.startswith("## "):
        heading = line[3:].strip()
        continue
    if line.strip() == "```mermaid":
        inside = True
        buf = []
        continue
    if inside and line.strip() == "```":
        if heading is None:
            raise SystemExit("Found mermaid block without preceding level-2 heading")
        blocks.append((heading, "\n".join(buf).rstrip() + "\n"))
        inside = False
        buf = []
        continue
    if inside:
        buf.append(line)

def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value

for heading, block in blocks:
    slug = slugify(heading)
    out_path = out_dir / f"{slug}.mmd"
    out_path.write_text(block)
    print(out_path)
PY

while IFS= read -r mmd_path; do
  base="${mmd_path%.mmd}"
  npx -y "@mermaid-js/mermaid-cli@${MERMAID_CLI_VERSION}" \
    -i "${mmd_path}" \
    -o "${base}.svg"

  npx -y "@mermaid-js/mermaid-cli@${MERMAID_CLI_VERSION}" \
    -i "${mmd_path}" \
    -o "${base}.png" \
    -w 2400 \
    -b transparent

  echo "Wrote ${base}.svg"
  echo "Wrote ${base}.png"
done < <(find "${OUT_DIR}" -maxdepth 1 -name '*.mmd' | sort)
