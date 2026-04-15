#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


EVENT_RE = re.compile(r"^\[event:(?P<event>[^\]]+)\](?P<body>.*)$")


def _normalize_request_ids(value: str) -> str:
    parts = [p.strip() for p in value.split(";") if p.strip()]
    out: list[int | str] = []
    for part in parts:
        try:
            out.append(int(part))
        except ValueError:
            out.append(part)
    return json.dumps(out)


def parse_debug_log(path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    event_index = 0
    with path.open() as f:
        for raw_line in f:
            line = raw_line.strip()
            match = EVENT_RE.match(line)
            if not match:
                continue
            row: dict[str, object] = {
                "event_index": event_index,
                "event": match.group("event"),
            }
            event_index += 1

            body = match.group("body").strip()
            if body:
                for token in body.split():
                    if "=" not in token:
                        continue
                    key, value = token.split("=", 1)
                    row[key] = value

            if "request_ids" in row:
                row["request_ids"] = _normalize_request_ids(str(row["request_ids"]))
            if "admission_queue_wait_ms" in row and "queue_wait_ms" not in row:
                row["queue_wait_ms"] = row["admission_queue_wait_ms"]

            rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert realtime-serving debug log lines into a debug-events CSV.")
    parser.add_argument("--debug-log", type=Path, required=True, help="Realtime frontend debug log path")
    parser.add_argument("--out-csv", type=Path, required=True, help="Output CSV path")
    args = parser.parse_args()

    if not args.debug_log.exists():
      raise FileNotFoundError(f"Debug log not found: {args.debug_log}")

    df = parse_debug_log(args.debug_log)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote realtime debug events CSV: {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
