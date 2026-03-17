#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

import pandas as pd


def _scenario_overlap() -> List[Tuple[int, int, int]]:
    # (arrival_ms, context_tokens, generated_tokens)
    # Deterministic micro-trace designed to expose queueing behavior:
    # - early burst
    # - mixed decode lengths
    # - short gap, then another burst
    return [
        (0, 867, 50),
        (1, 867, 50),
        (2, 867, 10),
        (3, 867, 12),
        (5, 867, 5),
        (40, 867, 50),
        (41, 867, 20),
        (42, 867, 6),
    ]


def _scenario_heavy_sparse() -> List[Tuple[int, int, int]]:
    # (arrival_ms, context_tokens, generated_tokens)
    # Heavier and sparser than overlap:
    # - first short burst of 3 requests
    # - long decode lengths to keep system busy
    # - later arrivals occur while earlier decodes are running
    return [
        (0, 867, 140),
        (2, 867, 120),
        (4, 867, 100),
        (70, 867, 130),
        (95, 867, 110),
        (130, 867, 90),
        (180, 867, 80),
        (260, 867, 70),
    ]


def main() -> int:
    p = argparse.ArgumentParser(
        description="Generate a deterministic request trace CSV compatible with run_trace_replay.py")
    p.add_argument("--out",
                   type=Path,
                   default=Path("cluster_outputs/debug_trace_deterministic.csv"),
                   help="Output CSV path")
    p.add_argument("--scenario",
                   choices=["overlap", "heavy_sparse"],
                   default="overlap",
                   help="Deterministic scenario preset")
    p.add_argument("--base-time",
                   type=str,
                   default="2025-01-01 00:00:00",
                   help="Base timestamp in '%Y-%m-%d %H:%M:%S' format")
    args = p.parse_args()

    if args.scenario == "overlap":
        rows = _scenario_overlap()
    elif args.scenario == "heavy_sparse":
        rows = _scenario_heavy_sparse()
    else:
        raise ValueError(f"Unsupported scenario: {args.scenario}")

    t0 = datetime.strptime(args.base_time, "%Y-%m-%d %H:%M:%S")
    recs = []
    for ms, ctx, gen in rows:
        recs.append({
            "TIMESTAMP": t0 + timedelta(milliseconds=int(ms)),
            "ContextTokens": int(ctx),
            "GeneratedTokens": int(gen),
        })

    df = pd.DataFrame(recs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, date_format="%Y-%m-%d %H:%M:%S.%f")
    print(f"Wrote deterministic trace: {args.out}")
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
