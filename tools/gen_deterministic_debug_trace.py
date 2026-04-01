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


def _scenario_heavy_sparse16() -> List[Tuple[int, int, int]]:
    # (arrival_ms, context_tokens, generated_tokens)
    # 16-request extension of heavy_sparse designed for larger decode batches:
    # - first 8 arrive in a tight burst to build decode pressure
    # - next 8 arrive while the system is already busy
    # - outputs stay long enough that larger max batch sizes can matter
    return [
        (0, 867, 180),
        (2, 867, 170),
        (4, 867, 160),
        (6, 867, 150),
        (8, 867, 140),
        (10, 867, 130),
        (12, 867, 120),
        (14, 867, 110),
        (60, 867, 150),
        (70, 867, 140),
        (80, 867, 130),
        (90, 867, 120),
        (110, 867, 110),
        (130, 867, 100),
        (160, 867, 90),
        (200, 867, 80),
    ]


def _scenario_tiny_prefill_long_decode() -> List[Tuple[int, int, int]]:
    # (arrival_ms, context_tokens, generated_tokens)
    # Small-prefill / long-decode debug case:
    # - context lengths stay small, so requests reach decode quickly
    # - decode lengths are long, so batching and decode-route effects dominate
    # - two bursts create overlap without making the trace too dense to inspect
    return [
        (0, 34, 224),
        (2, 48, 208),
        (4, 64, 192),
        (6, 96, 176),
        (40, 34, 160),
        (50, 48, 144),
        (60, 64, 128),
        (70, 96, 112),
    ]


def _scenario_tiny_prefill_long_decode16() -> List[Tuple[int, int, int]]:
    # (arrival_ms, context_tokens, generated_tokens)
    # 16-request extension of tiny_prefill_long_decode:
    # - first 8 arrive in a tight burst to build decode pressure early
    # - later 8 arrive while decodes are still active
    # - contexts remain small while outputs stay long to emphasize decode behavior
    return [
        (0, 34, 224),
        (2, 48, 208),
        (4, 64, 192),
        (6, 96, 176),
        (8, 34, 160),
        (10, 48, 144),
        (12, 64, 128),
        (14, 96, 112),
        (40, 34, 224),
        (50, 48, 208),
        (60, 64, 192),
        (70, 96, 176),
        (90, 34, 160),
        (110, 48, 144),
        (140, 64, 128),
        (180, 96, 112),
    ]


def _scenario_fixed_prefill_varied_long_decode16() -> List[Tuple[int, int, int]]:
    # (arrival_ms, context_tokens, generated_tokens)
    # 16-request decode-heavy case with a fixed, short prefill:
    # - all requests use the same modest context length
    # - generated lengths vary widely and stay long to emphasize decode effects
    # - two bursts create overlap while keeping the trace readable
    return [
        (0, 64, 128),
        (2, 64, 160),
        (4, 64, 192),
        (6, 64, 224),
        (8, 64, 256),
        (10, 64, 288),
        (12, 64, 320),
        (14, 64, 352),
        (40, 64, 352),
        (50, 64, 320),
        (60, 64, 288),
        (70, 64, 256),
        (90, 64, 224),
        (110, 64, 192),
        (140, 64, 160),
        (180, 64, 128),
    ]


def main() -> int:
    p = argparse.ArgumentParser(
        description="Generate a deterministic request trace CSV compatible with run_trace_replay.py")
    p.add_argument("--out",
                   type=Path,
                   default=Path("cluster_outputs/debug_trace_deterministic.csv"),
                   help="Output CSV path")
    p.add_argument("--scenario",
                   choices=[
                       "overlap",
                       "heavy_sparse",
                       "heavy_sparse16",
                       "tiny_prefill_long_decode",
                       "tiny_prefill_long_decode16",
                       "fixed_prefill_varied_long_decode16",
                   ],
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
    elif args.scenario == "heavy_sparse16":
        rows = _scenario_heavy_sparse16()
    elif args.scenario == "tiny_prefill_long_decode":
        rows = _scenario_tiny_prefill_long_decode()
    elif args.scenario == "tiny_prefill_long_decode16":
        rows = _scenario_tiny_prefill_long_decode16()
    elif args.scenario == "fixed_prefill_varied_long_decode16":
        rows = _scenario_fixed_prefill_varied_long_decode16()
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
