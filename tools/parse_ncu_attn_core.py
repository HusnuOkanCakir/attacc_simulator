#!/usr/bin/env python3
"""Parse NCU CSV and compute attention-core GPU time and optional DRAM bandwidth."""

from __future__ import annotations

import argparse
import pandas as pd
from typing import Iterable, Optional


def _to_us(value: float, unit: str) -> float:
    unit_to_us = {
        "ns": 1e-3,
        "us": 1.0,
        "ms": 1e3,
        "s": 1e6,
    }
    if unit not in unit_to_us:
        # Nsight Compute commonly exports gpu__time_duration.sum in microseconds.
        unit = "us"
    return value * unit_to_us[unit]


def _to_bytes(value: float, unit: str) -> float:
    unit_to_bytes = {
        "byte": 1.0,
        "bytes": 1.0,
        "kbyte": 1e3,
        "mbyte": 1e6,
        "gbyte": 1e9,
        "tbyte": 1e12,
        "kbytes": 1e3,
        "mbytes": 1e6,
        "gbytes": 1e9,
        "tbytes": 1e12,
        "kib": 1024.0,
        "mib": 1024.0**2,
        "gib": 1024.0**3,
        "tib": 1024.0**4,
    }
    u = unit.strip().lower()
    if u not in unit_to_bytes:
        # Nsight Compute byte counters usually use "byte".
        u = "byte"
    return value * unit_to_bytes[u]


def _clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("'", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def _detect_unit(series: pd.Series, unit_set: set[str], default_unit: str) -> str:
    text = series.astype(str).str.strip()
    for v in text.head(64):
        if v in unit_set:
            return v
    return default_unit


def _first_existing(candidates: Iterable[str], columns: Iterable[str]) -> Optional[str]:
    col_set = set(columns)
    for c in candidates:
        if c in col_set:
            return c
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse NCU CSV for attention core timing.")
    parser.add_argument("--input", type=str, required=True, help="Path to NCU CSV file.")
    parser.add_argument(
        "--include-pattern",
        type=str,
        default="BLOCK.attn.core_layer",
        help="Substring to include from NVTX ranges.",
    )
    parser.add_argument(
        "--exclude-pattern",
        type=str,
        default=None,
        help="Substring to exclude from NVTX ranges.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=None,
        help="If provided, compute per-iteration time (total / iters).",
    )
    parser.add_argument(
        "--dram-total-col",
        type=str,
        default=None,
        help="Optional NCU metric column for total DRAM bytes (e.g., dram__bytes.sum).",
    )
    parser.add_argument(
        "--dram-read-col",
        type=str,
        default=None,
        help="Optional NCU metric column for DRAM read bytes.",
    )
    parser.add_argument(
        "--dram-write-col",
        type=str,
        default=None,
        help="Optional NCU metric column for DRAM write bytes.",
    )
    parser.add_argument(
        "--show-mem-cols",
        action="store_true",
        help="Print available columns that look memory/bytes related.",
    )
    args = parser.parse_args()

    base_cols = [
        "gpu__time_duration.sum",
        "thread Domain:Push/Pop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg",
        "Id:Domain:Start/Stop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg",
    ]

    all_cols = list(pd.read_csv(args.input, nrows=0, low_memory=False).columns)
    missing = [c for c in base_cols if c not in all_cols]
    if missing:
        raise SystemExit(f"Missing required columns in CSV: {missing}")

    auto_total_candidates = [
        "dram__bytes.sum",
        "gpu__dram_bytes.sum",
    ]
    auto_read_candidates = [
        "dram__bytes_read.sum",
        "gpu__dram_bytes_read.sum",
    ]
    auto_write_candidates = [
        "dram__bytes_write.sum",
        "gpu__dram_bytes_write.sum",
    ]

    dram_total_col = args.dram_total_col or _first_existing(auto_total_candidates, all_cols)
    dram_read_col = args.dram_read_col or _first_existing(auto_read_candidates, all_cols)
    dram_write_col = args.dram_write_col or _first_existing(auto_write_candidates, all_cols)

    usecols = list(base_cols)
    for c in [dram_total_col, dram_read_col, dram_write_col]:
        if c and c not in usecols:
            usecols.append(c)

    df = pd.read_csv(args.input, usecols=usecols, low_memory=False, dtype=str)

    # Nsight Compute CSV has a unit row; gpu__time_duration.sum is often "us".
    duration_col = "gpu__time_duration.sum"
    unit = _detect_unit(
        df[duration_col], {"ns", "us", "ms", "s"}, default_unit="us"
    )
    df[duration_col] = _clean_numeric(df[duration_col])

    text = df[base_cols[1]].astype(str) + " " + df[base_cols[2]].astype(str)
    mask = text.str.contains(args.include_pattern, regex=False)
    if args.exclude_pattern:
        mask = mask & ~text.str.contains(args.exclude_pattern, regex=False)
    core = df[mask]

    # Drop non-numeric rows (e.g., unit row), then convert to us/ms explicitly.
    total_native = core[duration_col].dropna().astype(float).sum()
    total_us = _to_us(total_native, unit)
    total_ms = total_us / 1000.0

    print(f"duration_unit_in_csv {unit}")
    print(f"matched_rows {len(core[duration_col].dropna())}")
    print(f"core_time_us_total {total_us}")
    print(f"core_time_ms_total {total_ms}")

    if args.iters:
        print(f"core_time_us_per_iter_({args.iters}_iters) {total_us / args.iters}")
        print(f"core_time_ms_per_iter_({args.iters}_iters) {total_ms / args.iters}")

    if args.show_mem_cols:
        mem_cols = [
            c
            for c in all_cols
            if any(k in c.lower() for k in ["dram", "hbm", "bytes", "throughput", "lts", "l2"])
        ]
        print("available_mem_like_columns")
        for c in mem_cols:
            print(f"  {c}")

    total_dram_bytes = None
    bytes_unit = "byte"

    if dram_total_col:
        # Detect from full CSV (unit row is often outside NVTX-filtered rows).
        bytes_unit = _detect_unit(
            df[dram_total_col],
            {"byte", "bytes", "Kbyte", "Mbyte", "Gbyte", "Tbyte", "KiB", "MiB", "GiB", "TiB"},
            default_unit="byte",
        )
        total_native = _clean_numeric(core[dram_total_col]).dropna().astype(float).sum()
        total_dram_bytes = _to_bytes(total_native, bytes_unit)
    elif dram_read_col and dram_write_col:
        # Detect from full CSV (unit row is often outside NVTX-filtered rows).
        read_unit = _detect_unit(
            df[dram_read_col],
            {"byte", "bytes", "Kbyte", "Mbyte", "Gbyte", "Tbyte", "KiB", "MiB", "GiB", "TiB"},
            default_unit="byte",
        )
        write_unit = _detect_unit(
            df[dram_write_col],
            {"byte", "bytes", "Kbyte", "Mbyte", "Gbyte", "Tbyte", "KiB", "MiB", "GiB", "TiB"},
            default_unit="byte",
        )
        read_native = _clean_numeric(core[dram_read_col]).dropna().astype(float).sum()
        write_native = _clean_numeric(core[dram_write_col]).dropna().astype(float).sum()
        total_dram_bytes = _to_bytes(read_native, read_unit) + _to_bytes(write_native, write_unit)
        bytes_unit = f"{read_unit}+{write_unit}"

    if total_dram_bytes is not None and total_us > 0:
        bw_gbps = total_dram_bytes / (total_us * 1e-6) / 1e9
        bw_gibps = total_dram_bytes / (total_us * 1e-6) / (1024.0**3)
        print(f"dram_bytes_source {dram_total_col or (dram_read_col + '+' + dram_write_col)}")
        print(f"dram_bytes_unit_in_csv {bytes_unit}")
        print(f"dram_bytes_total {total_dram_bytes}")
        print(f"dram_bw_effective_GBps {bw_gbps}")
        print(f"dram_bw_effective_GiBps {bw_gibps}")
        if args.iters:
            print(f"dram_bytes_per_iter_({args.iters}_iters) {total_dram_bytes / args.iters}")
    else:
        print("dram_bw_note No DRAM byte metrics found in CSV. Re-run NCU with dram byte metrics.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
