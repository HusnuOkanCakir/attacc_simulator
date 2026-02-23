#!/usr/bin/env python3
"""
Parse Ramulator2 stdout and compute:
  - estimated time in microseconds
  - estimated bytes moved (using prefetch size)
  - per-layer and per-model scaled totals
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, Optional


@dataclass
class RamulatorStats:
    cycles: int = 0
    softmax: int = 0
    mvgb: int = 0
    mvsb: int = 0
    wrgb: int = 0
    mac_ab: int = 0
    mac_sb: int = 0
    mac_pb: int = 0


LPDDR5_TCK_PS_BY_PRESET = {
    "LPDDR5_6400": 1250.0,
}


def _parse_lines(lines: Iterable[str]) -> RamulatorStats:
    stats = RamulatorStats()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if "memory_system_cycles" in line:
            stats.cycles = int(line.split()[-1])
        elif "pim_softmax_requests" in line:
            stats.softmax = int(line.split()[-1])
        elif "pim_move_to_gemv_buffer_requests" in line:
            stats.mvgb = int(line.split()[-1])
        elif "pim_move_to_softmax_buffer_requests" in line:
            stats.mvsb = int(line.split()[-1])
        elif "pim_write_to_gemv_buffer_requests" in line:
            stats.wrgb = int(line.split()[-1])
        elif "pim_mac_all_bank_requests" in line:
            stats.mac_ab = int(line.split()[-1])
        elif "pim_mac_same_bank_requests" in line:
            stats.mac_sb = int(line.split()[-1])
        elif "pim_mac_per_bank_requests" in line:
            stats.mac_pb = int(line.split()[-1])
    return stats


def _tck_ps_from_rate(rate_mtps: int) -> float:
    # HBM3 uses QDR pins in this codebase (see HBM3-PIM.cpp)
    return 1e6 / (rate_mtps / 4.0)


def _hbm_rate_mtps_from_preset(preset: str) -> Optional[int]:
    # Example: HBM3_5.2Gbps -> 5200 MT/s
    m = re.search(r"HBM3_([0-9]+(?:\.[0-9]+)?)Gbps", preset)
    if not m:
        return None
    return int(round(float(m.group(1)) * 1000.0))


def _parse_dram_info_from_yaml(config_yaml: str) -> tuple[Optional[str], Optional[str]]:
    if not os.path.exists(config_yaml):
        return None, None

    dram_impl = None
    timing_preset = None
    in_dram = False
    dram_indent = -1
    in_timing = False
    timing_indent = -1

    with open(config_yaml, "r", encoding="utf-8") as f:
        for raw in f:
            # Remove comments and trailing newline
            line = raw.split("#", 1)[0].rstrip("\n")
            if not line.strip():
                continue

            indent = len(line) - len(line.lstrip(" "))
            stripped = line.strip()

            if in_timing and indent <= timing_indent:
                in_timing = False
            if in_dram and indent <= dram_indent:
                in_dram = False
                in_timing = False

            if stripped == "DRAM:":
                in_dram = True
                dram_indent = indent
                in_timing = False
                continue

            if in_dram and stripped == "timing:":
                in_timing = True
                timing_indent = indent
                continue

            if in_dram and not in_timing and stripped.startswith("impl:"):
                dram_impl = stripped.split(":", 1)[1].strip()
                continue

            if in_timing and stripped.startswith("preset:"):
                timing_preset = stripped.split(":", 1)[1].strip()
                continue

    return dram_impl, timing_preset


def _resolve_tck_ps(
    rate_mtps: int,
    tck_ps_override: Optional[float],
    config_yaml: Optional[str],
) -> tuple[float, str]:
    if tck_ps_override is not None:
        return float(tck_ps_override), "user-override(--tck-ps)"

    dram_impl = None
    timing_preset = None
    if config_yaml:
        dram_impl, timing_preset = _parse_dram_info_from_yaml(config_yaml)

    if dram_impl is not None and dram_impl.startswith("LPDDR5"):
        if timing_preset in LPDDR5_TCK_PS_BY_PRESET:
            return LPDDR5_TCK_PS_BY_PRESET[timing_preset], f"yaml({dram_impl}/{timing_preset})"
        # Fallback for unknown LPDDR5 preset
        return _tck_ps_from_rate(rate_mtps), f"fallback-rate({rate_mtps} MT/s, unknown LPDDR5 preset)"

    if dram_impl is not None and dram_impl.startswith("HBM3"):
        if timing_preset:
            parsed_rate = _hbm_rate_mtps_from_preset(timing_preset)
            if parsed_rate is not None:
                return _tck_ps_from_rate(parsed_rate), f"yaml({dram_impl}/{timing_preset})"
        return _tck_ps_from_rate(rate_mtps), f"fallback-rate({rate_mtps} MT/s, unknown HBM3 preset)"

    # Default behavior (backward compatible)
    return _tck_ps_from_rate(rate_mtps), f"default-rate({rate_mtps} MT/s)"


def _estimate_bytes(stats: RamulatorStats, prefetch_bytes: int) -> Dict[str, int]:
    # Following ramulator_wrapper.py conventions:
    # - Each WRGB/MVGB/MVSB is a 32B (prefetch) movement
    # - MACs also touch prefetch_bytes
    si_io = stats.wrgb * prefetch_bytes
    tsv_io = (stats.wrgb + stats.mvsb + stats.mvgb) * prefetch_bytes
    giomux_io = tsv_io
    bgmux_io = tsv_io
    mem_acc = (stats.mac_ab + stats.mac_sb + stats.mac_pb) * prefetch_bytes
    return {
        "si_io": si_io,
        "tsv_io": tsv_io,
        "giomux_io": giomux_io,
        "bgmux_io": bgmux_io,
        "mem_acc": mem_acc,
    }


def _format_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    v = float(n)
    idx = 0
    while v >= 1024.0 and idx < len(units) - 1:
        v /= 1024.0
        idx += 1
    return f"{v:.3f} {units[idx]}"


def _bw_gbps(bytes_count: float, time_us: float) -> float:
    if time_us <= 0.0:
        return 0.0
    # GB/s (decimal): bytes / sec / 1e9
    return bytes_count / (time_us * 1e-6) / 1e9


def _read_file(path: str) -> Iterable[str]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield line


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse ramulator output.")
    parser.add_argument(
        "--input",
        type=str,
        default="-",
        help="Path to ramulator stdout log, or '-' for stdin.",
    )
    parser.add_argument(
        "--dump-dir",
        type=str,
        default=None,
        help="PI0 dump directory containing pi0_attn_info.json.",
    )
    parser.add_argument(
        "--info-json",
        type=str,
        default=None,
        help="Path to pi0_attn_info.json (overrides --dump-dir).",
    )
    parser.add_argument("--rate-mtps", type=int, default=5200, help="HBM3 rate in MT/s.")
    parser.add_argument("--prefetch-bytes", type=int, default=32, help="Prefetch size in bytes.")
    parser.add_argument("--layers", type=int, default=1, help="Number of layers to scale.")
    parser.add_argument("--batch", type=int, default=1, help="Batch size to scale.")
    parser.add_argument(
        "--heads-per-hbm", type=int, default=1, help="Heads per HBM used in trace."
    )
    parser.add_argument(
        "--total-heads", type=int, default=None, help="Total heads in model (optional)."
    )
    parser.add_argument(
        "--config-yaml",
        type=str,
        default=None,
        help="Ramulator YAML config used for the run (used to infer DRAM/timing preset).",
    )
    parser.add_argument(
        "--tck-ps",
        type=float,
        default=None,
        help="Override tCK in picoseconds directly (highest priority).",
    )
    args = parser.parse_args()

    info = {}
    info_path = None
    if args.info_json:
        info_path = args.info_json
    elif args.dump_dir:
        info_path = os.path.join(args.dump_dir, "pi0_attn_info.json")
    if info_path and os.path.exists(info_path):
        try:
            with open(info_path, "r", encoding="utf-8") as f:
                info = json.load(f)
        except Exception:
            info = {}
    if args.layers == 1 and "num_layers" in info:
        args.layers = int(info["num_layers"])
    if args.batch == 1 and "batch_size" in info:
        args.batch = int(info["batch_size"])
    if args.total_heads is None and "num_heads" in info:
        args.total_heads = int(info["num_heads"])

    if args.input == "-":
        import sys

        lines = sys.stdin
    else:
        lines = _read_file(args.input)

    stats = _parse_lines(lines)
    if stats.cycles == 0:
        raise SystemExit("No memory_system_cycles found in input.")

    tck_ps, timing_source = _resolve_tck_ps(
        rate_mtps=args.rate_mtps,
        tck_ps_override=args.tck_ps,
        config_yaml=args.config_yaml,
    )
    time_us = stats.cycles * tck_ps / 1e6

    bytes_dict = _estimate_bytes(stats, args.prefetch_bytes)

    # Scaling
    scale = args.layers * args.batch
    head_scale = 1.0
    if args.total_heads is not None and args.heads_per_hbm > 0:
        head_scale = args.total_heads / args.heads_per_hbm
        scale *= head_scale

    def scaled(v: float) -> float:
        return v * scale

    print("Parsed ramulator output")
    print(f"  cycles: {stats.cycles}")
    print(f"  tCK_ps: {tck_ps:.3f}")
    print(f"  timing_source: {timing_source}")
    print(f"  time_us (per trace): {time_us:.3f}")
    print("  command counts:")
    print(f"    softmax: {stats.softmax}")
    print(f"    mvgb: {stats.mvgb}")
    print(f"    mvsb: {stats.mvsb}")
    print(f"    wrgb: {stats.wrgb}")
    print(f"    mac_ab: {stats.mac_ab}")
    print(f"    mac_sb: {stats.mac_sb}")
    print(f"    mac_pb: {stats.mac_pb}")
    print("  bytes (per trace):")
    for k, v in bytes_dict.items():
        print(f"    {k}: {v} ({_format_bytes(v)})")
    mem_acc_bw_trace_gbps = _bw_gbps(bytes_dict["mem_acc"], time_us)
    print("  effective bandwidth:")
    print(f"    mem_acc_bw_GBps (per trace): {mem_acc_bw_trace_gbps:.3f}")

    print("  scaled totals:")
    if args.total_heads is not None:
        print(f"    head_scale: {head_scale:.3f} (total_heads / heads_per_hbm)")
    print(f"    layers: {args.layers}, batch: {args.batch}")
    time_us_total = scaled(time_us)
    print(f"    time_us_total: {time_us_total:.3f}")
    for k, v in bytes_dict.items():
        sv = int(scaled(v))
        print(f"    {k}_total: {sv} ({_format_bytes(sv)})")
    mem_acc_total = scaled(bytes_dict["mem_acc"])
    mem_acc_bw_total_gbps = _bw_gbps(mem_acc_total, time_us_total)
    print(f"    mem_acc_bw_GBps_total: {mem_acc_bw_total_gbps:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
