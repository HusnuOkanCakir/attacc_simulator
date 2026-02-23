#!/usr/bin/env python3
"""
Hybrid roofline plot:
  - GPU-only kernels from Nsight Compute CSV
  - LPDDR5/HBM PIM attention point from Ramulator stdout

This is a pragmatic comparison plot, not a pure NCU-vs-NCU roofline:
the PIM point is derived from simulator command counts and timing.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import pandas as pd


NVTX_COLS = [
    "thread Domain:Push/Pop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg",
    "Id:Domain:Start/Stop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg",
]


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


def _detect_encoding(csv_path: Path) -> str:
    start = csv_path.read_bytes()[:4]
    if start.startswith(b"\xff\xfe") or start.startswith(b"\xfe\xff"):
        return "utf-16"
    return "utf-8"


def _to_float(value) -> float:
    if value is None:
        return math.nan
    s = str(value).strip().strip('"')
    if not s or s.lower() in {"nan", "na", "n/a", "inf", "-inf"}:
        return math.nan
    s = s.replace(" ", "")
    if "," in s:
        if "." in s and s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        elif "." not in s:
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")
    elif s.count(".") > 1:
        s = s.replace(".", "")
    try:
        return float(s)
    except ValueError:
        return math.nan


def _unit_scale(unit) -> float:
    if unit is None:
        return 1.0
    u = str(unit).strip().lower()
    table = {
        "hz": 1.0,
        "khz": 1e3,
        "mhz": 1e6,
        "ghz": 1e9,
        "byte/s": 1.0,
        "bytes/s": 1.0,
        "kbyte/s": 1e3,
        "kbytes/s": 1e3,
        "mbyte/s": 1e6,
        "mbytes/s": 1e6,
        "gbyte/s": 1e9,
        "gbytes/s": 1e9,
        "tbyte/s": 1e12,
        "tbytes/s": 1e12,
        "byte/cycle": 1.0,
        "bytes/cycle": 1.0,
        "kbyte/cycle": 1e3,
        "kbytes/cycle": 1e3,
        "mbyte/cycle": 1e6,
        "mbytes/cycle": 1e6,
        "gbyte/cycle": 1e9,
        "gbytes/cycle": 1e9,
        "ns": 1e-9,
        "us": 1e-6,
        "ms": 1e-3,
        "s": 1.0,
    }
    return table.get(u, 1.0)


def _load_ncu_points(csv_path: Path) -> pd.DataFrame:
    enc = _detect_encoding(csv_path)
    header = pd.read_csv(csv_path, nrows=0, encoding=enc).columns.tolist()

    per_cycle_cols = [
        "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed",
        "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed",
        "derived__smsp__sass_thread_inst_executed_op_ffma_pred_on_x2",
    ]
    bytes_col = None
    for c in ["dram__bytes.sum.per_second", "dram__bytes.sum", "dram__bytes.avg"]:
        if c in header:
            bytes_col = c
            break
    if bytes_col is None:
        raise RuntimeError("No DRAM bytes column found in NCU CSV.")

    cycles_per_sec_col = None
    for c in ["smsp__cycles_elapsed.avg.per_second", "sm__cycles_elapsed.avg.per_second"]:
        if c in header:
            cycles_per_sec_col = c
            break
    if cycles_per_sec_col is None:
        raise RuntimeError("No cycles-per-second column found in NCU CSV.")

    usecols = set(per_cycle_cols + [bytes_col, cycles_per_sec_col, "gpu__time_duration.sum", "Kernel Name", "ID"])
    for c in NVTX_COLS:
        if c in header:
            usecols.add(c)
    for c in [
        "dram__bytes.sum.peak_sustained",
        "derived__sm__sass_thread_inst_executed_op_ffma_pred_on_x2",
        "sm__cycles_elapsed.avg.per_second",
        "dram__cycles_elapsed.avg.per_second",
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    ]:
        if c in header:
            usecols.add(c)

    df = pd.read_csv(csv_path, usecols=list(usecols), encoding=enc, engine="python")

    # Unit row handling (NCU CSV often has first row with units).
    units_row = df.iloc[0] if len(df) > 0 else None
    has_units = False
    if units_row is not None:
        probe_cols = [bytes_col, cycles_per_sec_col, "gpu__time_duration.sum"]
        for c in probe_cols:
            if c in df.columns:
                v = units_row.get(c)
                if isinstance(v, str) and any(ch.isalpha() for ch in v):
                    has_units = True
                    break
    if has_units:
        df = df.iloc[1:].copy()

    def scale_col(col: str) -> float:
        if not has_units or units_row is None:
            return 1.0
        return _unit_scale(units_row.get(col))

    for c in per_cycle_cols:
        if c not in df.columns:
            raise RuntimeError(f"Missing required column: {c}")
        df[c] = df[c].map(_to_float)

    df[bytes_col] = df[bytes_col].map(_to_float) * scale_col(bytes_col)
    df[cycles_per_sec_col] = df[cycles_per_sec_col].map(_to_float) * scale_col(cycles_per_sec_col)
    if "gpu__time_duration.sum" in df.columns:
        df["gpu__time_duration.sum"] = df["gpu__time_duration.sum"].map(_to_float) * scale_col("gpu__time_duration.sum")
    else:
        df["gpu__time_duration.sum"] = math.nan

    # Keep optional peaks for roofline ceiling inference.
    for c in [
        "dram__bytes.sum.peak_sustained",
        "derived__sm__sass_thread_inst_executed_op_ffma_pred_on_x2",
        "sm__cycles_elapsed.avg.per_second",
        "dram__cycles_elapsed.avg.per_second",
    ]:
        if c in df.columns:
            df[c] = df[c].map(_to_float) * scale_col(c)

    ops_per_cycle = (
        df["smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed"]
        + df["smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed"]
        + df["derived__smsp__sass_thread_inst_executed_op_ffma_pred_on_x2"]
    )
    df["ops_per_sec"] = ops_per_cycle * df[cycles_per_sec_col]
    df["bytes_per_sec"] = df[bytes_col]
    df["time_s"] = df["gpu__time_duration.sum"]
    df["ai"] = df["ops_per_sec"] / df["bytes_per_sec"]
    df["perf"] = df["ops_per_sec"]

    df = df[(df["ops_per_sec"] > 0) & (df["bytes_per_sec"] > 0) & (df["ai"] > 0)].copy()
    if df.empty:
        raise RuntimeError("No valid NCU rows after filtering.")

    return df


def _export_rep_to_csv(rep_path: Path, ncu_bin: str) -> Path:
    out_csv = rep_path.with_suffix(".csv")
    cmd = [ncu_bin, "--import", str(rep_path), "--page", "raw", "--csv"]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        subprocess.run(cmd, stdout=f, check=True)
    return out_csv


def _row_text(row: pd.Series) -> str:
    vals = []
    if "Kernel Name" in row and pd.notna(row["Kernel Name"]):
        vals.append(str(row["Kernel Name"]))
    for c in NVTX_COLS:
        if c in row and pd.notna(row[c]):
            vals.append(str(row[c]))
    return " ".join(vals)


def _aggregate_ncu_attention(df: pd.DataFrame, patterns: list[re.Pattern]) -> dict:
    matched = []
    for _, row in df.iterrows():
        text = _row_text(row)
        if any(p.search(text) for p in patterns):
            matched.append(row)
    if not matched:
        # Fallback to all rows if filtering misses due to different NVTX naming.
        mdf = df
        work = (mdf["ops_per_sec"] * mdf["time_s"]).sum()
        traffic = (mdf["bytes_per_sec"] * mdf["time_s"]).sum()
        time_s = mdf["time_s"].sum()
        if work <= 0 or traffic <= 0 or time_s <= 0:
            raise RuntimeError("No NCU rows matched attention regex and fallback aggregate is invalid.")
        return {
            "name": "GPU aggregate (fallback: all kernels)",
            "time_s": float(time_s),
            "work_ops": float(work),
            "traffic_bytes": float(traffic),
            "perf": float(work / time_s),
            "ai": float(work / traffic),
            "bw": float(traffic / time_s),
            "num_rows": int(len(mdf)),
        }

    mdf = pd.DataFrame(matched)
    # Aggregate by work/traffic over total time.
    work = (mdf["ops_per_sec"] * mdf["time_s"]).sum()
    traffic = (mdf["bytes_per_sec"] * mdf["time_s"]).sum()
    time_s = mdf["time_s"].sum()
    if work <= 0 or traffic <= 0 or time_s <= 0:
        raise RuntimeError("Invalid aggregate values from matched attention rows.")
    return {
        "name": "GPU attention (NCU aggregate)",
        "time_s": float(time_s),
        "work_ops": float(work),
        "traffic_bytes": float(traffic),
        "perf": float(work / time_s),
        "ai": float(work / traffic),
        "bw": float(traffic / time_s),
        "num_rows": int(len(mdf)),
    }


def _parse_lines(lines: Iterable[str]) -> RamulatorStats:
    stats = RamulatorStats()
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if "memory_system_cycles" in s:
            stats.cycles = int(s.split()[-1])
        elif "pim_softmax_requests" in s:
            stats.softmax = int(s.split()[-1])
        elif "pim_move_to_gemv_buffer_requests" in s:
            stats.mvgb = int(s.split()[-1])
        elif "pim_move_to_softmax_buffer_requests" in s:
            stats.mvsb = int(s.split()[-1])
        elif "pim_write_to_gemv_buffer_requests" in s:
            stats.wrgb = int(s.split()[-1])
        elif "pim_mac_all_bank_requests" in s:
            stats.mac_ab = int(s.split()[-1])
        elif "pim_mac_same_bank_requests" in s:
            stats.mac_sb = int(s.split()[-1])
        elif "pim_mac_per_bank_requests" in s:
            stats.mac_pb = int(s.split()[-1])
    return stats


def _parse_dram_info_from_yaml(config_yaml: Path) -> tuple[Optional[str], Optional[str]]:
    if not config_yaml.exists():
        return None, None

    dram_impl = None
    timing_preset = None
    in_dram = False
    dram_indent = -1
    in_timing = False
    timing_indent = -1

    for raw in config_yaml.read_text(encoding="utf-8").splitlines():
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


def _hbm_rate_mtps_from_preset(preset: str) -> Optional[int]:
    m = re.search(r"HBM3_([0-9]+(?:\.[0-9]+)?)Gbps", preset)
    if not m:
        return None
    return int(round(float(m.group(1)) * 1000.0))


def _tck_ps_from_rate(rate_mtps: int) -> float:
    # HBM3 model in this repo uses QDR pin rate.
    return 1e6 / (rate_mtps / 4.0)


def _resolve_tck_ps(
    rate_mtps: int,
    tck_ps_override: Optional[float],
    config_yaml: Optional[Path],
) -> tuple[float, str]:
    if tck_ps_override is not None:
        return float(tck_ps_override), "user-override"
    if config_yaml:
        dram_impl, timing_preset = _parse_dram_info_from_yaml(config_yaml)
        if dram_impl and dram_impl.startswith("LPDDR5"):
            if timing_preset in LPDDR5_TCK_PS_BY_PRESET:
                return LPDDR5_TCK_PS_BY_PRESET[timing_preset], f"yaml({dram_impl}/{timing_preset})"
            return 1250.0, f"fallback-lpddr5({timing_preset})"
        if dram_impl and dram_impl.startswith("HBM3"):
            if timing_preset:
                rate = _hbm_rate_mtps_from_preset(timing_preset)
                if rate is not None:
                    return _tck_ps_from_rate(rate), f"yaml({dram_impl}/{timing_preset})"
    return _tck_ps_from_rate(rate_mtps), f"default-rate({rate_mtps} MT/s)"


def _read_pi0_dump_info(dump_dir: Optional[Path]) -> dict:
    if not dump_dir:
        return {}
    info_path = dump_dir / "pi0_attn_info.json"
    if not info_path.exists():
        return {}
    try:
        return json.loads(info_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _build_pim_point(
    ram_output: Path,
    config_yaml: Optional[Path],
    prefetch_bytes: int,
    dbyte: int,
    layers: int,
    batch: int,
    heads_per_hbm: int,
    total_heads: Optional[int],
    rate_mtps: int,
    tck_ps_override: Optional[float],
) -> dict:
    stats = _parse_lines(ram_output.read_text(encoding="utf-8").splitlines())
    if stats.cycles <= 0:
        raise RuntimeError("No memory_system_cycles found in ramulator output.")
    mac_total = stats.mac_ab + stats.mac_sb + stats.mac_pb
    if mac_total <= 0:
        raise RuntimeError("No MAC commands found in ramulator output.")

    tck_ps, tck_src = _resolve_tck_ps(rate_mtps, tck_ps_override, config_yaml)
    head_scale = 1.0
    if total_heads is not None and heads_per_hbm > 0:
        head_scale = float(total_heads) / float(heads_per_hbm)
    scale = float(layers) * float(batch) * head_scale

    # Command-model estimate:
    #  - each MAC command consumes prefetch_bytes / dbyte elements
    #  - each element is one FMA => 2 FLOPs
    elems_per_mac = float(prefetch_bytes) / float(dbyte)
    ops_per_trace = float(mac_total) * elems_per_mac * 2.0
    bytes_per_trace = float(mac_total) * float(prefetch_bytes)
    time_s_per_trace = float(stats.cycles) * tck_ps * 1e-12

    work = ops_per_trace * scale
    traffic = bytes_per_trace * scale
    time_s = time_s_per_trace * scale
    if work <= 0 or traffic <= 0 or time_s <= 0:
        raise RuntimeError("Invalid PIM aggregate values.")

    return {
        "name": "LPDDR5/HBM PIM attention (sim)",
        "time_s": float(time_s),
        "work_ops": float(work),
        "traffic_bytes": float(traffic),
        "perf": float(work / time_s),
        "ai": float(work / traffic),
        "bw": float(traffic / time_s),
        "cycles_per_trace": int(stats.cycles),
        "mac_per_trace": int(mac_total),
        "scale": float(scale),
        "tck_ps": float(tck_ps),
        "tck_source": tck_src,
    }


def _infer_roofline_peaks(df: pd.DataFrame, use_peak_formula: bool, peak_flops: float, peak_bw: float) -> tuple[float, float]:
    pf = peak_flops
    pb = peak_bw
    if use_peak_formula:
        req = [
            "derived__sm__sass_thread_inst_executed_op_ffma_pred_on_x2",
            "sm__cycles_elapsed.avg.per_second",
            "dram__bytes.sum.peak_sustained",
            "dram__cycles_elapsed.avg.per_second",
        ]
        if all(c in df.columns for c in req):
            pf = max(pf, (df[req[0]] * df[req[1]]).max())
            pb = max(pb, (df[req[2]] * df[req[3]]).max())
    if pb <= 0:
        pb = float(df["bytes_per_sec"].max()) * 1.2
    if pf <= 0:
        pf = float(df["ops_per_sec"].max()) * 1.2
    return pf, pb


def main() -> int:
    parser = argparse.ArgumentParser(description="Hybrid roofline: GPU NCU + PIM simulation point.")
    parser.add_argument("--gpu", required=True, type=Path, help="GPU profile input: NCU CSV or .ncu-rep from GPU-only run.")
    parser.add_argument("--ncu", type=str, default="ncu", help="ncu binary path (used if --gpu is .ncu-rep).")
    parser.add_argument("--ram-output", required=True, type=Path, help="Ramulator stdout log (e.g., ram_out.txt).")
    parser.add_argument("--config-yaml", type=Path, default=None, help="Ramulator YAML used for the ramulator run.")
    parser.add_argument("--dump-dir", type=Path, default=None, help="Optional PI0 dump dir (for default layers/batch/heads).")
    parser.add_argument("--layers", type=int, default=1, help="Scale factor: number of layers.")
    parser.add_argument("--batch", type=int, default=1, help="Scale factor: batch size.")
    parser.add_argument("--heads-per-hbm", type=int, default=1, help="Heads represented by one generated trace.")
    parser.add_argument("--total-heads", type=int, default=None, help="Total model heads (for head scaling).")
    parser.add_argument("--dbyte", type=int, default=2, help="Data bytes per element in PIM trace (e.g., 2 for fp16).")
    parser.add_argument("--prefetch-bytes", type=int, default=32, help="Prefetch granularity bytes per MAC command.")
    parser.add_argument("--rate-mtps", type=int, default=5200, help="Fallback HBM rate if YAML is missing.")
    parser.add_argument("--tck-ps", type=float, default=None, help="Optional direct tCK override.")
    parser.add_argument(
        "--attn-regex",
        action="append",
        default=[r"BLOCK\.attn\.core_layer", r"attn", r"attention"],
        help="Regex for attention kernels in NCU CSV (repeatable).",
    )
    parser.add_argument("--out", type=Path, default=Path("hybrid_roofline.png"), help="Output PNG path.")
    parser.add_argument("--out-summary", type=Path, default=Path("hybrid_roofline_summary.csv"), help="Output summary CSV.")
    parser.add_argument("--title", type=str, default="Hybrid Roofline: GPU-only vs PIM+GPU", help="Plot title.")
    parser.add_argument("--peak-flops", type=float, default=0.0, help="Optional roofline compute peak (FLOP/s).")
    parser.add_argument("--peak-bw", type=float, default=0.0, help="Optional roofline bandwidth peak (B/s).")
    parser.add_argument("--use-peak-formula", action="store_true", help="Use Nsight peak formula columns if present.")
    args = parser.parse_args()

    info = _read_pi0_dump_info(args.dump_dir)
    layers = args.layers if args.layers != 1 else int(info.get("num_layers", args.layers))
    batch = args.batch if args.batch != 1 else int(info.get("batch_size", args.batch))
    total_heads = args.total_heads if args.total_heads is not None else info.get("num_heads")
    if total_heads is not None:
        total_heads = int(total_heads)

    gpu_path = args.gpu
    if not gpu_path.exists():
        raise FileNotFoundError(f"--gpu path not found: {gpu_path}")

    if gpu_path.suffix.lower() == ".ncu-rep":
        ncu_bin = shutil.which(args.ncu) if args.ncu == "ncu" else args.ncu
        if not ncu_bin:
            raise FileNotFoundError("ncu not found on PATH. Pass --ncu /path/to/ncu.")
        csv_path = _export_rep_to_csv(gpu_path, ncu_bin)
    else:
        csv_path = gpu_path

    ncu_df = _load_ncu_points(csv_path)
    patterns = [re.compile(p) for p in args.attn_regex]
    gpu_attn = _aggregate_ncu_attention(ncu_df, patterns)
    pim_attn = _build_pim_point(
        ram_output=args.ram_output,
        config_yaml=args.config_yaml,
        prefetch_bytes=args.prefetch_bytes,
        dbyte=args.dbyte,
        layers=layers,
        batch=batch,
        heads_per_hbm=args.heads_per_hbm,
        total_heads=total_heads,
        rate_mtps=args.rate_mtps,
        tck_ps_override=args.tck_ps,
    )

    peak_flops, peak_bw = _infer_roofline_peaks(
        ncu_df, args.use_peak_formula, args.peak_flops, args.peak_bw
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(
        ncu_df["ai"],
        ncu_df["perf"],
        s=18,
        alpha=0.35,
        color="gray",
        edgecolors="none",
        label="GPU-only kernels (NCU)",
    )

    ax.scatter(
        [gpu_attn["ai"]],
        [gpu_attn["perf"]],
        s=160,
        marker="*",
        color="tab:blue",
        edgecolors="black",
        linewidths=0.5,
        label="GPU-only attention aggregate",
        zorder=5,
    )
    ax.scatter(
        [pim_attn["ai"]],
        [pim_attn["perf"]],
        s=120,
        marker="X",
        color="tab:red",
        edgecolors="black",
        linewidths=0.5,
        label="PIM attention aggregate (sim)",
        zorder=5,
    )

    ridge_x = peak_flops / peak_bw
    xmin = min(float(ncu_df["ai"].min()), gpu_attn["ai"], pim_attn["ai"]) / 5.0
    xmax = max(float(ncu_df["ai"].max()), gpu_attn["ai"], pim_attn["ai"]) * 5.0
    xmin = min(xmin, ridge_x / 100.0)
    xmax = max(xmax, ridge_x * 100.0)
    xs = [xmin, xmax]
    ax.plot(xs, [peak_bw * x for x in xs], "--", color="orange", linewidth=1.2, label="Memory roof")
    ax.plot(xs, [peak_flops, peak_flops], "--", color="red", linewidth=1.2, label="Compute roof")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic Intensity [FLOP/byte]")
    ax.set_ylabel("Performance [FLOP/s]")
    ax.set_title(args.title)
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(loc="best")
    ax.set_xlim(xmin, xmax)
    ymin = min(float(ncu_df["perf"].min()), gpu_attn["perf"], pim_attn["perf"]) / 5.0
    ymax = max(float(ncu_df["perf"].max()), gpu_attn["perf"], pim_attn["perf"], peak_flops) * 2.0
    ax.set_ylim(ymin, ymax)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=220)

    summary = pd.DataFrame(
        [
            {
                "name": gpu_attn["name"],
                "ai_flop_per_byte": gpu_attn["ai"],
                "perf_flop_per_s": gpu_attn["perf"],
                "bw_byte_per_s": gpu_attn["bw"],
                "time_s": gpu_attn["time_s"],
                "work_ops": gpu_attn["work_ops"],
                "traffic_bytes": gpu_attn["traffic_bytes"],
                "extra": f"matched_ncu_rows={gpu_attn['num_rows']}",
            },
            {
                "name": pim_attn["name"],
                "ai_flop_per_byte": pim_attn["ai"],
                "perf_flop_per_s": pim_attn["perf"],
                "bw_byte_per_s": pim_attn["bw"],
                "time_s": pim_attn["time_s"],
                "work_ops": pim_attn["work_ops"],
                "traffic_bytes": pim_attn["traffic_bytes"],
                "extra": (
                    f"cycles_per_trace={pim_attn['cycles_per_trace']};"
                    f"mac_per_trace={pim_attn['mac_per_trace']};"
                    f"scale={pim_attn['scale']:.3f};"
                    f"tck_ps={pim_attn['tck_ps']:.3f};"
                    f"tck_source={pim_attn['tck_source']}"
                ),
            },
        ]
    )
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_summary, index=False)

    print(f"Wrote plot: {args.out}")
    print(f"Wrote summary: {args.out_summary}")
    print("GPU attention aggregate:")
    print(f"  AI={gpu_attn['ai']:.6g} FLOP/B, Perf={gpu_attn['perf']:.6g} FLOP/s, BW={gpu_attn['bw']:.6g} B/s")
    print("PIM attention aggregate (sim):")
    print(f"  AI={pim_attn['ai']:.6g} FLOP/B, Perf={pim_attn['perf']:.6g} FLOP/s, BW={pim_attn['bw']:.6g} B/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
