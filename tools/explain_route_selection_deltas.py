#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _fmt_ms(value: float) -> str:
    return f"{abs(float(value)):.1f} ms"


def _fmt_nj(value: float) -> str:
    return f"{abs(float(value)):.3g} nJ"


def _who_is_faster(delta_ms: float) -> str:
    if delta_ms < 0:
        return "hybrid"
    if delta_ms > 0:
        return "gpu"
    return "same_finish"


def _who_is_lower_energy(delta_nj: float) -> str:
    if delta_nj < 0:
        return "hybrid"
    if delta_nj > 0:
        return "gpu"
    return "same_energy"


def _guard_region(delta_ms: float, guard_ms: float | None) -> str:
    if guard_ms is None:
        return "no_guard_info"
    if abs(float(delta_ms)) <= float(guard_ms) + 1e-9:
        return f"inside {guard_ms:g} ms guard"
    return f"outside {guard_ms:g} ms guard"


def _load_guard_ms(summary_json: Path | None) -> float | None:
    if summary_json is None or not summary_json.exists():
        return None
    with summary_json.open() as f:
        summary = json.load(f)
    local = summary.get("local_scheduling", {})
    if not isinstance(local, dict):
        return None
    value = local.get("energy_latency_guard_ms")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_explanations(df: pd.DataFrame, guard_ms: float | None) -> list[str]:
    lines: list[str] = []
    for _, row in df.sort_values("request_id").iterrows():
        rid = int(row["request_id"])
        chosen_route = str(row.get("chosen_route", ""))
        reason = str(row.get("decision_reason", ""))
        dt = float(row["finish_delta_hybrid_minus_gpu_ms"])
        de = float(row["scheduler_energy_delta_hybrid_minus_gpu_nj"])

        faster = _who_is_faster(dt)
        lower_energy = _who_is_lower_energy(de)
        faster_clause = {
            "hybrid": f"hybrid faster by {_fmt_ms(dt)}",
            "gpu": f"gpu faster by {_fmt_ms(dt)}",
            "same_finish": "same predicted finish",
        }[faster]
        energy_clause = {
            "hybrid": f"hybrid lower scheduler energy by {_fmt_nj(de)}",
            "gpu": f"gpu lower scheduler energy by {_fmt_nj(de)}",
            "same_energy": "same scheduler energy",
        }[lower_energy]
        region_clause = _guard_region(dt, guard_ms)
        gpu_wait = float(row.get("gpu_predicted_gpu_wait_ms", float("nan")))
        hybrid_gpu_wait = float(row.get("hybrid_predicted_gpu_wait_ms", float("nan")))
        hybrid_pim_wait = float(row.get("hybrid_predicted_pim_wait_ms", float("nan")))
        gpu_active = int(float(row.get("gpu_route_active_requests", 0) or 0))
        hybrid_active = int(float(row.get("hybrid_route_active_requests", 0) or 0))
        gpu_decode_waiting = int(float(row.get("gpu_route_waiting_decode_count", 0) or 0))
        hybrid_decode_waiting = int(float(row.get("hybrid_route_waiting_decode_count", 0) or 0))
        gpu_pending_tokens = int(float(row.get("gpu_route_pending_decode_tokens", 0) or 0))
        hybrid_pending_tokens = int(float(row.get("hybrid_route_pending_decode_tokens", 0) or 0))
        line = (
            f"r{rid}: {faster_clause}, {energy_clause}, {region_clause} -> "
            f"chose {chosen_route}"
        )
        if reason and reason != "nan":
            line += f" ({reason})"
        line += (
            f"; waits[gpu={gpu_wait:.1f} ms, hybrid_gpu={hybrid_gpu_wait:.1f} ms, "
            f"hybrid_pim={hybrid_pim_wait:.1f} ms]"
        )
        line += (
            f"; route_load[gpu_active={gpu_active}, hybrid_active={hybrid_active}, "
            f"gpu_waiting_decode={gpu_decode_waiting}, hybrid_waiting_decode={hybrid_decode_waiting}, "
            f"gpu_pending_decode_tokens={gpu_pending_tokens}, "
            f"hybrid_pending_decode_tokens={hybrid_pending_tokens}]"
        )
        lines.append(line)
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description="Write compact human-readable route decision explanations.")
    ap.add_argument("--route-delta-csv", type=Path, required=True)
    ap.add_argument("--summary-json", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.route_delta_csv)
    required = {
        "request_id",
        "chosen_route",
        "decision_reason",
        "finish_delta_hybrid_minus_gpu_ms",
        "scheduler_energy_delta_hybrid_minus_gpu_nj",
    }
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")

    guard_ms = _load_guard_ms(args.summary_json)
    lines = build_explanations(df, guard_ms)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print(f"Wrote route explanations to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
