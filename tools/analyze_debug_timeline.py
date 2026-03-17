#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


def _is_nan(v: Any) -> bool:
    try:
        return bool(pd.isna(v))
    except Exception:
        return False


def _to_int(v: Any) -> Optional[int]:
    if _is_nan(v):
        return None
    try:
        return int(v)
    except Exception:
        try:
            return int(float(v))
        except Exception:
            return None


def _to_float(v: Any) -> Optional[float]:
    if _is_nan(v):
        return None
    try:
        out = float(v)
    except Exception:
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _loads_jsonish(v: Any) -> Any:
    if _is_nan(v):
        return None
    if isinstance(v, (list, dict)):
        return v
    s = str(v).strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def _int_list(v: Any) -> List[int]:
    obj = _loads_jsonish(v)
    if isinstance(obj, list):
        out: List[int] = []
        for x in obj:
            i = _to_int(x)
            if i is not None:
                out.append(i)
        return out
    i = _to_int(v)
    return [] if i is None else [i]


def _find_state_entry(state_field: Any, request_id: int) -> Dict[str, Any]:
    entries = _loads_jsonish(state_field)
    if not isinstance(entries, list):
        return {}
    for ent in entries:
        if not isinstance(ent, dict):
            continue
        rid = _to_int(ent.get("request_id"))
        if rid == request_id:
            return ent
    return {}


def _fmt_ms(v: Optional[float]) -> str:
    if v is None:
        return "-"
    return f"{v:.3f}ms"


def _fmt_int(v: Optional[int]) -> str:
    if v is None:
        return "-"
    return str(v)


def build_timelines(events_df: pd.DataFrame) -> Dict[int, List[Dict[str, Any]]]:
    timelines: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    decode_step_no: Dict[int, int] = defaultdict(int)

    for _, row in events_df.sort_values("event_index").iterrows():
        event = str(row.get("event", ""))
        idx = _to_int(row.get("event_index"))
        now_ms = _to_float(row.get("sim_now_ms"))
        if idx is None:
            continue

        if event in ("admit_route", "admit_drop"):
            rid = _to_int(row.get("request_id"))
            if rid is None:
                continue
            timelines[rid].append({
                "idx": idx,
                "time_ms": now_ms,
                "type": event,
                "route": row.get("chosen_route"),
                "decision_reason": row.get("decision_reason"),
                "pred_finish_ms": _to_float(row.get("predicted_finish_ms")),
                "pred_e2e_ms": _to_float(row.get("predicted_e2e_ms")),
                "pred_gpu_wait_ms": _to_float(row.get("predicted_gpu_wait_ms")),
                "pred_pim_wait_ms": _to_float(row.get("predicted_pim_wait_ms")),
                "pred_qp_ms": _to_float(row.get("predicted_queue_pressure_ms")),
            })
            continue

        if event == "dispatch_task":
            rids = _int_list(row.get("request_ids"))
            phase = str(row.get("phase", ""))
            for rid in rids:
                step = None
                if phase == "decode":
                    decode_step_no[rid] += 1
                    step = decode_step_no[rid]
                timelines[rid].append({
                    "idx": idx,
                    "time_ms": now_ms,
                    "type": "dispatch",
                    "phase": phase,
                    "decode_step": step,
                    "batch_size": _to_int(row.get("batch_size")),
                    "task_start_ms": _to_float(row.get("task_earliest_start_ms")),
                    "task_ready_ms": _to_float(row.get("task_ready_ms")),
                    "task_gpu_ms": _to_float(row.get("task_gpu_ms")),
                    "task_pim_ms": _to_float(row.get("task_pim_ms")),
                    "task_e2e_ms": _to_float(row.get("task_e2e_ms")),
                })
            continue

        if event == "complete_task":
            rids = _int_list(row.get("request_ids"))
            phase = str(row.get("phase", ""))
            finish_ms = _to_float(row.get("finish_ms"))
            for rid in rids:
                state_after = _find_state_entry(row.get("request_states_after"), rid)
                timelines[rid].append({
                    "idx": idx,
                    "time_ms": now_ms,
                    "type": "complete",
                    "phase": phase,
                    "batch_size": _to_int(row.get("batch_size")),
                    "finish_ms": finish_ms,
                    "state_after": state_after.get("state"),
                    "remaining_decode_tokens": _to_int(state_after.get("remaining_decode_tokens")),
                    "ready_ms_after": _to_float(state_after.get("ready_ms")),
                    "completion_time_ms": _to_float(state_after.get("completion_time_ms")),
                    "prefill_start_ms": _to_float(state_after.get("prefill_start_ms")),
                    "prefill_end_ms": _to_float(state_after.get("prefill_end_ms")),
                    "last_token_completion_ms": _to_float(state_after.get("last_token_completion_ms")),
                })
            continue

    for rid in list(timelines.keys()):
        timelines[rid] = sorted(timelines[rid], key=lambda x: (x["idx"], x.get("time_ms") or 0.0))
    return timelines


def print_timeline_for_request(request_id: int,
                               events: List[Dict[str, Any]],
                               req_row: Optional[pd.Series]) -> None:
    print(f"\n=== Request {request_id} ===")
    if req_row is not None:
        print(
            f"arrival={_fmt_ms(_to_float(req_row.get('arrival_ms')))} "
            f"Lin={_fmt_int(_to_int(req_row.get('context_tokens')))} "
            f"Lout={_fmt_int(_to_int(req_row.get('generated_tokens')))} "
            f"route={req_row.get('route', '-')}"
        )
        print(
            f"prefill_wait={_fmt_ms(_to_float(req_row.get('prefill_wait_ms')))} "
            f"ttft={_fmt_ms(_to_float(req_row.get('ttft_ms')))} "
            f"e2e={_fmt_ms(_to_float(req_row.get('e2e_ms')))}"
        )

    for ev in events:
        idx = ev["idx"]
        t = _fmt_ms(ev.get("time_ms"))
        et = ev["type"]
        if et in ("admit_route", "admit_drop"):
            print(
                f"[{idx:04d}] t={t} {et} "
                f"route={ev.get('route', '-')} reason={ev.get('decision_reason', '-')}"
            )
            print(
                f"       pred_finish={_fmt_ms(ev.get('pred_finish_ms'))} "
                f"pred_e2e={_fmt_ms(ev.get('pred_e2e_ms'))} "
                f"pred_gpu_wait={_fmt_ms(ev.get('pred_gpu_wait_ms'))} "
                f"pred_pim_wait={_fmt_ms(ev.get('pred_pim_wait_ms'))} "
                f"pred_qp={_fmt_ms(ev.get('pred_qp_ms'))}"
            )
        elif et == "dispatch":
            phase = ev.get("phase", "-")
            dstep = ev.get("decode_step")
            dstep_txt = f" step={dstep}" if dstep is not None else ""
            print(
                f"[{idx:04d}] t={t} dispatch {phase}{dstep_txt} "
                f"batch={_fmt_int(ev.get('batch_size'))} "
                f"start={_fmt_ms(ev.get('task_start_ms'))} ready={_fmt_ms(ev.get('task_ready_ms'))}"
            )
            print(
                f"       task_gpu={_fmt_ms(ev.get('task_gpu_ms'))} "
                f"task_pim={_fmt_ms(ev.get('task_pim_ms'))} "
                f"task_e2e={_fmt_ms(ev.get('task_e2e_ms'))}"
            )
        elif et == "complete":
            print(
                f"[{idx:04d}] t={t} complete {ev.get('phase', '-')} "
                f"batch={_fmt_int(ev.get('batch_size'))} finish={_fmt_ms(ev.get('finish_ms'))} "
                f"state_after={ev.get('state_after', '-')}"
            )
            print(
                f"       rem_decode={_fmt_int(ev.get('remaining_decode_tokens'))} "
                f"ready_after={_fmt_ms(ev.get('ready_ms_after'))} "
                f"completion={_fmt_ms(ev.get('completion_time_ms'))}"
            )


def main() -> int:
    p = argparse.ArgumentParser(
        description="Analyze replay debug timeline and print per-request event traces.")
    p.add_argument("--events-csv",
                   type=Path,
                   default=Path("cluster_outputs/debug_events_deterministic.csv"),
                   help="Debug events CSV written by run_trace_replay.py --debug-events-csv")
    p.add_argument("--requests-csv",
                   type=Path,
                   default=None,
                   help="Optional per-request CSV for final metrics/context")
    p.add_argument("--request-id",
                   type=int,
                   action="append",
                   default=[],
                   help="Request id to print (can repeat). Default: print all.")
    p.add_argument("--max-requests",
                   type=int,
                   default=0,
                   help="If > 0 and --request-id is not set, print only first N request timelines.")
    args = p.parse_args()

    if not args.events_csv.exists():
        raise FileNotFoundError(f"Debug events CSV not found: {args.events_csv}")
    events_df = pd.read_csv(args.events_csv)
    timelines = build_timelines(events_df)
    if not timelines:
        print("No per-request events found.")
        return 0

    req_df: Optional[pd.DataFrame] = None
    if args.requests_csv is not None:
        if not args.requests_csv.exists():
            raise FileNotFoundError(f"Requests CSV not found: {args.requests_csv}")
        req_df = pd.read_csv(args.requests_csv).set_index("request_id", drop=False)

    if args.request_id:
        request_ids = sorted(set(args.request_id))
    else:
        request_ids = sorted(timelines.keys())
        if args.max_requests > 0:
            request_ids = request_ids[:args.max_requests]

    print(f"Loaded {len(timelines)} request timelines from {args.events_csv}")
    if req_df is not None:
        print(f"Loaded per-request metrics from {args.requests_csv}")

    for rid in request_ids:
        evs = timelines.get(rid, [])
        if not evs:
            print(f"\n=== Request {rid} ===\n(no events)")
            continue
        req_row = None
        if req_df is not None and rid in req_df.index:
            req_row = req_df.loc[rid]
        print_timeline_for_request(rid, evs, req_row)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

