#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


STATE_COLORS = {
    "waiting_admission": "#bdbdbd",
    "prefill_window": "#9ecae1",
    "prefill_running": "#2b6cb0",
    "decode_window": "#a1d99b",
    "decode_running": "#2f7d32",
}
MAX_FIGURE_HEIGHT_IN = 40.0
TIMELINE_IN_PER_REQUEST = 0.015
QUEUE_SLOT_IN_PER_LEVEL = 0.3
DEFAULT_TIMELINE_WIDTH_IN = 12.0
DEFAULT_TIMELINE_DPI = 120


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


def _to_str(v: Any) -> str:
    if _is_nan(v):
        return ""
    try:
        return str(v).strip()
    except Exception:
        return ""


def _safe_prefix(path: Path, override: Optional[str]) -> str:
    return override if override else path.stem


def _queue_str(queue: List[int]) -> str:
    return "[" + ",".join(str(x) for x in queue) + "]"


def _color_for_request(request_id: int):
    cmap = plt.get_cmap("tab20")
    return cmap(request_id % 20)


def _route_kind(route: Optional[str]) -> Optional[str]:
    if not route:
        return None
    if route == "gpu_only":
        return "gpu"
    if "pim" in route:
        return "gpu+pim"
    return route


def _parse_request_ids(value: Any) -> List[int]:
    if _is_nan(value):
        return []
    if isinstance(value, list):
        out = []
        for item in value:
            item_i = _to_int(item)
            if item_i is not None:
                out.append(item_i)
        return out
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = None
        if isinstance(parsed, list):
            out = []
            for item in parsed:
                item_i = _to_int(item)
                if item_i is not None:
                    out.append(item_i)
            return out
        text = text.strip("[]")
        if not text:
            return []
        out = []
        for part in text.split(","):
            item_i = _to_int(part.strip())
            if item_i is not None:
                out.append(item_i)
        return out
    item_i = _to_int(value)
    return [item_i] if item_i is not None else []


def reconstruct_queue_snapshots(events_df: pd.DataFrame) -> pd.DataFrame:
    queue: List[int] = []
    rows: List[Dict[str, Any]] = []
    for _, row in events_df.sort_values("event_index").iterrows():
        event = str(row.get("event", ""))
        rid = _to_int(row.get("request_id"))
        changed = False
        note = ""
        reason = _to_str(row.get("reason", ""))
        decision_reason = _to_str(row.get("decision_reason", ""))
        chosen_route = _to_str(row.get("chosen_route", ""))
        chosen_route_kind = _route_kind(chosen_route if chosen_route else None)
        queue_wait_ms = _to_float(row.get("queue_wait_ms"))

        if event == "arrival_enqueued" and rid is not None:
            if rid not in queue:
                queue.append(rid)
                changed = True
                note = f"enqueue r{rid}"
        elif event in ("admit_route", "admit_drop") and rid is not None:
            if rid in queue:
                queue.remove(rid)
                changed = True
                note = f"dequeue r{rid} ({event})"
                if chosen_route_kind is not None:
                    note += f" route={chosen_route_kind}"
                    if chosen_route:
                        note += f"[{chosen_route}]"
                if decision_reason:
                    note += f" decision={decision_reason}"
        elif event == "admission_hold" and rid is not None:
            changed = True
            note = f"hold r{rid}"
            if reason:
                note += f" reason={reason}"
            if queue_wait_ms is not None:
                note += f" wait={queue_wait_ms:.1f}ms"
        elif event == "admission_best_effort" and rid is not None:
            changed = True
            note = f"best_effort r{rid}"
            if queue_wait_ms is not None:
                note += f" wait={queue_wait_ms:.1f}ms"
        elif event == "admission_bypass":
            changed = True
            b = _to_int(row.get("bypassed_request_id"))
            a = _to_int(row.get("admitted_request_id"))
            note = f"bypass r{b} by r{a}"

        if not changed:
            continue

        rows.append({
            "event_index": _to_int(row.get("event_index")),
            "sim_now_ms": _to_float(row.get("sim_now_ms")),
            "event": event,
            "request_id": rid,
            "bypassed_request_id": _to_int(row.get("bypassed_request_id")),
            "admitted_request_id": _to_int(row.get("admitted_request_id")),
            "bypass_count": _to_int(row.get("bypass_count")),
            "reason": reason if reason else None,
            "decision_reason": decision_reason if decision_reason else None,
            "chosen_route": chosen_route if chosen_route else None,
            "chosen_route_kind": chosen_route_kind,
            "queue_wait_ms": queue_wait_ms,
            "waiting_admission_count": len(queue),
            "waiting_queue_order": json.dumps(queue),
            "waiting_queue_order_pretty": _queue_str(queue),
            "note": note,
        })
    return pd.DataFrame(rows)


def plot_queue_counts(events_df: pd.DataFrame, out_path: Path) -> None:
    cols = ["sim_now_ms", "waiting_admission_count", "waiting_prefill_count", "waiting_decode_count"]
    if not set(cols).issubset(events_df.columns):
        return
    df = events_df.copy()
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["sim_now_ms"])
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.step(df["sim_now_ms"], df["waiting_admission_count"], where="post", label="waiting_admission", linewidth=2.0)
    ax.step(df["sim_now_ms"], df["waiting_prefill_count"], where="post", label="waiting_prefill", linewidth=2.0)
    ax.step(df["sim_now_ms"], df["waiting_decode_count"], where="post", label="waiting_decode", linewidth=2.0)

    bypass = df[df["event"] == "admission_bypass"]
    if not bypass.empty:
        ax.scatter(bypass["sim_now_ms"], bypass["waiting_admission_count"], marker="v", s=50, label="bypass", color="#d62728")
    admit = df[df["event"] == "admit_route"]
    if not admit.empty:
        ax.scatter(admit["sim_now_ms"], admit["waiting_admission_count"], marker="o", s=24, label="admit", color="#2ca02c", alpha=0.7)

    ax.set_xlabel("Simulation time (ms)")
    ax.set_ylabel("Request count")
    ax.set_title("Admission / Prefill / Decode Queue Counts")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_waiting_queue_slots(snapshots_df: pd.DataFrame, out_path: Path) -> None:
    if snapshots_df.empty:
        return
    queues = [json.loads(v) for v in snapshots_df["waiting_queue_order"]]
    max_len = max((len(q) for q in queues), default=0)
    if max_len == 0:
        return

    fig_h = min(MAX_FIGURE_HEIGHT_IN, max(4, QUEUE_SLOT_IN_PER_LEVEL * max_len + 1.5))
    fig, ax = plt.subplots(figsize=(11, fig_h))

    event_indices = snapshots_df["event_index"].tolist()
    for i, queue in enumerate(queues):
        x0 = event_indices[i]
        x1 = event_indices[i + 1] if i + 1 < len(event_indices) else x0 + 1
        if x1 <= x0:
            x1 = x0 + 1
        width = x1 - x0
        for slot, request_id in enumerate(queue):
            y = max_len - slot - 1
            color = _color_for_request(request_id)
            ax.broken_barh([(x0, width)], (y + 0.08, 0.84), facecolors=color, edgecolors="black", linewidth=0.5)
            if width >= 1:
                ax.text(x0 + width / 2, y + 0.5, f"r{request_id}", ha="center", va="center", fontsize=8)

    bypass_rows = snapshots_df[snapshots_df["event"] == "admission_bypass"]
    if not bypass_rows.empty:
        ax.scatter(bypass_rows["event_index"], [max_len + 0.35] * len(bypass_rows), marker="v", color="#d62728", s=40, label="bypass event")

    ax.set_xlabel("Debug event index")
    ax.set_ylabel("Waiting-admission slot (head at top)")
    ax.set_title("Waiting-Admission Queue Order")
    ax.set_xlim(event_indices[0], event_indices[-1] + 1)
    ax.set_ylim(0, max_len + 0.8)
    tick_step = max(1, int(math.ceil(max_len / 50.0)))
    tick_positions = []
    tick_labels = []
    for i in range(0, max_len, tick_step):
        tick_positions.append(i + 0.5)
        tick_labels.append(str(max_len - i - 1))
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)
    ax.grid(True, axis="x", alpha=0.25)
    if not bypass_rows.empty:
        ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_request_state_timeline(requests_df: pd.DataFrame,
                                out_path: Path,
                                *,
                                width_in: float = DEFAULT_TIMELINE_WIDTH_IN,
                                in_per_request: float = TIMELINE_IN_PER_REQUEST,
                                dpi: int = DEFAULT_TIMELINE_DPI,
                                max_height_in: float = MAX_FIGURE_HEIGHT_IN) -> None:
    required = {"request_id", "arrival_ms", "admission_time_ms", "ttft_ms", "e2e_ms", "route"}
    if not required.issubset(requests_df.columns):
        return
    df = requests_df.copy().sort_values(["arrival_ms", "request_id"])
    fig_h = min(max_height_in, max(4, in_per_request * len(df) + 1.5))
    fig, ax = plt.subplots(figsize=(width_in, fig_h))
    annotate_rows = len(df) <= 300

    yticks = []
    ylabels = []
    for row_idx, (_, row) in enumerate(df.iterrows()):
        request_id = int(row["request_id"])
        arrival = _to_float(row.get("arrival_ms"))
        admission = _to_float(row.get("admission_time_ms"))
        ttft = _to_float(row.get("ttft_ms"))
        e2e = _to_float(row.get("e2e_ms"))
        if arrival is None or e2e is None:
            continue
        if admission is None:
            admission = arrival
        prefill_end = arrival + ttft if ttft is not None else admission
        finish = arrival + e2e
        y = row_idx
        yticks.append(y + 0.4)
        forced = bool(row.get("admission_forced_best_effort", False))
        route = str(row.get("route", "-"))
        label = f"r{request_id}"
        if forced:
            label += " *"
        ylabels.append(label)

        if admission > arrival:
            ax.broken_barh([(arrival, admission - arrival)], (y + 0.05, 0.7), facecolors=STATE_COLORS["waiting_admission"], edgecolors="black", linewidth=0.4)
        if prefill_end > admission:
            ax.broken_barh([(admission, prefill_end - admission)], (y + 0.05, 0.7), facecolors=STATE_COLORS["prefill_window"], edgecolors="black", linewidth=0.4)
        if finish > prefill_end:
            ax.broken_barh([(prefill_end, finish - prefill_end)], (y + 0.05, 0.7), facecolors=STATE_COLORS["decode_window"], edgecolors="black", linewidth=0.4)
        if annotate_rows:
            ax.text(finish + 0.5, y + 0.4, route, va="center", fontsize=8)
        if forced and annotate_rows:
            ax.scatter([admission], [y + 0.4], marker="*", s=70, color="#d62728", zorder=3)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=STATE_COLORS[k], edgecolor="black", linewidth=0.4) for k in ("waiting_admission", "prefill_window", "decode_window")]
    ax.legend(handles, ["waiting_admission", "to_first_token_window", "post_first_token_window"], loc="lower right")
    ax.set_xlabel("Simulation time (ms)")
    ax.set_ylabel("Request")
    ax.set_title("Request Lifecycle Timeline (* = forced best-effort admission)")
    tick_step = max(1, int(math.ceil(len(yticks) / 80.0)))
    ax.set_yticks(yticks[::tick_step])
    ax.set_yticklabels(ylabels[::tick_step], fontsize=8 if len(yticks) <= 200 else 6)
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _build_running_segments(events_df: pd.DataFrame) -> Dict[int, Dict[str, List[tuple[float, float]]]]:
    segments: Dict[int, Dict[str, List[tuple[float, float]]]] = {}
    dispatch_df = events_df[events_df["event"] == "dispatch_task"].copy()
    if dispatch_df.empty:
        return segments
    for _, row in dispatch_df.iterrows():
        phase = str(row.get("phase", "")).strip()
        if phase not in ("prefill", "decode"):
            continue
        start = _to_float(row.get("task_earliest_start_ms"))
        dur = _to_float(row.get("task_e2e_ms"))
        if start is None or dur is None or dur < 0:
            continue
        for request_id in _parse_request_ids(row.get("request_ids")):
            req_segments = segments.setdefault(request_id, {"prefill": [], "decode": []})
            req_segments[phase].append((start, dur))
    for req_segments in segments.values():
        for phase in ("prefill", "decode"):
            req_segments[phase].sort(key=lambda item: (item[0], item[1]))
    return segments


def plot_request_execution_timeline(requests_df: pd.DataFrame,
                                    events_df: pd.DataFrame,
                                    out_path: Path,
                                    *,
                                    width_in: float = DEFAULT_TIMELINE_WIDTH_IN,
                                    in_per_request: float = TIMELINE_IN_PER_REQUEST,
                                    dpi: int = DEFAULT_TIMELINE_DPI,
                                    max_height_in: float = MAX_FIGURE_HEIGHT_IN) -> None:
    required = {"request_id", "arrival_ms", "admission_time_ms", "ttft_ms", "e2e_ms", "route"}
    if not required.issubset(requests_df.columns):
        return
    df = requests_df.copy().sort_values(["arrival_ms", "request_id"])
    running_segments = _build_running_segments(events_df)
    fig_h = min(max_height_in, max(4, in_per_request * len(df) + 1.5))
    fig, ax = plt.subplots(figsize=(width_in, fig_h))
    annotate_rows = len(df) <= 300

    yticks = []
    ylabels = []
    for row_idx, (_, row) in enumerate(df.iterrows()):
        request_id = int(row["request_id"])
        arrival = _to_float(row.get("arrival_ms"))
        admission = _to_float(row.get("admission_time_ms"))
        ttft = _to_float(row.get("ttft_ms"))
        e2e = _to_float(row.get("e2e_ms"))
        if arrival is None or e2e is None:
            continue
        if admission is None:
            admission = arrival
        prefill_end = arrival + ttft if ttft is not None else admission
        finish = arrival + e2e
        y = row_idx
        yticks.append(y + 0.4)
        forced = bool(row.get("admission_forced_best_effort", False))
        route = str(row.get("route", "-"))
        label = f"r{request_id}"
        if forced:
            label += " *"
        ylabels.append(label)

        if admission > arrival:
            ax.broken_barh([(arrival, admission - arrival)], (y + 0.05, 0.7),
                           facecolors=STATE_COLORS["waiting_admission"],
                           edgecolors="black", linewidth=0.4)

        if prefill_end > admission:
            ax.broken_barh([(admission, prefill_end - admission)], (y + 0.05, 0.7),
                           facecolors=STATE_COLORS["prefill_window"],
                           edgecolors="black", linewidth=0.4)

        if finish > prefill_end:
            ax.broken_barh([(prefill_end, finish - prefill_end)], (y + 0.05, 0.7),
                           facecolors=STATE_COLORS["decode_window"],
                           edgecolors="black", linewidth=0.4)

        req_segments = running_segments.get(request_id, {"prefill": [], "decode": []})
        for start, dur in req_segments.get("prefill", []):
            if dur > 0:
                ax.broken_barh([(start, dur)], (y + 0.05, 0.7),
                               facecolors=STATE_COLORS["prefill_running"],
                               edgecolors="black", linewidth=0.4)
            else:
                ax.scatter([start], [y + 0.4], marker="|", s=120,
                           color=STATE_COLORS["prefill_running"], zorder=3)
        for start, dur in req_segments.get("decode", []):
            if dur > 0:
                ax.broken_barh([(start, dur)], (y + 0.05, 0.7),
                               facecolors=STATE_COLORS["decode_running"],
                               edgecolors="black", linewidth=0.4)
            else:
                ax.scatter([start], [y + 0.4], marker="|", s=120,
                           color=STATE_COLORS["decode_running"], zorder=3)

        if annotate_rows:
            ax.text(finish + 0.5, y + 0.4, route, va="center", fontsize=8)
        if forced and annotate_rows:
            ax.scatter([admission], [y + 0.4], marker="*", s=70, color="#d62728", zorder=3)

    legend_keys = [
        "waiting_admission",
        "prefill_window",
        "prefill_running",
        "decode_window",
        "decode_running",
    ]
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=STATE_COLORS[k], edgecolor="black", linewidth=0.4)
        for k in legend_keys
    ]
    ax.legend(handles,
              ["waiting_admission", "waiting_prefill", "running_prefill", "waiting_decode", "running_decode"],
              loc="lower right")
    ax.set_xlabel("Simulation time (ms)")
    ax.set_ylabel("Request")
    ax.set_title("Request Execution Timeline (dark = running, light = waiting)")
    tick_step = max(1, int(math.ceil(len(yticks) / 80.0)))
    ax.set_yticks(yticks[::tick_step])
    ax.set_yticklabels(ylabels[::tick_step], fontsize=8 if len(yticks) <= 200 else 6)
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def write_snapshot_text(snapshots_df: pd.DataFrame, out_path: Path) -> None:
    with out_path.open("w") as f:
        for _, row in snapshots_df.iterrows():
            t = row.get("sim_now_ms")
            idx = row.get("event_index")
            queue = row.get("waiting_queue_order_pretty")
            note = row.get("note", "")
            f.write(f"[{idx}] t={t:.3f}ms queue={queue} {note}\n")


def main() -> int:
    p = argparse.ArgumentParser(description="Plot admission/debug queue dynamics from debug-events CSV.")
    p.add_argument("--events-csv", type=Path, required=True, help="Debug events CSV")
    p.add_argument("--requests-csv", type=Path, default=None, help="Optional per-request CSV for state timeline")
    p.add_argument("--out-dir", type=Path, default=Path("cluster_outputs/replay_plots"), help="Output directory")
    p.add_argument("--prefix", type=str, default=None, help="Output prefix")
    p.add_argument("--timeline-width-in",
                   type=float,
                   default=DEFAULT_TIMELINE_WIDTH_IN,
                   help="Figure width in inches for request timeline plots")
    p.add_argument("--timeline-in-per-request",
                   type=float,
                   default=TIMELINE_IN_PER_REQUEST,
                   help="Figure height budget in inches per request for request timeline plots")
    p.add_argument("--timeline-dpi",
                   type=int,
                   default=DEFAULT_TIMELINE_DPI,
                   help="Raster DPI for request timeline plots")
    p.add_argument("--timeline-max-height-in",
                   type=float,
                   default=MAX_FIGURE_HEIGHT_IN,
                   help="Maximum figure height in inches for request timeline plots")
    args = p.parse_args()

    if not args.events_csv.exists():
        raise FileNotFoundError(f"Debug events CSV not found: {args.events_csv}")
    events_df = pd.read_csv(args.events_csv, low_memory=False)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prefix = _safe_prefix(args.events_csv, args.prefix)

    snapshots_df = reconstruct_queue_snapshots(events_df)
    snapshots_csv = args.out_dir / f"{prefix}_queue_snapshots.csv"
    snapshots_txt = args.out_dir / f"{prefix}_queue_snapshots.txt"
    snapshots_df.to_csv(snapshots_csv, index=False)
    write_snapshot_text(snapshots_df, snapshots_txt)

    plot_queue_counts(events_df, args.out_dir / f"{prefix}_queue_counts.png")
    plot_waiting_queue_slots(snapshots_df, args.out_dir / f"{prefix}_waiting_queue_slots.png")

    if args.requests_csv is not None and args.requests_csv.exists():
        requests_df = pd.read_csv(args.requests_csv)
        plot_request_state_timeline(
            requests_df,
            args.out_dir / f"{prefix}_request_timeline.png",
            width_in=max(1.0, float(args.timeline_width_in)),
            in_per_request=max(0.001, float(args.timeline_in_per_request)),
            dpi=max(1, int(args.timeline_dpi)),
            max_height_in=max(1.0, float(args.timeline_max_height_in)),
        )
        plot_request_execution_timeline(requests_df,
                                        events_df,
                                        args.out_dir / f"{prefix}_request_execution_timeline.png",
                                        width_in=max(1.0, float(args.timeline_width_in)),
                                        in_per_request=max(0.001, float(args.timeline_in_per_request)),
                                        dpi=max(1, int(args.timeline_dpi)),
                                        max_height_in=max(1.0, float(args.timeline_max_height_in)))

    print(f"Wrote queue snapshots: {snapshots_csv}")
    print(f"Wrote queue timeline text: {snapshots_txt}")
    print(f"Wrote plots to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
