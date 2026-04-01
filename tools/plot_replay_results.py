#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def _format_bar_value(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:.0f}"
    if abs(value) >= 100:
        return f"{value:.1f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.3f}"


def _ecdf(values: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.array([]), np.array([])
    x = np.sort(arr)
    y = np.arange(1, x.size + 1) / x.size
    return x, y


def _safe_prefix(path: Path, prefix: Optional[str]) -> str:
    if prefix:
        return prefix
    return path.stem


def _plot_latency_ecdf(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False
    for col, label in [
        ("ttft_ms", "TTFT"),
        ("e2e_ms", "E2E"),
        ("mean_tbt_ms", "Mean TBT"),
    ]:
        if col not in df:
            continue
        x, y = _ecdf(df[col].dropna())
        if x.size == 0:
            continue
        ax.plot(x, y, label=label, linewidth=2)
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xscale("log")
    ax.set_xlabel("Latency (ms, log scale)")
    ax.set_ylabel("ECDF")
    ax.set_title("Replay Latency ECDF")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_arrival_scatter(df: pd.DataFrame, out_path: Path) -> None:
    cols = [c for c in ["ttft_ms", "e2e_ms"] if c in df]
    if not cols or "arrival_ms" not in df:
        return
    fig, axes = plt.subplots(1, len(cols), figsize=(7 * len(cols), 5), sharex=True)
    if len(cols) == 1:
        axes = [axes]
    clipped = (df.get("clipped_lin", False).astype(str).str.lower() == "true") | (
        df.get("clipped_lout", False).astype(str).str.lower() == "true"
    )
    for ax, col in zip(axes, cols):
        ax.scatter(df.loc[~clipped, "arrival_ms"], df.loc[~clipped, col], s=8, alpha=0.35, label="in-range")
        if clipped.any():
            ax.scatter(df.loc[clipped, "arrival_ms"], df.loc[clipped, col], s=10, alpha=0.6, label="clipped")
        ax.set_xlabel("Arrival time (ms)")
        ax.set_ylabel(f"{col} (ms)")
        ax.set_title(f"Arrival vs {col}")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_predicted_vs_actual(df: pd.DataFrame, out_path: Path) -> None:
    pred_col = "latest_predicted_finish_ms" if "latest_predicted_finish_ms" in df else "chosen_predicted_finish_ms"
    if (pred_col not in df or "arrival_ms" not in df or "e2e_ms" not in df):
        return
    cols = [pred_col, "arrival_ms", "e2e_ms"]
    has_lost = "is_lost" in df.columns
    if has_lost:
        cols.append("is_lost")
    tmp = df[cols].dropna(subset=[pred_col, "arrival_ms", "e2e_ms"]).copy()
    if tmp.empty:
        return
    tmp["predicted_e2e_ms"] = tmp[pred_col] - tmp["arrival_ms"]
    tmp = tmp[tmp["predicted_e2e_ms"].notna()]
    if tmp.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    if has_lost:
        lost_mask = tmp["is_lost"].astype(str).str.lower().isin({"1", "true", "yes"})
        ax.scatter(tmp.loc[~lost_mask, "predicted_e2e_ms"],
                   tmp.loc[~lost_mask, "e2e_ms"],
                   s=10,
                   alpha=0.35,
                   label="non-lost")
        if lost_mask.any():
            ax.scatter(tmp.loc[lost_mask, "predicted_e2e_ms"],
                       tmp.loc[lost_mask, "e2e_ms"],
                       s=16,
                       alpha=0.7,
                       color="tab:orange",
                       label="lost")
    else:
        ax.scatter(tmp["predicted_e2e_ms"], tmp["e2e_ms"], s=10, alpha=0.35)
    lo = min(tmp["predicted_e2e_ms"].min(), tmp["e2e_ms"].min())
    hi = max(tmp["predicted_e2e_ms"].max(), tmp["e2e_ms"].max())
    ax.plot([lo, hi], [lo, hi], "--", color="black", linewidth=1.5, label="ideal")
    ax.set_xlabel("Predicted E2E (ms)")
    ax.set_ylabel("Actual E2E (ms)")
    ax.set_title("Predicted E2E vs Actual E2E")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_predicted_finish_vs_actual_finish(df: pd.DataFrame, out_path: Path) -> None:
    pred_col = "latest_predicted_finish_ms" if "latest_predicted_finish_ms" in df else "chosen_predicted_finish_ms"
    if pred_col not in df or "arrival_ms" not in df or "e2e_ms" not in df:
        return
    cols = [pred_col, "arrival_ms", "e2e_ms"]
    has_lost = "is_lost" in df.columns
    if has_lost:
        cols.append("is_lost")
    tmp = df[cols].dropna(subset=[pred_col, "arrival_ms", "e2e_ms"]).copy()
    if tmp.empty:
        return
    tmp["actual_finish_ms"] = tmp["arrival_ms"] + tmp["e2e_ms"]
    fig, ax = plt.subplots(figsize=(6, 6))
    if has_lost:
        lost_mask = tmp["is_lost"].astype(str).str.lower().isin({"1", "true", "yes"})
        ax.scatter(tmp.loc[~lost_mask, pred_col],
                   tmp.loc[~lost_mask, "actual_finish_ms"],
                   s=10,
                   alpha=0.35,
                   label="non-lost")
        if lost_mask.any():
            ax.scatter(tmp.loc[lost_mask, pred_col],
                       tmp.loc[lost_mask, "actual_finish_ms"],
                       s=16,
                       alpha=0.7,
                       color="tab:orange",
                       label="lost")
    else:
        ax.scatter(tmp[pred_col], tmp["actual_finish_ms"], s=10, alpha=0.35)
    lo = min(tmp[pred_col].min(), tmp["actual_finish_ms"].min())
    hi = max(tmp[pred_col].max(), tmp["actual_finish_ms"].max())
    ax.plot([lo, hi], [lo, hi], "--", color="black", linewidth=1.5, label="ideal")
    ax.set_xlabel("Predicted Finish (ms)")
    ax.set_ylabel("Actual Finish (ms)")
    ax.set_title("Predicted Finish vs Actual Finish")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_predicted_vs_actual_loglog(df: pd.DataFrame, out_path: Path) -> None:
    pred_col = "latest_predicted_finish_ms" if "latest_predicted_finish_ms" in df else "chosen_predicted_finish_ms"
    if (pred_col not in df or "arrival_ms" not in df or "e2e_ms" not in df):
        return
    cols = [pred_col, "arrival_ms", "e2e_ms"]
    has_lost = "is_lost" in df.columns
    if has_lost:
        cols.append("is_lost")
    tmp = df[cols].dropna(subset=[pred_col, "arrival_ms", "e2e_ms"]).copy()
    if tmp.empty:
        return
    tmp["predicted_e2e_ms"] = tmp[pred_col] - tmp["arrival_ms"]
    tmp = tmp[(tmp["predicted_e2e_ms"] > 0) & (tmp["e2e_ms"] > 0)]
    if tmp.empty:
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    if has_lost:
        lost_mask = tmp["is_lost"].astype(str).str.lower().isin({"1", "true", "yes"})
        ax.scatter(tmp.loc[~lost_mask, "predicted_e2e_ms"],
                   tmp.loc[~lost_mask, "e2e_ms"],
                   s=10,
                   alpha=0.35,
                   label="non-lost")
        if lost_mask.any():
            ax.scatter(tmp.loc[lost_mask, "predicted_e2e_ms"],
                       tmp.loc[lost_mask, "e2e_ms"],
                       s=16,
                       alpha=0.7,
                       color="tab:orange",
                       label="lost")
    else:
        ax.scatter(tmp["predicted_e2e_ms"], tmp["e2e_ms"], s=10, alpha=0.35)
    lo = min(tmp["predicted_e2e_ms"].min(), tmp["e2e_ms"].min())
    hi = max(tmp["predicted_e2e_ms"].max(), tmp["e2e_ms"].max())
    ax.plot([lo, hi], [lo, hi], "--", color="black", linewidth=1.5, label="ideal")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Predicted E2E (ms, log)")
    ax.set_ylabel("Actual E2E (ms, log)")
    ax.set_title("Predicted E2E vs Actual E2E (Log-Log)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_route_mix(df: pd.DataFrame, summary: Optional[dict], out_path: Path) -> None:
    route_counts = df["route"].fillna("None").value_counts().sort_index()
    reason_counts = df["route_decision_reason"].fillna("None").value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    route_bars = axes[0].bar(route_counts.index.astype(str), route_counts.values)
    axes[0].set_title("Route Counts")
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].grid(True, axis="y", alpha=0.3)
    ymax = float(route_counts.max()) if len(route_counts) else 0.0
    if ymax > 0:
        axes[0].set_ylim(top=ymax * 1.15)
    axes[0].bar_label(route_bars, labels=[_format_bar_value(v) for v in route_counts.values], padding=3, fontsize=8)

    reason_bars = axes[1].bar(reason_counts.index.astype(str), reason_counts.values)
    axes[1].set_title("Decision Reasons")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].grid(True, axis="y", alpha=0.3)
    ymax = float(reason_counts.max()) if len(reason_counts) else 0.0
    if ymax > 0:
        axes[1].set_ylim(top=ymax * 1.15)
    axes[1].bar_label(reason_bars, labels=[_format_bar_value(v) for v in reason_counts.values], padding=3, fontsize=8)

    if summary:
        fig.suptitle(
            f"Replay Summary: tok/s={summary.get('throughput_tokps', float('nan')):.2f}, "
            f"GPU util={summary.get('gpu_util', float('nan')):.3f}, "
            f"PIM util={summary.get('pim_util', float('nan')):.3f}",
            fontsize=11,
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_slack_hist(df: pd.DataFrame, out_path: Path) -> None:
    if "chosen_slack_ms" not in df:
        return
    vals = pd.to_numeric(df["chosen_slack_ms"], errors="coerce").dropna()
    if vals.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(vals, bins=50, alpha=0.8)
    ax.axvline(0.0, color="red", linestyle="--", linewidth=2, label="deadline boundary")
    ax.set_xlabel("Chosen slack (ms)")
    ax.set_ylabel("Request count")
    ax.set_title("Slack Distribution")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_queue_pressure_hist(df: pd.DataFrame, out_path: Path) -> None:
    if "chosen_queue_pressure_ms" not in df:
        return
    vals = pd.to_numeric(df["chosen_queue_pressure_ms"], errors="coerce").dropna()
    if vals.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(vals, bins=50, alpha=0.8)
    ax.set_xlabel("Chosen queue pressure (ms)")
    ax.set_ylabel("Request count")
    ax.set_title("Queue Pressure Distribution")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_queue_pressure_vs_e2e(df: pd.DataFrame, out_path: Path) -> None:
    if "chosen_queue_pressure_ms" not in df or "e2e_ms" not in df:
        return
    tmp = df[["chosen_queue_pressure_ms", "e2e_ms"]].dropna()
    if tmp.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(tmp["chosen_queue_pressure_ms"], tmp["e2e_ms"], s=10, alpha=0.35)
    ax.set_xlabel("Chosen queue pressure (ms)")
    ax.set_ylabel("Actual E2E (ms)")
    ax.set_title("Queue Pressure vs Actual E2E")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _parse_route_candidates(raw: object) -> list[dict]:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return []
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    text = str(raw).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return []
    if not isinstance(parsed, list):
        return []
    return [x for x in parsed if isinstance(x, dict)]


def _as_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def _build_route_selection_delta_df(df: pd.DataFrame,
                                    events_df: pd.DataFrame) -> pd.DataFrame:
    required_req_cols = {"request_id", "arrival_ms", "context_tokens", "generated_tokens", "route"}
    required_evt_cols = {"event", "request_id"}
    if not required_req_cols.issubset(df.columns) or not required_evt_cols.issubset(events_df.columns):
        return pd.DataFrame()

    req_df = df.copy()
    req_df["request_id"] = pd.to_numeric(req_df["request_id"], errors="coerce")
    req_df = req_df.dropna(subset=["request_id"]).copy()
    req_df["request_id"] = req_df["request_id"].astype(int)
    req_rows = req_df.set_index("request_id").to_dict(orient="index")

    events = events_df.copy()
    events["_row_order"] = range(len(events))
    events["request_id"] = pd.to_numeric(events["request_id"], errors="coerce")
    events["sim_now_ms"] = pd.to_numeric(events.get("sim_now_ms"), errors="coerce")
    events = events.dropna(subset=["request_id"]).copy()
    events["request_id"] = events["request_id"].astype(int)
    events = events.sort_values(["sim_now_ms", "_row_order"], na_position="last")

    latest_eval_candidates: dict[int, list[dict]] = {}
    rows: list[dict[str, object]] = []

    for _, row in events.iterrows():
        request_id = int(row["request_id"])
        event = str(row.get("event", ""))
        if event == "admission_eval":
            candidates = _parse_route_candidates(row.get("route_candidates"))
            if candidates:
                latest_eval_candidates[request_id] = candidates
            continue
        if event != "admit_route":
            continue

        candidates = _parse_route_candidates(row.get("route_candidates"))
        source = "admit_route"
        if not candidates:
            candidates = latest_eval_candidates.get(request_id, [])
            source = "admission_eval_fallback"
        if not candidates:
            continue

        req = req_rows.get(request_id, {})
        arrival_ms = _as_float(row.get("request_arrival_ms"))
        if not np.isfinite(arrival_ms):
            arrival_ms = _as_float(req.get("arrival_ms"))

        cand_by_route = {str(c.get("route")): c for c in candidates}
        gpu = cand_by_route.get("gpu_only", {})
        hybrid = cand_by_route.get("lpddr5_pim_bank", {})

        gpu_finish = _as_float(gpu.get("predicted_finish_ms"))
        hybrid_finish = _as_float(hybrid.get("predicted_finish_ms"))
        gpu_e2e = gpu_finish - arrival_ms if np.isfinite(gpu_finish) and np.isfinite(arrival_ms) else float("nan")
        hybrid_e2e = hybrid_finish - arrival_ms if np.isfinite(hybrid_finish) and np.isfinite(arrival_ms) else float("nan")

        gpu_prefill_energy = _as_float(gpu.get("prefill_energy_nj"))
        hybrid_prefill_energy = _as_float(hybrid.get("prefill_energy_nj"))
        gpu_decode_total_energy = _as_float(gpu.get("decode_total_energy_nj"))
        hybrid_decode_total_energy = _as_float(hybrid.get("decode_total_energy_nj"))
        gpu_total_energy = _as_float(gpu.get("request_total_energy_nj"))
        hybrid_total_energy = _as_float(hybrid.get("request_total_energy_nj"))
        gpu_incremental_energy = _as_float(gpu.get("predicted_incremental_energy_nj"))
        hybrid_incremental_energy = _as_float(hybrid.get("predicted_incremental_energy_nj"))
        gpu_incremental_decode_energy = _as_float(gpu.get("predicted_incremental_decode_energy_nj"))
        hybrid_incremental_decode_energy = _as_float(hybrid.get("predicted_incremental_decode_energy_nj"))

        gpu_scheduler_energy = gpu_incremental_energy
        if not np.isfinite(gpu_scheduler_energy):
            gpu_scheduler_energy = gpu_incremental_decode_energy
        hybrid_scheduler_energy = hybrid_incremental_energy
        if not np.isfinite(hybrid_scheduler_energy):
            hybrid_scheduler_energy = hybrid_incremental_decode_energy

        rows.append({
            "request_id": request_id,
            "arrival_ms": arrival_ms,
            "context_tokens": req.get("context_tokens"),
            "generated_tokens": req.get("generated_tokens"),
            "chosen_route": row.get("chosen_route", req.get("route")),
            "decision_reason": row.get("decision_reason", req.get("route_decision_reason")),
            "candidate_source": source,
            "gpu_predicted_finish_ms": gpu_finish,
            "hybrid_predicted_finish_ms": hybrid_finish,
            "gpu_predicted_e2e_ms": gpu_e2e,
            "hybrid_predicted_e2e_ms": hybrid_e2e,
            "finish_delta_hybrid_minus_gpu_ms": hybrid_finish - gpu_finish,
            "e2e_delta_hybrid_minus_gpu_ms": hybrid_e2e - gpu_e2e,
            "gpu_predicted_gpu_wait_ms": _as_float(gpu.get("predicted_gpu_wait_ms")),
            "hybrid_predicted_gpu_wait_ms": _as_float(hybrid.get("predicted_gpu_wait_ms")),
            "gpu_predicted_pim_wait_ms": _as_float(gpu.get("predicted_pim_wait_ms")),
            "hybrid_predicted_pim_wait_ms": _as_float(hybrid.get("predicted_pim_wait_ms")),
            "gpu_route_active_requests": _as_float(gpu.get("route_active_requests")),
            "hybrid_route_active_requests": _as_float(hybrid.get("route_active_requests")),
            "gpu_route_waiting_prefill_count": _as_float(gpu.get("route_waiting_prefill_count")),
            "hybrid_route_waiting_prefill_count": _as_float(hybrid.get("route_waiting_prefill_count")),
            "gpu_route_waiting_decode_count": _as_float(gpu.get("route_waiting_decode_count")),
            "hybrid_route_waiting_decode_count": _as_float(hybrid.get("route_waiting_decode_count")),
            "gpu_route_pending_decode_tokens": _as_float(gpu.get("route_pending_decode_tokens")),
            "hybrid_route_pending_decode_tokens": _as_float(hybrid.get("route_pending_decode_tokens")),
            "gpu_prefill_energy_nj": gpu_prefill_energy,
            "hybrid_prefill_energy_nj": hybrid_prefill_energy,
            "gpu_decode_total_energy_nj": gpu_decode_total_energy,
            "hybrid_decode_total_energy_nj": hybrid_decode_total_energy,
            "gpu_predicted_incremental_energy_nj": gpu_incremental_energy,
            "hybrid_predicted_incremental_energy_nj": hybrid_incremental_energy,
            "gpu_predicted_incremental_decode_energy_nj": gpu_incremental_decode_energy,
            "hybrid_predicted_incremental_decode_energy_nj": hybrid_incremental_decode_energy,
            "gpu_scheduler_energy_nj": gpu_scheduler_energy,
            "hybrid_scheduler_energy_nj": hybrid_scheduler_energy,
            "gpu_total_request_energy_nj": gpu_total_energy,
            "hybrid_total_request_energy_nj": hybrid_total_energy,
            "scheduler_energy_delta_hybrid_minus_gpu_nj": (
                hybrid_scheduler_energy - gpu_scheduler_energy
            ),
            "prefill_energy_delta_hybrid_minus_gpu_nj": hybrid_prefill_energy - gpu_prefill_energy,
            "decode_total_energy_delta_hybrid_minus_gpu_nj": (
                hybrid_decode_total_energy - gpu_decode_total_energy
            ),
            "total_energy_delta_hybrid_minus_gpu_nj": hybrid_total_energy - gpu_total_energy,
            "gpu_eligible": gpu.get("eligible"),
            "hybrid_eligible": hybrid.get("eligible"),
            "gpu_reason": gpu.get("reason"),
            "hybrid_reason": hybrid.get("reason"),
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("request_id").reset_index(drop=True)


def _plot_route_selection_delta_scatter(route_df: pd.DataFrame,
                                        out_path: Path,
                                        energy_latency_guard_ms: Optional[float] = None,
                                        route_policy: Optional[str] = None) -> None:
    if route_df.empty:
        return
    tmp = route_df[["finish_delta_hybrid_minus_gpu_ms",
                    "scheduler_energy_delta_hybrid_minus_gpu_nj",
                    "chosen_route"]].dropna()
    if tmp.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {
        "gpu_only": "tab:blue",
        "lpddr5_pim_bank": "tab:orange",
    }
    for route, grp in tmp.groupby("chosen_route"):
        ax.scatter(grp["finish_delta_hybrid_minus_gpu_ms"],
                   grp["scheduler_energy_delta_hybrid_minus_gpu_nj"],
                   s=30,
                   alpha=0.8,
                   color=colors.get(str(route), "tab:gray"),
                   label=str(route))
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:blue", markersize=8, label="chosen gpu_only"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:orange", markersize=8, label="chosen lpddr5_pim_bank"),
    ]
    if energy_latency_guard_ms is not None and np.isfinite(energy_latency_guard_ms):
        guard = max(0.0, float(energy_latency_guard_ms))
        if guard > 0.0:
            ax.axvline(-guard, color="tab:green", linestyle=":", linewidth=1.5)
            ax.axvline(+guard, color="tab:green", linestyle=":", linewidth=1.5)
            ax.axvspan(-guard, +guard, color="tab:green", alpha=0.08)
            ylim = ax.get_ylim()
            y_text = ylim[1] - 0.06 * (ylim[1] - ylim[0])
            ax.text(0.0, y_text, f"energy-choice band |Δfinish| <= {guard:g} ms",
                    ha="center", va="top", fontsize=9, color="tab:green")
            legend_handles.append(Line2D([0], [0],
                                         color="tab:green",
                                         linestyle=":",
                                         linewidth=1.5,
                                         label=f"latency guard ±{guard:g} ms"))

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_pad = 0.04 * (xlim[1] - xlim[0])
    y_pad = 0.05 * (ylim[1] - ylim[0])
    label_box = dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="none")
    ax.text(xlim[0] + x_pad, ylim[1] - y_pad,
            "Hybrid faster\nGPU lower energy",
            ha="left", va="top", fontsize=8.5, color="tab:purple", bbox=label_box)
    ax.text(xlim[1] - x_pad, ylim[1] - y_pad,
            "GPU wins both",
            ha="right", va="top", fontsize=8.5, color="tab:blue", bbox=label_box)
    ax.text(xlim[0] + x_pad, ylim[0] + y_pad,
            "Hybrid wins both",
            ha="left", va="bottom", fontsize=8.5, color="tab:orange", bbox=label_box)
    ax.text(xlim[1] - x_pad, ylim[0] + y_pad,
            "GPU faster\nHybrid lower energy",
            ha="right", va="bottom", fontsize=8.5, color="tab:green", bbox=label_box)

    ax.set_xlabel("Finish delta: hybrid - gpu (ms)")
    ax.set_ylabel("Scheduler energy delta: hybrid - gpu (nJ)")
    ax.set_title("Route Selection: Predicted Time / Scheduler-Energy Delta Per Request")
    ax.grid(True, alpha=0.3)
    ax.legend(handles=legend_handles, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_route_selection_delta_by_request(route_df: pd.DataFrame, out_path: Path) -> None:
    if route_df.empty:
        return
    tmp = route_df[["request_id",
                    "finish_delta_hybrid_minus_gpu_ms",
                    "scheduler_energy_delta_hybrid_minus_gpu_nj",
                    "chosen_route"]].dropna(subset=["request_id"])
    if tmp.empty:
        return
    colors = tmp["chosen_route"].map({
        "gpu_only": "tab:blue",
        "lpddr5_pim_bank": "tab:orange",
    }).fillna("tab:gray")

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].bar(tmp["request_id"], tmp["finish_delta_hybrid_minus_gpu_ms"], color=colors)
    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("hybrid - gpu finish (ms)")
    axes[0].set_title("Route Selection Delta by Request")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(tmp["request_id"], tmp["scheduler_energy_delta_hybrid_minus_gpu_nj"], color=colors)
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_xlabel("request_id")
    axes[1].set_ylabel("hybrid - gpu scheduler energy (nJ)")
    axes[1].grid(True, axis="y", alpha=0.3)

    legend_handles = [
        Patch(facecolor="tab:blue", label="chosen gpu_only"),
        Patch(facecolor="tab:orange", label="chosen lpddr5_pim_bank"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.0, label="zero delta"),
    ]
    axes[0].legend(handles=legend_handles, loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_prediction_evolution(df: pd.DataFrame,
                               events_df: pd.DataFrame,
                               min_request_id: int,
                               max_request_id: int,
                               out_path: Path) -> None:
    if max_request_id < min_request_id:
        return
    required_event_cols = {"event", "sim_now_ms", "request_id"}
    if not required_event_cols.issubset(events_df.columns):
        return
    if not {"request_id", "arrival_ms", "e2e_ms"}.issubset(df.columns):
        return

    req_df = df.copy()
    req_df["request_id"] = pd.to_numeric(req_df["request_id"], errors="coerce")
    req_df["arrival_ms"] = pd.to_numeric(req_df["arrival_ms"], errors="coerce")
    req_df["e2e_ms"] = pd.to_numeric(req_df["e2e_ms"], errors="coerce")
    req_df = req_df.dropna(subset=["request_id", "arrival_ms", "e2e_ms"])
    req_df["request_id"] = req_df["request_id"].astype(int)
    req_df = req_df[(req_df["request_id"] >= min_request_id) &
                    (req_df["request_id"] <= max_request_id)]
    if req_df.empty:
        return

    events = events_df.copy()
    events["request_id"] = pd.to_numeric(events["request_id"], errors="coerce")
    events["sim_now_ms"] = pd.to_numeric(events["sim_now_ms"], errors="coerce")
    events = events.dropna(subset=["request_id", "sim_now_ms"])
    events["request_id"] = events["request_id"].astype(int)
    events = events[(events["request_id"] >= min_request_id) &
                    (events["request_id"] <= max_request_id)]
    if events.empty:
        return

    history: dict[int, list[tuple[float, float]]] = {}
    for _, row in events.iterrows():
        rid = int(row["request_id"])
        event = str(row.get("event", ""))
        sim_now_ms = float(row["sim_now_ms"])
        predicted_finish_ms: Optional[float] = None

        if event == "admit_route":
            predicted_finish_ms = pd.to_numeric(row.get("predicted_finish_ms"), errors="coerce")
        elif event == "prediction_refresh":
            predicted_finish_ms = pd.to_numeric(row.get("new_latest_predicted_finish_ms"), errors="coerce")
        elif event == "decode_rebind_decision":
            chosen_route = row.get("chosen_route")
            candidates = _parse_route_candidates(row.get("route_candidates"))
            for cand in candidates:
                if cand.get("route") == chosen_route:
                    predicted_finish_ms = pd.to_numeric(cand.get("predicted_finish_ms"), errors="coerce")
                    break

        if predicted_finish_ms is None or not np.isfinite(predicted_finish_ms):
            continue
        history.setdefault(rid, []).append((sim_now_ms, float(predicted_finish_ms)))

    if not history:
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.get_cmap("tab20")
    plotted = False

    for color_idx, request_id in enumerate(sorted(history)):
        points = sorted(history[request_id], key=lambda x: x[0])
        if not points:
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        color = cmap(color_idx % 20)
        ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.8, color=color, label=f"req {request_id}")
        req_row = req_df[req_df["request_id"] == request_id]
        if not req_row.empty:
            actual_finish_ms = float(req_row.iloc[0]["arrival_ms"] + req_row.iloc[0]["e2e_ms"])
            if np.isfinite(actual_finish_ms):
                last_x = xs[-1]
                last_pred_finish_ms = ys[-1]
                if np.isclose(last_pred_finish_ms, actual_finish_ms, rtol=0.0, atol=1e-6):
                    connector_color = "gray"
                elif last_pred_finish_ms < actual_finish_ms:
                    connector_color = "red"
                else:
                    connector_color = "green"
                ax.plot([last_x, actual_finish_ms],
                        [last_pred_finish_ms, actual_finish_ms],
                        linestyle="--",
                        linewidth=1.5,
                        color=connector_color,
                        alpha=0.9)
                ax.scatter([actual_finish_ms],
                           [actual_finish_ms],
                           s=42,
                           color=color,
                           marker="D",
                           edgecolor="black",
                           linewidth=0.5)
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_xlabel("Simulation time of prediction update (ms)")
    ax.set_ylabel("Predicted / actual finish time (ms)")
    ax.set_title(f"Prediction Evolution for request_id in [{min_request_id}, {max_request_id}]")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2 if len(history) > 6 else 1, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description="Plot replay result diagnostics from per-request CSV and summary JSON.")
    p.add_argument("--requests-csv", type=Path, required=True, help="Per-request replay CSV")
    p.add_argument("--summary-json", type=Path, default=None, help="Optional replay summary JSON")
    p.add_argument("--debug-events-csv", type=Path, default=None, help="Optional debug-events CSV for prediction-evolution plot")
    p.add_argument("--prediction-history-min-request-id",
                   type=int,
                   default=0,
                   help="If set together with --debug-events-csv, lower bound of request ids to include in prediction-evolution plot")
    p.add_argument("--prediction-history-max-request-id",
                   type=int,
                   default=None,
                   help="If set together with --debug-events-csv, upper bound of request ids to include in prediction-evolution plot")
    p.add_argument("--out-dir", type=Path, default=Path("cluster_outputs/replay_plots"), help="Output directory")
    p.add_argument("--prefix", type=str, default=None, help="Output file prefix")
    args = p.parse_args()

    df = pd.read_csv(args.requests_csv)
    summary = None
    if args.summary_json and args.summary_json.exists():
        with args.summary_json.open() as f:
            summary = json.load(f)
    energy_latency_guard_ms = None
    route_policy = None
    if summary is not None:
        local = summary.get("local_scheduling", {})
        if isinstance(local, dict):
            value = local.get("route_policy")
            if value is not None:
                route_policy = str(value)
            value = local.get("energy_latency_guard_ms")
            if value is not None:
                try:
                    energy_latency_guard_ms = float(value)
                except (TypeError, ValueError):
                    energy_latency_guard_ms = None

    args.out_dir.mkdir(parents=True, exist_ok=True)
    prefix = _safe_prefix(args.requests_csv, args.prefix)

    _plot_latency_ecdf(df, args.out_dir / f"{prefix}_latency_ecdf.png")
    _plot_arrival_scatter(df, args.out_dir / f"{prefix}_arrival_scatter.png")
    _plot_predicted_vs_actual(df, args.out_dir / f"{prefix}_predicted_vs_actual.png")
    _plot_predicted_finish_vs_actual_finish(df, args.out_dir / f"{prefix}_predicted_finish_vs_actual_finish.png")
    _plot_predicted_vs_actual_loglog(df, args.out_dir / f"{prefix}_predicted_vs_actual_loglog.png")
    _plot_route_mix(df, summary, args.out_dir / f"{prefix}_route_mix.png")
    _plot_slack_hist(df, args.out_dir / f"{prefix}_slack_hist.png")
    _plot_queue_pressure_hist(df, args.out_dir / f"{prefix}_queue_pressure_hist.png")
    _plot_queue_pressure_vs_e2e(df, args.out_dir / f"{prefix}_queue_pressure_vs_e2e.png")
    if args.debug_events_csv and args.debug_events_csv.exists():
        events_df = pd.read_csv(args.debug_events_csv, low_memory=False)
        route_df = _build_route_selection_delta_df(df, events_df)
        if not route_df.empty:
            route_df.to_csv(args.out_dir / f"{prefix}_route_selection_deltas.csv", index=False)
            _plot_route_selection_delta_scatter(
                route_df,
                args.out_dir / f"{prefix}_route_selection_delta_scatter.png",
                energy_latency_guard_ms=energy_latency_guard_ms,
                route_policy=route_policy,
            )
            _plot_route_selection_delta_by_request(
                route_df,
                args.out_dir / f"{prefix}_route_selection_delta_by_request.png",
            )
        if args.prediction_history_max_request_id is not None:
            _plot_prediction_evolution(df,
                                       events_df,
                                       args.prediction_history_min_request_id,
                                       args.prediction_history_max_request_id,
                                       args.out_dir / f"{prefix}_prediction_evolution.png")

    print(f"Wrote plots to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
