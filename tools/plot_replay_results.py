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
    if args.debug_events_csv and args.debug_events_csv.exists() and args.prediction_history_max_request_id is not None:
        events_df = pd.read_csv(args.debug_events_csv, low_memory=False)
        _plot_prediction_evolution(df,
                                   events_df,
                                   args.prediction_history_min_request_id,
                                   args.prediction_history_max_request_id,
                                   args.out_dir / f"{prefix}_prediction_evolution.png")

    print(f"Wrote plots to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
