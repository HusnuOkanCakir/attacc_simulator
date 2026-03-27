#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.learned_cost_model import LearnedCostModel
from src.azure_trace import load_azure_llm_trace
from src.trace_replay import OutputCsvCostModel


DEFAULT_ROUTES = ("gpu_only", "lpddr5_pim_bank")


def _parse_profile_args(values: Sequence[str] | None) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for item in values or []:
        if "=" not in item:
            raise ValueError(f"Expected route=path format, got: {item}")
        route, path = item.split("=", 1)
        route = route.strip()
        path = path.strip()
        if not route or not path:
            raise ValueError(f"Invalid route=path mapping: {item}")
        mapping[route] = path
    return mapping


def _friendly_route(route: str) -> str:
    if route == "gpu_only":
        return "GPU"
    if "pim" in route:
        return "GPU+PIM"
    return route


def _load_cost_model(args: argparse.Namespace):
    if args.cost_model == "ml":
        route_to_model = _parse_profile_args(args.ml_profile)
        if not route_to_model:
            raise ValueError("--cost-model ml requires at least one --ml-profile route=path.pkl")
        return LearnedCostModel.from_pickles(route_to_model)

    route_to_csv = _parse_profile_args(args.profile)
    if not route_to_csv:
        raise ValueError("--cost-model table requires at least one --profile route=path.csv")
    return OutputCsvCostModel.from_route_csvs(route_to_csv)


def _intersection_bounds(cost_model, routes: Sequence[str]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    if isinstance(cost_model, LearnedCostModel):
        lin_lo = max(int(math.ceil(cost_model.route_bundles[r].bounds["Lin"][0])) for r in routes)
        lin_hi = min(int(math.floor(cost_model.route_bundles[r].bounds["Lin"][1])) for r in routes)
        lout_lo = max(int(math.ceil(cost_model.route_bundles[r].bounds["Lout"][0])) for r in routes)
        lout_hi = min(int(math.floor(cost_model.route_bundles[r].bounds["Lout"][1])) for r in routes)
        return (lin_lo, lin_hi), (lout_lo, lout_hi)

    lin_lo = max(min(p.lin for p in cost_model.route_points[r]) for r in routes)
    lin_hi = min(max(p.lin for p in cost_model.route_points[r]) for r in routes)
    lout_lo = max(min(p.lout for p in cost_model.route_points[r]) for r in routes)
    lout_hi = min(max(p.lout for p in cost_model.route_points[r]) for r in routes)
    return (int(lin_lo), int(lin_hi)), (int(lout_lo), int(lout_hi))


def _default_grid(lo: int, hi: int, count: int) -> List[int]:
    if count <= 1 or lo >= hi:
        return [int(lo)]
    values = np.geomspace(max(1, lo), hi, num=count)
    out = sorted({int(round(v)) for v in values} | {int(lo), int(hi)})
    out = [v for v in out if lo <= v <= hi]
    return out


def _estimate_grid(cost_model,
                   routes: Sequence[str],
                   lin_values: Sequence[int],
                   lout_values: Sequence[int],
                   bs: int,
                   unsupported_policy: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for route in routes:
        for lin in lin_values:
            for lout in lout_values:
                est = cost_model.estimate_request(route=route,
                                                  context_tokens=int(lin),
                                                  generated_tokens=int(lout),
                                                  bs=int(bs),
                                                  unsupported_policy=unsupported_policy,
                                                  lin_bucket=1,
                                                  lout_bucket=1)
                if est is None:
                    continue
                decode_total_energy_nj = float(est.decode_energy_nj) * float(est.decode_tokens)
                total_request_energy_nj = float(est.prefill_energy_nj) + decode_total_energy_nj
                rows.append({
                    "route": route,
                    "route_label": _friendly_route(route),
                    "lin": int(lin),
                    "lout": int(lout),
                    "bs": int(bs),
                    "mapped_lin": int(est.mapping.mapped_lin),
                    "mapped_lout": int(est.mapping.mapped_lout),
                    "mapped_bs": int(est.mapping.mapped_bs),
                    "prefill_energy_nj": float(est.prefill_energy_nj),
                    "decode_energy_per_token_nj": float(est.decode_energy_nj),
                    "decode_tokens": int(est.decode_tokens),
                    "decode_total_energy_nj": float(decode_total_energy_nj),
                    "total_request_energy_nj": float(total_request_energy_nj),
                    "prefill_e2e_ms": float(est.prefill_e2e_ms),
                    "decode_e2e_ms": float(est.decode_e2e_ms),
                })
    if not rows:
        raise RuntimeError("No estimates produced for the requested grid.")
    return pd.DataFrame(rows)


def _build_comparison(df: pd.DataFrame, routes: Sequence[str]) -> pd.DataFrame:
    gpu_route, hybrid_route = routes
    gpu = df[df["route"] == gpu_route].copy()
    hybrid = df[df["route"] == hybrid_route].copy()
    merged = gpu.merge(hybrid,
                       on=["lin", "lout", "bs"],
                       suffixes=("_gpu", "_hybrid"),
                       how="inner")
    if merged.empty:
        raise RuntimeError("No overlapping comparison points between routes.")

    metric_map = {
        "prefill": "prefill_energy_nj",
        "decode_total": "decode_total_energy_nj",
        "total": "total_request_energy_nj",
    }
    for key, metric in metric_map.items():
        diff_col = f"{key}_energy_diff_hybrid_minus_gpu_nj"
        winner_col = f"{key}_energy_winner"
        merged[diff_col] = merged[f"{metric}_hybrid"] - merged[f"{metric}_gpu"]
        winners: List[str] = []
        for diff in merged[diff_col]:
            if abs(float(diff)) <= 1e-9:
                winners.append("tie")
            elif float(diff) < 0.0:
                winners.append("gpu+pim")
            else:
                winners.append("gpu")
        merged[winner_col] = winners
    return merged


def _pivot(df: pd.DataFrame, value_col: str) -> Tuple[np.ndarray, List[int], List[int]]:
    pivot = df.pivot(index="lin", columns="lout", values=value_col).sort_index().sort_index(axis=1)
    return pivot.to_numpy(dtype=float), pivot.index.astype(int).tolist(), pivot.columns.astype(int).tolist()


def _plot_heatmap(arr: np.ndarray,
                  lin_values: Sequence[int],
                  lout_values: Sequence[int],
                  title: str,
                  cbar_label: str,
                  out_path: Path,
                  cmap: str = "viridis",
                  vmin: float | None = None,
                  vmax: float | None = None) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(arr, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(lout_values)))
    ax.set_xticklabels([str(v) for v in lout_values], rotation=45, ha="right")
    ax.set_yticks(range(len(lin_values)))
    ax.set_yticklabels([str(v) for v in lin_values])
    ax.set_xlabel("Generated Tokens (Lout)")
    ax.set_ylabel("Context Tokens (Lin)")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_route_pair(df: pd.DataFrame,
                     routes: Sequence[str],
                     metric: str,
                     title: str,
                     cbar_label: str,
                     out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    route_frames = [df[df["route"] == route] for route in routes]
    pivots = [_pivot(route_df, metric) for route_df in route_frames]
    vals = [p[0] for p in pivots]
    vmin = min(float(np.nanmin(v)) for v in vals)
    vmax = max(float(np.nanmax(v)) for v in vals)

    for ax, route, (arr, lin_values, lout_values) in zip(axes, routes, pivots):
        im = ax.imshow(arr, origin="lower", aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(lout_values)))
        ax.set_xticklabels([str(v) for v in lout_values], rotation=45, ha="right")
        ax.set_yticks(range(len(lin_values)))
        ax.set_yticklabels([str(v) for v in lin_values])
        ax.set_xlabel("Generated Tokens (Lout)")
        ax.set_ylabel("Context Tokens (Lin)")
        ax.set_title(_friendly_route(route))

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.95)
    cbar.set_label(cbar_label)
    fig.suptitle(title)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_diff_map(compare_df: pd.DataFrame,
                   diff_col: str,
                   title: str,
                   out_path: Path) -> None:
    arr, lin_values, lout_values = _pivot(compare_df, diff_col)
    max_abs = max(1e-9, float(np.nanmax(np.abs(arr))))
    _plot_heatmap(arr,
                  lin_values=lin_values,
                  lout_values=lout_values,
                  title=title,
                  cbar_label="hybrid - gpu (nJ)",
                  out_path=out_path,
                  cmap="coolwarm",
                  vmin=-max_abs,
                  vmax=max_abs)


def _plot_winner_map(compare_df: pd.DataFrame,
                     winner_col: str,
                     title: str,
                     out_path: Path) -> None:
    mapping = {"gpu+pim": 0, "tie": 1, "gpu": 2}
    labels = {0: "GPU+PIM lower", 1: "Tie", 2: "GPU lower"}
    arr, lin_values, lout_values = _pivot(compare_df.assign(_winner_code=compare_df[winner_col].map(mapping)),
                                          "_winner_code")

    cmap = ListedColormap(["#55a868", "#c7c7c7", "#4c72b0"])
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(arr, origin="lower", aspect="auto", cmap=cmap, vmin=-0.5, vmax=2.5)
    ax.set_xticks(range(len(lout_values)))
    ax.set_xticklabels([str(v) for v in lout_values], rotation=45, ha="right")
    ax.set_yticks(range(len(lin_values)))
    ax.set_yticklabels([str(v) for v in lin_values])
    ax.set_xlabel("Generated Tokens (Lout)")
    ax.set_ylabel("Context Tokens (Lin)")
    ax.set_title(title)

    token_map = {0: "H", 1: "=", 2: "G"}
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            code = int(arr[i, j])
            ax.text(j, i, token_map[code], ha="center", va="center", fontsize=9, color="black")

    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels([labels[i] for i in [0, 1, 2]])
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _winner_counts(compare_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for prefix, label in [("prefill", "Prefill energy"),
                          ("decode_total", "Decode total energy"),
                          ("total", "Total request energy")]:
        counts = compare_df[f"{prefix}_energy_winner"].value_counts().to_dict()
        rows.append({
            "metric": label,
            "gpu_lower_count": int(counts.get("gpu", 0)),
            "hybrid_lower_count": int(counts.get("gpu+pim", 0)),
            "tie_count": int(counts.get("tie", 0)),
        })
    return pd.DataFrame(rows)


def _nearest_grid_value(value: int, grid_values: Sequence[int]) -> int:
    if len(grid_values) == 1:
        return int(grid_values[0])
    log_v = math.log(max(1.0, float(value)))
    return min(grid_values, key=lambda g: abs(math.log(max(1.0, float(g))) - log_v))


def _load_trace_distribution(azure_trace: Path | str,
                             lin_values: Sequence[int],
                             lout_values: Sequence[int],
                             limit: int | None,
                             start_offset: int) -> pd.DataFrame:
    arrivals = load_azure_llm_trace(str(azure_trace),
                                    limit=limit,
                                    start_offset=start_offset,
                                    arrival_time_scale=1.0)
    rows: List[Dict[str, object]] = []
    for req in arrivals:
        mapped_lin = _nearest_grid_value(int(req.context_tokens), lin_values)
        mapped_lout = _nearest_grid_value(int(req.generated_tokens), lout_values)
        rows.append({
            "request_id": int(req.request_id),
            "lin": int(req.context_tokens),
            "lout": int(req.generated_tokens),
            "grid_lin": int(mapped_lin),
            "grid_lout": int(mapped_lout),
        })
    return pd.DataFrame(rows)


def _distribution_counts(trace_df: pd.DataFrame,
                         lin_values: Sequence[int],
                         lout_values: Sequence[int]) -> pd.DataFrame:
    if trace_df.empty:
        cols = ["lin", "lout", "count", "fraction"]
        return pd.DataFrame(columns=cols)
    grouped = (trace_df.groupby(["grid_lin", "grid_lout"])
                      .size()
                      .rename("count")
                      .reset_index()
                      .rename(columns={"grid_lin": "lin", "grid_lout": "lout"}))
    full_index = pd.MultiIndex.from_product([lin_values, lout_values], names=["lin", "lout"])
    grouped = (grouped.set_index(["lin", "lout"])
                      .reindex(full_index, fill_value=0)
                      .reset_index())
    total = max(1, int(grouped["count"].sum()))
    grouped["fraction"] = grouped["count"] / float(total)
    return grouped


def _plot_distribution_map(dist_df: pd.DataFrame,
                           value_col: str,
                           title: str,
                           cbar_label: str,
                           out_path: Path) -> None:
    arr, lin_values, lout_values = _pivot(dist_df, value_col)
    _plot_heatmap(arr,
                  lin_values=lin_values,
                  lout_values=lout_values,
                  title=title,
                  cbar_label=cbar_label,
                  out_path=out_path,
                  cmap="magma")


def _plot_distribution_over_winner(compare_df: pd.DataFrame,
                                   dist_df: pd.DataFrame,
                                   winner_col: str,
                                   title: str,
                                   out_path: Path) -> None:
    mapping = {"gpu+pim": 0, "tie": 1, "gpu": 2}
    labels = {0: "GPU+PIM lower", 1: "Tie", 2: "GPU lower"}
    winner_arr, lin_values, lout_values = _pivot(
        compare_df.assign(_winner_code=compare_df[winner_col].map(mapping)),
        "_winner_code",
    )
    count_arr, _, _ = _pivot(dist_df, "count")
    cmap = ListedColormap(["#55a868", "#c7c7c7", "#4c72b0"])

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(winner_arr, origin="lower", aspect="auto", cmap=cmap, vmin=-0.5, vmax=2.5)
    ax.set_xticks(range(len(lout_values)))
    ax.set_xticklabels([str(v) for v in lout_values], rotation=45, ha="right")
    ax.set_yticks(range(len(lin_values)))
    ax.set_yticklabels([str(v) for v in lin_values])
    ax.set_xlabel("Generated Tokens (Lout)")
    ax.set_ylabel("Context Tokens (Lin)")
    ax.set_title(title)

    token_map = {0: "H", 1: "=", 2: "G"}
    for i in range(winner_arr.shape[0]):
        for j in range(winner_arr.shape[1]):
            code = int(winner_arr[i, j])
            cnt = int(count_arr[i, j])
            text = f"{token_map[code]}\n{cnt}" if cnt > 0 else token_map[code]
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color="black")

    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels([labels[i] for i in [0, 1, 2]])
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description="Compare GPU vs GPU+PIM predicted energies on a Lin/Lout grid.")
    p.add_argument("--cost-model", choices=["ml", "table"], default="ml")
    p.add_argument("--ml-profile", action="append", default=[],
                   help="Route/profile mapping in route=path.pkl format. Repeatable.")
    p.add_argument("--profile", action="append", default=[],
                   help="Route/profile mapping in route=path.csv format. Repeatable.")
    p.add_argument("--routes", nargs=2, default=list(DEFAULT_ROUTES),
                   help="Exactly two routes to compare, default: gpu_only lpddr5_pim_bank")
    p.add_argument("--bs", type=int, default=1, help="Batch size used for the static estimate grid")
    p.add_argument("--unsupported-policy", default="clip", choices=["clip", "nearest", "drop"])
    p.add_argument("--lin-values", nargs="*", type=int, default=None,
                   help="Explicit Lin grid values. If omitted, derived from the common route bounds.")
    p.add_argument("--lout-values", nargs="*", type=int, default=None,
                   help="Explicit Lout grid values. If omitted, derived from the common route bounds.")
    p.add_argument("--lin-count", type=int, default=8,
                   help="Number of default Lin samples if --lin-values is omitted")
    p.add_argument("--lout-count", type=int, default=6,
                   help="Number of default Lout samples if --lout-values is omitted")
    p.add_argument("--azure-trace", type=Path, default=None,
                   help="Optional Azure trace CSV to summarize as a Lin/Lout grid distribution")
    p.add_argument("--trace-limit", type=int, default=None,
                   help="Optional request limit when building the Azure trace distribution")
    p.add_argument("--trace-start-offset", type=int, default=0,
                   help="Optional start offset when building the Azure trace distribution")
    p.add_argument("--out-dir", type=Path,
                   default=Path("cluster_outputs/route_energy_grid"))
    p.add_argument("--prefix", default="route_energy_grid")
    args = p.parse_args()

    routes = list(args.routes)
    if len(routes) != 2:
        raise ValueError("--routes must contain exactly two route names")

    cost_model = _load_cost_model(args)
    missing = [route for route in routes if route not in set(cost_model.routes())]
    if missing:
        raise ValueError(f"Routes unavailable in cost model: {missing}")

    (lin_lo, lin_hi), (lout_lo, lout_hi) = _intersection_bounds(cost_model, routes)
    lin_values = sorted(set(args.lin_values)) if args.lin_values else _default_grid(lin_lo, lin_hi, args.lin_count)
    lout_values = sorted(set(args.lout_values)) if args.lout_values else _default_grid(lout_lo, lout_hi, args.lout_count)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    grid_df = _estimate_grid(cost_model=cost_model,
                             routes=routes,
                             lin_values=lin_values,
                             lout_values=lout_values,
                             bs=int(args.bs),
                             unsupported_policy=args.unsupported_policy)
    compare_df = _build_comparison(grid_df, routes=routes)
    winner_df = _winner_counts(compare_df)

    grid_csv = args.out_dir / f"{args.prefix}_grid.csv"
    compare_csv = args.out_dir / f"{args.prefix}_comparison.csv"
    winners_csv = args.out_dir / f"{args.prefix}_winner_counts.csv"
    meta_json = args.out_dir / f"{args.prefix}_meta.json"
    grid_df.to_csv(grid_csv, index=False)
    compare_df.to_csv(compare_csv, index=False)
    winner_df.to_csv(winners_csv, index=False)
    meta_json.write_text(json.dumps({
        "cost_model": args.cost_model,
        "routes": routes,
        "bs": int(args.bs),
        "unsupported_policy": args.unsupported_policy,
        "lin_values": lin_values,
        "lout_values": lout_values,
        "common_bounds": {
            "Lin": [lin_lo, lin_hi],
            "Lout": [lout_lo, lout_hi],
        },
    }, indent=2))

    metric_specs = [
        ("prefill_energy_nj", "Prefill Energy By Route", "Prefill energy (nJ)", "prefill"),
        ("decode_total_energy_nj", "Decode Total Energy By Route", "Decode total energy (nJ)", "decode_total"),
        ("total_request_energy_nj", "Total Request Energy By Route", "Total request energy (nJ)", "total"),
    ]
    for metric, title, cbar, tag in metric_specs:
        _plot_route_pair(grid_df,
                         routes=routes,
                         metric=metric,
                         title=title,
                         cbar_label=cbar,
                         out_path=args.out_dir / f"{args.prefix}_{tag}_by_route.png")
        _plot_diff_map(compare_df,
                       diff_col=f"{tag}_energy_diff_hybrid_minus_gpu_nj",
                       title=f"{title}: Hybrid - GPU",
                       out_path=args.out_dir / f"{args.prefix}_{tag}_diff_hybrid_minus_gpu.png")
        _plot_winner_map(compare_df,
                         winner_col=f"{tag}_energy_winner",
                         title=f"{title}: Lower-Energy Route",
                         out_path=args.out_dir / f"{args.prefix}_{tag}_winner_map.png")

    if args.azure_trace is not None:
        trace_df = _load_trace_distribution(args.azure_trace,
                                            lin_values=lin_values,
                                            lout_values=lout_values,
                                            limit=args.trace_limit,
                                            start_offset=max(0, int(args.trace_start_offset)))
        dist_df = _distribution_counts(trace_df, lin_values=lin_values, lout_values=lout_values)
        trace_csv = args.out_dir / f"{args.prefix}_azure_trace_grid.csv"
        dist_csv = args.out_dir / f"{args.prefix}_azure_distribution_grid.csv"
        trace_df.to_csv(trace_csv, index=False)
        dist_df.to_csv(dist_csv, index=False)

        _plot_distribution_map(dist_df,
                               value_col="count",
                               title="Azure Trace Distribution on Lin/Lout Grid (Counts)",
                               cbar_label="Request count",
                               out_path=args.out_dir / f"{args.prefix}_azure_distribution_counts.png")
        _plot_distribution_map(dist_df,
                               value_col="fraction",
                               title="Azure Trace Distribution on Lin/Lout Grid (Fraction)",
                               cbar_label="Fraction of requests",
                               out_path=args.out_dir / f"{args.prefix}_azure_distribution_fraction.png")
        _plot_distribution_over_winner(compare_df,
                                       dist_df,
                                       winner_col="total_energy_winner",
                                       title="Total-Energy Winner With Azure Trace Counts",
                                       out_path=args.out_dir / f"{args.prefix}_total_winner_with_azure_counts.png")
    else:
        trace_csv = None
        dist_csv = None

    print(f"Wrote grid CSV: {grid_csv}")
    print(f"Wrote comparison CSV: {compare_csv}")
    print(f"Wrote winner-count CSV: {winners_csv}")
    if trace_csv is not None and dist_csv is not None:
        print(f"Wrote Azure trace grid CSV: {trace_csv}")
        print(f"Wrote Azure distribution grid CSV: {dist_csv}")
    print(f"Wrote plots to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
