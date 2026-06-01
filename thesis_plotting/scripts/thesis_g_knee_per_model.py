#!/usr/bin/env python3
"""Per-model G main-claim plot: hybrid vs gpu-only at the knee.

One figure per (model, dataset) tuple. 1 row × 3 columns
(Throughput, TTFT p99, E2E p99). This is the methodologically clean
version of fig_g_knee_main_claim_* — different models live at
different knee scales, so they shouldn't share a figure.

Usage:
    # Single (sweep, model) -> single figure
    python thesis_plotting/scripts/thesis_g_knee_per_model.py \\
        --sweep-dir cluster_outputs/online_serving_runs/g_knee_main_claim_<TS> \\
        --model pi0 \\
        --dataset-label "Azure full-19k (real)" \\
        --scale 4.24 \\
        --out-name fig_g_pi0_azure_full19k

    # Auto: scan a set of G sweep dirs and render per-(model,dataset)
    python thesis_plotting/scripts/thesis_g_knee_per_model.py --auto
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO))

from plot_sweep_compare import parse_summary  # noqa: E402

from thesis_plotting.style import (  # noqa: E402
    configure_plotting, save_fig, set_spines,
    COLOR_ROUTE, COLOR_MODEL, FONT_LABEL, FONT_TITLE, FONT_TICK,
)


MODE_LABEL  = {"hybrid": "Hybrid", "gpuonly": "GPU-only"}
MODE_COLOR  = {"hybrid": COLOR_ROUTE["lpddr5_pim_bank"],
               "gpuonly": COLOR_ROUTE["gpu_only"]}
MODEL_DISPLAY = {"pi0": "Pi0", "openvla": "OpenVLA"}


# Knee-scaled G sweeps we know about. Add new ones here.
AUTO_SWEEPS = [
    # (sweep_dir_basename, dataset_label, scale, valid_models)
    # `valid_models` = models whose knee SCALE matches this sweep's
    # arrival_scale (so rendering the figure is honest). Sweeps that
    # ran both models but at one model's scale should mark only that
    # model — the other model is in saturation here and shouldn't get
    # an "at-knee" figure from this dir.
    ("g_knee_main_claim_20260530_164544",
     "Azure full-19k  (real, heavy-tail)", 4.24, "pi0"),
    ("g_knee_azure_poisson_20260530_174416",
     "azure_poisson  (Poisson shape)", 3.75, "pi0"),
    ("g_knee_vla_poisson_20260531_152009",
     "vla_poisson  (robotic shape)", 2.44, "pi0"),
    ("g_knee_azure_empirical_20260531_152009",
     "azure_empirical  (heavy-tail synth)", 3.77, "pi0"),
    ("g_knee_azure_poisson_wide_20260531_173946",
     "azure_poisson_wide  (Poisson, mixture v2)", 6.5, "pi0"),
    # OpenVLA-only sweeps at OpenVLA's own knee:
    ("g_knee_azure_poisson_20260531_160014",
     "azure_poisson  (Poisson shape)", 15.0, "openvla"),
    ("g_knee_vla_poisson_20260531_160023",
     "vla_poisson  (robotic shape)", 8.0, "openvla"),
    ("g_knee_azure_poisson_wide_20260601_152403",
     "azure_poisson_wide  (Poisson, mixture v2)", 22.9, "openvla"),
]


def discover_cells(sweep_dir: Path, model: str) -> dict:
    """Returns {mode: metrics_dict} for the cells of a given model."""
    out = {}
    label_re = re.compile(
        rf"^\d+_{model}_(?P<mode>hybrid|gpuonly)$")
    for sub in sorted(sweep_dir.iterdir()):
        if not sub.is_dir():
            continue
        m = label_re.match(sub.name)
        if not m:
            continue
        sumfile = sub / "policy_compare_summary.txt"
        if not sumfile.is_file():
            for child in sub.iterdir():
                if child.is_dir():
                    cand = child / "policy_compare_summary.txt"
                    if cand.is_file():
                        sumfile = cand
                        break
        if not sumfile.is_file():
            continue
        out[m["mode"]] = parse_summary(sumfile)
    return out


def render(cells: dict, model: str, dataset_label: str, scale: float,
           out_dir: Path, out_name: str):
    configure_plotting()
    if not cells.get("hybrid") or not cells.get("gpuonly"):
        print(f"[skip] {out_name}: missing hybrid or gpuonly cell")
        return

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))

    metrics = [
        ("throughput", "Throughput  (req/s)",  lambda v: v,
         False, "higher"),
        ("ttft_p99",   "TTFT p99  (ms)",       lambda v: v,
         True,  "lower"),
        ("e2e_p99",    "E2E p99  (s)",         lambda v: v / 1000.0,
         True,  "lower"),
    ]
    modes = ["gpuonly", "hybrid"]
    xs = np.array([0.35, 0.65])

    for col, (key, ylabel, transform, log, winning) in enumerate(metrics):
        ax = axes[col]
        vals = []
        for mode in modes:
            v = cells[mode].get(key)
            vals.append(transform(v) if v is not None else 0.0)

        colors = [MODE_COLOR[m] for m in modes]
        bars = ax.bar(xs, vals, width=0.28, color=colors,
                      edgecolor="black", linewidth=0.5, zorder=3)
        if log:
            ax.set_yscale("log")

        gpu_v, hyb_v = vals
        if gpu_v > 0:
            delta_signed = (hyb_v - gpu_v) / gpu_v * 100.0
            if abs(delta_signed) < 0.5:
                label_str = "= 0%"
                color = "#5a5a5a"
            else:
                arrow = "↑" if delta_signed > 0 else "↓"
                label_str = f"{arrow}{abs(delta_signed):.0f}%"
                hybrid_wins = ((winning == "lower" and delta_signed < 0)
                               or (winning == "higher" and delta_signed > 0))
                color = "#2ca02c" if hybrid_wins else "#c44e52"
            top_y = max(vals)
            ax.text(0.5, top_y * (2.0 if log else 1.18),
                    label_str,
                    ha="center", va="bottom",
                    fontsize=FONT_LABEL + 4, fontweight="bold",
                    color=color)

        for bar, v in zip(bars, vals):
            label = f"{v:.2f}" if v < 100 else f"{v:.0f}"
            if log:
                y_lab = bar.get_height() * 1.07
            else:
                y_lab = bar.get_height() + max(vals) * 0.015
            ax.text(bar.get_x() + bar.get_width() / 2, y_lab, label,
                    ha="center", va="bottom",
                    fontsize=FONT_LABEL, fontweight="bold")

        ax.set_xticks(xs)
        ax.set_xticklabels([MODE_LABEL[m] for m in modes],
                           fontsize=FONT_TICK, fontweight="bold")
        ax.set_xlim(0.10, 0.90)
        ax.set_ylabel(ylabel, fontsize=FONT_LABEL, fontweight="bold")
        if log:
            ax.set_ylim(top=max(vals) * 4.5)
        else:
            ax.set_ylim(0, max(vals) * 1.32)
        set_spines(ax)

    sup = (f"{MODEL_DISPLAY[model]} @ knee   "
           f"({dataset_label}, arrival_scale = {scale:g})")
    fig.suptitle(sup, fontsize=FONT_TITLE, fontweight="bold", y=1.00)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    save_fig(fig, out_name, out_dir)
    plt.close(fig)


def short_dataset_token(label: str) -> str:
    """Slug-safe short token for the filename, derived from a label."""
    token = label.split("(")[0].strip().lower()
    token = re.sub(r"\s+", "_", token)
    token = re.sub(r"[^a-z0-9_]+", "", token)
    return token


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep-dir", type=Path, default=None)
    ap.add_argument("--model", choices=["pi0", "openvla"], default="pi0")
    ap.add_argument("--dataset-label", default="")
    ap.add_argument("--scale", type=float, default=0.0)
    ap.add_argument("--out-dir", type=Path,
                    default=REPO / "thesis_plotting/figures")
    ap.add_argument("--out-name", default=None)
    ap.add_argument("--auto", action="store_true",
                    help="iterate the AUTO_SWEEPS list and render all of them")
    ap.add_argument("--models", default="pi0 openvla",
                    help="space-sep list of models to render in --auto mode")
    args = ap.parse_args()

    if args.auto:
        runs_root = REPO / "cluster_outputs/online_serving_runs"
        for entry in AUTO_SWEEPS:
            # Backward compat: support both 3-tuple and 4-tuple AUTO_SWEEPS
            if len(entry) == 4:
                basename, label, scale, models_present = entry
            else:
                basename, label, scale = entry
                models_present = "pi0 openvla"
            sweep = runs_root / basename
            if not sweep.is_dir():
                print(f"[skip] {basename}: directory not found")
                continue
            slug = short_dataset_token(label)
            wanted = set(args.models.split())
            available = set(models_present.split())
            for model in sorted(wanted & available):
                cells = discover_cells(sweep, model)
                if not cells:
                    print(f"[skip] {basename} {model}: no cells")
                    continue
                out_name = f"fig_g_{model}_knee_{slug}"
                print(f"[render] {out_name}  scale={scale}")
                render(cells, model, label, scale, args.out_dir, out_name)
        return

    if args.sweep_dir is None:
        sys.exit("[error] --sweep-dir required (or use --auto)")
    cells = discover_cells(args.sweep_dir, args.model)
    print(f"[info] {args.model} cells: {sorted(cells.keys())}")
    out_name = args.out_name or (
        f"fig_g_{args.model}_knee_"
        f"{short_dataset_token(args.dataset_label or args.sweep_dir.name)}")
    render(cells, args.model, args.dataset_label or args.sweep_dir.name,
           args.scale, args.out_dir, out_name)


if __name__ == "__main__":
    main()
