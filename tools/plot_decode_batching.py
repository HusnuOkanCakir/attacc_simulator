#!/usr/bin/env python3
"""E6 / batching diagnostic plotter.

Reads each sub-run's `ramulator_summary.yaml` and `policy_compare_summary.txt`
to produce three sweep-level figures:

  - bs_actual_vs_configured.png  — grouped bars: configured max_bs vs
    achieved mean_bs, per model.  Answers "is the batch filling".
  - bs_throughput_vs_config.png  — line plot: throughput (req/s) vs
    configured max_bs, one line per model.  Visualizes the 1/B regression
    if it exists.
  - bs_histogram.png             — small-multiples grid (one panel per
    cell) showing the % distribution of actual batch sizes observed.

Expects sub-run labels of the form `<NN>_<model>_bs<N>`, e.g. `01_pi0_bs1`,
`02_pi0_bs4`, ..., `10_openvla_bs16`.

Usage:
  python tools/plot_decode_batching.py --sweep-dir <SWEEP_DIR>
"""

import argparse
import re
import sys
import yaml
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from plot_sweep_compare import parse_summary, discover_runs  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LABEL_RE = re.compile(r"^(?:\d+_)?(?P<model>[a-z0-9_]+?)_bs(?P<bs>\d+)$")


def parse_label(label: str):
    m = LABEL_RE.match(label)
    if not m:
        return None
    return m.group("model"), int(m.group("bs"))


def load_histogram(sweep_dir: Path, sub_name: str):
    """Return the decode_batching histogram list from a cell's
    ramulator_summary.yaml. `sub_name` may be the stripped form
    (`pi0_bs4`) returned by `discover_runs`; we resolve to the actual
    subdir whose name ends with `_<sub_name>` (e.g. `01_pi0_bs4`)."""
    sub = sweep_dir / sub_name
    if not sub.is_dir():
        for d in sweep_dir.iterdir():
            if d.is_dir() and (d.name == sub_name or d.name.endswith("_" + sub_name)):
                sub = d
                break
    if not sub.is_dir():
        return []
    candidates = [sub / "ramulator_summary.yaml"]
    for child in sub.iterdir():
        if child.is_dir():
            candidates.append(child / "ramulator_summary.yaml")
    for path in candidates:
        if not path.is_file():
            continue
        try:
            data = yaml.safe_load(path.read_text()) or {}
        except yaml.YAMLError:
            continue
        fe = data.get("Frontend", {}) or {}
        db = fe.get("decode_batching", {}) or {}
        hist = db.get("histogram", [])
        if hist:
            return list(hist)
    return []


def group_by_model(rows):
    grouped: dict[str, list[tuple[int, dict, str]]] = {}
    for label, summary in rows:
        parsed = parse_label(label)
        if parsed is None:
            print(f"[skip] label {label!r} not in <NN>_<model>_bs<N> form")
            continue
        model, bs = parsed
        grouped.setdefault(model, []).append((bs, summary, label))
    for model in grouped:
        grouped[model].sort(key=lambda x: x[0])
    return grouped


def plot_actual_vs_configured(grouped, out_path: Path):
    models = list(grouped.keys())
    if not models:
        return
    fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 4.5),
                             squeeze=False)
    for ax, model in zip(axes[0], models):
        points = grouped[model]
        cfgs   = [p[0] for p in points]
        means  = [p[1].get("decode_mean_bs", 0.0) for p in points]
        x = np.arange(len(cfgs))
        width = 0.38
        ax.bar(x - width / 2, cfgs, width, label="configured max_bs",
               color="#9ec1ce")
        ax.bar(x + width / 2, means, width, label="achieved mean_bs",
               color="#1f6e8c")
        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in cfgs])
        ax.set_xlabel("configured max_decode_batch_size")
        ax.set_ylabel("batch size")
        ax.set_title(f"{model} — actual vs configured")
        ax.set_ylim(0, max(cfgs + [1]) * 1.15)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
        # Annotate fill ratio on each pair.
        for i, (cfg, mean) in enumerate(zip(cfgs, means)):
            ratio = (mean / cfg * 100.0) if cfg > 0 else 0.0
            ax.text(x[i], max(cfg, mean) * 1.02, f"{ratio:.0f}%",
                    ha="center", va="bottom", fontsize=8, color="#444")
    fig.suptitle("Decode batch — configured vs achieved", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[plot] {out_path}")


def plot_throughput_vs_config(grouped, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model, points in grouped.items():
        cfgs = [p[0] for p in points]
        thrus = [p[1].get("throughput", float("nan")) for p in points]
        ax.plot(cfgs, thrus, marker="o", linewidth=2, label=model)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("configured max_decode_batch_size")
    ax.set_ylabel("throughput (req/s)")
    ax.set_title("Throughput vs max_decode_batch_size")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[plot] {out_path}")


def plot_histogram(sweep_dir, grouped, out_path: Path):
    # Small multiples — one panel per (model, configured_bs) cell.
    cells = []
    for model, points in grouped.items():
        for bs, _summary, label in points:
            hist = load_histogram(sweep_dir, label)
            if not hist:
                continue
            cells.append((model, bs, hist))
    if not cells:
        return
    n = len(cells)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 2.6 * rows),
                             squeeze=False)
    axes_flat = axes.flatten()
    for ax in axes_flat[n:]:
        ax.axis("off")
    for ax, (model, bs, hist) in zip(axes_flat, cells):
        total = float(sum(hist))
        pct = [100.0 * h / total if total > 0 else 0.0 for h in hist]
        bs_x = list(range(1, len(hist) + 1))
        ax.bar(bs_x, pct, color="#3b6e8c")
        ax.set_xticks(bs_x)
        ax.set_xlabel("actual bs")
        ax.set_ylabel("% of decode steps")
        ax.set_title(f"{model}  cfg={bs}", fontsize=9)
        ax.set_ylim(0, max(pct + [1]) * 1.15)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Decode batch-size distribution per cell", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[plot] {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-dir", type=Path, required=True)
    args = ap.parse_args()

    if not args.sweep_dir.is_dir():
        sys.exit(f"[error] sweep-dir not a directory: {args.sweep_dir}")

    rows = discover_runs(args.sweep_dir)
    if not rows:
        sys.exit("[error] no sub-runs with policy_compare_summary.txt found")
    print(f"[scan] {len(rows)} sub-runs found")

    grouped = group_by_model(rows)
    if not grouped:
        sys.exit("[error] no labels matched <NN>_<model>_bs<N>")

    for model, points in grouped.items():
        cfgs   = [p[0] for p in points]
        means  = [round(p[1].get("decode_mean_bs", 0.0), 2) for p in points]
        thrus  = [round(p[1].get("throughput", 0.0), 2) for p in points]
        print(f"  {model:<8}  cfgs={cfgs}  mean_bs={means}  thru={thrus}")

    plots_dir = args.sweep_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_actual_vs_configured(grouped,
                              plots_dir / "bs_actual_vs_configured.png")
    plot_throughput_vs_config(grouped,
                              plots_dir / "bs_throughput_vs_config.png")
    plot_histogram(args.sweep_dir, grouped, plots_dir / "bs_histogram.png")


if __name__ == "__main__":
    main()
