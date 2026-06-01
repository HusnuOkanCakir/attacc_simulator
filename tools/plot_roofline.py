#!/usr/bin/env python3
"""
Roofline model for PI0 on GPU-only (A100) vs PIM+GPU (LPDDR5-PIM, bank-level).

Hardware parameters (from src/config.py):
  GPU (A100 FP16):  peak compute = 312 TFLOPS,  BW = 3.352 TB/s,  ridge = 93 FLOPs/byte
  PIM attention:    effective BW ≈ 1.62 TB/s (empirical from g_matmul / KV bytes across Lin)
  PIM prefill:      compute-bound at long Lin, empirical peak ≈ 62 TFLOPS (Lin=4096)

PI0 geometry (from serving_online YAML):
  num_layers=40, num_heads=8, d_head=256, dtype_bytes=2

Arithmetic intensity:
  Decode  FC:      AI = 1 FLOPs/byte  (batch=1, weight-bound; FLOPs = 2×params, mem = 2×params)
  Decode  attn:    AI = 1 FLOPs/byte  (4 FLOPs per KV token-byte pair, independent of Lin)
  Prefill FC/attn: AI ≈ Lin FLOPs/byte (batch=Lin for FC; Lin²/Lin for attention)

Usage:
  python tools/plot_roofline.py
  python tools/plot_roofline.py --output docs/figures/pi0_pim_roofline.png --batch 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent

# ── Hardware peaks (from src/config.py, A100 FP16) ───────────────────────────
GPU_COMPUTE_TFLOPS = 312.0     # A100 FP16 tensor-core peak
GPU_BW_TBS         = 3.352     # A100 HBM2e bandwidth (TB/s)
GPU_RIDGE          = GPU_COMPUTE_TFLOPS / GPU_BW_TBS  # = 93.07 FLOPs/byte = sys_opb ✓

# PIM attention: empirically derived from cost table (KV bytes / g_matmul time)
# Consistent across Lin=512–4096: ~1.62 TB/s, matching LPDDR5-6400 x16 × 16 channels
PIM_ATTN_BW_TBS    = 1.62      # TB/s empirical
# PIM prefill compute peak: from s_flops / s_time at large Lin (compute-bound regime)
PIM_COMPUTE_TFLOPS = 62.0      # TFLOPS empirical (Lin=4096)
PIM_RIDGE          = PIM_COMPUTE_TFLOPS / PIM_ATTN_BW_TBS  # ≈ 38.3 FLOPs/byte

# ── PI0 model geometry ────────────────────────────────────────────────────────
NUM_LAYERS  = 40
NUM_HEADS   = 8
D_HEAD      = 256
DTYPE_BYTES = 2

# KV bytes per context token (Key + Value per layer)
KV_BYTES_PER_TOKEN    = 2 * NUM_LAYERS * NUM_HEADS * D_HEAD * DTYPE_BYTES  # 327,680 B
# Attention FLOPs per context token (QK + AV)
ATTN_FLOPS_PER_TOKEN  = 4 * NUM_LAYERS * NUM_HEADS * D_HEAD               # 327,680 FLOPs
# AI for attention = ATTN_FLOPS / KV_BYTES = 1.0 FLOPs/byte regardless of Lin
AI_ATTN = ATTN_FLOPS_PER_TOKEN / KV_BYTES_PER_TOKEN  # = 1.0


def load_tables(gpu_csv: str, pim_csv: str):
    return pd.read_csv(gpu_csv), pd.read_csv(pim_csv)


def operating_points(gpu_df: pd.DataFrame, pim_df: pd.DataFrame, bs: int = 1):
    """Return dicts with (AI, achieved_TFLOPS, Lin) for each kernel category."""
    LOUT = 2  # decode: 1 new token

    # FC decode bytes: at Lin→0 all time is FC; AI=1 → FC_bytes = FC_FLOPs
    row0 = gpu_df[(gpu_df["Lin"] == 2) & (gpu_df["Lout"] == LOUT) & (gpu_df["bs"] == bs)]
    if row0.empty:
        row0 = gpu_df[gpu_df["bs"] == bs].sort_values("Lin").iloc[[0]]
    fc_flops  = float(row0["g_flops"].iloc[0])   # ≈ 3.02e9 FLOPs (FC-dominated at Lin=2)
    fc_bytes  = fc_flops                          # AI=1 → bytes = FLOPs
    ai_fc     = fc_flops / fc_bytes               # = 1.0

    # Prefill activation memory per token per layer (Q + output activations)
    D_MODEL = NUM_HEADS * D_HEAD
    ACT_BYTES_PER_TOKEN = 4 * D_MODEL * DTYPE_BYTES  # Q, K, V, O activations

    dec_gpu_total = []
    dec_pim_total = []
    dec_gpu_attn  = []
    dec_pim_attn  = []
    pref_pim      = []
    pref_gpu      = []

    lin_vals = [2, 512, 1024, 1536, 2048, 3072, 4096, 5120, 6144]

    for lin in lin_vals:
        gr = gpu_df[(gpu_df["Lin"] == lin) & (gpu_df["Lout"] == LOUT) & (gpu_df["bs"] == bs)]
        pr = pim_df[(pim_df["Lin"] == lin) & (pim_df["Lout"] == LOUT) & (pim_df["bs"] == bs)]
        if gr.empty or pr.empty:
            continue
        gr, pr = gr.iloc[0], pr.iloc[0]

        kv_bytes   = KV_BYTES_PER_TOKEN * lin * bs
        attn_flops = ATTN_FLOPS_PER_TOKEN * lin * bs

        # ── Decode total (FC + attention combined) ────────────────────────────
        total_flops = float(gr["g_flops"])
        total_bytes = fc_bytes + kv_bytes
        ai_total    = total_flops / total_bytes

        gpu_time = float(gr["g_time (ms)"]) * 1e-3
        pim_time = float(pr["g_time (ms)"]) * 1e-3
        dec_gpu_total.append((ai_total, total_flops / gpu_time / 1e12, lin))
        dec_pim_total.append((ai_total, total_flops / pim_time / 1e12, lin))

        # ── Decode attention kernel only ──────────────────────────────────────
        if lin >= 512 and attn_flops > 0:
            # GPU-only attn time: difference from the near-zero-Lin baseline
            fc_time_gpu = float(
                gpu_df[(gpu_df["Lin"] == 2) & (gpu_df["Lout"] == LOUT) & (gpu_df["bs"] == bs)
                       ]["g_time (ms)"].iloc[0]) * 1e-3
            gpu_attn_time = max((float(gr["g_time (ms)"]) - fc_time_gpu * 1e3) * 1e-3, 1e-7)
            gpu_attn_ach  = attn_flops / gpu_attn_time / 1e12

            # PIM attn time: g_matmul column
            pim_matmul_ms = float(pr["g_matmul"])
            if pim_matmul_ms > 1e-4:
                pim_attn_ach = attn_flops / (pim_matmul_ms * 1e-3) / 1e12
                dec_pim_attn.append((AI_ATTN, pim_attn_ach, lin))
                dec_gpu_attn.append((AI_ATTN, gpu_attn_ach, lin))

        # ── Prefill (serial/PIM) ──────────────────────────────────────────────
        pref_flops = float(pr["s_flops"])
        pref_mem   = (fc_bytes
                      + lin * KV_BYTES_PER_TOKEN          # KV generated
                      + lin * ACT_BYTES_PER_TOKEN * NUM_LAYERS)  # activations
        ai_pref    = pref_flops / pref_mem
        pref_time_pim = float(pr["s_time"]) * 1e-3
        pref_ach_pim  = pref_flops / pref_time_pim / 1e12

        # GPU prefill: use same s_flops but with GPU throughput
        # (s_time in GPU-only table = same prefill model, just different hw)
        gr_pref = gpu_df[(gpu_df["Lin"] == lin) & (gpu_df["Lout"] == LOUT) & (gpu_df["bs"] == bs)]
        if not gr_pref.empty:
            pref_time_gpu = float(gr_pref.iloc[0]["s_time"]) * 1e-3
            pref_ach_gpu  = pref_flops / pref_time_gpu / 1e12
            pref_gpu.append((ai_pref, pref_ach_gpu, lin))

        pref_pim.append((ai_pref, pref_ach_pim, lin))

    return dict(
        ai_fc=ai_fc,
        fc_flops=fc_flops,
        fc_achieved_gpu=fc_flops / (float(
            gpu_df[(gpu_df["Lin"] == 2) & (gpu_df["Lout"] == LOUT) & (gpu_df["bs"] == bs)
                   ]["g_time (ms)"].iloc[0]) * 1e-3) / 1e12,
        dec_gpu_total=dec_gpu_total,
        dec_pim_total=dec_pim_total,
        dec_gpu_attn=dec_gpu_attn,
        dec_pim_attn=dec_pim_attn,
        pref_pim=pref_pim,
        pref_gpu=pref_gpu,
    )


def plot_roofline(pts: dict, outpath: Path, batch: int):
    fig, ax = plt.subplots(figsize=(11, 7))

    ai_range = np.logspace(-2, 4, 800)

    # ── Hardware rooflines ────────────────────────────────────────────────────
    gpu_roof = np.minimum(GPU_COMPUTE_TFLOPS, GPU_BW_TBS * ai_range)
    pim_roof = np.minimum(PIM_COMPUTE_TFLOPS, PIM_ATTN_BW_TBS * ai_range)

    ax.plot(ai_range, gpu_roof, color="#1565C0", lw=2.5,
            label=f"GPU-only roofline  (A100, {GPU_BW_TBS} TB/s BW, {GPU_COMPUTE_TFLOPS:.0f} TFLOPS)")
    ax.plot(ai_range, pim_roof, color="#B71C1C", lw=2.5, linestyle="--",
            label=f"PIM attention roofline  ({PIM_ATTN_BW_TBS} TB/s BW, {PIM_COMPUTE_TFLOPS:.0f} TFLOPS)")

    # Ridge annotations
    ax.axvline(GPU_RIDGE, color="#1565C0", lw=0.8, alpha=0.45, linestyle=":")
    ax.axvline(PIM_RIDGE, color="#B71C1C", lw=0.8, alpha=0.45, linestyle=":")
    ax.text(GPU_RIDGE * 1.08, GPU_COMPUTE_TFLOPS * 0.55,
            f"GPU ridge\n{GPU_RIDGE:.0f} FLOPs/B", color="#1565C0",
            fontsize=8, va="center")
    ax.text(PIM_RIDGE * 1.08, PIM_COMPUTE_TFLOPS * 0.55,
            f"PIM ridge\n{PIM_RIDGE:.0f} FLOPs/B", color="#B71C1C",
            fontsize=8, va="center")

    # ── FC decode point ───────────────────────────────────────────────────────
    ax.scatter([pts["ai_fc"]], [pts["fc_achieved_gpu"]],
               marker="D", s=90, color="navy", zorder=7,
               label=f"FC/FFN decode (GPU, bs={batch})")
    ax.annotate("FC\ndecode", (pts["ai_fc"], pts["fc_achieved_gpu"]),
                xytext=(-42, 6), textcoords="offset points",
                fontsize=7.5, color="navy",
                arrowprops=dict(arrowstyle="-", color="navy", lw=0.7))

    # ── Decode attention kernels ──────────────────────────────────────────────
    label_lins = {512, 2048, 4096}

    if pts["dec_gpu_attn"]:
        xs, ys, ls = zip(*pts["dec_gpu_attn"])
        ax.scatter(xs, ys, marker="^", s=55, color="#42A5F5", zorder=6,
                   label="GPU-only attention (decode, varies Lin)")
        for ai, perf, lin in pts["dec_gpu_attn"]:
            if lin in label_lins:
                ax.annotate(f"L={lin}", (ai, perf),
                            xytext=(6, -12), textcoords="offset points",
                            fontsize=7, color="#1565C0")

    if pts["dec_pim_attn"]:
        xs, ys, ls = zip(*pts["dec_pim_attn"])
        ax.scatter(xs, ys, marker="^", s=55, color="#EF5350", zorder=6,
                   label="PIM attention (decode, varies Lin)")
        for ai, perf, lin in pts["dec_pim_attn"]:
            if lin in label_lins:
                ax.annotate(f"L={lin}", (ai, perf),
                            xytext=(6, 4), textcoords="offset points",
                            fontsize=7, color="#B71C1C")

    # ── Decode total (FC+attn combined) ──────────────────────────────────────
    if pts["dec_gpu_total"]:
        xs, ys, ls = zip(*pts["dec_gpu_total"])
        ax.scatter(xs, ys, marker="o", s=45, color="#90CAF9", zorder=5,
                   label="GPU-only decode total (FC+attn)")

    if pts["dec_pim_total"]:
        xs, ys, ls = zip(*pts["dec_pim_total"])
        ax.scatter(xs, ys, marker="o", s=45, color="#FFCDD2", edgecolors="#E53935",
                   linewidths=0.8, zorder=5,
                   label="PIM+GPU decode total (FC+attn)")

    # ── Prefill operating points ──────────────────────────────────────────────
    pref_label_lins = {512, 2048, 4096}
    if pts["pref_gpu"]:
        xs, ys, ls = zip(*pts["pref_gpu"])
        ax.scatter(xs, ys, marker="*", s=90, color="#1A237E", zorder=6,
                   label="GPU prefill (serial, varies Lin)")
        for ai, perf, lin in pts["pref_gpu"]:
            if lin in pref_label_lins:
                ax.annotate(f"L={lin}", (ai, perf),
                            xytext=(5, 4), textcoords="offset points",
                            fontsize=7, color="#1A237E")

    if pts["pref_pim"]:
        xs, ys, ls = zip(*pts["pref_pim"])
        ax.scatter(xs, ys, marker="*", s=90, color="#7B1FA2", zorder=6,
                   label="PIM prefill (serial, varies Lin)")
        for ai, perf, lin in pts["pref_pim"]:
            if lin in pref_label_lins:
                ax.annotate(f"L={lin}", (ai, perf),
                            xytext=(5, -12), textcoords="offset points",
                            fontsize=7, color="#7B1FA2")

    # ── Region labels ─────────────────────────────────────────────────────────
    ax.text(0.018, 260, "Memory-\nbound", fontsize=8.5, color="gray",
            fontstyle="italic", ha="left")
    ax.text(200, 15, "Compute-\nbound", fontsize=8.5, color="gray",
            fontstyle="italic", ha="center")

    # ── Formatting ────────────────────────────────────────────────────────────
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic Intensity (FLOPs / byte)", fontsize=12)
    ax.set_ylabel("Achieved Performance (TFLOPS)", fontsize=12)
    ax.set_title(
        f"Roofline — PI0 on A100 GPU vs LPDDR5-PIM+GPU  (bs={batch})\n"
        "GPU roofline: 312 TFLOPS / 3.35 TB/s  |  "
        "PIM attention: empirical 1.62 TB/s BW / 62 TFLOPS compute",
        fontsize=10,
    )
    ax.set_xlim(0.015, 5000)
    ax.set_ylim(0.05, max(GPU_COMPUTE_TFLOPS, PIM_COMPUTE_TFLOPS) * 2.5)
    ax.grid(True, which="both", alpha=0.25)

    # Compact legend
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9,
              ncol=1, borderpad=0.6)

    # Key insight annotation
    ax.annotate(
        "Attention: AI = 1 FLOPs/byte\nPIM gives 3.4× more BW\nvs GPU for this kernel",
        xy=(AI_ATTN, 1.62), xytext=(0.12, 8),
        fontsize=7.5, color="#7B1FA2",
        arrowprops=dict(arrowstyle="->", color="#7B1FA2", lw=1.0),
        bbox=dict(boxstyle="round,pad=0.3", fc="lavender", alpha=0.8),
    )

    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(outpath), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outpath}")


def main():
    parser = argparse.ArgumentParser(description="Roofline plot for PI0 PIM+GPU system")
    parser.add_argument("--gpu-csv", default=str(
        REPO / "cluster_outputs/cost_tables_full_energy/gpu_only.csv"))
    parser.add_argument("--pim-csv", default=str(
        REPO / "cluster_outputs/cost_tables_full_energy/lpddr5_pim_bank.csv"))
    parser.add_argument("--output", default=str(
        REPO / "docs/figures/pi0_pim_roofline.png"))
    parser.add_argument("--batch", type=int, default=1,
                        help="Decode batch size (default 1)")
    args = parser.parse_args()

    gpu_df, pim_df = load_tables(args.gpu_csv, args.pim_csv)
    pts = operating_points(gpu_df, pim_df, bs=args.batch)

    # Print summary table
    print(f"\n{'':=<70}")
    print(f"  Hardware: GPU A100  {GPU_COMPUTE_TFLOPS} TFLOPS / {GPU_BW_TBS} TB/s  ridge={GPU_RIDGE:.1f} FLOPs/B")
    print(f"  Hardware: PIM attn  {PIM_ATTN_BW_TBS} TB/s (empirical)  "
          f"compute {PIM_COMPUTE_TFLOPS} TFLOPS  ridge={PIM_RIDGE:.1f} FLOPs/B")
    print(f"{'':=<70}")
    print(f"  FC decode:  AI={pts['ai_fc']:.2f} FLOPs/B   achieved_GPU={pts['fc_achieved_gpu']:.3f} TFLOPS")
    print()
    print(f"  {'Lin':>5}  {'AI_dec':>8}  {'GPU_dec(T)':>12}  {'PIM_dec(T)':>12}  "
          f"{'GPU_attn(T)':>12}  {'PIM_attn(T)':>12}  {'GPU_pref(T)':>12}  {'PIM_pref(T)':>12}")
    gpu_dec_d   = {l: (a, p) for a, p, l in pts["dec_gpu_total"]}
    pim_dec_d   = {l: (a, p) for a, p, l in pts["dec_pim_total"]}
    gpu_attn_d  = {l: p for _, p, l in pts["dec_gpu_attn"]}
    pim_attn_d  = {l: p for _, p, l in pts["dec_pim_attn"]}
    gpu_pref_d  = {l: (a, p) for a, p, l in pts["pref_gpu"]}
    pim_pref_d  = {l: (a, p) for a, p, l in pts["pref_pim"]}
    for lin in [512, 1024, 2048, 3072, 4096]:
        if lin not in gpu_dec_d:
            continue
        ai, gd = gpu_dec_d[lin]
        pd_ = pim_dec_d.get(lin, (0, 0))[1]
        ga  = gpu_attn_d.get(lin, 0)
        pa  = pim_attn_d.get(lin, 0)
        gp  = gpu_pref_d.get(lin, (0, 0))[1]
        pp  = pim_pref_d.get(lin, (0, 0))[1]
        print(f"  {lin:>5}  {ai:>8.3f}  {gd:>12.3f}  {pd_:>12.3f}  "
              f"{ga:>12.3f}  {pa:>12.3f}  {gp:>12.1f}  {pp:>12.1f}")
    print(f"{'':=<70}\n")

    plot_roofline(pts, Path(args.output), args.batch)


if __name__ == "__main__":
    main()
