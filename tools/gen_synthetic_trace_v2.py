#!/usr/bin/env python3
"""Two-preset Poisson-shaped synthetic trace generator.

Output schema is identical to the Azure CSV reader + the v1 generator:

    TIMESTAMP,ContextTokens,GeneratedTokens

Per-request distributions:

  * **ContextTokens** ~ Poisson(λ_Lin), clipped to [min_Lin, max_Lin]
  * **GeneratedTokens** ~ Poisson(λ_Lout), clipped to [1, max_Lout]
  * **Inter-arrival** ~ Exponential(rate_hz)  (Poisson process)

Two presets:

  * `robotic_vla`     — Lin~Poisson(306), Lout~Poisson(7) (OpenVLA)
                        or Lout~Poisson(50) (Pi0, via --lout-mode pi0).
                        Mimics the v1 synthetic VLA distribution but
                        Poisson-sampled rather than fixed+jitter.
  * `conversational`  — λ values lifted from the FULL 19 366-row Azure
                        conv CSV (mean Lin=1155, Lout=211; see
                        AZURE_CONV_STATS below). Poisson std (√λ) is
                        much tighter than Azure's true heavy-tail std,
                        so this preset captures the MEAN shape but not
                        the heavy-tail. Use `--shape-distribution
                        empirical` (future) for the heavy-tail version.

Usage:
    python tools/gen_synthetic_trace_v2.py \\
        --preset conversational --n-requests 5000 --rate-hz 5 \\
        --out-csv cluster_outputs/synthetic/conv_n5000_r5.csv

NB: the v1 generator (tools/gen_vla_synthetic_trace.py) stays in
place — E1, E2, E3 sweeps depend on it for reproducibility. This
file is its forward-compatible successor.
"""

import argparse
import datetime as _dt
import math
import random
import sys
from pathlib import Path


# ── Azure conv stats (FULL 19 366-row trace) ──────────────────────────
# Source: cluster_outputs/azure/AzureLLMInferenceTrace_conv.csv
# Profiled via tools/profile_trace_rps.py on 2026-05-29.
# Stored here for reproducibility — no runtime profiling.
AZURE_CONV_STATS = {
    "n_requests":       19366,
    "span_s":           3501.7,
    "mean_rps":         5.53,
    "peak_rps_1s":      16.0,
    "lin": {
        "mean":   1154.7,
        "std":    1108.8,
        "p50":    1020,
        "p90":    2734,
        "p99":    4142,
        "min":    2,
        "max":    14050,
    },
    "lout": {
        "mean":   211.1,
        "std":    162.9,
        "p50":    129,
        "p90":    424,
        "p99":    601,
        "min":    7,
        "max":    1000,
    },
    "interarrival_cv":  1.094,   # ≈ 1 ⇒ Poisson process confirmed
}

# 101-point empirical CDF (p0..p100) from the full 19 366-row Azure conv.
# Used by --shape-distribution empirical for the conversational preset
# so the generated trace matches Azure's heavy-tail Lin / Lout
# distribution (Poisson(λ=mean) underestimates the tail ~30×).
AZURE_LIN_CDF = [
    2, 29, 91, 161, 181, 181, 191, 197, 201, 203, 207, 212, 222, 242,
    317, 372, 375, 378, 381, 384, 387, 390, 392, 394, 395, 396, 398,
    399, 400, 402, 403, 405, 406, 408, 411, 414, 419, 424, 433, 456,
    703, 863, 903, 946, 972, 984, 994, 1002, 1009, 1014, 1020, 1025,
    1031, 1036, 1041, 1045, 1050, 1055, 1060, 1065, 1071, 1076, 1081,
    1086, 1092, 1097, 1102, 1109, 1116, 1122, 1130, 1139, 1147, 1159,
    1171, 1189, 1214, 1249, 1311, 1314, 1315, 1321, 1368, 1508, 1690,
    1876, 2032, 2221, 2401, 2584, 2734, 3291, 4073, 4076, 4079, 4083,
    4086, 4090, 4098, 4142, 14050,
]
AZURE_LOUT_CDF = [
    7, 16, 24, 32, 38, 41, 44, 47, 49, 52, 54, 57, 59, 62, 64, 67, 69,
    71, 73, 75, 77, 79, 81, 82, 84, 85, 86, 87, 89, 90, 91, 92, 93,
    94, 95, 96, 97, 99, 100, 101, 103, 105, 107, 108, 110, 113, 116,
    118, 122, 125, 129, 133, 137, 141, 145, 151, 156, 161, 163, 169,
    177, 182, 190, 202, 216, 217, 237, 371, 375, 378, 384, 387, 390,
    393, 394, 395, 396, 396, 397, 398, 400, 402, 404, 407, 410, 411,
    414, 415, 417, 420, 424, 428, 429, 433, 442, 451, 468, 509, 540,
    601, 1000,
]

# Mixture-of-Poissons modes for the "wide" conversational variant.
# Each request picks a mode by weight, then draws Lin and Lout from a
# Poisson centered at that mode's λ. The aggregate distribution gets
# wide spread approximating Azure's percentile structure while every
# individual draw is still Poisson.
#
# Revision history:
#   v1 (2026-05-31 16:00) — weights 50/40/8/2 matched Azure F(x) at the
#     p50/p90/p99 thresholds but produced only 1.7% Lin>4 000. PIM
#     hybrid showed NO TTFT advantage at this tail density.
#   v2 (2026-05-31 17:00) — weights 50/30/15/5 push more density into
#     the long/tail modes. Target: ~5–7% Lin>4 000 (intermediate
#     between v1 and azure_empirical's 8.2%). This is the "Option A"
#     bump for the tail-density ladder.
AZURE_WIDE_MIXTURE = [
    # (weight, lin_lambda, lout_lambda, label)
    (0.50,  510,  65, "short  (≤p50)"),
    (0.30, 1700, 265, "medium (p50–p90)"),
    (0.15, 3400, 520, "long   (p90–p99)"),
    (0.05, 9000, 850, "tail   (>p99)"),
]


def empirical_sample(cdf: list[int], rng: random.Random) -> int:
    """Inverse-CDF sample: u ~ Uniform(0,1) → linear-interp the
    101-point cdf and round to int. Preserves heavy-tail shape that
    Poisson(λ=mean) misses."""
    u = rng.random() * 100.0
    i = int(u)
    if i >= 100:
        return cdf[100]
    frac = u - i
    return max(1, int(round(cdf[i] + (cdf[i + 1] - cdf[i]) * frac)))


def mixture_pick(rng: random.Random):
    """Pick a (lin_lambda, lout_lambda) by AZURE_WIDE_MIXTURE weights."""
    u = rng.random()
    acc = 0.0
    for w, lin_l, lout_l, _label in AZURE_WIDE_MIXTURE:
        acc += w
        if u <= acc:
            return lin_l, lout_l
    return AZURE_WIDE_MIXTURE[-1][1], AZURE_WIDE_MIXTURE[-1][2]


# Real Azure conv trace used by the 'bootstrap' shape distribution.
AZURE_CONV_CSV = (Path(__file__).resolve().parents[1]
                  / "cluster_outputs/azure/AzureLLMInferenceTrace_conv.csv")


def load_azure_rows(csv_path: Path) -> list[tuple[int, int]]:
    """Load (ContextTokens, GeneratedTokens) pairs from the real Azure
    conv trace for joint bootstrap resampling."""
    pairs = []
    with open(csv_path) as f:
        header = f.readline().strip().split(",")
        idx_lin = header.index("ContextTokens")
        idx_lout = header.index("GeneratedTokens")
        for line in f:
            parts = line.rstrip("\n").split(",")
            try:
                pairs.append((int(parts[idx_lin]), int(parts[idx_lout])))
            except (ValueError, IndexError):
                continue
    if not pairs:
        raise ValueError(f"no rows parsed from {csv_path}")
    return pairs


PRESETS = {
    "robotic_vla": {
        "lambda_lin":   306,    # vision_prefix(256) + text_instruction(~50)
        "lambda_lout":  7,      # OpenVLA action tokens (override via --lout-mode pi0)
        "min_lin":      60,     # avoid impossibly short prompts
        "max_lin":      700,
        "max_lout":     200,
        "rate_hz":      30.0,   # robot ctrl loop frequency, high enough to stress
    },
    "conversational": {
        # λ from the FULL Azure conv 19k profile above.
        "lambda_lin":   int(round(AZURE_CONV_STATS["lin"]["mean"])),    # 1155
        "lambda_lout":  int(round(AZURE_CONV_STATS["lout"]["mean"])),   # 211
        "min_lin":      32,
        "max_lin":      6144,   # cost-table sweep upper bound
        "max_lout":     800,    # cost-table sweep upper bound
        "rate_hz":      5.5,    # ≈ Azure mean rate
    },
}


def gen_trace(preset_name: str,
              n: int,
              rate_hz: float,
              start_time: _dt.datetime,
              lout_mode: str | None,
              shape_dist: str,
              rng: random.Random):
    """Return N (timestamp_str, ContextTokens, GeneratedTokens) rows.

    `shape_dist` is either 'poisson' (per-request Poisson(λ) on each
    metric) or 'empirical' (inverse-CDF sample from AZURE_*_CDF). The
    empirical mode preserves Azure's heavy tail; Poisson does not.
    """
    if preset_name not in PRESETS:
        raise ValueError(f"unknown preset {preset_name!r} — "
                         f"choose from {sorted(PRESETS)}")
    if shape_dist not in ("poisson", "empirical", "mixture_poisson",
                          "bootstrap"):
        raise ValueError(f"unknown shape_dist {shape_dist!r}")
    if shape_dist in ("empirical", "mixture_poisson", "bootstrap") \
            and preset_name != "conversational":
        # robotic_vla has no analogue source trace for these shapes.
        raise ValueError(f"--shape-distribution {shape_dist} only valid "
                         "with --preset conversational")

    # Bootstrap: joint (Lin, Lout) row resampling from the real trace.
    # Unlike 'empirical' (independent marginals), this preserves the
    # Lin-Lout correlation of the real workload exactly.
    azure_pairs = load_azure_rows(AZURE_CONV_CSV) \
        if shape_dist == "bootstrap" else None

    cfg = dict(PRESETS[preset_name])

    # Pi0 override for robotic preset (Lout=50 instead of 7).
    if preset_name == "robotic_vla" and lout_mode == "pi0":
        cfg["lambda_lout"] = 50

    lambda_lin  = cfg["lambda_lin"]
    lambda_lout = cfg["lambda_lout"]
    min_lin, max_lin   = cfg["min_lin"], cfg["max_lin"]
    max_lout            = cfg["max_lout"]

    # NumPy not required — manual Knuth Poisson sampler is fast enough
    # for n in the tens-of-thousands range we use here.
    def poisson(lam: float) -> int:
        if lam <= 30:
            L = math.exp(-lam)
            k = 0
            p = 1.0
            while True:
                k += 1
                p *= rng.random()
                if p <= L:
                    return k - 1
        return max(0, int(round(rng.gauss(lam, math.sqrt(lam)))))

    def sample_lin_lout_independent():
        """Used by 'poisson' (fixed λ) and 'empirical' (CDF)."""
        if shape_dist == "empirical":
            return (empirical_sample(AZURE_LIN_CDF, rng),
                    empirical_sample(AZURE_LOUT_CDF, rng))
        return poisson(lambda_lin), poisson(lambda_lout)

    rows = []
    cumulative_s = 0.0

    for _ in range(n):
        gap = rng.expovariate(rate_hz)
        cumulative_s += gap

        if shape_dist == "mixture_poisson":
            # Pick a mode (one shared mode per request → correlates Lin
            # and Lout, matching Azure's "short prompts → short completions").
            lin_l, lout_l = mixture_pick(rng)
            lin_raw, lout_raw = poisson(lin_l), poisson(lout_l)
        elif shape_dist == "bootstrap":
            lin_raw, lout_raw = azure_pairs[rng.randrange(len(azure_pairs))]
        else:
            lin_raw, lout_raw = sample_lin_lout_independent()

        lin  = max(min_lin, min(lin_raw,  max_lin))
        lout = max(1,        min(lout_raw, max_lout))

        ts = start_time + _dt.timedelta(seconds=cumulative_s)
        ts_str = ts.strftime("%Y-%m-%d %H:%M:%S.") + f"{ts.microsecond:06d}0"
        rows.append((ts_str, lin, lout))

    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=sorted(PRESETS), required=True,
                    help="Workload preset.")
    ap.add_argument("--n-requests", type=int, required=True,
                    help="Number of requests to generate.")
    ap.add_argument("--rate-hz", type=float, default=None,
                    help="Mean arrival rate (Hz). Default = preset's rate_hz "
                         "(robotic=30, conversational=5.5). Smaller rate_hz "
                         "= sparser arrivals; the runner's --arrival-scale "
                         "can compress further at sim time.")
    ap.add_argument("--lout-mode", choices=["openvla", "pi0"], default="openvla",
                    help="Only meaningful for --preset robotic_vla. "
                         "openvla=Lout~Poisson(7), pi0=Lout~Poisson(50).")
    ap.add_argument("--shape-distribution",
                    choices=["poisson", "empirical", "mixture_poisson",
                             "bootstrap"],
                    default="poisson",
                    help="poisson = Lin~Poisson(λ), Lout~Poisson(λ); "
                         "empirical = inverse-CDF sample from AZURE_*_CDF "
                         "(preserves Azure's heavy tail; conversational "
                         "preset only); mixture_poisson = pick a mode by "
                         "weighted choice, then Poisson within (4 modes "
                         "match Azure's p50/p90/p99/tail; conversational "
                         "preset only); bootstrap = sample (Lin, Lout) "
                         "ROWS jointly with replacement from the real "
                         "Azure 19k conv trace (preserves the Lin-Lout "
                         "correlation; conversational preset only). "
                         "Default poisson.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start-time", type=str,
                    default="2026-05-14 00:00:00")
    ap.add_argument("--out-csv", type=Path, required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    start = _dt.datetime.fromisoformat(args.start_time)
    rate_hz = args.rate_hz if args.rate_hz is not None \
              else PRESETS[args.preset]["rate_hz"]

    rows = gen_trace(args.preset, args.n_requests, rate_hz,
                     start, args.lout_mode, args.shape_distribution, rng)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w") as f:
        f.write("TIMESTAMP,ContextTokens,GeneratedTokens\n")
        for ts, lin, lout in rows:
            f.write(f"{ts},{lin},{lout}\n")

    cfg = PRESETS[args.preset]
    lambda_lout = (50 if (args.preset == "robotic_vla" and args.lout_mode == "pi0")
                   else cfg["lambda_lout"])

    span_s = (_dt.datetime.fromisoformat(rows[-1][0].split('.')[0]) - start).total_seconds() + 1
    print(f"[gen_synthetic_trace_v2] preset={args.preset}  n={args.n_requests}  "
          f"rate={rate_hz} Hz  seed={args.seed}  shape={args.shape_distribution}")
    if args.shape_distribution == "poisson":
        print(f"  λ_Lin  = {cfg['lambda_lin']}  (clipped to [{cfg['min_lin']}, {cfg['max_lin']}])")
        print(f"  λ_Lout = {lambda_lout}        (clipped to [1, {cfg['max_lout']}])")
    elif args.shape_distribution == "empirical":
        print(f"  Lin   ~ empirical CDF (101-point) from full 19k Azure conv")
        print(f"  Lout  ~ empirical CDF (101-point) from full 19k Azure conv")
    elif args.shape_distribution == "bootstrap":
        print(f"  (Lin, Lout) rows resampled jointly from {AZURE_CONV_CSV.name}"
              f" (clipped to Lin<=6144, Lout<=800)")
    else:  # mixture_poisson
        print("  Shape ~ mixture of 4 Poissons matching Azure percentiles:")
        for w, lin_l, lout_l, label in AZURE_WIDE_MIXTURE:
            print(f"    w={w:.2f}  λ_Lin={lin_l:>4}  λ_Lout={lout_l:>4}  {label}")
    print(f"  arrival span ≈ {span_s:.1f} s "
          f"(effective {args.n_requests / max(span_s, 1e-9):.2f} req/s)")
    print(f"  wrote {args.out_csv}")


if __name__ == "__main__":
    main()
