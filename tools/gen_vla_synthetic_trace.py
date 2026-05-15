#!/usr/bin/env python3
"""
Generate a synthetic VLA workload trace compatible with the Azure CSV reader.

Output schema matches `cluster_outputs/azure/AzureLLMInferenceTrace_conv_*.csv`:

    TIMESTAMP,ContextTokens,GeneratedTokens
    2023-11-16 18:15:46.6805900,374,44
    ...

Per-request distribution (paper-grounded VLA defaults):

  - **OpenVLA**: vision_prefix=256 + text_instruction~50 → Lin≈306, Lout=7
    (one action token per dim). See OpenVLA paper §3.
  - **Pi0**:     vision_prefix=256 + text_instruction~50 → Lin≈306, Lout=50
    (action chunk H=50). Simulator caps autoregressive decode here — the real
    Pi0 inference is 10 flow-matching steps over a 50-token suffix, modeled
    here as autoregressive decode (per plan scope: architecture-only).

Arrival pattern: Poisson(rate_hz) for cumulative inter-arrival times. Default
6 Hz (typical robot control loop). Use --rate-hz for sustained-load sweeps.

Usage:
    python tools/gen_vla_synthetic_trace.py \\
        --model openvla --n-requests 200 --rate-hz 6 --seed 42 \\
        --out-csv cluster_outputs/synthetic/vla_openvla_n200.csv
"""

import argparse
import datetime as _dt
import math
import random
from pathlib import Path


# Paper-grounded defaults per model.
MODEL_DEFAULTS = {
    "openvla": dict(
        vision_prefix_tokens=256,
        text_instruction_mean=50,
        text_instruction_jitter=10,
        generated_tokens=7,           # paper: 7 action tokens per inference
    ),
    "pi0": dict(
        vision_prefix_tokens=256,
        text_instruction_mean=50,
        text_instruction_jitter=10,
        generated_tokens=50,          # action chunk H=50; flow matching not modeled
    ),
}


def gen_trace(model: str, n: int, rate_hz: float,
              start_time: _dt.datetime, jitter: float,
              rng: random.Random) -> list[tuple[str, int, int]]:
    """Return N (timestamp, ctx_tokens, gen_tokens) rows."""
    if model not in MODEL_DEFAULTS:
        raise ValueError(f"Unknown model '{model}' — expected one of "
                         f"{sorted(MODEL_DEFAULTS)}")
    cfg = MODEL_DEFAULTS[model]

    mean_interarrival_s = 1.0 / rate_hz
    rows: list[tuple[str, int, int]] = []
    cumulative_s = 0.0

    for _ in range(n):
        # Exponential inter-arrival time → Poisson process at rate `rate_hz`.
        gap = rng.expovariate(rate_hz) if jitter > 0 else mean_interarrival_s
        cumulative_s += gap

        # Per-request context tokens: vision prefix (deterministic) + jittered text length.
        text_tokens = max(1, int(rng.gauss(cfg["text_instruction_mean"],
                                            cfg["text_instruction_jitter"])))
        ctx_tokens = cfg["vision_prefix_tokens"] + text_tokens

        # Output tokens fixed per inference; jitter ±1 around mean for stochasticity.
        gen_jitter = rng.randint(-1, 1) if jitter > 0 else 0
        gen_tokens = max(1, cfg["generated_tokens"] + gen_jitter)

        ts = start_time + _dt.timedelta(seconds=cumulative_s)
        # Azure trace fractional-second format: %Y-%m-%d %H:%M:%S.%f7  (7-digit µs)
        ts_str = ts.strftime("%Y-%m-%d %H:%M:%S.") + f"{ts.microsecond:06d}0"
        rows.append((ts_str, ctx_tokens, gen_tokens))

    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(MODEL_DEFAULTS), required=True,
                    help="VLA model — determines per-request token shape "
                         "(OpenVLA Lout=7, Pi0 Lout=50).")
    ap.add_argument("--n-requests", type=int, required=True,
                    help="Number of requests to generate.")
    ap.add_argument("--rate-hz", type=float, default=6.0,
                    help="Mean arrival rate (Hz). 6 Hz ~ typical robot control "
                         "frequency. Use 30+ for stress / saturation sweeps.")
    ap.add_argument("--jitter", type=float, default=1.0,
                    help="If 0, deterministic spacing + tokens. >0 enables "
                         "Poisson arrivals + Gaussian context jitter. Default 1.")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for reproducibility. Default 0.")
    ap.add_argument("--start-time", type=str,
                    default="2026-05-14 00:00:00",
                    help="ISO timestamp anchoring the first arrival. Default 2026-05-14 00:00:00.")
    ap.add_argument("--out-csv", type=Path, required=True,
                    help="Destination CSV path. Parent dirs created automatically.")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    start = _dt.datetime.fromisoformat(args.start_time)

    rows = gen_trace(args.model, args.n_requests, args.rate_hz,
                     start, args.jitter, rng)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w") as f:
        f.write("TIMESTAMP,ContextTokens,GeneratedTokens\n")
        for ts, ctx, gen in rows:
            f.write(f"{ts},{ctx},{gen}\n")

    cfg = MODEL_DEFAULTS[args.model]
    span_s = (_dt.datetime.fromisoformat(rows[-1][0].split('.')[0])
              - start).total_seconds() + 1
    print(f"[gen_vla_synthetic_trace] model={args.model}  n={args.n_requests}  "
          f"rate={args.rate_hz} Hz  jitter={args.jitter}  seed={args.seed}")
    print(f"  vision_prefix={cfg['vision_prefix_tokens']}  "
          f"text_mean={cfg['text_instruction_mean']}  Lout={cfg['generated_tokens']}")
    print(f"  arrival span ≈ {span_s:.1f}s "
          f"(effective rate {args.n_requests/max(span_s,1e-9):.2f} req/s)")
    print(f"  wrote {args.out_csv}")


if __name__ == "__main__":
    main()
