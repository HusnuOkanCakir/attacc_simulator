#!/usr/bin/env python3
"""Compute the `arrival_scale` factor that brings a trace's peak
instantaneous RPS to a desired target `a`.

Semantics: the simulator's `--arrival-scale` multiplies all trace
timestamps. A scale of 0.5 → arrivals happen at half the original
inter-arrival time → 2× the offered RPS. So
    target_peak / current_peak = 1 / scale
    ⇒ scale = current_peak / target_peak

Closes the loop on the knee-finder pipeline: once thesis_rps_knee.py
returns `a = 14 req/s` for a given (model, policy), and
profile_trace_rps.py says the conversational synthetic peaks at 4 RPS,
this helper says "use arrival_scale = 4 / 14 ≈ 0.286" to push the trace
to operate right at the knee.

Optionally writes out a pre-scaled CSV (timestamps already multiplied)
so downstream sweeps can use `--arrival-scale 1.0` instead of carrying
the scale separately.

Usage:
    python tools/scale_trace_to_peak.py \\
        --csv cluster_outputs/synthetic_v2/conv_n5000_r5_seed0.csv \\
        --target-peak-rps 14
    # prints suggested arrival_scale + new peak

    python tools/scale_trace_to_peak.py \\
        --csv <in.csv> --target-peak-rps 14 \\
        --write-scaled <out.csv>
    # also writes a pre-scaled CSV
"""

import argparse
import csv
import datetime as _dt
import sys
from pathlib import Path

# Reuse profiler helpers — they parse timestamps + bucket RPS the same way.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from profile_trace_rps import parse_ts, bucket_rps  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--target-peak-rps", type=float, required=True,
                    help="Desired peak instantaneous RPS (e.g. the `a` "
                         "value from the knee-finder figure).")
    ap.add_argument("--bucket-ms", type=int, default=1000,
                    help="Bucket for measuring current peak (default 1000ms).")
    ap.add_argument("--write-scaled", type=Path, default=None,
                    help="If given, write a pre-scaled copy of the trace "
                         "with all timestamps multiplied by the suggested "
                         "scale factor.")
    args = ap.parse_args()

    if not args.csv.is_file():
        sys.exit(f"[error] CSV not found: {args.csv}")
    if args.target_peak_rps <= 0:
        sys.exit("[error] --target-peak-rps must be positive")

    # Read rows.
    rows = []
    with open(args.csv, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append((parse_ts(r["TIMESTAMP"]),
                         int(r["ContextTokens"]),
                         int(r["GeneratedTokens"])))
    if not rows:
        sys.exit(f"[error] empty CSV: {args.csv}")

    timestamps = [r[0] for r in rows]
    n = len(rows)

    rps_rows = bucket_rps(timestamps, args.bucket_ms)
    current_peak = max(c for _, c in rps_rows) * 1000.0 / args.bucket_ms

    scale = current_peak / args.target_peak_rps
    new_peak = current_peak / scale

    span_s = (timestamps[-1] - timestamps[0]).total_seconds()
    print(f"== scale_trace_to_peak  {args.csv.name} ==")
    print(f"  n_requests         {n}")
    print(f"  current span       {span_s:.1f} s")
    print(f"  current peak RPS   {current_peak:.2f}")
    print(f"  target peak RPS    {args.target_peak_rps:.2f}")
    print()
    print(f"  → suggested --arrival-scale {scale:.4f}")
    print(f"    (new peak after scaling: {new_peak:.2f} ≈ target)")
    if scale > 1:
        print(f"    NB: scale>1 ⇒ arrivals SPREAD OUT, offered load DROPS.")
    elif scale < 1:
        print(f"    NB: scale<1 ⇒ arrivals COMPRESS, offered load RISES.")

    if args.write_scaled is not None:
        t0 = timestamps[0]
        out_rows = []
        for ts, lin, lout in rows:
            new_ts = t0 + (ts - t0) * scale
            out_rows.append((new_ts, lin, lout))
        args.write_scaled.parent.mkdir(parents=True, exist_ok=True)
        with open(args.write_scaled, "w") as f:
            f.write("TIMESTAMP,ContextTokens,GeneratedTokens\n")
            for ts, lin, lout in out_rows:
                ts_str = ts.strftime("%Y-%m-%d %H:%M:%S.") + f"{ts.microsecond:06d}0"
                f.write(f"{ts_str},{lin},{lout}\n")
        new_span = (out_rows[-1][0] - out_rows[0][0]).total_seconds()
        print()
        print(f"  → wrote pre-scaled CSV: {args.write_scaled}")
        print(f"    new span: {new_span:.1f} s   "
              f"(effective mean RPS {n / new_span:.2f})")


if __name__ == "__main__":
    main()
