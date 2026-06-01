#!/usr/bin/env python3
"""Characterize a TIMESTAMP,ContextTokens,GeneratedTokens trace by:

  - per-bucket instantaneous RPS over time (default 1 s buckets)
  - shape distribution stats (Lin, Lout: mean, std, percentiles)
  - inter-arrival distribution + Poisson-fit diagnostic
  - headline: total reqs, span (s), mean / peak / p99 RPS

Outputs:
  - JSON profile sidecar:  <csv>.profile.json
  - Stdout summary suitable for copy-paste into Poisson λ constants

Usage:
    python tools/profile_trace_rps.py \\
        --csv cluster_outputs/azure/AzureLLMInferenceTrace_conv.csv \\
        --bucket-ms 1000
"""

import argparse
import csv
import datetime as _dt
import json
import math
import statistics
import sys
from pathlib import Path


TS_FORMATS = [
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
]


def parse_ts(s: str) -> _dt.datetime:
    s = s.strip()
    # Azure / synthetic trace fractional-second has 7 digits (...µs0).
    # Python's %f only takes up to 6; trim if needed.
    if "." in s:
        head, frac = s.split(".", 1)
        frac = frac[:6]
        s = f"{head}.{frac}"
    for fmt in TS_FORMATS:
        try:
            return _dt.datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise ValueError(f"unrecognized timestamp: {s!r}")


def load_rows(csv_path: Path):
    """Yield (ts: datetime, lin: int, lout: int) rows."""
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield (parse_ts(row["TIMESTAMP"]),
                   int(row["ContextTokens"]),
                   int(row["GeneratedTokens"]))


def percentiles(xs, ps=(50, 90, 95, 99)):
    xs = sorted(xs)
    out = {}
    for p in ps:
        if not xs:
            out[f"p{p}"] = 0.0
            continue
        k = (len(xs) - 1) * p / 100.0
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            out[f"p{p}"] = float(xs[f])
        else:
            out[f"p{p}"] = float(xs[f] + (xs[c] - xs[f]) * (k - f))
    return out


def bucket_rps(timestamps, bucket_ms: int):
    """Return list of (t_start_s, count_in_bucket) tuples covering the
    whole timespan, padded with zeros for empty buckets so the timeline
    is dense."""
    if not timestamps:
        return []
    t0 = timestamps[0]
    bucket_s = bucket_ms / 1000.0
    counts: dict[int, int] = {}
    for ts in timestamps:
        idx = int((ts - t0).total_seconds() // bucket_s)
        counts[idx] = counts.get(idx, 0) + 1
    last_idx = int((timestamps[-1] - t0).total_seconds() // bucket_s)
    rows = []
    for i in range(last_idx + 1):
        rows.append((i * bucket_s, counts.get(i, 0)))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, required=True,
                    help="Trace CSV path (TIMESTAMP,ContextTokens,GeneratedTokens)")
    ap.add_argument("--bucket-ms", type=int, default=1000,
                    help="Bucket size in ms for the RPS timeline (default 1000).")
    ap.add_argument("--out-json", type=Path, default=None,
                    help="Output JSON sidecar (default <csv>.profile.json).")
    args = ap.parse_args()

    if not args.csv.is_file():
        sys.exit(f"[error] CSV not found: {args.csv}")

    rows = list(load_rows(args.csv))
    if not rows:
        sys.exit(f"[error] empty CSV: {args.csv}")

    timestamps = [r[0] for r in rows]
    lins  = [r[1] for r in rows]
    louts = [r[2] for r in rows]

    span_s = (timestamps[-1] - timestamps[0]).total_seconds()
    n = len(rows)
    mean_rps = n / span_s if span_s > 0 else float("nan")

    # Inter-arrival times
    gaps_s = [(timestamps[i] - timestamps[i - 1]).total_seconds()
              for i in range(1, n)]
    gap_mean = statistics.fmean(gaps_s) if gaps_s else 0.0
    gap_std  = statistics.pstdev(gaps_s) if len(gaps_s) >= 2 else 0.0
    # For Poisson process: std == mean. CV (coeff. of variation) ≈ 1.
    cv = (gap_std / gap_mean) if gap_mean > 0 else float("nan")

    rps_buckets = bucket_rps(timestamps, args.bucket_ms)
    bucket_counts = [c for _, c in rps_buckets]
    # Convert counts/bucket -> requests/second
    rps_series = [c * 1000.0 / args.bucket_ms for c in bucket_counts]
    peak_rps = max(rps_series) if rps_series else 0.0
    rps_pct = percentiles(rps_series, (50, 90, 95, 99))

    lin_stats = {
        "mean": statistics.fmean(lins),
        "std":  statistics.pstdev(lins),
        "min":  min(lins),
        "max":  max(lins),
        **percentiles(lins),
    }
    lout_stats = {
        "mean": statistics.fmean(louts),
        "std":  statistics.pstdev(louts),
        "min":  min(louts),
        "max":  max(louts),
        **percentiles(louts),
    }

    summary = {
        "csv":         str(args.csv),
        "n_requests":  n,
        "span_s":      span_s,
        "mean_rps":    mean_rps,
        "bucket_ms":   args.bucket_ms,
        "peak_rps":    peak_rps,
        "rps_pct":     rps_pct,
        "n_buckets":   len(rps_series),
        "interarrival_mean_s": gap_mean,
        "interarrival_std_s":  gap_std,
        "interarrival_cv":     cv,
        "lin":         lin_stats,
        "lout":        lout_stats,
        "rps_timeline": [
            {"t_s": t, "rps": r}
            for t, r in zip([t for t, _ in rps_buckets], rps_series)
        ],
    }

    out_json = args.out_json or args.csv.with_suffix(args.csv.suffix + ".profile.json")
    out_json.write_text(json.dumps(summary, indent=2, default=str))

    # Human summary -- copy-paste friendly for the generator constants.
    print(f"== profile  {args.csv.name} ==")
    print(f"  n_requests           {n}")
    print(f"  span_s               {span_s:,.1f}  ({span_s / 3600:.2f} h)")
    print(f"  mean RPS             {mean_rps:.3f}")
    print(f"  peak RPS  (bucket={args.bucket_ms}ms)  {peak_rps:.2f}")
    print(f"  p99 RPS               {rps_pct['p99']:.2f}")
    print(f"  p95 RPS               {rps_pct['p95']:.2f}")
    print(f"  inter-arrival mean    {gap_mean:.3f} s  std {gap_std:.3f} s  "
          f"CV {cv:.3f}  (Poisson => CV≈1)")
    print()
    print(f"  ContextTokens (Lin)")
    print(f"    mean / std         {lin_stats['mean']:.1f} / {lin_stats['std']:.1f}")
    print(f"    p50 / p90 / p99    "
          f"{lin_stats['p50']:.0f} / {lin_stats['p90']:.0f} / {lin_stats['p99']:.0f}")
    print(f"    min / max          {lin_stats['min']} / {lin_stats['max']}")
    print()
    print(f"  GeneratedTokens (Lout)")
    print(f"    mean / std         {lout_stats['mean']:.1f} / {lout_stats['std']:.1f}")
    print(f"    p50 / p90 / p99    "
          f"{lout_stats['p50']:.0f} / {lout_stats['p90']:.0f} / {lout_stats['p99']:.0f}")
    print(f"    min / max          {lout_stats['min']} / {lout_stats['max']}")
    print()
    print(f"  -> wrote {out_json}")


if __name__ == "__main__":
    main()
