#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.azure_trace import load_azure_llm_trace
from src.scheduler_policy import QueueAwareFinishTimePolicy
from src.trace_replay_sim import (OutputCsvCostModel, ReplayConfig,
                                  TraceReplaySimulator)


DEFAULT_AZURE_TRACE = (
    "https://raw.githubusercontent.com/Azure/AzurePublicDataset/refs/heads/master/"
    "data/AzureLLMInferenceTrace_code.csv")


def _parse_profile_args(items: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(
                f"Invalid --profile '{item}'. Expected format route=/path/to/output.csv")
        route, path = item.split("=", 1)
        route = route.strip()
        path = path.strip()
        if not route or not path:
            raise ValueError(f"Invalid --profile '{item}'. Empty route/path.")
        out[route] = path
    return out


def main() -> int:
    p = argparse.ArgumentParser(
        description="Replay request arrivals from Azure LLM trace using calibrated attacc_simulator cost profiles.")
    p.add_argument("--azure-trace",
                   type=str,
                   default=DEFAULT_AZURE_TRACE,
                   help="Path or URL to Azure LLM inference trace CSV")
    p.add_argument("--limit", type=int, default=1000, help="Max requests to replay")
    p.add_argument("--start-offset", type=int, default=0, help="Skip this many trace rows first")
    p.add_argument("--arrival-time-scale",
                   type=float,
                   default=1.0,
                   help="Scale normalized arrivals in time (e.g., 0.5 doubles offered load)")

    p.add_argument("--profile",
                   action="append",
                   required=True,
                   help=("Route calibration in form route=output.csv. "
                         "Example: gpu_only=cluster_outputs/output_gpu.csv "
                         "lpddr5_pim_bank=cluster_outputs/output_lpddr5.csv"))
    p.add_argument("--routes",
                   nargs="*",
                   default=None,
                   help="Optional explicit route order (subset of --profile keys)")

    p.add_argument("--unsupported-policy",
                   choices=["nearest", "clip", "drop"],
                   default="nearest",
                   help="How to map requests whose (Lin,Lout) are not in the cost table")
    p.add_argument("--lin-bucket", type=int, default=1, help="Bucket size for Lin before lookup")
    p.add_argument("--lout-bucket", type=int, default=1, help="Bucket size for Lout before lookup")
    p.add_argument("--batch-size", type=int, default=1, help="Cost-table batch size for lookup")
    p.add_argument("--sum-offload-to-pim",
                   action="store_true",
                   help="Interpret hybrid profiles as if sum/prefill stage is split across GPU+PIM")

    p.add_argument("--pim-wait-threshold-ms",
                   type=float,
                   default=None,
                   help="If predicted PIM wait exceeds threshold, route to GPU-only")
    p.add_argument("--no-prompt-priority",
                   action="store_true",
                   help="Disable prompt/prefill prioritization in local scheduling")

    p.add_argument("--slo-ttft-ms", type=float, default=None, help="Optional TTFT SLO threshold")
    p.add_argument("--slo-e2e-ms", type=float, default=None, help="Optional E2E SLO threshold")

    p.add_argument("--requests-csv",
                   type=str,
                   default=None,
                   help="Optional path to write per-request replay results")
    p.add_argument("--summary-json",
                   type=str,
                   default=None,
                   help="Optional path to write summary JSON")

    args = p.parse_args()

    route_to_csv = _parse_profile_args(args.profile)
    route_order = args.routes if args.routes else list(route_to_csv.keys())
    for r in route_order:
        if r not in route_to_csv:
            raise ValueError(f"Route '{r}' in --routes missing from --profile")

    arrivals = load_azure_llm_trace(args.azure_trace,
                                    limit=args.limit,
                                    start_offset=args.start_offset,
                                    arrival_time_scale=args.arrival_time_scale)
    if not arrivals:
        raise RuntimeError("No trace requests loaded after filtering.")

    cost_model = OutputCsvCostModel.from_route_csvs(route_to_csv,
                                                    sum_offload_to_pim=args.sum_offload_to_pim)
    policy = QueueAwareFinishTimePolicy(
        pim_wait_threshold_ms=args.pim_wait_threshold_ms)
    cfg = ReplayConfig(unsupported_policy=args.unsupported_policy,
                       lin_bucket=args.lin_bucket,
                       lout_bucket=args.lout_bucket,
                       batch_size=args.batch_size,
                       prompt_priority=(not args.no_prompt_priority),
                       slo_e2e_ms=args.slo_e2e_ms,
                       slo_ttft_ms=args.slo_ttft_ms)

    sim = TraceReplaySimulator(arrivals=arrivals,
                               cost_model=cost_model,
                               policy=policy,
                               config=cfg,
                               route_order=route_order)
    summary = sim.run()

    print("Trace Replay Summary")
    print(summary.to_json())

    if args.requests_csv:
        sim.requests_dataframe().to_csv(args.requests_csv, index=False)
        print(f"Wrote per-request results: {args.requests_csv}")

    if args.summary_json:
        with open(args.summary_json, "w") as f:
            f.write(summary.to_json() + "\n")
        print(f"Wrote summary JSON: {args.summary_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
