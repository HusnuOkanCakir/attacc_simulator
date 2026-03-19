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
from src.learned_cost_model import LearnedCostModel
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
                   default=[],
                   help=("Route calibration in form route=output.csv. "
                         "Example: gpu_only=cluster_outputs/output_gpu.csv "
                         "lpddr5_pim_bank=cluster_outputs/output_lpddr5.csv"))
    p.add_argument("--ml-profile",
                   action="append",
                   default=[],
                   help=("Learned model bundle in form route=path/to/model.pkl. "
                         "Used with --cost-model ml."))
    p.add_argument("--routes",
                   nargs="*",
                   default=None,
                   help="Optional explicit route order (subset of --profile keys)")
    p.add_argument("--cost-model",
                   choices=["table", "ml"],
                   default="table",
                   help=("table: exact/nearest/clip CSV lookup; "
                         "ml: learned interpolation model"))

    p.add_argument("--unsupported-policy",
                   choices=["nearest", "clip", "drop"],
                   default="nearest",
                   help="How to map requests whose (Lin,Lout) are not in the cost table")
    p.add_argument("--lin-bucket", type=int, default=1, help="Bucket size for Lin before lookup")
    p.add_argument("--lout-bucket", type=int, default=1, help="Bucket size for Lout before lookup")
    p.add_argument("--batch-size", type=int, default=1, help="Cost-table batch size for lookup")
    p.add_argument("--enable-prefill-batching",
                   action="store_true",
                   help="Enable same-route batching for prefill steps")
    p.add_argument("--max-prefill-batch-size",
                   type=int,
                   default=4,
                   help="Maximum prefill batch size when --enable-prefill-batching is set")
    p.add_argument("--enable-decode-batching",
                   action="store_true",
                   help="Enable continuous batching for decode steps")
    p.add_argument("--max-decode-batch-size",
                   type=int,
                   default=4,
                   help="Maximum decode batch size when --enable-decode-batching is set")
    p.add_argument("--prefill-guard-ms",
                   type=float,
                   default=None,
                   help=("If any waiting prefill exceeds this wait time, schedule prefill next. "
                         "Used for v3.1 prompt-aware decode batching."))
    p.add_argument("--max-consecutive-decode-batches",
                   type=int,
                   default=0,
                   help=("If > 0, force a prefill task after this many consecutive decode batches "
                         "when prefills are waiting."))
    p.add_argument("--decode-batch-cap-with-prefill",
                   type=int,
                   default=0,
                   help=("If > 0 and prefills are waiting, cap decode batch size to this value."))
    p.add_argument("--local-scheduling-policy",
                   choices=["priority", "fcfs_strict", "prefill_priority_fcfs_decode"],
                   default="priority",
                   help=("Local task execution policy. "
                         "priority: current prompt-aware policy; "
                         "fcfs_strict: strict request-level FCFS run-to-completion; "
                         "prefill_priority_fcfs_decode: prefills are priority, decode queue is FCFS "
                         "with continuous prefix batching."))
    p.add_argument("--fcfs-decode-prediction-mode",
                   choices=["shadow", "legacy", "incremental", "heuristic_refresh"],
                   default="shadow",
                   help=("Prediction mode used only by prefill_priority_fcfs_decode. "
                         "shadow: scheduler-aligned forward projection; "
                         "legacy: reuse the earlier analytic v3-style predictor; "
                         "incremental: deterministic FCFS-decode forecaster without the generic shadow sim; "
                         "heuristic_refresh: FCFS queue-aware analytic predictor with rolling refresh."))
    p.add_argument("--enable-decode-rebind",
                   action="store_true",
                   help=("Re-evaluate route once at prefill completion and allow "
                         "hybrid->gpu_only switch before first decode token."))
    p.add_argument("--decode-rebind-margin-ms",
                   type=float,
                   default=0.0,
                   help=("Minimum predicted finish-time improvement required to switch "
                         "decode route during rebind."))
    p.add_argument("--enable-predictive-admission",
                   action="store_true",
                   help=("Enable throttLLeM-style predictive admission gate. "
                         "Arrivals enter waiting_admission and are admitted only when feasible."))
    p.add_argument("--enable-slo-guarded-admission",
                   action="store_true",
                   help=("Enable deferred SLO-aware admission with admissible bypass on top of "
                         "prefill_priority_fcfs_decode + incremental prediction."))
    p.add_argument("--slo-tbt-ms",
                   type=float,
                   default=None,
                   help="Optional TBT SLO threshold used by admission control")
    p.add_argument("--admission-shadow-max-steps",
                   type=int,
                   default=10000,
                   help="Safety bound for step-level shadow admission projection")
    p.add_argument("--admission-retry-interval-ms",
                   type=float,
                   default=0.0,
                   help="Delay before retrying rejected admission queue head (0=recheck every cycle)")
    p.add_argument("--admission-max-harmed-requests",
                   type=int,
                   default=0,
                   help="Maximum number of existing active requests that may see increased E2E miss")
    p.add_argument("--admission-max-total-harm-ms",
                   type=float,
                   default=0.0,
                   help="Maximum total incremental E2E miss across existing active requests")
    p.add_argument("--admission-max-single-harm-ms",
                   type=float,
                   default=0.0,
                   help="Maximum incremental E2E miss allowed for any one existing active request")
    p.add_argument("--admission-max-own-miss-ms",
                   type=float,
                   default=0.0,
                   help="Maximum E2E miss allowed for the candidate request itself")
    p.add_argument("--admission-max-bypass-count",
                   type=int,
                   default=0,
                   help="Force best-effort admission once a queued request has been bypassed this many times")
    p.add_argument("--admission-max-wait-ms",
                   type=float,
                   default=0.0,
                   help="Force best-effort admission once queue wait reaches this threshold")
    p.add_argument("--admission-aging-harm-ms-per-ms",
                   type=float,
                   default=0.0,
                   help="Linear relaxation rate for total/single harm budgets as queue wait grows")
    p.add_argument("--admission-aging-harmed-requests-per-ms",
                   type=float,
                   default=0.0,
                   help="Linear relaxation rate for harmed-request-count budget as queue wait grows")
    p.add_argument("--sum-offload-to-pim",
                   action="store_true",
                   help="Interpret hybrid profiles as if sum/prefill stage is split across GPU+PIM")

    p.add_argument("--pim-wait-threshold-ms",
                   type=float,
                   default=None,
                   help="If predicted PIM wait exceeds threshold, route to GPU-only")
    p.add_argument("--gpu-queue-alpha",
                   type=float,
                   default=0.0,
                   help=("v2.1 queue-pressure term for pending GPU work. "
                         "Larger values penalize routes that depend more on GPU backlog."))
    p.add_argument("--pim-queue-alpha",
                   type=float,
                   default=0.0,
                   help=("v2.1 queue-pressure term for pending PIM work. "
                         "Larger values penalize routes that depend more on PIM backlog."))
    p.add_argument("--active-request-alpha",
                   type=float,
                   default=0.0,
                   help="v2.1 additive penalty per active request already in the system.")
    p.add_argument("--decode-token-alpha",
                   type=float,
                   default=0.0,
                   help=("v2.1 additive penalty proportional to pending decode tokens, "
                         "scaled by the route's per-token decode latency."))
    p.add_argument("--route-policy",
                   choices=["min_finish", "slack_then_finish", "latency_guarded_energy"],
                   default="min_finish",
                   help=("Admission-time route policy. "
                         "min_finish: smallest predicted finish time. "
                         "slack_then_finish: prefer routes that meet the E2E deadline, "
                         "otherwise choose least lateness. "
                         "latency_guarded_energy: choose the lowest-energy route within "
                         "an absolute finish-time guard of the fastest route."))
    p.add_argument("--energy-latency-guard-ms",
                   type=float,
                   default=5.0,
                   help=("Absolute finish-time guard used by --route-policy "
                         "latency_guarded_energy. Among routes within this guard "
                         "of the fastest predicted finish, choose the lowest "
                         "incremental forecasted energy route."))
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
    p.add_argument("--debug-events-csv",
                   type=str,
                   default=None,
                   help=("Optional path to write debug event timeline CSV "
                         "(admissions, dispatches, completions, and resource states)"))

    args = p.parse_args()

    route_to_csv = _parse_profile_args(args.profile) if args.profile else {}
    route_to_ml = _parse_profile_args(args.ml_profile) if args.ml_profile else {}

    if args.cost_model == "table":
        if not route_to_csv:
            raise ValueError("--cost-model table requires at least one --profile route=csv")
        known_routes = set(route_to_csv.keys())
    else:
        if not route_to_ml:
            raise ValueError("--cost-model ml requires at least one --ml-profile route=model.pkl")
        known_routes = set(route_to_ml.keys())

    route_order = args.routes if args.routes else sorted(known_routes)
    for r in route_order:
        if r not in known_routes:
            raise ValueError(f"Route '{r}' is unavailable for --cost-model {args.cost_model}")
    if args.route_policy == "slack_then_finish" and args.slo_e2e_ms is None:
        raise ValueError("--route-policy slack_then_finish requires --slo-e2e-ms")
    if args.route_policy == "latency_guarded_energy":
        if args.local_scheduling_policy != "prefill_priority_fcfs_decode":
            raise ValueError("--route-policy latency_guarded_energy requires --local-scheduling-policy prefill_priority_fcfs_decode")
        if args.fcfs_decode_prediction_mode != "incremental":
            raise ValueError("--route-policy latency_guarded_energy requires --fcfs-decode-prediction-mode incremental")
    if args.enable_predictive_admission:
        if args.slo_e2e_ms is None:
            raise ValueError("--enable-predictive-admission requires --slo-e2e-ms")
        if args.slo_tbt_ms is None:
            raise ValueError("--enable-predictive-admission requires --slo-tbt-ms")
    if args.enable_slo_guarded_admission:
        if args.slo_e2e_ms is None:
            raise ValueError("--enable-slo-guarded-admission requires --slo-e2e-ms")
        if args.slo_tbt_ms is None:
            raise ValueError("--enable-slo-guarded-admission requires --slo-tbt-ms")
    if args.local_scheduling_policy == "prefill_priority_fcfs_decode":
        if args.enable_predictive_admission:
            raise SystemExit("prefill_priority_fcfs_decode does not support --enable-predictive-admission")
        if args.enable_slo_guarded_admission and args.fcfs_decode_prediction_mode != "incremental":
            raise SystemExit("slo_guarded_admission requires --fcfs-decode-prediction-mode incremental")
        if args.enable_prefill_batching:
            raise SystemExit("prefill_priority_fcfs_decode does not support --enable-prefill-batching")
    else:
        if args.fcfs_decode_prediction_mode != "shadow":
            raise SystemExit("--fcfs-decode-prediction-mode only applies to --local-scheduling-policy prefill_priority_fcfs_decode")
        if args.enable_slo_guarded_admission:
            raise SystemExit("--enable-slo-guarded-admission requires --local-scheduling-policy prefill_priority_fcfs_decode")
    if args.enable_predictive_admission and args.enable_slo_guarded_admission:
        raise SystemExit("--enable-predictive-admission and --enable-slo-guarded-admission are mutually exclusive")

    arrivals = load_azure_llm_trace(args.azure_trace,
                                    limit=args.limit,
                                    start_offset=args.start_offset,
                                    arrival_time_scale=args.arrival_time_scale)
    if not arrivals:
        raise RuntimeError("No trace requests loaded after filtering.")

    table_model = None
    if route_to_csv:
        table_model = OutputCsvCostModel.from_route_csvs(route_to_csv,
                                                         sum_offload_to_pim=args.sum_offload_to_pim)

    if args.cost_model == "table":
        cost_model = table_model
    else:
        cost_model = LearnedCostModel.from_pickles(route_to_ml)
    policy = QueueAwareFinishTimePolicy(
        pim_wait_threshold_ms=args.pim_wait_threshold_ms,
        route_policy=args.route_policy,
        energy_latency_guard_ms=max(0.0, args.energy_latency_guard_ms))
    cfg = ReplayConfig(unsupported_policy=args.unsupported_policy,
                       lin_bucket=args.lin_bucket,
                       lout_bucket=args.lout_bucket,
                       batch_size=args.batch_size,
                       enable_prefill_batching=args.enable_prefill_batching,
                       max_prefill_batch_size=max(1, args.max_prefill_batch_size),
                       enable_decode_batching=args.enable_decode_batching,
                       max_decode_batch_size=max(1, args.max_decode_batch_size),
                       prefill_guard_ms=args.prefill_guard_ms,
                       max_consecutive_decode_batches=max(0, args.max_consecutive_decode_batches),
                       decode_batch_cap_with_prefill=max(0, args.decode_batch_cap_with_prefill),
                       local_scheduling_policy=args.local_scheduling_policy,
                       fcfs_decode_prediction_mode=args.fcfs_decode_prediction_mode,
                       enable_decode_rebind=args.enable_decode_rebind,
                       decode_rebind_margin_ms=max(0.0, args.decode_rebind_margin_ms),
                       enable_predictive_admission=args.enable_predictive_admission,
                       enable_slo_guarded_admission=args.enable_slo_guarded_admission,
                       slo_tbt_ms=args.slo_tbt_ms,
                       admission_shadow_max_steps=max(1, args.admission_shadow_max_steps),
                       admission_retry_interval_ms=max(0.0, args.admission_retry_interval_ms),
                       admission_max_harmed_requests=max(0, args.admission_max_harmed_requests),
                       admission_max_total_harm_ms=max(0.0, args.admission_max_total_harm_ms),
                       admission_max_single_harm_ms=max(0.0, args.admission_max_single_harm_ms),
                       admission_max_own_miss_ms=max(0.0, args.admission_max_own_miss_ms),
                       admission_max_bypass_count=max(0, args.admission_max_bypass_count),
                       admission_max_wait_ms=max(0.0, args.admission_max_wait_ms),
                       admission_aging_harm_ms_per_ms=max(0.0, args.admission_aging_harm_ms_per_ms),
                       admission_aging_harmed_requests_per_ms=max(0.0, args.admission_aging_harmed_requests_per_ms),
                       prompt_priority=(not args.no_prompt_priority),
                       slo_e2e_ms=args.slo_e2e_ms,
                       slo_ttft_ms=args.slo_ttft_ms,
                       gpu_queue_alpha=args.gpu_queue_alpha,
                       pim_queue_alpha=args.pim_queue_alpha,
                       active_request_alpha=args.active_request_alpha,
                       decode_token_alpha=args.decode_token_alpha,
                       capture_debug_events=bool(args.debug_events_csv))

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

    if args.debug_events_csv:
        sim.debug_events_dataframe().to_csv(args.debug_events_csv, index=False)
        print(f"Wrote debug events: {args.debug_events_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
