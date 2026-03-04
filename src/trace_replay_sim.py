from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Protocol, Tuple

import pandas as pd

from src.azure_trace import AzureTraceRequest
from src.scheduler_policy import QueueAwareFinishTimePolicy, RouteCandidate


@dataclass
class RouteServicePoint:
    route: str
    lin: int
    lout: int
    bs: int
    prefill_e2e_ms: float
    prefill_gpu_ms: float
    prefill_pim_ms: float
    decode_e2e_ms: float
    decode_gpu_ms: float
    decode_pim_ms: float
    source_file: str

    @property
    def uses_pim(self) -> bool:
        return self.prefill_pim_ms > 0 or self.decode_pim_ms > 0


@dataclass
class LookupResult:
    point: RouteServicePoint
    requested_lin: int
    requested_lout: int
    mapped_lin: int
    mapped_lout: int
    requested_bs: int
    mapped_bs: int
    clipped_lin: bool
    clipped_lout: bool


@dataclass
class RequestServiceEstimate:
    route: str
    uses_pim: bool
    lin: int
    lout: int
    generated_tokens: int
    decode_tokens: int
    prefill_e2e_ms: float
    prefill_gpu_ms: float
    prefill_pim_ms: float
    decode_e2e_ms: float
    decode_gpu_ms: float
    decode_pim_ms: float
    mapping: LookupResult


@dataclass
class ReplayRequest:
    request_id: int
    arrival_ms: float
    context_tokens: int
    generated_tokens: int
    route: Optional[str] = None
    route_decision_reason: str = ""
    state: str = "new"  # new, waiting_prefill, waiting_decode, done, dropped
    ready_ms: float = 0.0
    remaining_decode_tokens: int = 0
    decoded_tokens_done: int = 0
    deadline_ms: Optional[float] = None
    chosen_predicted_finish_ms: Optional[float] = None
    chosen_predicted_gpu_wait_ms: Optional[float] = None
    chosen_predicted_pim_wait_ms: Optional[float] = None
    chosen_queue_pressure_ms: Optional[float] = None
    chosen_slack_ms: Optional[float] = None

    # Service estimate chosen at admission.
    svc: Optional[RequestServiceEstimate] = None

    # Timing stats
    prefill_start_ms: Optional[float] = None
    prefill_end_ms: Optional[float] = None
    first_token_time_ms: Optional[float] = None
    completion_time_ms: Optional[float] = None
    last_token_completion_ms: Optional[float] = None
    tbt_intervals_ms: List[float] = field(default_factory=list)

    # Diagnostics
    dropped_reason: str = ""


@dataclass
class ReplayConfig:
    unsupported_policy: str = "nearest"  # nearest, clip, drop
    lin_bucket: int = 1
    lout_bucket: int = 1
    batch_size: int = 1
    enable_prefill_batching: bool = False
    max_prefill_batch_size: int = 1
    enable_decode_batching: bool = False
    max_decode_batch_size: int = 1
    prefill_guard_ms: Optional[float] = None
    max_consecutive_decode_batches: int = 0
    decode_batch_cap_with_prefill: int = 0
    prompt_priority: bool = True
    slo_e2e_ms: Optional[float] = None
    slo_ttft_ms: Optional[float] = None
    gpu_queue_alpha: float = 0.0
    pim_queue_alpha: float = 0.0
    active_request_alpha: float = 0.0
    decode_token_alpha: float = 0.0


@dataclass
class QueuePressureSnapshot:
    active_requests: int = 0
    pending_decode_tokens: int = 0
    pending_gpu_work_ms: float = 0.0
    pending_pim_work_ms: float = 0.0


class CostModelProtocol(Protocol):
    def routes(self) -> List[str]:
        ...

    def estimate_request(self,
                         route: str,
                         context_tokens: int,
                         generated_tokens: int,
                         bs: int,
                         unsupported_policy: str,
                         lin_bucket: int,
                         lout_bucket: int) -> Optional["RequestServiceEstimate"]:
        ...


class OutputCsvCostModel:
    """Cost model built from one or more `main.py` output CSV files.

    Each row provides:
    - `s_time`: prefill (sum stage) end-to-end latency in ms
    - `g_time (ms)`: per-token decode latency in ms (averaged over `lout-1`)

    For hybrid routes, GPU/PIM occupancy split is approximated using the stage
    breakdown fields in the CSV. This is sufficient for queue-aware routing.
    """

    REQUIRED_COLUMNS = {
        "Lin",
        "Lout",
        "bs",
        "s_time",
        "g_time (ms)",
        "g_matmul",
        "g_fc",
        "g_comm",
        "g_etc",
        "g_softmax",
        "g2g_comm",
        "c2g_comm",
        "s_matmul",
        "s_fc",
        "s_comm",
        "s_softmax",
        "s_act",
        "s_lnorm",
    }

    def __init__(self,
                 route_points: Dict[str, List[RouteServicePoint]],
                 sum_offload_to_pim: bool = False):
        self.route_points = route_points
        self.sum_offload_to_pim = sum_offload_to_pim

    @classmethod
    def from_route_csvs(cls,
                        route_to_csv: Dict[str, str],
                        sum_offload_to_pim: bool = False) -> "OutputCsvCostModel":
        route_points: Dict[str, List[RouteServicePoint]] = {}
        for route, csv_path in route_to_csv.items():
            df = pd.read_csv(csv_path)
            missing = cls.REQUIRED_COLUMNS - set(df.columns)
            if missing:
                raise ValueError(f"{csv_path}: missing columns {sorted(missing)}")

            points: List[RouteServicePoint] = []
            for _, row in df.iterrows():
                points.append(cls._row_to_point(route, row, csv_path, sum_offload_to_pim))
            if not points:
                raise ValueError(f"{csv_path}: no rows found")
            route_points[route] = points
        return cls(route_points, sum_offload_to_pim=sum_offload_to_pim)

    @staticmethod
    def _safe_float(row, key: str) -> float:
        v = row.get(key, 0.0)
        try:
            if pd.isna(v):
                return 0.0
        except Exception:
            pass
        return float(v)

    @classmethod
    def _row_to_point(cls, route: str, row, csv_path: str,
                      sum_offload_to_pim: bool) -> RouteServicePoint:
        lin = int(row["Lin"])
        lout = int(row["Lout"])
        bs = int(row["bs"])

        s_time = cls._safe_float(row, "s_time")
        g_time = cls._safe_float(row, "g_time (ms)")

        # Prefill split.
        if route == "gpu_only":
            prefill_gpu = s_time
            prefill_pim = 0.0
        elif sum_offload_to_pim:
            # Optional hook for future experiments where sum stage is also offloaded.
            prefill_pim = cls._safe_float(row, "s_matmul") + cls._safe_float(row, "s_softmax")
            prefill_gpu = (cls._safe_float(row, "s_fc") + cls._safe_float(row, "s_comm") +
                           cls._safe_float(row, "s_act") + cls._safe_float(row, "s_lnorm"))
        else:
            # Current default in this repo: sum stage runs on GPU.
            prefill_gpu = s_time
            prefill_pim = 0.0

        # Decode split.
        if route == "gpu_only":
            decode_gpu = g_time
            decode_pim = 0.0
        else:
            # Approximate occupancy split using stage-level timing breakdown.
            decode_gpu = (cls._safe_float(row, "g_fc") + cls._safe_float(row, "g_etc") +
                          cls._safe_float(row, "g2g_comm"))
            decode_pim = (cls._safe_float(row, "g_matmul") + cls._safe_float(row, "g_softmax") +
                          cls._safe_float(row, "c2g_comm"))

            # If a row lacks detailed splits, fail safe to end-to-end on GPU side.
            if decode_gpu <= 0 and decode_pim <= 0:
                decode_gpu = g_time
                decode_pim = 0.0

        prefill_e2e = max(s_time, prefill_gpu, prefill_pim)
        decode_e2e = max(g_time, decode_gpu, decode_pim)

        return RouteServicePoint(route=route,
                                 lin=lin,
                                 lout=lout,
                                 bs=bs,
                                 prefill_e2e_ms=prefill_e2e,
                                 prefill_gpu_ms=prefill_gpu,
                                 prefill_pim_ms=prefill_pim,
                                 decode_e2e_ms=decode_e2e,
                                 decode_gpu_ms=decode_gpu,
                                 decode_pim_ms=decode_pim,
                                 source_file=csv_path)

    def routes(self) -> List[str]:
        return sorted(self.route_points.keys())

    def _bucket(self, value: int, bucket: int) -> int:
        if bucket <= 1:
            return int(value)
        return max(1, int(round(value / bucket) * bucket))

    def lookup(self,
               route: str,
               lin: int,
               lout: int,
               bs: int,
               unsupported_policy: str = "nearest",
               lin_bucket: int = 1,
               lout_bucket: int = 1) -> Optional[LookupResult]:
        if route not in self.route_points:
            return None
        points = self.route_points[route]
        if not points:
            return None

        target_lin = self._bucket(int(lin), lin_bucket)
        target_lout = self._bucket(int(lout), lout_bucket)
        target_bs = int(bs)

        route_lins = [p.lin for p in points]
        route_louts = [p.lout for p in points]
        clipped_lin = False
        clipped_lout = False

        if unsupported_policy == "clip":
            min_lin, max_lin = min(route_lins), max(route_lins)
            min_lout, max_lout = min(route_louts), max(route_louts)
            new_lin = min(max(target_lin, min_lin), max_lin)
            new_lout = min(max(target_lout, min_lout), max_lout)
            clipped_lin = new_lin != target_lin
            clipped_lout = new_lout != target_lout
            target_lin, target_lout = new_lin, new_lout
        elif unsupported_policy not in ("nearest", "drop"):
            raise ValueError(f"Unsupported policy: {unsupported_policy}")

        exact = [
            p for p in points
            if p.lin == target_lin and p.lout == target_lout and p.bs == target_bs
        ]
        if exact:
            point = exact[0]
            return LookupResult(point=point,
                                requested_lin=int(lin),
                                requested_lout=int(lout),
                                mapped_lin=point.lin,
                                mapped_lout=point.lout,
                                requested_bs=target_bs,
                                mapped_bs=point.bs,
                                clipped_lin=clipped_lin,
                                clipped_lout=clipped_lout)

        if unsupported_policy == "drop":
            return None

        def _dist(p: RouteServicePoint) -> Tuple[int, int, int]:
            return (abs(p.lin - target_lin) + abs(p.lout - target_lout),
                    abs(p.bs - target_bs), p.lin + p.lout)

        point = min(points, key=_dist)
        return LookupResult(point=point,
                            requested_lin=int(lin),
                            requested_lout=int(lout),
                            mapped_lin=point.lin,
                            mapped_lout=point.lout,
                            requested_bs=target_bs,
                            mapped_bs=point.bs,
                            clipped_lin=clipped_lin or (point.lin != target_lin),
                            clipped_lout=clipped_lout or (point.lout != target_lout))

    def estimate_request(self,
                         route: str,
                         context_tokens: int,
                         generated_tokens: int,
                         bs: int,
                         unsupported_policy: str,
                         lin_bucket: int,
                         lout_bucket: int) -> Optional[RequestServiceEstimate]:
        lin = int(context_tokens)
        lout = int(generated_tokens)
        if lout <= 0:
            return None

        lookup = self.lookup(route,
                             lin=lin,
                             lout=lout,
                             bs=bs,
                             unsupported_policy=unsupported_policy,
                             lin_bucket=lin_bucket,
                             lout_bucket=lout_bucket)
        if lookup is None:
            return None

        return self._estimate_from_lookup(route=route,
                                          lin=lin,
                                          lout=lout,
                                          generated_tokens=int(generated_tokens),
                                          lookup=lookup)

    def _estimate_from_lookup(self,
                              route: str,
                              lin: int,
                              lout: int,
                              generated_tokens: int,
                              lookup: LookupResult) -> RequestServiceEstimate:
        p = lookup.point
        decode_tokens = max(0, int(generated_tokens) - 1)
        return RequestServiceEstimate(route=route,
                                      uses_pim=p.uses_pim,
                                      lin=int(lin),
                                      lout=int(lout),
                                      generated_tokens=int(generated_tokens),
                                      decode_tokens=decode_tokens,
                                      prefill_e2e_ms=p.prefill_e2e_ms,
                                      prefill_gpu_ms=p.prefill_gpu_ms,
                                      prefill_pim_ms=p.prefill_pim_ms,
                                      decode_e2e_ms=p.decode_e2e_ms,
                                      decode_gpu_ms=p.decode_gpu_ms,
                                      decode_pim_ms=p.decode_pim_ms,
                                      mapping=lookup)


@dataclass
class _TaskCandidate:
    requests: List[ReplayRequest]
    route: str
    phase: str  # prefill or decode
    ready_ms: float
    gpu_ms: float
    pim_ms: float
    e2e_ms: float
    earliest_start_ms: float
    predicted_finish_ms: float
    batch_size: int


@dataclass
class ReplaySummary:
    total_requests: int
    admitted_requests: int
    completed_requests: int
    dropped_requests: int
    makespan_ms: float
    throughput_rps: float
    throughput_tokps: float
    gpu_util: float
    pim_util: float
    route_counts: Dict[str, int]
    route_fallback_count: int
    mapping_clipped_lin_count: int
    mapping_clipped_lout_count: int
    ttft_ms: Dict[str, float]
    prefill_wait_ms: Dict[str, float]
    e2e_ms: Dict[str, float]
    tbt_ms: Dict[str, float]
    slo_ttft_miss_rate: Optional[float]
    slo_e2e_miss_rate: Optional[float]
    prefill_batching: Dict[str, float]
    decode_batching: Dict[str, float]
    local_scheduling: Dict[str, float]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def _percentiles(values: Iterable[float]) -> Dict[str, float]:
    vals = list(values)
    if not vals:
        return {"mean": math.nan, "p50": math.nan, "p95": math.nan, "p99": math.nan}
    s = pd.Series(vals, dtype=float)
    return {
        "mean": float(s.mean()),
        "p50": float(s.quantile(0.50)),
        "p95": float(s.quantile(0.95)),
        "p99": float(s.quantile(0.99)),
    }


class TraceReplaySimulator:
    def __init__(self,
                 arrivals: List[AzureTraceRequest],
                 cost_model: CostModelProtocol,
                 policy: QueueAwareFinishTimePolicy,
                 config: ReplayConfig,
                 route_order: Optional[List[str]] = None):
        self.arrivals = sorted(arrivals, key=lambda r: (r.arrival_ms, r.request_id))
        self.cost_model = cost_model
        self.policy = policy
        self.config = config
        self.route_order = route_order or cost_model.routes()

        self.gpu_free_ms = 0.0
        self.pim_free_ms = 0.0
        self.gpu_busy_ms = 0.0
        self.pim_busy_ms = 0.0
        self.now_ms = 0.0

        self.requests: List[ReplayRequest] = []
        self._arrival_idx = 0

        self.route_counts: Dict[str, int] = {}
        self.route_fallback_count = 0
        self.mapping_clipped_lin_count = 0
        self.mapping_clipped_lout_count = 0
        self.prefill_batch_sizes: List[int] = []
        self.decode_batch_sizes: List[int] = []
        self.consecutive_decode_batches = 0
        self.prefill_guard_trigger_count = 0
        self.decode_limit_trigger_count = 0

    def _resource_ready(self, gpu_ms: float, pim_ms: float) -> float:
        t = self.now_ms
        if gpu_ms > 0:
            t = max(t, self.gpu_free_ms)
        if pim_ms > 0:
            t = max(t, self.pim_free_ms)
        return t

    def _predict_stage_start(self, ready_ms: float, gpu_ms: float,
                             pim_ms: float, gpu_free_ms: float,
                             pim_free_ms: float) -> float:
        t = ready_ms
        if gpu_ms > 0:
            t = max(t, gpu_free_ms)
        if pim_ms > 0:
            t = max(t, pim_free_ms)
        return t

    def _request_decode_context_tokens(self, rr: ReplayRequest) -> int:
        # Prefill emits the first token; each decode step appends one more token.
        return int(rr.context_tokens + 1 + rr.decoded_tokens_done)

    def _predicted_lookup_batch_size(self, route: str) -> int:
        bs = int(self.config.batch_size)
        if not self.config.enable_decode_batching:
            return bs
        same_route_decode = sum(
            1 for rr in self.requests
            if rr.route == route and rr.state == "waiting_decode" and rr.svc is not None
        )
        predicted_bs = min(int(self.config.max_decode_batch_size), same_route_decode + 1)
        if self._has_waiting_prefill() and self.config.decode_batch_cap_with_prefill > 0:
            predicted_bs = min(predicted_bs, int(self.config.decode_batch_cap_with_prefill))
        return max(bs, predicted_bs)

    def _waiting_prefill_requests(self) -> List[ReplayRequest]:
        return [
            rr for rr in self.requests
            if rr.state == "waiting_prefill" and rr.svc is not None
        ]

    def _has_waiting_prefill(self) -> bool:
        return any(rr.state == "waiting_prefill" and rr.svc is not None for rr in self.requests)

    def _pick_prefill_task(self, prefill_tasks: List[_TaskCandidate]) -> Optional[_TaskCandidate]:
        if not prefill_tasks:
            return None
        return min(
            prefill_tasks,
            key=lambda tc: (tc.earliest_start_ms, tc.ready_ms, tc.requests[0].request_id),
        )

    def _queue_pressure_snapshot(self) -> QueuePressureSnapshot:
        snap = QueuePressureSnapshot()
        for rr in self.requests:
            if rr.state not in ("waiting_prefill", "waiting_decode"):
                continue
            if rr.svc is None:
                continue
            snap.active_requests += 1
            if rr.state == "waiting_prefill":
                snap.pending_gpu_work_ms += rr.svc.prefill_gpu_ms
                snap.pending_pim_work_ms += rr.svc.prefill_pim_ms
            if rr.remaining_decode_tokens > 0:
                snap.pending_decode_tokens += rr.remaining_decode_tokens
                snap.pending_gpu_work_ms += rr.remaining_decode_tokens * rr.svc.decode_gpu_ms
                snap.pending_pim_work_ms += rr.remaining_decode_tokens * rr.svc.decode_pim_ms
        return snap

    def _predict_route_candidate(self,
                                 req: AzureTraceRequest,
                                 route: str,
                                 deadline_ms: Optional[float] = None) -> Tuple[RouteCandidate, Optional[RequestServiceEstimate]]:
        est = self.cost_model.estimate_request(route,
                                               context_tokens=req.context_tokens,
                                               generated_tokens=req.generated_tokens,
                                               bs=self._predicted_lookup_batch_size(route),
                                               unsupported_policy=self.config.unsupported_policy,
                                               lin_bucket=self.config.lin_bucket,
                                               lout_bucket=self.config.lout_bucket)
        if est is None:
            return (RouteCandidate(route=route,
                                   eligible=False,
                                   uses_pim=("pim" in route),
                                   predicted_finish_ms=math.inf,
                                   predicted_gpu_wait_ms=math.inf,
                                    predicted_pim_wait_ms=math.inf,
                                   deadline_ms=deadline_ms,
                                   slack_ms=None,
                                   reason="unsupported_length"), None)

        # Aggregate admission time prediction (simple, no batching)
        prefill_start = self._predict_stage_start(req.arrival_ms, est.prefill_gpu_ms,
                                                  est.prefill_pim_ms, self.gpu_free_ms,
                                                  self.pim_free_ms)
        prefill_end = prefill_start + est.prefill_e2e_ms

        gpu_free_after_prefill = self.gpu_free_ms
        pim_free_after_prefill = self.pim_free_ms
        if est.prefill_gpu_ms > 0:
            gpu_free_after_prefill = max(gpu_free_after_prefill, prefill_start + est.prefill_gpu_ms)
        if est.prefill_pim_ms > 0:
            pim_free_after_prefill = max(pim_free_after_prefill, prefill_start + est.prefill_pim_ms)

        decode_gpu_total = est.decode_gpu_ms * est.decode_tokens
        decode_pim_total = est.decode_pim_ms * est.decode_tokens
        decode_e2e_total = est.decode_e2e_ms * est.decode_tokens
        decode_start = self._predict_stage_start(prefill_end, decode_gpu_total,
                                                 decode_pim_total, gpu_free_after_prefill,
                                                 pim_free_after_prefill)
        finish = decode_start + decode_e2e_total

        pred_gpu_wait = max(0.0, prefill_start - req.arrival_ms)
        pred_pim_wait = 0.0
        if est.uses_pim and est.decode_tokens > 0 and est.decode_pim_ms > 0:
            pred_pim_wait = max(0.0, pim_free_after_prefill - prefill_end)
        queue_pressure_ms = 0.0
        if (self.config.gpu_queue_alpha > 0 or self.config.pim_queue_alpha > 0 or
                self.config.active_request_alpha > 0 or self.config.decode_token_alpha > 0):
            snap = self._queue_pressure_snapshot()
            req_gpu_work_ms = est.prefill_gpu_ms + decode_gpu_total
            req_pim_work_ms = est.prefill_pim_ms + decode_pim_total
            req_total_ms = max(est.prefill_e2e_ms + decode_e2e_total, 1e-9)
            gpu_share = req_gpu_work_ms / req_total_ms if req_gpu_work_ms > 0 else 0.0
            pim_share = req_pim_work_ms / req_total_ms if req_pim_work_ms > 0 else 0.0
            queue_pressure_ms += self.config.gpu_queue_alpha * snap.pending_gpu_work_ms * gpu_share
            queue_pressure_ms += self.config.pim_queue_alpha * snap.pending_pim_work_ms * pim_share
            queue_pressure_ms += self.config.active_request_alpha * snap.active_requests
            queue_pressure_ms += (self.config.decode_token_alpha *
                                  snap.pending_decode_tokens *
                                  max(est.decode_e2e_ms, 0.0))
            finish += queue_pressure_ms
        slack_ms = None if deadline_ms is None else (deadline_ms - finish)

        cand = RouteCandidate(route=route,
                              eligible=True,
                              uses_pim=est.uses_pim,
                              predicted_finish_ms=finish,
                              predicted_gpu_wait_ms=pred_gpu_wait,
                              predicted_pim_wait_ms=pred_pim_wait,
                              queue_pressure_ms=queue_pressure_ms,
                              deadline_ms=deadline_ms,
                              slack_ms=slack_ms)
        return cand, est

    def _admit_arrivals_up_to_now(self):
        while self._arrival_idx < len(self.arrivals) and self.arrivals[self._arrival_idx].arrival_ms <= self.now_ms:
            src_req = self.arrivals[self._arrival_idx]
            self._arrival_idx += 1

            rr = ReplayRequest(request_id=src_req.request_id,
                               arrival_ms=src_req.arrival_ms,
                               context_tokens=src_req.context_tokens,
                               generated_tokens=src_req.generated_tokens,
                               ready_ms=src_req.arrival_ms)
            if self.config.slo_e2e_ms is not None:
                rr.deadline_ms = rr.arrival_ms + self.config.slo_e2e_ms

            candidates: List[RouteCandidate] = []
            est_by_route: Dict[str, RequestServiceEstimate] = {}
            cand_by_route: Dict[str, RouteCandidate] = {}
            for route in self.route_order:
                cand, est = self._predict_route_candidate(src_req,
                                                          route,
                                                          deadline_ms=rr.deadline_ms)
                candidates.append(cand)
                cand_by_route[route] = cand
                if est is not None:
                    est_by_route[route] = est

            decision = self.policy.choose_route(candidates,
                                               now_ms=self.now_ms,
                                               deadline_ms=rr.deadline_ms)
            if decision.dropped or decision.route is None or decision.route not in est_by_route:
                rr.state = "dropped"
                rr.dropped_reason = decision.reason
                rr.route_decision_reason = decision.reason
                self.requests.append(rr)
                continue

            rr.route = decision.route
            rr.route_decision_reason = decision.reason
            if "fallback_gpu" in decision.reason:
                self.route_fallback_count += 1
            rr.svc = est_by_route[decision.route]
            chosen_cand = cand_by_route.get(decision.route)
            if chosen_cand is not None:
                rr.chosen_predicted_finish_ms = chosen_cand.predicted_finish_ms
                rr.chosen_predicted_gpu_wait_ms = chosen_cand.predicted_gpu_wait_ms
                rr.chosen_predicted_pim_wait_ms = chosen_cand.predicted_pim_wait_ms
                rr.chosen_queue_pressure_ms = chosen_cand.queue_pressure_ms
                rr.chosen_slack_ms = chosen_cand.slack_ms
            rr.remaining_decode_tokens = rr.svc.decode_tokens
            rr.state = "waiting_prefill"
            rr.ready_ms = rr.arrival_ms

            if rr.svc.mapping.clipped_lin:
                self.mapping_clipped_lin_count += 1
            if rr.svc.mapping.clipped_lout:
                self.mapping_clipped_lout_count += 1

            self.route_counts[rr.route] = self.route_counts.get(rr.route, 0) + 1
            self.requests.append(rr)

    def _prefill_task_for_request(self, rr: ReplayRequest) -> Optional[_TaskCandidate]:
        if rr.state != "waiting_prefill" or rr.svc is None:
            return None
        gpu_ms = rr.svc.prefill_gpu_ms
        pim_ms = rr.svc.prefill_pim_ms
        e2e_ms = rr.svc.prefill_e2e_ms
        earliest_start = self._predict_stage_start(rr.ready_ms, gpu_ms, pim_ms,
                                                   self.gpu_free_ms, self.pim_free_ms)
        return _TaskCandidate(requests=[rr],
                              route=(rr.route or "unknown"),
                              phase="prefill",
                              ready_ms=rr.ready_ms,
                              gpu_ms=gpu_ms,
                              pim_ms=pim_ms,
                              e2e_ms=e2e_ms,
                              earliest_start_ms=earliest_start,
                              predicted_finish_ms=earliest_start + e2e_ms,
                              batch_size=1)

    def _estimate_prefill_batch(self,
                                route: str,
                                batch_reqs: List[ReplayRequest]) -> Optional[RequestServiceEstimate]:
        if not batch_reqs:
            return None
        batch_lin = max(req.context_tokens for req in batch_reqs)
        batch_lout = max(req.generated_tokens for req in batch_reqs)
        return self.cost_model.estimate_request(
            route,
            context_tokens=batch_lin,
            generated_tokens=batch_lout,
            bs=len(batch_reqs),
            unsupported_policy=self.config.unsupported_policy,
            lin_bucket=self.config.lin_bucket,
            lout_bucket=self.config.lout_bucket,
        )

    def _build_prefill_batch_candidates(self) -> List[_TaskCandidate]:
        grouped: Dict[str, List[ReplayRequest]] = {}
        for rr in self.requests:
            if rr.state != "waiting_prefill" or rr.svc is None or not rr.route:
                continue
            grouped.setdefault(rr.route, []).append(rr)

        cands: List[_TaskCandidate] = []
        for route, route_reqs in grouped.items():
            route_reqs.sort(key=lambda r: (r.ready_ms, r.arrival_ms, r.request_id))
            cutoff_ms = max(self.now_ms, route_reqs[0].ready_ms)
            cohort = [r for r in route_reqs if r.ready_ms <= cutoff_ms]
            if not cohort:
                continue

            batch_reqs = cohort[:min(len(cohort), int(self.config.max_prefill_batch_size))]
            batch_est = self._estimate_prefill_batch(route, batch_reqs)
            if batch_est is None:
                continue
            earliest_start = self._predict_stage_start(
                cutoff_ms,
                batch_est.prefill_gpu_ms,
                batch_est.prefill_pim_ms,
                self.gpu_free_ms,
                self.pim_free_ms,
            )
            cands.append(_TaskCandidate(requests=list(batch_reqs),
                                        route=route,
                                        phase="prefill",
                                        ready_ms=cutoff_ms,
                                        gpu_ms=batch_est.prefill_gpu_ms,
                                        pim_ms=batch_est.prefill_pim_ms,
                                        e2e_ms=batch_est.prefill_e2e_ms,
                                        earliest_start_ms=earliest_start,
                                        predicted_finish_ms=earliest_start + batch_est.prefill_e2e_ms,
                                        batch_size=len(batch_reqs)))
        return cands

    def _single_decode_task_for_request(self, rr: ReplayRequest) -> Optional[_TaskCandidate]:
        if rr.state != "waiting_decode" or rr.svc is None or rr.remaining_decode_tokens <= 0:
            return None
        gpu_ms = rr.svc.decode_gpu_ms
        pim_ms = rr.svc.decode_pim_ms
        e2e_ms = rr.svc.decode_e2e_ms
        earliest_start = self._predict_stage_start(rr.ready_ms, gpu_ms, pim_ms,
                                                   self.gpu_free_ms, self.pim_free_ms)
        return _TaskCandidate(requests=[rr],
                              route=(rr.route or "unknown"),
                              phase="decode",
                              ready_ms=rr.ready_ms,
                              gpu_ms=gpu_ms,
                              pim_ms=pim_ms,
                              e2e_ms=e2e_ms,
                              earliest_start_ms=earliest_start,
                              predicted_finish_ms=earliest_start + e2e_ms,
                              batch_size=1)

    def _build_decode_batch_candidates(self) -> List[_TaskCandidate]:
        grouped: Dict[str, List[ReplayRequest]] = {}
        for rr in self.requests:
            if rr.state != "waiting_decode" or rr.svc is None or rr.remaining_decode_tokens <= 0:
                continue
            if not rr.route:
                continue
            grouped.setdefault(rr.route, []).append(rr)

        cands: List[_TaskCandidate] = []
        has_waiting_prefill = self._has_waiting_prefill()
        for route, route_reqs in grouped.items():
            route_reqs.sort(key=lambda r: (r.ready_ms, r.arrival_ms, r.request_id))
            cutoff_ms = max(self.now_ms, route_reqs[0].ready_ms)
            cohort = [r for r in route_reqs if r.ready_ms <= cutoff_ms]
            if not cohort:
                continue

            max_batch = min(len(cohort), int(self.config.max_decode_batch_size))
            if has_waiting_prefill and self.config.decode_batch_cap_with_prefill > 0:
                max_batch = min(max_batch, int(self.config.decode_batch_cap_with_prefill))
            for batch_size in range(max_batch, 0, -1):
                batch_reqs = cohort[:batch_size]
                batch_lin = max(self._request_decode_context_tokens(r) for r in batch_reqs)
                batch_est = self.cost_model.estimate_request(
                    route,
                    context_tokens=batch_lin,
                    generated_tokens=2,
                    bs=batch_size,
                    unsupported_policy=self.config.unsupported_policy,
                    lin_bucket=self.config.lin_bucket,
                    lout_bucket=self.config.lout_bucket,
                )
                if batch_est is None:
                    continue
                earliest_start = self._predict_stage_start(
                    cutoff_ms,
                    batch_est.decode_gpu_ms,
                    batch_est.decode_pim_ms,
                    self.gpu_free_ms,
                    self.pim_free_ms,
                )
                cands.append(_TaskCandidate(requests=list(batch_reqs),
                                            route=route,
                                            phase="decode",
                                            ready_ms=cutoff_ms,
                                            gpu_ms=batch_est.decode_gpu_ms,
                                            pim_ms=batch_est.decode_pim_ms,
                                            e2e_ms=batch_est.decode_e2e_ms,
                                            earliest_start_ms=earliest_start,
                                            predicted_finish_ms=earliest_start + batch_est.decode_e2e_ms,
                                            batch_size=batch_size))
                break
        return cands

    def _choose_next_task(self) -> Optional[_TaskCandidate]:
        prefill_tasks: List[_TaskCandidate] = []
        if self.config.enable_prefill_batching:
            prefill_tasks.extend(self._build_prefill_batch_candidates())
        else:
            for rr in self.requests:
                tc = self._prefill_task_for_request(rr)
                if tc is not None:
                    prefill_tasks.append(tc)
        decode_tasks: List[_TaskCandidate] = []
        if self.config.enable_decode_batching:
            decode_tasks.extend(self._build_decode_batch_candidates())
        else:
            for rr in self.requests:
                tc = self._single_decode_task_for_request(rr)
                if tc is not None:
                    decode_tasks.append(tc)
        cands = prefill_tasks + decode_tasks
        if not cands:
            return None

        if prefill_tasks:
            if self.config.prefill_guard_ms is not None:
                max_prefill_wait_ms = max(max(0.0, self.now_ms - tc.ready_ms) for tc in prefill_tasks)
                if max_prefill_wait_ms >= float(self.config.prefill_guard_ms):
                    self.prefill_guard_trigger_count += 1
                    return self._pick_prefill_task(prefill_tasks)

            if (self.config.max_consecutive_decode_batches > 0 and
                    self.consecutive_decode_batches >= int(self.config.max_consecutive_decode_batches)):
                self.decode_limit_trigger_count += 1
                return self._pick_prefill_task(prefill_tasks)

        def _key(tc: _TaskCandidate):
            phase_prio = 0
            if self.config.prompt_priority:
                phase_prio = 0 if tc.phase == "prefill" else 1
            first_req_id = min(r.request_id for r in tc.requests)
            return (tc.earliest_start_ms, phase_prio, tc.ready_ms, first_req_id)

        return min(cands, key=_key)

    def _reserve_resources(self, start_ms: float, gpu_ms: float, pim_ms: float):
        if gpu_ms > 0:
            self.gpu_free_ms = max(self.gpu_free_ms, start_ms) + gpu_ms
            self.gpu_busy_ms += gpu_ms
        if pim_ms > 0:
            self.pim_free_ms = max(self.pim_free_ms, start_ms) + pim_ms
            self.pim_busy_ms += pim_ms

    def _complete_task(self, tc: _TaskCandidate):
        finish_ms = tc.earliest_start_ms + tc.e2e_ms

        if tc.phase == "prefill":
            self.prefill_batch_sizes.append(tc.batch_size)
            self.consecutive_decode_batches = 0
            for rr in tc.requests:
                rr.prefill_start_ms = tc.earliest_start_ms
                rr.prefill_end_ms = finish_ms
                rr.first_token_time_ms = finish_ms
                rr.last_token_completion_ms = finish_ms
                if rr.remaining_decode_tokens > 0:
                    rr.state = "waiting_decode"
                    rr.ready_ms = finish_ms
                else:
                    rr.state = "done"
                    rr.completion_time_ms = finish_ms
        else:
            self.decode_batch_sizes.append(tc.batch_size)
            self.consecutive_decode_batches += 1
            for rr in tc.requests:
                if rr.last_token_completion_ms is not None:
                    rr.tbt_intervals_ms.append(finish_ms - rr.last_token_completion_ms)
                rr.last_token_completion_ms = finish_ms
                rr.remaining_decode_tokens -= 1
                rr.decoded_tokens_done += 1
                if rr.remaining_decode_tokens <= 0:
                    rr.state = "done"
                    rr.completion_time_ms = finish_ms
                else:
                    rr.state = "waiting_decode"
                    rr.ready_ms = finish_ms

    def run(self) -> ReplaySummary:
        self.now_ms = 0.0
        if self.arrivals:
            self.now_ms = min(0.0, self.arrivals[0].arrival_ms)

        while True:
            self._admit_arrivals_up_to_now()

            task = self._choose_next_task()
            next_arrival_ms = None
            if self._arrival_idx < len(self.arrivals):
                next_arrival_ms = self.arrivals[self._arrival_idx].arrival_ms

            if task is None:
                if next_arrival_ms is None:
                    break
                self.now_ms = max(self.now_ms, next_arrival_ms)
                continue

            # Let earlier arrivals enter before scheduling a later-start task.
            if next_arrival_ms is not None and next_arrival_ms < task.earliest_start_ms:
                self.now_ms = max(self.now_ms, next_arrival_ms)
                continue

            self.now_ms = max(self.now_ms, task.earliest_start_ms)
            self._reserve_resources(task.earliest_start_ms, task.gpu_ms, task.pim_ms)
            self._complete_task(task)

        makespan_ms = 0.0
        if self.requests:
            makespan_ms = max([
                0.0,
                max((r.completion_time_ms or r.arrival_ms) for r in self.requests),
                max((r.arrival_ms for r in self.requests), default=0.0),
            ])

        completed = [r for r in self.requests if r.state == "done"]
        dropped = [r for r in self.requests if r.state == "dropped"]
        admitted = [r for r in self.requests if r.state != "dropped"]

        ttft_vals = [
            (r.first_token_time_ms - r.arrival_ms)
            for r in completed
            if r.first_token_time_ms is not None
        ]
        prefill_wait_vals = [
            (r.prefill_start_ms - r.arrival_ms)
            for r in admitted
            if r.prefill_start_ms is not None
        ]
        e2e_vals = [
            (r.completion_time_ms - r.arrival_ms)
            for r in completed
            if r.completion_time_ms is not None
        ]
        tbt_vals: List[float] = []
        for r in completed:
            tbt_vals.extend(r.tbt_intervals_ms)

        ttft_miss = None
        if self.config.slo_ttft_ms is not None and completed:
            misses = 0
            total = 0
            for r in completed:
                if r.first_token_time_ms is None:
                    continue
                total += 1
                if (r.first_token_time_ms - r.arrival_ms) > self.config.slo_ttft_ms:
                    misses += 1
            ttft_miss = (misses / total) if total > 0 else math.nan

        e2e_miss = None
        if self.config.slo_e2e_ms is not None and completed:
            misses = 0
            total = 0
            for r in completed:
                if r.completion_time_ms is None:
                    continue
                total += 1
                if (r.completion_time_ms - r.arrival_ms) > self.config.slo_e2e_ms:
                    misses += 1
            e2e_miss = (misses / total) if total > 0 else math.nan

        makespan_s = makespan_ms / 1000.0 if makespan_ms > 0 else math.nan
        total_out_tokens = sum(r.generated_tokens for r in completed)
        prefill_batching = {
            "enabled": float(self.config.enable_prefill_batching),
            "mean_batch_size": float(sum(self.prefill_batch_sizes) / len(self.prefill_batch_sizes))
            if self.prefill_batch_sizes else 0.0,
            "max_batch_size": float(max(self.prefill_batch_sizes)) if self.prefill_batch_sizes else 0.0,
            "num_prefill_steps": float(len(self.prefill_batch_sizes)),
        }
        decode_batching = {
            "enabled": float(self.config.enable_decode_batching),
            "mean_batch_size": float(sum(self.decode_batch_sizes) / len(self.decode_batch_sizes))
            if self.decode_batch_sizes else 0.0,
            "max_batch_size": float(max(self.decode_batch_sizes)) if self.decode_batch_sizes else 0.0,
            "num_decode_steps": float(len(self.decode_batch_sizes)),
        }
        local_scheduling = {
            "prefill_guard_ms": (float(self.config.prefill_guard_ms)
                                  if self.config.prefill_guard_ms is not None else math.nan),
            "max_consecutive_decode_batches": float(self.config.max_consecutive_decode_batches),
            "decode_batch_cap_with_prefill": float(self.config.decode_batch_cap_with_prefill),
            "prefill_guard_trigger_count": float(self.prefill_guard_trigger_count),
            "decode_limit_trigger_count": float(self.decode_limit_trigger_count),
        }

        return ReplaySummary(
            total_requests=len(self.requests),
            admitted_requests=len(admitted),
            completed_requests=len(completed),
            dropped_requests=len(dropped),
            makespan_ms=makespan_ms,
            throughput_rps=(len(completed) / makespan_s) if makespan_ms > 0 else math.nan,
            throughput_tokps=(total_out_tokens / makespan_s) if makespan_ms > 0 else math.nan,
            gpu_util=(self.gpu_busy_ms / makespan_ms) if makespan_ms > 0 else math.nan,
            pim_util=(self.pim_busy_ms / makespan_ms) if makespan_ms > 0 else math.nan,
            route_counts=dict(sorted(self.route_counts.items())),
            route_fallback_count=self.route_fallback_count,
            mapping_clipped_lin_count=self.mapping_clipped_lin_count,
            mapping_clipped_lout_count=self.mapping_clipped_lout_count,
            ttft_ms=_percentiles(ttft_vals),
            prefill_wait_ms=_percentiles(prefill_wait_vals),
            e2e_ms=_percentiles(e2e_vals),
            tbt_ms=_percentiles(tbt_vals),
            slo_ttft_miss_rate=ttft_miss,
            slo_e2e_miss_rate=e2e_miss,
            prefill_batching=prefill_batching,
            decode_batching=decode_batching,
            local_scheduling=local_scheduling,
        )

    def requests_dataframe(self) -> pd.DataFrame:
        rows = []
        for r in self.requests:
            ttft = None
            if r.first_token_time_ms is not None:
                ttft = r.first_token_time_ms - r.arrival_ms
            prefill_wait = None
            if r.prefill_start_ms is not None:
                prefill_wait = r.prefill_start_ms - r.arrival_ms
            e2e = None
            if r.completion_time_ms is not None:
                e2e = r.completion_time_ms - r.arrival_ms
            rows.append({
                "request_id": r.request_id,
                "arrival_ms": r.arrival_ms,
                "context_tokens": r.context_tokens,
                "generated_tokens": r.generated_tokens,
                "route": r.route,
                "state": r.state,
                "route_decision_reason": r.route_decision_reason,
                "dropped_reason": r.dropped_reason,
                "ttft_ms": ttft,
                "prefill_wait_ms": prefill_wait,
                "e2e_ms": e2e,
                "decode_tokens": (r.svc.decode_tokens if r.svc else None),
                "mean_tbt_ms": (sum(r.tbt_intervals_ms) / len(r.tbt_intervals_ms)
                                if r.tbt_intervals_ms else None),
                "tbt_count": len(r.tbt_intervals_ms),
                "mapped_lin": (r.svc.mapping.mapped_lin if r.svc else None),
                "mapped_lout": (r.svc.mapping.mapped_lout if r.svc else None),
                "clipped_lin": (r.svc.mapping.clipped_lin if r.svc else None),
                "clipped_lout": (r.svc.mapping.clipped_lout if r.svc else None),
                "deadline_ms": r.deadline_ms,
                "chosen_predicted_finish_ms": r.chosen_predicted_finish_ms,
                "chosen_predicted_gpu_wait_ms": r.chosen_predicted_gpu_wait_ms,
                "chosen_predicted_pim_wait_ms": r.chosen_predicted_pim_wait_ms,
                "chosen_queue_pressure_ms": r.chosen_queue_pressure_ms,
                "chosen_slack_ms": r.chosen_slack_ms,
                "decoded_tokens_done": r.decoded_tokens_done,
            })
        return pd.DataFrame(rows)
