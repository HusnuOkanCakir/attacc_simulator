from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

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
    prefill_energy_nj: float
    decode_e2e_ms: float
    decode_gpu_ms: float
    decode_pim_ms: float
    decode_energy_nj: float
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
    prefill_energy_nj: float
    decode_e2e_ms: float
    decode_gpu_ms: float
    decode_pim_ms: float
    decode_energy_nj: float
    mapping: LookupResult


@dataclass
class ReplayRequest:
    request_id: int
    arrival_ms: float
    context_tokens: int
    generated_tokens: int
    route: Optional[str] = None
    route_decision_reason: str = ""
    admission_route: Optional[str] = None
    state: str = "new"  # new, waiting_admission, waiting_prefill, waiting_decode, done, dropped
    ready_ms: float = 0.0
    admission_next_retry_ms: float = 0.0
    admission_time_ms: Optional[float] = None
    admission_queue_wait_ms: Optional[float] = None
    admission_attempts: int = 0
    admission_bypass_count: int = 0
    admission_last_reject_reason: str = ""
    admission_last_harmed_count: int = 0
    admission_last_total_harm_ms: float = 0.0
    admission_last_single_harm_ms: float = 0.0
    admission_last_own_miss_ms: float = 0.0
    admission_forced_best_effort: bool = False
    is_lost: bool = False
    remaining_decode_tokens: int = 0
    decoded_tokens_done: int = 0
    deadline_ms: Optional[float] = None
    chosen_predicted_finish_ms: Optional[float] = None
    latest_predicted_finish_ms: Optional[float] = None
    chosen_predicted_gpu_wait_ms: Optional[float] = None
    chosen_predicted_pim_wait_ms: Optional[float] = None
    chosen_predicted_incremental_energy_nj: Optional[float] = None
    chosen_predicted_incremental_decode_energy_nj: Optional[float] = None
    chosen_queue_pressure_ms: Optional[float] = None
    chosen_slack_ms: Optional[float] = None
    decode_bind_time_ms: Optional[float] = None
    decode_rebind_attempted: bool = False
    decode_rebound: bool = False
    decode_rebind_reason: str = ""
    decode_enqueue_seq: Optional[int] = None
    prediction_refresh_count: int = 0

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
    local_scheduling_policy: str = "priority"  # priority, fcfs_strict, prefill_priority_fcfs_decode
    fcfs_decode_prediction_mode: str = "incremental"  # shadow, legacy, incremental, heuristic_refresh
    enable_decode_rebind: bool = False
    decode_rebind_margin_ms: float = 0.0
    enable_predictive_admission: bool = False
    enable_slo_guarded_admission: bool = False
    slo_tbt_ms: Optional[float] = None
    admission_shadow_max_steps: int = 10000
    admission_retry_interval_ms: float = 0.0
    admission_max_harmed_requests: int = 0
    admission_max_total_harm_ms: float = 0.0
    admission_max_single_harm_ms: float = 0.0
    admission_max_own_miss_ms: float = 0.0
    admission_max_bypass_count: int = 0
    admission_max_wait_ms: float = 0.0
    admission_aging_harm_ms_per_ms: float = 0.0
    admission_aging_harmed_requests_per_ms: float = 0.0
    prompt_priority: bool = True
    slo_e2e_ms: Optional[float] = None
    slo_ttft_ms: Optional[float] = None
    gpu_queue_alpha: float = 0.0
    pim_queue_alpha: float = 0.0
    active_request_alpha: float = 0.0
    decode_token_alpha: float = 0.0
    capture_debug_events: bool = False


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

    def supports_decode_energy(self, route: Optional[str] = None) -> bool:
        ...

    def supports_prefill_energy(self, route: Optional[str] = None) -> bool:
        ...

    def estimate_requests_batch(self,
                                route: str,
                                requests: Sequence[Tuple[int, int, int]],
                                unsupported_policy: str,
                                lin_bucket: int,
                                lout_bucket: int) -> List[Optional["RequestServiceEstimate"]]:
        ...

    def stats(self) -> Dict[str, float]:
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
                 sum_offload_to_pim: bool = False,
                 route_has_decode_energy: Optional[Dict[str, bool]] = None,
                 route_has_prefill_energy: Optional[Dict[str, bool]] = None):
        self.route_points = route_points
        self.sum_offload_to_pim = sum_offload_to_pim
        if route_has_decode_energy is None:
            route_has_decode_energy = {
                route: any(p.decode_energy_nj > 0.0 for p in points)
                for route, points in route_points.items()
            }
        if route_has_prefill_energy is None:
            route_has_prefill_energy = {
                route: any(p.prefill_energy_nj > 0.0 for p in points)
                for route, points in route_points.items()
            }
        self.route_has_decode_energy = route_has_decode_energy
        self.route_has_prefill_energy = route_has_prefill_energy

    @classmethod
    def from_route_csvs(cls,
                        route_to_csv: Dict[str, str],
                        sum_offload_to_pim: bool = False) -> "OutputCsvCostModel":
        route_points: Dict[str, List[RouteServicePoint]] = {}
        route_has_decode_energy: Dict[str, bool] = {}
        route_has_prefill_energy: Dict[str, bool] = {}
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
            route_has_decode_energy[route] = "g_energy (nJ)" in df.columns
            route_has_prefill_energy[route] = "s_energy (nJ)" in df.columns
        return cls(route_points,
                   sum_offload_to_pim=sum_offload_to_pim,
                   route_has_decode_energy=route_has_decode_energy,
                   route_has_prefill_energy=route_has_prefill_energy)

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
        prefill_energy_nj = cls._safe_float(row, "s_energy (nJ)")
        decode_energy_nj = cls._safe_float(row, "g_energy (nJ)")

        return RouteServicePoint(route=route,
                                 lin=lin,
                                 lout=lout,
                                 bs=bs,
                                 prefill_e2e_ms=prefill_e2e,
                                 prefill_gpu_ms=prefill_gpu,
                                 prefill_pim_ms=prefill_pim,
                                 prefill_energy_nj=prefill_energy_nj,
                                 decode_e2e_ms=decode_e2e,
                                 decode_gpu_ms=decode_gpu,
                                 decode_pim_ms=decode_pim,
                                 decode_energy_nj=decode_energy_nj,
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

    def estimate_requests_batch(self,
                                route: str,
                                requests: Sequence[Tuple[int, int, int]],
                                unsupported_policy: str,
                                lin_bucket: int,
                                lout_bucket: int) -> List[Optional[RequestServiceEstimate]]:
        return [
            self.estimate_request(route,
                                  context_tokens=context_tokens,
                                  generated_tokens=generated_tokens,
                                  bs=bs,
                                  unsupported_policy=unsupported_policy,
                                  lin_bucket=lin_bucket,
                                  lout_bucket=lout_bucket)
            for context_tokens, generated_tokens, bs in requests
        ]

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
                                      prefill_energy_nj=p.prefill_energy_nj,
                                      decode_e2e_ms=p.decode_e2e_ms,
                                      decode_gpu_ms=p.decode_gpu_ms,
                                      decode_pim_ms=p.decode_pim_ms,
                                      decode_energy_nj=p.decode_energy_nj,
                                      mapping=lookup)

    def supports_decode_energy(self, route: Optional[str] = None) -> bool:
        if route is not None:
            return bool(self.route_has_decode_energy.get(route, False))
        return all(bool(self.route_has_decode_energy.get(r, False)) for r in self.route_points)

    def supports_prefill_energy(self, route: Optional[str] = None) -> bool:
        if route is not None:
            return bool(self.route_has_prefill_energy.get(route, False))
        return all(bool(self.route_has_prefill_energy.get(r, False)) for r in self.route_points)

    def stats(self) -> Dict[str, float]:
        return {}


@dataclass
class _TaskCandidate:
    requests: List[ReplayRequest]
    route: str
    phase: str  # prefill or decode
    ready_ms: float
    gpu_ms: float
    pim_ms: float
    e2e_ms: float
    prefill_energy_nj: float
    decode_energy_nj: float
    earliest_start_ms: float
    predicted_finish_ms: float
    batch_size: int


@dataclass
class _IncrementalForecastRequest:
    request_id: int
    arrival_ms: float
    context_tokens: int
    route: str
    svc: RequestServiceEstimate
    state: str
    ready_ms: float
    remaining_decode_tokens: int
    decoded_tokens_done: int
    decode_enqueue_seq: Optional[int]
    prefill_start_ms: Optional[float] = None
    prefill_end_ms: Optional[float] = None
    first_decode_start_ms: Optional[float] = None
    completion_time_ms: Optional[float] = None
    last_token_completion_ms: Optional[float] = None
    tbt_intervals_ms: List[float] = field(default_factory=list)


@dataclass
class _IncrementalForecastRun:
    requests: Dict[int, Dict[str, float]]
    total_prefill_energy_nj: float
    total_decode_energy_nj: float
    first_task_start_ms: float


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
    prefill_energy_nj: Dict[str, object]
    decode_energy_nj: Dict[str, object]
    energy_nj: Dict[str, object]
    energy_nj_per_request: float
    decode_energy_nj_per_decode_token: float
    admission_queue: Dict[str, float]
    admission: Dict[str, float]
    local_scheduling: Dict[str, object]
    cache_stats: Dict[str, float]

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
        valid_policies = {"priority", "fcfs_strict", "prefill_priority_fcfs_decode"}
        if self.config.local_scheduling_policy not in valid_policies:
            raise ValueError("ReplayConfig.local_scheduling_policy must be one of: priority, fcfs_strict, prefill_priority_fcfs_decode")
        if self.config.fcfs_decode_prediction_mode not in {"shadow", "legacy", "incremental", "heuristic_refresh"}:
            raise ValueError("ReplayConfig.fcfs_decode_prediction_mode must be one of: shadow, legacy, incremental, heuristic_refresh")
        if self.config.enable_predictive_admission and self.config.enable_slo_guarded_admission:
            raise ValueError("predictive admission and slo_guarded_admission are mutually exclusive")
        if self._is_prefill_priority_fcfs_decode():
            if self.config.enable_predictive_admission:
                raise ValueError("prefill_priority_fcfs_decode does not support predictive admission")
            if self.config.enable_prefill_batching:
                raise ValueError("prefill_priority_fcfs_decode does not support prefill batching")
            if self.config.enable_slo_guarded_admission:
                if self.config.slo_e2e_ms is None:
                    raise ValueError("slo_guarded_admission requires slo_e2e_ms")
                if self.config.slo_tbt_ms is None:
                    raise ValueError("slo_guarded_admission requires slo_tbt_ms")
                if self.config.fcfs_decode_prediction_mode != "incremental":
                    raise ValueError("slo_guarded_admission requires fcfs_decode_prediction_mode=incremental")
        elif self.config.fcfs_decode_prediction_mode != "shadow":
            raise ValueError("fcfs_decode_prediction_mode only applies to prefill_priority_fcfs_decode")
        elif self.config.enable_slo_guarded_admission:
            raise ValueError("slo_guarded_admission requires prefill_priority_fcfs_decode")
        if self._uses_latency_guarded_energy_policy():
            if not self._is_prefill_priority_fcfs_decode():
                raise ValueError("latency_guarded_energy requires prefill_priority_fcfs_decode")
            if not self._use_incremental_prediction_for_fcfs_decode():
                raise ValueError("latency_guarded_energy requires fcfs_decode_prediction_mode=incremental")
            missing_routes = [route for route in self.route_order
                              if not self.cost_model.supports_decode_energy(route)]
            if missing_routes:
                raise ValueError(
                    "latency_guarded_energy requires decode_energy_nj support for all configured routes; "
                    f"missing: {sorted(missing_routes)}")

        self.gpu_free_ms = 0.0
        self.pim_free_ms = 0.0
        self.gpu_busy_ms = 0.0
        self.pim_busy_ms = 0.0
        self.prefill_energy_total_nj = 0.0
        self.prefill_energy_by_route_nj: Dict[str, float] = {}
        self.decode_energy_total_nj = 0.0
        self.decode_energy_by_route_nj: Dict[str, float] = {}
        self.decode_token_count = 0
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
        self.fcfs_active_request_id: Optional[int] = None
        self.next_decode_enqueue_seq = 0
        self.debug_events: List[Dict[str, object]] = []
        self.admission_queue_wait_samples_ms: List[float] = []
        self.admission_reject_count = 0
        self.admission_lost_count = 0
        self.admission_shadow_timeout_count = 0
        self.admission_queue_max_len = 0
        self.admission_held_count = 0
        self.admission_admitted_from_queue_count = 0
        self.admission_bypassed_count = 0
        self.admission_best_effort_count = 0
        self.admission_blocked_request_ids: set[int] = set()
        self.admission_tbt_blocked_request_ids: set[int] = set()
        self._estimate_cache: Dict[Tuple[str, int, int, int, str, int, int], Optional[RequestServiceEstimate]] = {}
        self.estimate_cache_hits = 0
        self.estimate_cache_misses = 0
        self.estimate_prefetch_populated = 0
        self.incremental_active_forecast_cache_hits = 0
        self.incremental_active_forecast_cache_misses = 0
        self._incremental_active_forecast_cache_key: Optional[Tuple[object, ...]] = None
        self._incremental_active_forecast_cache_run: Optional[_IncrementalForecastRun] = None
        self.projection_cache_hits = 0
        self.projection_cache_misses = 0
        self._incremental_projection_cache_baseline_key: Optional[Tuple[object, ...]] = None
        self._incremental_projection_cache: Dict[Tuple[object, ...], _IncrementalForecastRun] = {}
        self._prefetch_arrival_request_estimates()

    def _inc_route_count(self, route: Optional[str], delta: int = 1):
        if not route:
            return
        new_val = self.route_counts.get(route, 0) + int(delta)
        if new_val <= 0:
            self.route_counts.pop(route, None)
        else:
            self.route_counts[route] = new_val

    def _move_route_count(self, old_route: Optional[str], new_route: Optional[str]):
        if old_route == new_route:
            return
        self._inc_route_count(old_route, -1)
        self._inc_route_count(new_route, 1)

    def _count_waiting_prefill(self) -> int:
        return sum(1 for rr in self.requests if rr.state == "waiting_prefill" and rr.svc is not None)

    def _count_waiting_admission(self) -> int:
        return sum(1 for rr in self.requests if rr.state == "waiting_admission")

    def _count_waiting_decode(self) -> int:
        return sum(1 for rr in self.requests
                   if rr.state == "waiting_decode" and rr.svc is not None and rr.remaining_decode_tokens > 0)

    def _record_debug_event(self, event: str, **payload):
        if not self.config.capture_debug_events:
            return
        row: Dict[str, object] = {
            "event_index": len(self.debug_events),
            "event": event,
            "sim_now_ms": self.now_ms,
            "gpu_free_ms": self.gpu_free_ms,
            "pim_free_ms": self.pim_free_ms,
            "waiting_admission_count": self._count_waiting_admission(),
            "waiting_prefill_count": self._count_waiting_prefill(),
            "waiting_decode_count": self._count_waiting_decode(),
            "arrival_index": self._arrival_idx,
        }
        for k, v in payload.items():
            if isinstance(v, (dict, list, tuple)):
                row[k] = json.dumps(v, sort_keys=True)
            else:
                row[k] = v
        self.debug_events.append(row)

    def _route_candidate_debug_dict(self, cand: RouteCandidate) -> Dict[str, object]:
        return {
            "route": cand.route,
            "eligible": cand.eligible,
            "uses_pim": cand.uses_pim,
            "predicted_finish_ms": cand.predicted_finish_ms,
            "predicted_gpu_wait_ms": cand.predicted_gpu_wait_ms,
            "predicted_pim_wait_ms": cand.predicted_pim_wait_ms,
            "predicted_incremental_energy_nj": cand.predicted_incremental_energy_nj,
            "predicted_incremental_decode_energy_nj": cand.predicted_incremental_decode_energy_nj,
            "queue_pressure_ms": cand.queue_pressure_ms,
            "deadline_ms": cand.deadline_ms,
            "slack_ms": cand.slack_ms,
            "candidate_mean_tbt_ms": cand.candidate_mean_tbt_ms,
            "harmed_count": cand.harmed_count,
            "total_harm_ms": cand.total_harm_ms,
            "max_single_harm_ms": cand.max_single_harm_ms,
            "own_miss_ms": cand.own_miss_ms,
            "reason": cand.reason,
        }

    def _resource_ready(self, gpu_ms: float, pim_ms: float) -> float:
        t = self.now_ms
        if gpu_ms > 0:
            t = max(t, self.gpu_free_ms)
        if pim_ms > 0:
            t = max(t, self.pim_free_ms)
        return t

    def _estimate_cache_key(self,
                            route: str,
                            context_tokens: int,
                            generated_tokens: int,
                            bs: int) -> Tuple[str, int, int, int, str, int, int]:
        return (str(route),
                int(context_tokens),
                int(generated_tokens),
                int(bs),
                self.config.unsupported_policy,
                int(self.config.lin_bucket),
                int(self.config.lout_bucket))

    def _estimate_request_cached(self,
                                 route: str,
                                 context_tokens: int,
                                 generated_tokens: int,
                                 bs: int) -> Optional[RequestServiceEstimate]:
        key = self._estimate_cache_key(route,
                                       context_tokens=context_tokens,
                                       generated_tokens=generated_tokens,
                                       bs=bs)
        if key not in self._estimate_cache:
            self.estimate_cache_misses += 1
            self._estimate_cache[key] = self.cost_model.estimate_request(
                route,
                context_tokens=int(context_tokens),
                generated_tokens=int(generated_tokens),
                bs=int(bs),
                unsupported_policy=self.config.unsupported_policy,
                lin_bucket=self.config.lin_bucket,
                lout_bucket=self.config.lout_bucket,
            )
        else:
            self.estimate_cache_hits += 1
        return self._estimate_cache[key]

    def _prefetch_request_estimates(self,
                                    requests: Iterable[object],
                                    routes: Optional[Iterable[str]] = None):
        route_list = list(routes) if routes is not None else list(self.route_order)
        if not route_list:
            return
        route_bs = {route: self._predicted_lookup_batch_size(route) for route in route_list}
        seen: set[Tuple[str, int, int, int, str, int, int]] = set()
        missing_by_route: Dict[str, List[Tuple[int, int, int]]] = {route: [] for route in route_list}
        missing_keys_by_route: Dict[str, List[Tuple[str, int, int, int, str, int, int]]] = {
            route: [] for route in route_list
        }
        for req in requests:
            context_tokens = int(getattr(req, "context_tokens"))
            generated_tokens = int(getattr(req, "generated_tokens"))
            for route in route_list:
                bs = route_bs[route]
                key = self._estimate_cache_key(route,
                                               context_tokens=context_tokens,
                                               generated_tokens=generated_tokens,
                                               bs=bs)
                if key in seen or key in self._estimate_cache:
                    continue
                seen.add(key)
                missing_by_route[route].append((context_tokens, generated_tokens, bs))
                missing_keys_by_route[route].append(key)

        for route in route_list:
            if not missing_by_route[route]:
                continue
            estimates = self.cost_model.estimate_requests_batch(
                route,
                missing_by_route[route],
                unsupported_policy=self.config.unsupported_policy,
                lin_bucket=self.config.lin_bucket,
                lout_bucket=self.config.lout_bucket,
            )
            for key, est in zip(missing_keys_by_route[route], estimates):
                self._estimate_cache[key] = est
                self.estimate_prefetch_populated += 1

    def _prefetch_arrival_request_estimates(self):
        if not self.arrivals:
            return
        if not (self._use_incremental_prediction_for_fcfs_decode() or
                self.config.enable_slo_guarded_admission or
                self._uses_latency_guarded_energy_policy()):
            return
        self._prefetch_request_estimates(self.arrivals)

    def _active_incremental_requests(self) -> List[ReplayRequest]:
        active = [
            rr for rr in self.requests
            if rr.state in ("waiting_prefill", "waiting_decode") and rr.svc is not None and rr.route is not None
        ]
        active.sort(key=lambda r: r.request_id)
        return active

    def _incremental_request_signature(self, rr: ReplayRequest) -> Tuple[object, ...]:
        if rr.svc is None or rr.route is None:
            raise ValueError("incremental forecast signature requires admitted requests")
        return (
            int(rr.request_id),
            float(rr.arrival_ms),
            int(rr.context_tokens),
            int(rr.generated_tokens),
            str(rr.route),
            str(rr.state),
            float(rr.ready_ms),
            int(rr.remaining_decode_tokens),
            int(rr.decoded_tokens_done),
            rr.decode_enqueue_seq,
            rr.prefill_start_ms,
            rr.prefill_end_ms,
            rr.first_token_time_ms,
            rr.completion_time_ms,
            rr.last_token_completion_ms,
            tuple(float(v) for v in rr.tbt_intervals_ms),
            float(rr.svc.prefill_e2e_ms),
            float(rr.svc.prefill_gpu_ms),
            float(rr.svc.prefill_pim_ms),
            float(rr.svc.prefill_energy_nj),
            float(rr.svc.decode_e2e_ms),
            float(rr.svc.decode_gpu_ms),
            float(rr.svc.decode_pim_ms),
            float(rr.svc.decode_energy_nj),
            int(rr.svc.decode_tokens),
        )

    def _incremental_active_forecast_signature(self,
                                               active: List[ReplayRequest]) -> Tuple[object, ...]:
        return (
            float(self.gpu_free_ms),
            float(self.pim_free_ms),
            int(self.next_decode_enqueue_seq),
            tuple(self._incremental_request_signature(rr) for rr in active),
        )

    def _sync_projection_cache_baseline(self,
                                        baseline_signature: Optional[Tuple[object, ...]]):
        if baseline_signature == self._incremental_projection_cache_baseline_key:
            return
        self._incremental_projection_cache_baseline_key = baseline_signature
        self._incremental_projection_cache = {}

    def _predict_stage_start(self, ready_ms: float, gpu_ms: float,
                             pim_ms: float, gpu_free_ms: float,
                             pim_free_ms: float) -> float:
        t = ready_ms
        if gpu_ms > 0:
            t = max(t, gpu_free_ms)
        if pim_ms > 0:
            t = max(t, pim_free_ms)
        return t

    def _request_decode_context_tokens(self, rr: object) -> int:
        # Prefill emits the first token; each decode step appends one more token.
        if isinstance(rr, dict):
            return int(rr["context_tokens"] + 1 + rr["decoded_tokens_done"])
        return int(rr.context_tokens + 1 + rr.decoded_tokens_done)  # type: ignore[attr-defined]

    def _predicted_lookup_batch_size(self, route: str) -> int:
        if self._is_fcfs_strict():
            return 1
        if (self._use_shadow_prediction_for_fcfs_decode() or
                self._use_incremental_prediction_for_fcfs_decode() or
                self._use_heuristic_refresh_prediction_for_fcfs_decode()):
            return 1
        bs = int(self.config.batch_size)
        if not self.config.enable_decode_batching:
            return bs
        same_route_decode = sum(
            1 for rr in self.requests
            if rr.route == route and rr.state == "waiting_decode" and rr.svc is not None
        )
        predicted_bs = min(int(self.config.max_decode_batch_size), same_route_decode + 1)
        if (not self._is_prefill_priority_fcfs_decode() and
                self._has_waiting_prefill() and
                self.config.decode_batch_cap_with_prefill > 0):
            predicted_bs = min(predicted_bs, int(self.config.decode_batch_cap_with_prefill))
        return max(bs, predicted_bs)

    def _waiting_prefill_requests(self) -> List[ReplayRequest]:
        return [
            rr for rr in self.requests
            if rr.state == "waiting_prefill" and rr.svc is not None
        ]

    def _has_waiting_prefill(self) -> bool:
        return any(rr.state == "waiting_prefill" and rr.svc is not None for rr in self.requests)

    def _is_fcfs_strict(self) -> bool:
        return self.config.local_scheduling_policy == "fcfs_strict"

    def _is_prefill_priority_fcfs_decode(self) -> bool:
        return self.config.local_scheduling_policy == "prefill_priority_fcfs_decode"

    def _uses_waiting_admission_queue(self) -> bool:
        return self.config.enable_predictive_admission or self.config.enable_slo_guarded_admission

    def _use_shadow_prediction_for_fcfs_decode(self) -> bool:
        return (self._is_prefill_priority_fcfs_decode() and
                self.config.fcfs_decode_prediction_mode == "shadow")

    def _use_incremental_prediction_for_fcfs_decode(self) -> bool:
        return (self._is_prefill_priority_fcfs_decode() and
                self.config.fcfs_decode_prediction_mode == "incremental")

    def _use_heuristic_refresh_prediction_for_fcfs_decode(self) -> bool:
        return (self._is_prefill_priority_fcfs_decode() and
                self.config.fcfs_decode_prediction_mode == "heuristic_refresh")

    def _uses_latency_guarded_energy_policy(self) -> bool:
        return getattr(self.policy, "route_policy", "") == "latency_guarded_energy"

    def _min_waiting_admission_retry_ms(self) -> Optional[float]:
        waits = [r.admission_next_retry_ms for r in self.requests if r.state == "waiting_admission"]
        if not waits:
            return None
        return min(float(v) for v in waits)

    def _queue_pressure_adjustment_ms(self,
                                      *,
                                      prefill_gpu_ms: float,
                                      prefill_pim_ms: float,
                                      prefill_e2e_ms: float,
                                      decode_gpu_ms: float,
                                      decode_pim_ms: float,
                                      decode_e2e_ms: float,
                                      decode_tokens: int) -> float:
        if (self.config.gpu_queue_alpha <= 0 and self.config.pim_queue_alpha <= 0 and
                self.config.active_request_alpha <= 0 and self.config.decode_token_alpha <= 0):
            return 0.0
        snap = self._queue_pressure_snapshot()
        decode_gpu_total = decode_gpu_ms * decode_tokens
        decode_pim_total = decode_pim_ms * decode_tokens
        decode_e2e_total = decode_e2e_ms * decode_tokens
        req_gpu_work_ms = prefill_gpu_ms + decode_gpu_total
        req_pim_work_ms = prefill_pim_ms + decode_pim_total
        req_total_ms = max(prefill_e2e_ms + decode_e2e_total, 1e-9)
        gpu_share = req_gpu_work_ms / req_total_ms if req_gpu_work_ms > 0 else 0.0
        pim_share = req_pim_work_ms / req_total_ms if req_pim_work_ms > 0 else 0.0
        queue_pressure_ms = 0.0
        queue_pressure_ms += self.config.gpu_queue_alpha * snap.pending_gpu_work_ms * gpu_share
        queue_pressure_ms += self.config.pim_queue_alpha * snap.pending_pim_work_ms * pim_share
        queue_pressure_ms += self.config.active_request_alpha * snap.active_requests
        queue_pressure_ms += (self.config.decode_token_alpha *
                              snap.pending_decode_tokens *
                              max(decode_e2e_ms, 0.0))
        return queue_pressure_ms

    def _fcfs_active_request(self) -> Optional[ReplayRequest]:
        if not self._is_fcfs_strict():
            return None

        if self.fcfs_active_request_id is not None:
            for rr in self.requests:
                if rr.request_id != self.fcfs_active_request_id:
                    continue
                if rr.state == "waiting_prefill" and rr.svc is not None:
                    return rr
                if rr.state == "waiting_decode" and rr.svc is not None and rr.remaining_decode_tokens > 0:
                    return rr
                break
            self.fcfs_active_request_id = None

        active: List[ReplayRequest] = []
        for rr in self.requests:
            if rr.svc is None:
                continue
            if rr.state == "waiting_prefill":
                active.append(rr)
            elif rr.state == "waiting_decode" and rr.remaining_decode_tokens > 0:
                active.append(rr)
        if not active:
            return None

        active.sort(
            key=lambda r: (
                (r.admission_time_ms if r.admission_time_ms is not None else r.arrival_ms),
                r.request_id,
            )
        )
        chosen = active[0]
        self.fcfs_active_request_id = chosen.request_id
        return chosen

    def _refresh_fcfs_active_request(self):
        if not self._is_fcfs_strict() or self.fcfs_active_request_id is None:
            return
        for rr in self.requests:
            if rr.request_id != self.fcfs_active_request_id:
                continue
            if rr.state == "waiting_prefill" and rr.svc is not None:
                return
            if rr.state == "waiting_decode" and rr.svc is not None and rr.remaining_decode_tokens > 0:
                return
            break
        self.fcfs_active_request_id = None

    def _pick_oldest_prefill_request(self) -> Optional[ReplayRequest]:
        prefills = [
            rr for rr in self.requests
            if rr.state == "waiting_prefill" and rr.svc is not None
        ]
        if not prefills:
            return None
        return min(prefills, key=lambda r: (r.ready_ms, r.arrival_ms, r.request_id))

    def _transition_to_waiting_decode(self, rr: ReplayRequest, ready_ms: float):
        rr.state = "waiting_decode"
        rr.ready_ms = ready_ms
        if self._is_prefill_priority_fcfs_decode() and rr.decode_enqueue_seq is None:
            rr.decode_enqueue_seq = self.next_decode_enqueue_seq
            self.next_decode_enqueue_seq += 1
            self._record_debug_event(
                "decode_enqueue",
                request_id=rr.request_id,
                route=rr.route,
                decode_enqueue_seq=rr.decode_enqueue_seq,
                ready_ms=ready_ms,
            )

    def _fcfs_decode_queue(self) -> List[ReplayRequest]:
        q = [
            rr for rr in self.requests
            if rr.state == "waiting_decode" and rr.svc is not None and rr.remaining_decode_tokens > 0
        ]
        for rr in q:
            if rr.decode_enqueue_seq is None:
                rr.decode_enqueue_seq = self.next_decode_enqueue_seq
                self.next_decode_enqueue_seq += 1
        q.sort(key=lambda r: ((r.decode_enqueue_seq if r.decode_enqueue_seq is not None else math.inf),
                              r.request_id))
        return q

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

    def _shadow_request_from_replay_request(self, rr: ReplayRequest) -> Dict[str, object]:
        return {
            "id": int(rr.request_id),
            "arrival_ms": float(rr.arrival_ms),
            "context_tokens": int(rr.context_tokens),
            "route": rr.route,
            "svc": rr.svc,
            "state": rr.state,
            "ready_ms": float(rr.ready_ms),
            "remaining_decode_tokens": int(rr.remaining_decode_tokens),
            "decoded_tokens_done": int(rr.decoded_tokens_done),
            "prefill_start_ms": rr.prefill_start_ms,
            "prefill_end_ms": rr.prefill_end_ms,
            "completion_time_ms": rr.completion_time_ms,
            "last_token_completion_ms": rr.last_token_completion_ms,
            "first_decode_start_ms": None,
            "decode_enqueue_seq": rr.decode_enqueue_seq,
        }

    def _shadow_request_for_candidate(self,
                                      req: AzureTraceRequest,
                                      route: str,
                                      est: RequestServiceEstimate) -> Dict[str, object]:
        return {
            "id": int(req.request_id),
            "arrival_ms": float(req.arrival_ms),
            "context_tokens": int(req.context_tokens),
            "route": route,
            "svc": est,
            "state": "waiting_prefill",
            "ready_ms": float(self.now_ms),
            "remaining_decode_tokens": int(est.decode_tokens),
            "decoded_tokens_done": 0,
            "prefill_start_ms": None,
            "prefill_end_ms": None,
            "completion_time_ms": None,
            "last_token_completion_ms": None,
            "first_decode_start_ms": None,
            "decode_enqueue_seq": None,
        }

    def _simulate_prefill_priority_fcfs_decode_shadow(self,
                                                      shadow: List[Dict[str, object]]) -> Dict[int, Dict[str, float]]:
        now = float(self.now_ms)
        gpu_free = float(self.gpu_free_ms)
        pim_free = float(self.pim_free_ms)
        max_decode_seq = max(
            [int(r["decode_enqueue_seq"]) for r in shadow if r.get("decode_enqueue_seq") is not None],
            default=-1,
        )
        next_decode_seq = max(self.next_decode_enqueue_seq, max_decode_seq + 1)

        def _waiting_prefills() -> List[Dict[str, object]]:
            prefills = [r for r in shadow if r["state"] == "waiting_prefill" and r["svc"] is not None]
            prefills.sort(key=lambda r: (float(r["ready_ms"]), float(r["arrival_ms"]), int(r["id"])))
            return prefills

        def _waiting_decodes() -> List[Dict[str, object]]:
            decodes = [
                r for r in shadow
                if r["state"] == "waiting_decode" and r["svc"] is not None and int(r["remaining_decode_tokens"]) > 0
            ]
            for r in decodes:
                if r.get("decode_enqueue_seq") is None:
                    nonlocal_next[0] = max(nonlocal_next[0], next_decode_seq)
                    r["decode_enqueue_seq"] = nonlocal_next[0]
                    nonlocal_next[0] += 1
            decodes.sort(key=lambda r: (int(r["decode_enqueue_seq"]), int(r["id"])))
            return decodes

        def _build_next_task() -> Optional[Dict[str, object]]:
            prefills = _waiting_prefills()
            if prefills:
                rr = prefills[0]
                svc: RequestServiceEstimate = rr["svc"]  # type: ignore[assignment]
                start = self._predict_stage_start(float(rr["ready_ms"]),
                                                  float(svc.prefill_gpu_ms),
                                                  float(svc.prefill_pim_ms),
                                                  gpu_free,
                                                  pim_free)
                return {
                    "phase": "prefill",
                    "requests": [rr],
                    "ready_ms": float(rr["ready_ms"]),
                    "gpu_ms": float(svc.prefill_gpu_ms),
                    "pim_ms": float(svc.prefill_pim_ms),
                    "e2e_ms": float(svc.prefill_e2e_ms),
                    "earliest_start_ms": start,
                }

            decodes = _waiting_decodes()
            if not decodes:
                return None

            head = decodes[0]
            route = str(head["route"])
            cutoff = max(now, float(head["ready_ms"]))
            contiguous_prefix = [head]
            if self.config.enable_decode_batching:
                for rr in decodes[1:]:
                    if len(contiguous_prefix) >= int(self.config.max_decode_batch_size):
                        break
                    if str(rr["route"]) != route:
                        break
                    if float(rr["ready_ms"]) > cutoff:
                        break
                    contiguous_prefix.append(rr)

            if not self.config.enable_decode_batching:
                contiguous_prefix = [head]

            task_reqs: List[Dict[str, object]] = []
            gpu_ms = pim_ms = e2e_ms = 0.0
            for batch_size in range(len(contiguous_prefix), 0, -1):
                task_reqs = contiguous_prefix[:batch_size]
                if batch_size == 1:
                    svc = task_reqs[0]["svc"]
                    if svc is None:
                        continue
                    svc = svc  # type: ignore[assignment]
                    gpu_ms = float(svc.decode_gpu_ms)
                    pim_ms = float(svc.decode_pim_ms)
                    e2e_ms = float(svc.decode_e2e_ms)
                    break
                batch_lin = max(self._request_decode_context_tokens(r) for r in task_reqs)
                batch_est = self._estimate_request_cached(route,
                                                          context_tokens=batch_lin,
                                                          generated_tokens=2,
                                                          bs=batch_size)
                if batch_est is None:
                    continue
                gpu_ms = float(batch_est.decode_gpu_ms)
                pim_ms = float(batch_est.decode_pim_ms)
                e2e_ms = float(batch_est.decode_e2e_ms)
                break
            if not task_reqs:
                return None

            start = self._predict_stage_start(cutoff, gpu_ms, pim_ms, gpu_free, pim_free)
            return {
                "phase": "decode",
                "requests": task_reqs,
                "ready_ms": cutoff,
                "gpu_ms": gpu_ms,
                "pim_ms": pim_ms,
                "e2e_ms": e2e_ms,
                "earliest_start_ms": start,
            }

        nonlocal_next = [next_decode_seq]
        while True:
            task = _build_next_task()
            if task is None:
                break
            start = float(task["earliest_start_ms"])
            finish = start + float(task["e2e_ms"])
            now = max(now, start)
            if float(task["gpu_ms"]) > 0:
                gpu_free = max(gpu_free, start) + float(task["gpu_ms"])
            if float(task["pim_ms"]) > 0:
                pim_free = max(pim_free, start) + float(task["pim_ms"])

            if task["phase"] == "prefill":
                rr = task["requests"][0]
                if rr["prefill_start_ms"] is None:
                    rr["prefill_start_ms"] = start
                rr["prefill_end_ms"] = finish
                rr["last_token_completion_ms"] = finish
                if int(rr["remaining_decode_tokens"]) > 0:
                    rr["state"] = "waiting_decode"
                    rr["ready_ms"] = finish
                    if rr.get("decode_enqueue_seq") is None:
                        rr["decode_enqueue_seq"] = nonlocal_next[0]
                        nonlocal_next[0] += 1
                else:
                    rr["state"] = "done"
                    rr["completion_time_ms"] = finish
            else:
                for rr in task["requests"]:
                    if rr["first_decode_start_ms"] is None:
                        rr["first_decode_start_ms"] = start
                    rr["last_token_completion_ms"] = finish
                    rr["remaining_decode_tokens"] = int(rr["remaining_decode_tokens"]) - 1
                    rr["decoded_tokens_done"] = int(rr["decoded_tokens_done"]) + 1
                    if int(rr["remaining_decode_tokens"]) <= 0:
                        rr["state"] = "done"
                        rr["completion_time_ms"] = finish
                    else:
                        rr["state"] = "waiting_decode"
                        rr["ready_ms"] = finish

        results: Dict[int, Dict[str, float]] = {}
        for rr in shadow:
            results[int(rr["id"])] = {
                "completion_time_ms": (float(rr["completion_time_ms"])
                                       if rr["completion_time_ms"] is not None else math.inf),
                "prefill_start_ms": (float(rr["prefill_start_ms"])
                                     if rr["prefill_start_ms"] is not None else math.nan),
                "prefill_end_ms": (float(rr["prefill_end_ms"])
                                   if rr["prefill_end_ms"] is not None else math.nan),
                "first_decode_start_ms": (float(rr["first_decode_start_ms"])
                                          if rr["first_decode_start_ms"] is not None else math.nan),
            }
        return results

    def _project_prefill_priority_fcfs_decode_candidate(self,
                                                        req: AzureTraceRequest,
                                                        route: str,
                                                        est: RequestServiceEstimate) -> Dict[str, float]:
        shadow = [
            self._shadow_request_from_replay_request(rr)
            for rr in self.requests
            if rr.state in ("waiting_prefill", "waiting_decode") and rr.svc is not None and rr.route is not None
        ]
        shadow.append(self._shadow_request_for_candidate(req, route, est))
        results = self._simulate_prefill_priority_fcfs_decode_shadow(shadow)
        out = results.get(req.request_id, {})
        prefill_start = out.get("prefill_start_ms", math.nan)
        prefill_end = out.get("prefill_end_ms", math.nan)
        first_decode_start = out.get("first_decode_start_ms", math.nan)
        pred_gpu_wait = 0.0 if math.isnan(prefill_start) else max(0.0, prefill_start - req.arrival_ms)
        pred_pim_wait = 0.0
        if est.uses_pim and not math.isnan(prefill_end) and not math.isnan(first_decode_start):
            pred_pim_wait = max(0.0, first_decode_start - prefill_end)
        return {
            "candidate_finish_ms": out.get("completion_time_ms", math.inf),
            "candidate_pred_gpu_wait_ms": pred_gpu_wait,
            "candidate_pred_pim_wait_ms": pred_pim_wait,
        }

    def _incremental_request_from_replay_request(self,
                                                 rr: ReplayRequest) -> _IncrementalForecastRequest:
        if rr.route is None or rr.svc is None:
            raise ValueError("incremental forecast requires admitted requests with route and svc")
        return _IncrementalForecastRequest(
            request_id=int(rr.request_id),
            arrival_ms=float(rr.arrival_ms),
            context_tokens=int(rr.context_tokens),
            route=str(rr.route),
            svc=rr.svc,
            state=str(rr.state),
            ready_ms=float(rr.ready_ms),
            remaining_decode_tokens=int(rr.remaining_decode_tokens),
            decoded_tokens_done=int(rr.decoded_tokens_done),
            decode_enqueue_seq=rr.decode_enqueue_seq,
            prefill_start_ms=rr.prefill_start_ms,
            prefill_end_ms=rr.prefill_end_ms,
            first_decode_start_ms=None,
            completion_time_ms=rr.completion_time_ms,
            last_token_completion_ms=rr.last_token_completion_ms,
            tbt_intervals_ms=list(rr.tbt_intervals_ms),
        )

    def _incremental_request_for_candidate(self,
                                           req: AzureTraceRequest,
                                           route: str,
                                           est: RequestServiceEstimate) -> _IncrementalForecastRequest:
        return _IncrementalForecastRequest(
            request_id=int(req.request_id),
            arrival_ms=float(req.arrival_ms),
            context_tokens=int(req.context_tokens),
            route=str(route),
            svc=est,
            state="waiting_prefill",
            ready_ms=float(self.now_ms),
            remaining_decode_tokens=int(est.decode_tokens),
            decoded_tokens_done=0,
            decode_enqueue_seq=None,
        )

    def _simulate_prefill_priority_fcfs_decode_incremental(
            self,
            forecast: List[_IncrementalForecastRequest]) -> _IncrementalForecastRun:
        now = float(self.now_ms)
        gpu_free = float(self.gpu_free_ms)
        pim_free = float(self.pim_free_ms)
        total_prefill_energy_nj = 0.0
        total_decode_energy_nj = 0.0
        first_task_start_ms = math.inf
        max_decode_seq = max(
            [int(r.decode_enqueue_seq) for r in forecast if r.decode_enqueue_seq is not None],
            default=-1,
        )
        next_decode_seq = max(self.next_decode_enqueue_seq, max_decode_seq + 1)

        def _waiting_prefills() -> List[_IncrementalForecastRequest]:
            prefills = [r for r in forecast if r.state == "waiting_prefill"]
            prefills.sort(key=lambda r: (r.ready_ms, r.arrival_ms, r.request_id))
            return prefills

        def _waiting_decodes() -> List[_IncrementalForecastRequest]:
            nonlocal next_decode_seq
            decodes = [
                r for r in forecast
                if r.state == "waiting_decode" and r.remaining_decode_tokens > 0
            ]
            for r in decodes:
                if r.decode_enqueue_seq is None:
                    r.decode_enqueue_seq = next_decode_seq
                    next_decode_seq += 1
            decodes.sort(key=lambda r: (r.decode_enqueue_seq if r.decode_enqueue_seq is not None else math.inf,
                                        r.request_id))
            return decodes

        while True:
            prefills = _waiting_prefills()
            if prefills:
                rr = prefills[0]
                start = self._predict_stage_start(rr.ready_ms,
                                                  rr.svc.prefill_gpu_ms,
                                                  rr.svc.prefill_pim_ms,
                                                  gpu_free,
                                                  pim_free)
                if not math.isfinite(first_task_start_ms):
                    first_task_start_ms = start
                finish = start + rr.svc.prefill_e2e_ms
                now = max(now, start)
                total_prefill_energy_nj += max(0.0, rr.svc.prefill_energy_nj)
                if rr.svc.prefill_gpu_ms > 0:
                    gpu_free = max(gpu_free, start) + rr.svc.prefill_gpu_ms
                if rr.svc.prefill_pim_ms > 0:
                    pim_free = max(pim_free, start) + rr.svc.prefill_pim_ms
                if rr.prefill_start_ms is None:
                    rr.prefill_start_ms = start
                rr.prefill_end_ms = finish
                rr.last_token_completion_ms = finish
                if rr.remaining_decode_tokens > 0:
                    rr.state = "waiting_decode"
                    rr.ready_ms = finish
                    if rr.decode_enqueue_seq is None:
                        rr.decode_enqueue_seq = next_decode_seq
                        next_decode_seq += 1
                else:
                    rr.state = "done"
                    rr.completion_time_ms = finish
                continue

            decodes = _waiting_decodes()
            if not decodes:
                break

            head = decodes[0]
            cutoff_ms = max(now, head.ready_ms)
            batch = [head]
            if self.config.enable_decode_batching:
                for rr in decodes[1:]:
                    if len(batch) >= int(self.config.max_decode_batch_size):
                        break
                    if rr.route != head.route:
                        break
                    if rr.ready_ms > cutoff_ms:
                        break
                    batch.append(rr)

            batch_reqs: List[_IncrementalForecastRequest] = []
            gpu_ms = pim_ms = e2e_ms = decode_energy_nj = 0.0
            for batch_size in range(len(batch), 0, -1):
                batch_reqs = batch[:batch_size]
                if batch_size == 1:
                    svc = batch_reqs[0].svc
                    gpu_ms = svc.decode_gpu_ms
                    pim_ms = svc.decode_pim_ms
                    e2e_ms = svc.decode_e2e_ms
                    decode_energy_nj = svc.decode_energy_nj
                    break
                batch_lin = max(self._request_decode_context_tokens(r) for r in batch_reqs)
                batch_est = self._estimate_request_cached(head.route,
                                                          context_tokens=batch_lin,
                                                          generated_tokens=2,
                                                          bs=batch_size)
                if batch_est is None:
                    continue
                gpu_ms = batch_est.decode_gpu_ms
                pim_ms = batch_est.decode_pim_ms
                e2e_ms = batch_est.decode_e2e_ms
                decode_energy_nj = batch_est.decode_energy_nj
                break
            if not batch_reqs:
                break

            start = self._predict_stage_start(cutoff_ms, gpu_ms, pim_ms, gpu_free, pim_free)
            if not math.isfinite(first_task_start_ms):
                first_task_start_ms = start
            finish = start + e2e_ms
            now = max(now, start)
            total_decode_energy_nj += max(0.0, decode_energy_nj)
            if gpu_ms > 0:
                gpu_free = max(gpu_free, start) + gpu_ms
            if pim_ms > 0:
                pim_free = max(pim_free, start) + pim_ms

            for rr in batch_reqs:
                if rr.first_decode_start_ms is None:
                    rr.first_decode_start_ms = start
                if rr.last_token_completion_ms is not None:
                    rr.tbt_intervals_ms.append(finish - rr.last_token_completion_ms)
                rr.last_token_completion_ms = finish
                rr.remaining_decode_tokens -= 1
                rr.decoded_tokens_done += 1
                if rr.remaining_decode_tokens <= 0:
                    rr.state = "done"
                    rr.completion_time_ms = finish
                else:
                    rr.state = "waiting_decode"
                    rr.ready_ms = finish

        results: Dict[int, Dict[str, float]] = {}
        for rr in forecast:
            results[rr.request_id] = {
                "completion_time_ms": rr.completion_time_ms if rr.completion_time_ms is not None else math.inf,
                "prefill_start_ms": rr.prefill_start_ms if rr.prefill_start_ms is not None else math.nan,
                "prefill_end_ms": rr.prefill_end_ms if rr.prefill_end_ms is not None else math.nan,
                "first_decode_start_ms": (rr.first_decode_start_ms
                                          if rr.first_decode_start_ms is not None else math.nan),
                "mean_tbt_ms": (float(sum(rr.tbt_intervals_ms) / len(rr.tbt_intervals_ms))
                                 if rr.tbt_intervals_ms else float(rr.svc.decode_e2e_ms)),
            }
        return _IncrementalForecastRun(requests=results,
                                       total_prefill_energy_nj=float(total_prefill_energy_nj),
                                       total_decode_energy_nj=float(total_decode_energy_nj),
                                       first_task_start_ms=float(first_task_start_ms))

    def _project_prefill_priority_fcfs_decode_incremental_candidate(
            self,
            req: AzureTraceRequest,
            route: str,
            est: RequestServiceEstimate,
            baseline_signature: Optional[Tuple[object, ...]] = None,
            baseline_total_energy_nj: float = 0.0,
            baseline_total_decode_energy_nj: float = 0.0) -> Dict[str, object]:
        cache_key: Optional[Tuple[object, ...]] = None
        run: Optional[_IncrementalForecastRun] = None
        if baseline_signature is not None:
            self._sync_projection_cache_baseline(baseline_signature)
            cache_key = (int(req.request_id),
                         float(req.arrival_ms),
                         int(req.context_tokens),
                         int(req.generated_tokens),
                         float(self.now_ms),
                         str(route))
            run = self._incremental_projection_cache.get(cache_key)
            if run is not None:
                self.projection_cache_hits += 1
            else:
                self.projection_cache_misses += 1

        if run is None:
            forecast = [
                self._incremental_request_from_replay_request(rr)
                for rr in self.requests
                if rr.state in ("waiting_prefill", "waiting_decode") and rr.svc is not None and rr.route is not None
            ]
            forecast.append(self._incremental_request_for_candidate(req, route, est))
            run = self._simulate_prefill_priority_fcfs_decode_incremental(forecast)
            if cache_key is not None:
                self._incremental_projection_cache[cache_key] = run

        out = run.requests.get(req.request_id, {})
        prefill_start = out.get("prefill_start_ms", math.nan)
        prefill_end = out.get("prefill_end_ms", math.nan)
        first_decode_start = out.get("first_decode_start_ms", math.nan)
        pred_gpu_wait = 0.0 if math.isnan(prefill_start) else max(0.0, prefill_start - req.arrival_ms)
        pred_pim_wait = 0.0
        if est.uses_pim and not math.isnan(prefill_end) and not math.isnan(first_decode_start):
            pred_pim_wait = max(0.0, first_decode_start - prefill_end)
        return {
            "run": run,
            "candidate_finish_ms": out.get("completion_time_ms", math.inf),
            "candidate_pred_gpu_wait_ms": pred_gpu_wait,
            "candidate_pred_pim_wait_ms": pred_pim_wait,
            "candidate_incremental_energy_nj": max(
                0.0,
                float(run.total_prefill_energy_nj + run.total_decode_energy_nj - baseline_total_energy_nj)),
            "candidate_incremental_decode_energy_nj": max(
                0.0, float(run.total_decode_energy_nj - baseline_total_decode_energy_nj)),
        }

    def _incremental_request_for_waiting_request(self,
                                                 rr: ReplayRequest,
                                                 route: str,
                                                 est: RequestServiceEstimate) -> _IncrementalForecastRequest:
        return _IncrementalForecastRequest(
            request_id=int(rr.request_id),
            arrival_ms=float(rr.arrival_ms),
            context_tokens=int(rr.context_tokens),
            route=str(route),
            svc=est,
            state="waiting_prefill",
            ready_ms=float(self.now_ms),
            remaining_decode_tokens=int(est.decode_tokens),
            decoded_tokens_done=0,
            decode_enqueue_seq=None,
        )

    def _incremental_active_forecast_results(self) -> _IncrementalForecastRun:
        active = self._active_incremental_requests()
        if not active:
            return _IncrementalForecastRun(requests={},
                                           total_prefill_energy_nj=0.0,
                                           total_decode_energy_nj=0.0,
                                           first_task_start_ms=math.inf)

        cache_key = self._incremental_active_forecast_signature(active)
        cached_run = self._incremental_active_forecast_cache_run
        if (cached_run is not None and
                self._incremental_active_forecast_cache_key == cache_key and
                self.now_ms <= cached_run.first_task_start_ms):
            self.incremental_active_forecast_cache_hits += 1
            return cached_run

        self.incremental_active_forecast_cache_misses += 1
        forecast = [self._incremental_request_from_replay_request(rr) for rr in active]
        run = self._simulate_prefill_priority_fcfs_decode_incremental(forecast)
        self._incremental_active_forecast_cache_key = cache_key
        self._incremental_active_forecast_cache_run = run
        return run

    def _effective_slo_guard_budgets(self, rr: ReplayRequest) -> Dict[str, float]:
        queue_wait_ms = max(0.0, self.now_ms - rr.arrival_ms)
        return {
            "max_harmed_requests": float(self.config.admission_max_harmed_requests) +
            queue_wait_ms * float(self.config.admission_aging_harmed_requests_per_ms),
            "max_total_harm_ms": float(self.config.admission_max_total_harm_ms) +
            queue_wait_ms * float(self.config.admission_aging_harm_ms_per_ms),
            "max_single_harm_ms": float(self.config.admission_max_single_harm_ms) +
            queue_wait_ms * float(self.config.admission_aging_harm_ms_per_ms),
            "max_own_miss_ms": float(self.config.admission_max_own_miss_ms),
        }

    def _evaluate_slo_guarded_route(self,
                                    rr: ReplayRequest,
                                    route: str,
                                    est: RequestServiceEstimate,
                                    baseline_signature: Optional[Tuple[object, ...]],
                                    baseline_results: Dict[int, Dict[str, float]],
                                    baseline_total_energy_nj: float,
                                    baseline_total_decode_energy_nj: float) -> RouteCandidate:
        projected = self._project_prefill_priority_fcfs_decode_incremental_candidate(
            rr,
            route,
            est,
            baseline_signature=baseline_signature,
            baseline_total_energy_nj=baseline_total_energy_nj,
            baseline_total_decode_energy_nj=baseline_total_decode_energy_nj,
        )
        projected_run = projected["run"]  # type: ignore[assignment]
        projected = projected_run.requests
        out = projected.get(rr.request_id, {})
        finish = float(out.get("completion_time_ms", math.inf))
        prefill_start = float(out.get("prefill_start_ms", math.nan))
        prefill_end = float(out.get("prefill_end_ms", math.nan))
        first_decode_start = float(out.get("first_decode_start_ms", math.nan))
        candidate_mean_tbt = float(out.get("mean_tbt_ms", est.decode_e2e_ms))
        pred_gpu_wait = 0.0 if math.isnan(prefill_start) else max(0.0, prefill_start - self.now_ms)
        pred_pim_wait = 0.0
        if est.uses_pim and not math.isnan(prefill_end) and not math.isnan(first_decode_start):
            pred_pim_wait = max(0.0, first_decode_start - prefill_end)

        own_miss_ms = 0.0
        slack_ms = None
        if rr.deadline_ms is not None:
            own_miss_ms = max(0.0, finish - rr.deadline_ms)
            slack_ms = rr.deadline_ms - finish

        harmed_count = 0
        total_harm_ms = 0.0
        max_single_harm_ms = 0.0
        for active in self.requests:
            if active.state not in ("waiting_prefill", "waiting_decode"):
                continue
            if active.svc is None or active.route is None or active.deadline_ms is None:
                continue
            if active.is_lost:
                continue
            before_finish = float(baseline_results.get(active.request_id, {}).get("completion_time_ms", math.inf))
            after_finish = float(projected.get(active.request_id, {}).get("completion_time_ms", math.inf))
            before_miss = max(0.0, before_finish - active.deadline_ms)
            after_miss = max(0.0, after_finish - active.deadline_ms)
            harm_ms = max(0.0, after_miss - before_miss)
            if harm_ms > 0.0:
                harmed_count += 1
                total_harm_ms += harm_ms
                max_single_harm_ms = max(max_single_harm_ms, harm_ms)

        budgets = self._effective_slo_guard_budgets(rr)
        own_ok = own_miss_ms <= budgets["max_own_miss_ms"]
        tbt_ok = candidate_mean_tbt <= float(self.config.slo_tbt_ms)
        harmed_ok = harmed_count <= budgets["max_harmed_requests"]
        total_harm_ok = total_harm_ms <= budgets["max_total_harm_ms"]
        single_harm_ok = max_single_harm_ms <= budgets["max_single_harm_ms"]
        eligible = own_ok and tbt_ok and harmed_ok and total_harm_ok and single_harm_ok

        if not own_ok:
            reason = "candidate_e2e_slo_miss"
        elif not tbt_ok:
            reason = "candidate_tbt_slo_miss"
        elif not harmed_ok:
            reason = "existing_harmed_count_exceeded"
        elif not total_harm_ok:
            reason = "existing_total_harm_exceeded"
        elif not single_harm_ok:
            reason = "existing_single_harm_exceeded"
        else:
            reason = ""

        return RouteCandidate(route=route,
                              eligible=eligible,
                              uses_pim=est.uses_pim,
                              predicted_finish_ms=finish,
                              predicted_gpu_wait_ms=pred_gpu_wait,
                              predicted_pim_wait_ms=pred_pim_wait,
                              predicted_incremental_energy_nj=max(
                                  0.0,
                                  float(projected_run.total_prefill_energy_nj +
                                        projected_run.total_decode_energy_nj -
                                        baseline_total_energy_nj)),
                              predicted_incremental_decode_energy_nj=max(
                                  0.0,
                                  float(projected_run.total_decode_energy_nj - baseline_total_decode_energy_nj)),
                              deadline_ms=rr.deadline_ms,
                              slack_ms=slack_ms,
                              candidate_mean_tbt_ms=candidate_mean_tbt,
                              harmed_count=harmed_count,
                              total_harm_ms=total_harm_ms,
                              max_single_harm_ms=max_single_harm_ms,
                              own_miss_ms=own_miss_ms,
                              reason=reason)

    def _heuristic_decode_candidate_from_replay_request(self,
                                                        rr: ReplayRequest,
                                                        route: str,
                                                        est: RequestServiceEstimate,
                                                        decision_time_ms: float) -> _IncrementalForecastRequest:
        return _IncrementalForecastRequest(
            request_id=int(rr.request_id),
            arrival_ms=float(rr.arrival_ms),
            context_tokens=int(rr.context_tokens),
            route=str(route),
            svc=est,
            state="waiting_decode",
            ready_ms=float(decision_time_ms),
            remaining_decode_tokens=int(rr.remaining_decode_tokens),
            decoded_tokens_done=int(rr.decoded_tokens_done),
            decode_enqueue_seq=rr.decode_enqueue_seq,
            prefill_start_ms=rr.prefill_start_ms,
            prefill_end_ms=rr.prefill_end_ms,
        )

    def _heuristic_decode_step_estimate(
            self,
            block: List[_IncrementalForecastRequest]) -> tuple[float, float, float, int]:
        if not block:
            return 0.0, 0.0, 0.0, 0

        if not self.config.enable_decode_batching:
            svc = block[0].svc
            return svc.decode_gpu_ms, svc.decode_pim_ms, svc.decode_e2e_ms, 1

        max_batch = min(int(self.config.max_decode_batch_size), len(block))
        batch_lin = max(self._request_decode_context_tokens(r) for r in block)
        for batch_size in range(max_batch, 0, -1):
            if batch_size == 1:
                svc = block[0].svc
                return svc.decode_gpu_ms, svc.decode_pim_ms, svc.decode_e2e_ms, 1
            batch_est = self._estimate_request_cached(block[0].route,
                                                      context_tokens=batch_lin,
                                                      generated_tokens=2,
                                                      bs=batch_size)
            if batch_est is None:
                continue
            return batch_est.decode_gpu_ms, batch_est.decode_pim_ms, batch_est.decode_e2e_ms, batch_size
        svc = block[0].svc
        return svc.decode_gpu_ms, svc.decode_pim_ms, svc.decode_e2e_ms, 1

    def _simulate_prefill_priority_fcfs_decode_heuristic(
            self,
            forecast: List[_IncrementalForecastRequest]) -> Dict[int, Dict[str, float]]:
        now = float(self.now_ms)
        gpu_free = float(self.gpu_free_ms)
        pim_free = float(self.pim_free_ms)
        max_decode_seq = max(
            [int(r.decode_enqueue_seq) for r in forecast if r.decode_enqueue_seq is not None],
            default=-1,
        )
        next_decode_seq = max(self.next_decode_enqueue_seq, max_decode_seq + 1)

        prefills = [r for r in forecast if r.state == "waiting_prefill"]
        prefills.sort(key=lambda r: (r.ready_ms, r.arrival_ms, r.request_id))
        for rr in prefills:
            start = self._predict_stage_start(rr.ready_ms,
                                              rr.svc.prefill_gpu_ms,
                                              rr.svc.prefill_pim_ms,
                                              gpu_free,
                                              pim_free)
            finish = start + rr.svc.prefill_e2e_ms
            now = max(now, start)
            if rr.svc.prefill_gpu_ms > 0:
                gpu_free = max(gpu_free, start) + rr.svc.prefill_gpu_ms
            if rr.svc.prefill_pim_ms > 0:
                pim_free = max(pim_free, start) + rr.svc.prefill_pim_ms
            rr.prefill_start_ms = start
            rr.prefill_end_ms = finish
            rr.last_token_completion_ms = finish
            if rr.remaining_decode_tokens > 0:
                rr.state = "waiting_decode"
                rr.ready_ms = finish
                if rr.decode_enqueue_seq is None:
                    rr.decode_enqueue_seq = next_decode_seq
                    next_decode_seq += 1
            else:
                rr.state = "done"
                rr.completion_time_ms = finish

        decodes = [
            r for r in forecast
            if r.state == "waiting_decode" and r.remaining_decode_tokens > 0
        ]
        for rr in decodes:
            if rr.decode_enqueue_seq is None:
                rr.decode_enqueue_seq = next_decode_seq
                next_decode_seq += 1
        decodes.sort(key=lambda r: (r.decode_enqueue_seq if r.decode_enqueue_seq is not None else math.inf,
                                    r.request_id))

        current_time = now
        idx = 0
        while idx < len(decodes):
            block: List[_IncrementalForecastRequest] = [decodes[idx]]
            idx += 1
            while idx < len(decodes) and decodes[idx].route == block[0].route:
                block.append(decodes[idx])
                idx += 1

            step_gpu_ms, step_pim_ms, step_e2e_ms, step_batch_size = self._heuristic_decode_step_estimate(block)
            head_ready_ms = block[0].ready_ms
            block_start_ms = self._predict_stage_start(max(current_time, head_ready_ms),
                                                       step_gpu_ms,
                                                       step_pim_ms,
                                                       gpu_free,
                                                       pim_free)

            busy_rounds = 0
            block_finish_ms = block_start_ms
            for pos, rr in enumerate(block):
                if step_batch_size <= 0:
                    excess_rounds = 0
                else:
                    excess_rounds = max(0, pos - step_batch_size + 1)
                first_decode_start_ms = max(block_start_ms + excess_rounds * step_e2e_ms, rr.ready_ms)
                rr.first_decode_start_ms = first_decode_start_ms
                finish_ms = first_decode_start_ms + rr.remaining_decode_tokens * step_e2e_ms
                rr.completion_time_ms = finish_ms
                rr.last_token_completion_ms = finish_ms
                busy_rounds = max(busy_rounds, rr.remaining_decode_tokens + excess_rounds)
                block_finish_ms = max(block_finish_ms, finish_ms)

            if step_gpu_ms > 0:
                gpu_free = max(gpu_free, block_start_ms) + busy_rounds * step_gpu_ms
            if step_pim_ms > 0:
                pim_free = max(pim_free, block_start_ms) + busy_rounds * step_pim_ms
            current_time = max(current_time, block_finish_ms)

        results: Dict[int, Dict[str, float]] = {}
        for rr in forecast:
            results[rr.request_id] = {
                "completion_time_ms": rr.completion_time_ms if rr.completion_time_ms is not None else math.inf,
                "prefill_start_ms": rr.prefill_start_ms if rr.prefill_start_ms is not None else math.nan,
                "prefill_end_ms": rr.prefill_end_ms if rr.prefill_end_ms is not None else math.nan,
                "first_decode_start_ms": (rr.first_decode_start_ms
                                          if rr.first_decode_start_ms is not None else math.nan),
            }
        return results

    def _project_prefill_priority_fcfs_decode_heuristic_candidate(
            self,
            req: AzureTraceRequest,
            route: str,
            est: RequestServiceEstimate) -> Dict[str, float]:
        forecast = [
            self._incremental_request_from_replay_request(rr)
            for rr in self.requests
            if rr.state in ("waiting_prefill", "waiting_decode") and rr.svc is not None and rr.route is not None
        ]
        forecast.append(self._incremental_request_for_candidate(req, route, est))
        results = self._simulate_prefill_priority_fcfs_decode_heuristic(forecast)
        out = results.get(req.request_id, {})
        prefill_start = float(out.get("prefill_start_ms", math.nan))
        prefill_end = float(out.get("prefill_end_ms", math.nan))
        first_decode_start = float(out.get("first_decode_start_ms", math.nan))
        finish = float(out.get("completion_time_ms", math.inf))
        pred_gpu_wait = 0.0 if math.isnan(prefill_start) else max(0.0, prefill_start - req.arrival_ms)
        pred_pim_wait = 0.0
        if est.uses_pim and not math.isnan(prefill_end) and not math.isnan(first_decode_start):
            pred_pim_wait = max(0.0, first_decode_start - prefill_end)
        finish += self._queue_pressure_adjustment_ms(prefill_gpu_ms=est.prefill_gpu_ms,
                                                     prefill_pim_ms=est.prefill_pim_ms,
                                                     prefill_e2e_ms=est.prefill_e2e_ms,
                                                     decode_gpu_ms=est.decode_gpu_ms,
                                                     decode_pim_ms=est.decode_pim_ms,
                                                     decode_e2e_ms=est.decode_e2e_ms,
                                                     decode_tokens=est.decode_tokens)
        return {
            "candidate_finish_ms": finish,
            "candidate_pred_gpu_wait_ms": pred_gpu_wait,
            "candidate_pred_pim_wait_ms": pred_pim_wait,
        }

    def _project_prefill_priority_fcfs_decode_heuristic_decode_candidate(
            self,
            rr: ReplayRequest,
            route: str,
            est: RequestServiceEstimate,
            decision_time_ms: float) -> Dict[str, float]:
        forecast = [
            self._incremental_request_from_replay_request(other)
            for other in self.requests
            if other.request_id != rr.request_id and
            other.state in ("waiting_prefill", "waiting_decode") and
            other.svc is not None and other.route is not None
        ]
        forecast.append(self._heuristic_decode_candidate_from_replay_request(rr,
                                                                             route=route,
                                                                             est=est,
                                                                             decision_time_ms=decision_time_ms))
        results = self._simulate_prefill_priority_fcfs_decode_heuristic(forecast)
        out = results.get(rr.request_id, {})
        first_decode_start = float(out.get("first_decode_start_ms", math.nan))
        finish = float(out.get("completion_time_ms", math.inf))
        decode_wait = 0.0 if math.isnan(first_decode_start) else max(0.0, first_decode_start - decision_time_ms)
        finish += self._queue_pressure_adjustment_ms(prefill_gpu_ms=0.0,
                                                     prefill_pim_ms=0.0,
                                                     prefill_e2e_ms=0.0,
                                                     decode_gpu_ms=est.decode_gpu_ms,
                                                     decode_pim_ms=est.decode_pim_ms,
                                                     decode_e2e_ms=est.decode_e2e_ms,
                                                     decode_tokens=rr.remaining_decode_tokens)
        return {
            "candidate_finish_ms": finish,
            "candidate_pred_gpu_wait_ms": decode_wait if est.decode_gpu_ms > 0 else 0.0,
            "candidate_pred_pim_wait_ms": decode_wait if est.uses_pim else 0.0,
        }

    def _refresh_active_predictions(self, cause: str):
        if not self._is_prefill_priority_fcfs_decode():
            return
        active = self._active_incremental_requests()
        if not active:
            return
        if self._use_shadow_prediction_for_fcfs_decode():
            forecast_results = self._simulate_prefill_priority_fcfs_decode_shadow(
                [self._shadow_request_from_replay_request(rr) for rr in active]
            )
        elif self._use_incremental_prediction_for_fcfs_decode():
            forecast_results = self._incremental_active_forecast_results().requests
        elif self._use_heuristic_refresh_prediction_for_fcfs_decode():
            forecast_results = self._simulate_prefill_priority_fcfs_decode_heuristic(
                [self._incremental_request_from_replay_request(rr) for rr in active]
            )
        else:
            return
        for rr in active:
            old_finish = rr.latest_predicted_finish_ms
            new_finish = forecast_results.get(rr.request_id, {}).get("completion_time_ms", math.inf)
            if math.isfinite(new_finish) and self._use_heuristic_refresh_prediction_for_fcfs_decode():
                prefill_gpu_ms = rr.svc.prefill_gpu_ms if rr.state == "waiting_prefill" else 0.0
                prefill_pim_ms = rr.svc.prefill_pim_ms if rr.state == "waiting_prefill" else 0.0
                prefill_e2e_ms = rr.svc.prefill_e2e_ms if rr.state == "waiting_prefill" else 0.0
                new_finish += self._queue_pressure_adjustment_ms(prefill_gpu_ms=prefill_gpu_ms,
                                                                 prefill_pim_ms=prefill_pim_ms,
                                                                 prefill_e2e_ms=prefill_e2e_ms,
                                                                 decode_gpu_ms=rr.svc.decode_gpu_ms,
                                                                 decode_pim_ms=rr.svc.decode_pim_ms,
                                                                 decode_e2e_ms=rr.svc.decode_e2e_ms,
                                                                 decode_tokens=rr.remaining_decode_tokens)
            rr.latest_predicted_finish_ms = new_finish if math.isfinite(new_finish) else old_finish
            rr.prediction_refresh_count += 1
            self._record_debug_event(
                "prediction_refresh",
                request_id=rr.request_id,
                cause=cause,
                old_latest_predicted_finish_ms=old_finish,
                new_latest_predicted_finish_ms=rr.latest_predicted_finish_ms,
            )

    def _predict_route_candidate(self,
                                 req: AzureTraceRequest,
                                 route: str,
                                 deadline_ms: Optional[float] = None,
                                 baseline_total_energy_nj: Optional[float] = None,
                                 baseline_total_decode_energy_nj: Optional[float] = None) -> Tuple[RouteCandidate, Optional[RequestServiceEstimate]]:
        est = self._estimate_request_cached(route,
                                            context_tokens=req.context_tokens,
                                            generated_tokens=req.generated_tokens,
                                            bs=self._predicted_lookup_batch_size(route))
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

        if self._use_shadow_prediction_for_fcfs_decode():
            proj = self._project_prefill_priority_fcfs_decode_candidate(req, route, est)
            finish = float(proj["candidate_finish_ms"])
            pred_gpu_wait = float(proj["candidate_pred_gpu_wait_ms"])
            pred_pim_wait = float(proj["candidate_pred_pim_wait_ms"])
            slack_ms = None if deadline_ms is None else (deadline_ms - finish)
            cand = RouteCandidate(route=route,
                                  eligible=True,
                                  uses_pim=est.uses_pim,
                                  predicted_finish_ms=finish,
                                  predicted_gpu_wait_ms=pred_gpu_wait,
                                  predicted_pim_wait_ms=pred_pim_wait,
                                  predicted_incremental_energy_nj=max(
                                      0.0,
                                      est.prefill_energy_nj +
                                      est.decode_energy_nj * est.decode_tokens),
                                  deadline_ms=deadline_ms,
                                  slack_ms=slack_ms)
            return cand, est
        if self._use_incremental_prediction_for_fcfs_decode():
            if baseline_total_energy_nj is None or baseline_total_decode_energy_nj is None:
                baseline_total_energy_nj = 0.0 if baseline_total_energy_nj is None else baseline_total_energy_nj
                baseline_total_decode_energy_nj = 0.0 if baseline_total_decode_energy_nj is None else baseline_total_decode_energy_nj
                if self._uses_latency_guarded_energy_policy():
                    baseline_run = self._incremental_active_forecast_results()
                    baseline_total_energy_nj = (
                        baseline_run.total_prefill_energy_nj + baseline_run.total_decode_energy_nj)
                    baseline_total_decode_energy_nj = baseline_run.total_decode_energy_nj
            proj = self._project_prefill_priority_fcfs_decode_incremental_candidate(
                req,
                route,
                est,
                baseline_total_energy_nj=float(baseline_total_energy_nj),
                baseline_total_decode_energy_nj=float(baseline_total_decode_energy_nj),
            )
            finish = float(proj["candidate_finish_ms"])
            pred_gpu_wait = float(proj["candidate_pred_gpu_wait_ms"])
            pred_pim_wait = float(proj["candidate_pred_pim_wait_ms"])
            slack_ms = None if deadline_ms is None else (deadline_ms - finish)
            cand = RouteCandidate(route=route,
                                  eligible=True,
                                  uses_pim=est.uses_pim,
                                  predicted_finish_ms=finish,
                                  predicted_gpu_wait_ms=pred_gpu_wait,
                                  predicted_pim_wait_ms=pred_pim_wait,
                                  predicted_incremental_energy_nj=float(
                                      proj["candidate_incremental_energy_nj"]),
                                  predicted_incremental_decode_energy_nj=float(
                                      proj["candidate_incremental_decode_energy_nj"]),
                                  deadline_ms=deadline_ms,
                                  slack_ms=slack_ms)
            return cand, est
        if self._use_heuristic_refresh_prediction_for_fcfs_decode():
            proj = self._project_prefill_priority_fcfs_decode_heuristic_candidate(req, route, est)
            finish = float(proj["candidate_finish_ms"])
            pred_gpu_wait = float(proj["candidate_pred_gpu_wait_ms"])
            pred_pim_wait = float(proj["candidate_pred_pim_wait_ms"])
            slack_ms = None if deadline_ms is None else (deadline_ms - finish)
            cand = RouteCandidate(route=route,
                                  eligible=True,
                                  uses_pim=est.uses_pim,
                                  predicted_finish_ms=finish,
                                  predicted_gpu_wait_ms=pred_gpu_wait,
                                  predicted_pim_wait_ms=pred_pim_wait,
                                  deadline_ms=deadline_ms,
                                  slack_ms=slack_ms)
            return cand, est

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
        queue_pressure_ms = self._queue_pressure_adjustment_ms(prefill_gpu_ms=est.prefill_gpu_ms,
                                                               prefill_pim_ms=est.prefill_pim_ms,
                                                               prefill_e2e_ms=est.prefill_e2e_ms,
                                                               decode_gpu_ms=est.decode_gpu_ms,
                                                               decode_pim_ms=est.decode_pim_ms,
                                                               decode_e2e_ms=est.decode_e2e_ms,
                                                               decode_tokens=est.decode_tokens)
        finish += queue_pressure_ms
        slack_ms = None if deadline_ms is None else (deadline_ms - finish)

        cand = RouteCandidate(route=route,
                              eligible=True,
                              uses_pim=est.uses_pim,
                              predicted_finish_ms=finish,
                              predicted_gpu_wait_ms=pred_gpu_wait,
                              predicted_pim_wait_ms=pred_pim_wait,
                              predicted_incremental_energy_nj=max(
                                  0.0,
                                  est.prefill_energy_nj +
                                  est.decode_energy_nj * est.decode_tokens),
                              queue_pressure_ms=queue_pressure_ms,
                              deadline_ms=deadline_ms,
                              slack_ms=slack_ms)
        return cand, est

    def _predict_decode_route_candidate(self,
                                        rr: ReplayRequest,
                                        route: str,
                                        decision_time_ms: float) -> Tuple[RouteCandidate, Optional[RequestServiceEstimate]]:
        est = self._estimate_request_cached(route,
                                            context_tokens=rr.context_tokens,
                                            generated_tokens=rr.generated_tokens,
                                            bs=self._predicted_lookup_batch_size(route))
        if est is None:
            return (RouteCandidate(route=route,
                                   eligible=False,
                                   uses_pim=("pim" in route),
                                   predicted_finish_ms=math.inf,
                                   predicted_gpu_wait_ms=math.inf,
                                   predicted_pim_wait_ms=math.inf,
                                   deadline_ms=rr.deadline_ms,
                                   slack_ms=None,
                                   reason="unsupported_length"), None)

        if self._use_heuristic_refresh_prediction_for_fcfs_decode():
            proj = self._project_prefill_priority_fcfs_decode_heuristic_decode_candidate(rr,
                                                                                         route=route,
                                                                                         est=est,
                                                                                         decision_time_ms=decision_time_ms)
            finish = float(proj["candidate_finish_ms"])
            pred_gpu_wait = float(proj["candidate_pred_gpu_wait_ms"])
            pred_pim_wait = float(proj["candidate_pred_pim_wait_ms"])
            slack_ms = None if rr.deadline_ms is None else (rr.deadline_ms - finish)
            cand = RouteCandidate(route=route,
                                  eligible=True,
                                  uses_pim=est.uses_pim,
                                  predicted_finish_ms=finish,
                                  predicted_gpu_wait_ms=pred_gpu_wait,
                                  predicted_pim_wait_ms=pred_pim_wait,
                                  predicted_incremental_energy_nj=max(
                                      0.0,
                                      est.prefill_energy_nj +
                                      est.decode_energy_nj * rr.remaining_decode_tokens),
                                  predicted_incremental_decode_energy_nj=max(
                                      0.0, est.decode_energy_nj * rr.remaining_decode_tokens),
                                  deadline_ms=rr.deadline_ms,
                                  slack_ms=slack_ms)
            return cand, est

        decode_gpu_total = est.decode_gpu_ms * rr.remaining_decode_tokens
        decode_pim_total = est.decode_pim_ms * rr.remaining_decode_tokens
        decode_e2e_total = est.decode_e2e_ms * rr.remaining_decode_tokens
        decode_start = self._predict_stage_start(decision_time_ms,
                                                 decode_gpu_total,
                                                 decode_pim_total,
                                                 self.gpu_free_ms,
                                                 self.pim_free_ms)
        finish = decode_start + decode_e2e_total
        pred_gpu_wait = max(0.0, self.gpu_free_ms - decision_time_ms) if decode_gpu_total > 0 else 0.0
        pred_pim_wait = max(0.0, self.pim_free_ms - decision_time_ms) if decode_pim_total > 0 else 0.0
        slack_ms = None if rr.deadline_ms is None else (rr.deadline_ms - finish)
        cand = RouteCandidate(route=route,
                              eligible=True,
                              uses_pim=est.uses_pim,
                              predicted_finish_ms=finish,
                              predicted_gpu_wait_ms=pred_gpu_wait,
                              predicted_pim_wait_ms=pred_pim_wait,
                              predicted_incremental_energy_nj=max(
                                  0.0,
                                  est.prefill_energy_nj +
                                  est.decode_energy_nj * rr.remaining_decode_tokens),
                              predicted_incremental_decode_energy_nj=max(
                                  0.0, est.decode_energy_nj * rr.remaining_decode_tokens),
                              deadline_ms=rr.deadline_ms,
                              slack_ms=slack_ms)
        return cand, est

    def _maybe_rebind_decode_route(self, rr: ReplayRequest, decision_time_ms: float):
        if not self.config.enable_decode_rebind:
            return
        if rr.svc is None or rr.route is None or rr.remaining_decode_tokens <= 0:
            return
        if rr.decoded_tokens_done > 0:
            return
        if not rr.svc.uses_pim:
            rr.decode_bind_time_ms = decision_time_ms
            rr.decode_rebind_reason = "skip_non_pim_route"
            return

        rr.decode_bind_time_ms = decision_time_ms
        rr.decode_rebind_attempted = True

        current_cand, current_est = self._predict_decode_route_candidate(rr, rr.route, decision_time_ms)
        if current_est is None:
            rr.decode_rebind_reason = "current_route_estimate_unavailable"
            return

        candidates = [current_cand]
        est_by_route: Dict[str, RequestServiceEstimate] = {rr.route: current_est}
        if current_est.uses_pim:
            gpu_cand, gpu_est = self._predict_decode_route_candidate(rr, "gpu_only", decision_time_ms)
            if gpu_est is not None:
                candidates.append(gpu_cand)
                est_by_route["gpu_only"] = gpu_est

        if len(candidates) == 1:
            rr.latest_predicted_finish_ms = current_cand.predicted_finish_ms
            rr.decode_rebind_reason = "no_gpu_candidate"
            self._record_debug_event(
                "decode_rebind_decision",
                request_id=rr.request_id,
                decision_time_ms=decision_time_ms,
                previous_route=rr.route,
                chosen_route=rr.route,
                rebound=False,
                reason=rr.decode_rebind_reason,
                improvement_ms=0.0,
                route_candidates=[self._route_candidate_debug_dict(current_cand)],
            )
            return

        decision = self.policy.choose_route(candidates,
                                            now_ms=decision_time_ms,
                                            deadline_ms=rr.deadline_ms)
        chosen_cand = next((c for c in candidates if c.route == decision.route), current_cand)
        rr.latest_predicted_finish_ms = chosen_cand.predicted_finish_ms

        improvement_ms = current_cand.predicted_finish_ms - chosen_cand.predicted_finish_ms
        old_route = rr.route
        margin = max(0.0, float(self.config.decode_rebind_margin_ms))
        if (decision.route == "gpu_only" and
                old_route != "gpu_only" and
                improvement_ms > margin and
                "gpu_only" in est_by_route):
            rr.route = "gpu_only"
            rr.svc = est_by_route["gpu_only"]
            rr.decode_rebound = True
            rr.decode_rebind_reason = f"rebound_to_gpu:{decision.reason}"
            self._move_route_count(old_route, rr.route)
        else:
            rr.decode_rebind_reason = f"keep_route:{decision.reason}"

        cand_debug = [self._route_candidate_debug_dict(c) for c in candidates]
        self._record_debug_event(
            "decode_rebind_decision",
            request_id=rr.request_id,
            decision_time_ms=decision_time_ms,
            previous_route=old_route,
            chosen_route=rr.route,
            rebound=rr.decode_rebound,
            reason=rr.decode_rebind_reason,
            improvement_ms=improvement_ms,
            route_candidates=cand_debug,
        )

    def _enqueue_arrivals_up_to_now(self):
        while self._arrival_idx < len(self.arrivals) and self.arrivals[self._arrival_idx].arrival_ms <= self.now_ms:
            src_req = self.arrivals[self._arrival_idx]
            self._arrival_idx += 1
            rr = ReplayRequest(request_id=src_req.request_id,
                               arrival_ms=src_req.arrival_ms,
                               context_tokens=src_req.context_tokens,
                               generated_tokens=src_req.generated_tokens,
                               state="waiting_admission",
                               ready_ms=src_req.arrival_ms,
                               admission_next_retry_ms=self.now_ms)
            if self.config.slo_e2e_ms is not None:
                rr.deadline_ms = rr.arrival_ms + self.config.slo_e2e_ms
            self.requests.append(rr)
            self._record_debug_event(
                "arrival_enqueued",
                request_id=rr.request_id,
                request_arrival_ms=rr.arrival_ms,
                context_tokens=rr.context_tokens,
                generated_tokens=rr.generated_tokens,
            )

    def _waiting_admission_requests(self) -> List[ReplayRequest]:
        return sorted(
            [r for r in self.requests if r.state == "waiting_admission"],
            key=lambda r: (r.arrival_ms, r.request_id),
        )

    def _project_admission_candidate(self,
                                     req: ReplayRequest,
                                     route: str,
                                     est: RequestServiceEstimate) -> Dict[str, object]:
        # Build immutable snapshot of currently admitted active requests + candidate.
        shadow: List[Dict[str, object]] = []
        for rr in self.requests:
            if rr.state not in ("waiting_prefill", "waiting_decode"):
                continue
            if rr.svc is None or rr.route is None:
                continue
            shadow.append({
                "id": int(rr.request_id),
                "arrival_ms": float(rr.arrival_ms),
                "context_tokens": int(rr.context_tokens),
                "deadline_ms": rr.deadline_ms,
                "is_lost": bool(rr.is_lost),
                "route": rr.route,
                "svc": rr.svc,
                "state": rr.state,
                "ready_ms": float(rr.ready_ms),
                "admission_time_ms": (float(rr.admission_time_ms)
                                       if rr.admission_time_ms is not None else float(rr.arrival_ms)),
                "remaining_decode_tokens": int(rr.remaining_decode_tokens),
                "decoded_tokens_done": int(rr.decoded_tokens_done),
                "prefill_start_ms": rr.prefill_start_ms,
                "prefill_end_ms": rr.prefill_end_ms,
                "first_token_time_ms": rr.first_token_time_ms,
                "completion_time_ms": rr.completion_time_ms,
                "last_token_completion_ms": rr.last_token_completion_ms,
                "tbt_intervals_ms": list(rr.tbt_intervals_ms),
                "first_decode_start_ms": None,
            })
        shadow.append({
            "id": int(req.request_id),
            "arrival_ms": float(req.arrival_ms),
            "context_tokens": int(req.context_tokens),
            "deadline_ms": req.deadline_ms,
            "is_lost": bool(req.is_lost),
            "route": route,
            "svc": est,
            "state": "waiting_prefill",
            "ready_ms": float(self.now_ms),
            "admission_time_ms": float(self.now_ms),
            "remaining_decode_tokens": int(est.decode_tokens),
            "decoded_tokens_done": 0,
            "prefill_start_ms": None,
            "prefill_end_ms": None,
            "first_token_time_ms": None,
            "completion_time_ms": None,
            "last_token_completion_ms": None,
            "tbt_intervals_ms": [],
            "first_decode_start_ms": None,
        })
        target_id = int(req.request_id)

        now = float(self.now_ms)
        gpu_free = float(self.gpu_free_ms)
        pim_free = float(self.pim_free_ms)
        steps = 0
        consecutive_decode_batches = 0
        timeout = False
        shadow_fcfs_active_request_id: Optional[int] = None

        def _has_waiting_prefill() -> bool:
            for r in shadow:
                if r["state"] == "waiting_prefill":
                    return True
            return False

        def _shadow_is_active(r: Dict[str, object]) -> bool:
            if r["state"] == "waiting_prefill":
                return True
            if r["state"] == "waiting_decode" and int(r["remaining_decode_tokens"]) > 0:
                return True
            return False

        def _shadow_fcfs_active() -> Optional[Dict[str, object]]:
            nonlocal shadow_fcfs_active_request_id
            if shadow_fcfs_active_request_id is not None:
                for r in shadow:
                    if int(r["id"]) == shadow_fcfs_active_request_id and _shadow_is_active(r):
                        return r
                shadow_fcfs_active_request_id = None

            active = [r for r in shadow if _shadow_is_active(r)]
            if not active:
                return None
            active.sort(
                key=lambda r: (
                    float(r["admission_time_ms"]) if r.get("admission_time_ms") is not None else float(r["arrival_ms"]),
                    int(r["id"]),
                )
            )
            chosen = active[0]
            shadow_fcfs_active_request_id = int(chosen["id"])
            return chosen

        def _pick_prefill_task(prefill_tasks: List[Dict[str, object]]) -> Optional[Dict[str, object]]:
            if not prefill_tasks:
                return None
            return min(prefill_tasks,
                       key=lambda t: (float(t["earliest_start_ms"]), float(t["ready_ms"]), int(t["first_req_id"])))

        def _build_prefill_tasks() -> List[Dict[str, object]]:
            tasks: List[Dict[str, object]] = []
            for idx, r in enumerate(shadow):
                if r["state"] != "waiting_prefill":
                    continue
                svc: RequestServiceEstimate = r["svc"]  # type: ignore
                gpu_ms = float(svc.prefill_gpu_ms)
                pim_ms = float(svc.prefill_pim_ms)
                e2e_ms = float(svc.prefill_e2e_ms)
                ready_ms = float(r["ready_ms"])
                start = self._predict_stage_start(ready_ms, gpu_ms, pim_ms, gpu_free, pim_free)
                tasks.append({
                    "phase": "prefill",
                    "idxs": [idx],
                    "route": r["route"],
                    "ready_ms": ready_ms,
                    "gpu_ms": gpu_ms,
                    "pim_ms": pim_ms,
                    "e2e_ms": e2e_ms,
                    "earliest_start_ms": start,
                    "batch_size": 1,
                    "first_req_id": int(r["id"]),
                })
            return tasks

        def _build_decode_tasks() -> List[Dict[str, object]]:
            tasks: List[Dict[str, object]] = []
            if self.config.enable_decode_batching:
                grouped: Dict[str, List[int]] = {}
                for idx, r in enumerate(shadow):
                    if r["state"] != "waiting_decode":
                        continue
                    if int(r["remaining_decode_tokens"]) <= 0:
                        continue
                    grouped.setdefault(str(r["route"]), []).append(idx)
                has_prefill = _has_waiting_prefill()
                for rname, idxs in grouped.items():
                    idxs.sort(key=lambda i: (float(shadow[i]["ready_ms"]),
                                             float(shadow[i]["arrival_ms"]),
                                             int(shadow[i]["id"])))
                    cutoff = max(now, float(shadow[idxs[0]]["ready_ms"]))
                    cohort = [i for i in idxs if float(shadow[i]["ready_ms"]) <= cutoff]
                    if not cohort:
                        continue
                    max_batch = min(len(cohort), int(self.config.max_decode_batch_size))
                    if has_prefill and self.config.decode_batch_cap_with_prefill > 0:
                        max_batch = min(max_batch, int(self.config.decode_batch_cap_with_prefill))
                    for bs in range(max_batch, 0, -1):
                        bidxs = cohort[:bs]
                        batch_lin = max(self._request_decode_context_tokens(shadow[i]) for i in bidxs)  # type: ignore[arg-type]
                        batch_est = self._estimate_request_cached(rname,
                                                                  context_tokens=batch_lin,
                                                                  generated_tokens=2,
                                                                  bs=bs)
                        if batch_est is None:
                            continue
                        gpu_ms = float(batch_est.decode_gpu_ms)
                        pim_ms = float(batch_est.decode_pim_ms)
                        e2e_ms = float(batch_est.decode_e2e_ms)
                        start = self._predict_stage_start(cutoff, gpu_ms, pim_ms, gpu_free, pim_free)
                        tasks.append({
                            "phase": "decode",
                            "idxs": bidxs,
                            "route": rname,
                            "ready_ms": cutoff,
                            "gpu_ms": gpu_ms,
                            "pim_ms": pim_ms,
                            "e2e_ms": e2e_ms,
                            "earliest_start_ms": start,
                            "batch_size": bs,
                            "first_req_id": min(int(shadow[i]["id"]) for i in bidxs),
                        })
                        break
            else:
                for idx, r in enumerate(shadow):
                    if r["state"] != "waiting_decode":
                        continue
                    if int(r["remaining_decode_tokens"]) <= 0:
                        continue
                    svc: RequestServiceEstimate = r["svc"]  # type: ignore
                    gpu_ms = float(svc.decode_gpu_ms)
                    pim_ms = float(svc.decode_pim_ms)
                    e2e_ms = float(svc.decode_e2e_ms)
                    ready_ms = float(r["ready_ms"])
                    start = self._predict_stage_start(ready_ms, gpu_ms, pim_ms, gpu_free, pim_free)
                    tasks.append({
                        "phase": "decode",
                        "idxs": [idx],
                        "route": r["route"],
                        "ready_ms": ready_ms,
                        "gpu_ms": gpu_ms,
                        "pim_ms": pim_ms,
                        "e2e_ms": e2e_ms,
                        "earliest_start_ms": start,
                        "batch_size": 1,
                        "first_req_id": int(r["id"]),
                    })
            return tasks

        def _choose_task() -> Optional[Dict[str, object]]:
            nonlocal consecutive_decode_batches
            if self._is_fcfs_strict():
                active = _shadow_fcfs_active()
                if active is None:
                    return None
                idx = next(i for i, r in enumerate(shadow) if int(r["id"]) == int(active["id"]))
                svc: RequestServiceEstimate = active["svc"]  # type: ignore
                if active["state"] == "waiting_prefill":
                    gpu_ms = float(svc.prefill_gpu_ms)
                    pim_ms = float(svc.prefill_pim_ms)
                    e2e_ms = float(svc.prefill_e2e_ms)
                else:
                    gpu_ms = float(svc.decode_gpu_ms)
                    pim_ms = float(svc.decode_pim_ms)
                    e2e_ms = float(svc.decode_e2e_ms)
                ready_ms = float(active["ready_ms"])
                start = self._predict_stage_start(ready_ms, gpu_ms, pim_ms, gpu_free, pim_free)
                return {
                    "phase": ("prefill" if active["state"] == "waiting_prefill" else "decode"),
                    "idxs": [idx],
                    "route": active["route"],
                    "ready_ms": ready_ms,
                    "gpu_ms": gpu_ms,
                    "pim_ms": pim_ms,
                    "e2e_ms": e2e_ms,
                    "earliest_start_ms": start,
                    "batch_size": 1,
                    "first_req_id": int(active["id"]),
                }

            prefill_tasks = _build_prefill_tasks()
            decode_tasks = _build_decode_tasks()
            cands = prefill_tasks + decode_tasks
            if not cands:
                return None
            if prefill_tasks:
                if self.config.prefill_guard_ms is not None:
                    max_prefill_wait_ms = max(max(0.0, now - float(tc["ready_ms"])) for tc in prefill_tasks)
                    if max_prefill_wait_ms >= float(self.config.prefill_guard_ms):
                        return _pick_prefill_task(prefill_tasks)
                if (self.config.max_consecutive_decode_batches > 0 and
                        consecutive_decode_batches >= int(self.config.max_consecutive_decode_batches)):
                    return _pick_prefill_task(prefill_tasks)

            def _key(tc: Dict[str, object]):
                phase_prio = 0
                if self.config.prompt_priority:
                    phase_prio = 0 if tc["phase"] == "prefill" else 1
                return (float(tc["earliest_start_ms"]),
                        phase_prio,
                        float(tc["ready_ms"]),
                        int(tc["first_req_id"]))

            return min(cands, key=_key)

        def _pending_existing_non_lost_with_deadline() -> bool:
            for r in shadow:
                if int(r["id"]) == target_id:
                    continue
                if bool(r["is_lost"]):
                    continue
                if r["deadline_ms"] is None:
                    continue
                if r["completion_time_ms"] is None:
                    return True
            return False

        while steps < int(self.config.admission_shadow_max_steps):
            target = next(r for r in shadow if int(r["id"]) == target_id)
            if target["completion_time_ms"] is not None and not _pending_existing_non_lost_with_deadline():
                break
            tc = _choose_task()
            if tc is None:
                break
            steps += 1
            start = float(tc["earliest_start_ms"])
            gpu_ms = float(tc["gpu_ms"])
            pim_ms = float(tc["pim_ms"])
            e2e_ms = float(tc["e2e_ms"])
            finish = start + e2e_ms

            now = max(now, start)
            if gpu_ms > 0:
                gpu_free = max(gpu_free, start) + gpu_ms
            if pim_ms > 0:
                pim_free = max(pim_free, start) + pim_ms

            if tc["phase"] == "prefill":
                consecutive_decode_batches = 0
                for i in tc["idxs"]:
                    r = shadow[i]
                    if r["prefill_start_ms"] is None:
                        r["prefill_start_ms"] = start
                    r["prefill_end_ms"] = finish
                    if r["first_token_time_ms"] is None:
                        r["first_token_time_ms"] = finish
                    r["last_token_completion_ms"] = finish
                    if int(r["remaining_decode_tokens"]) > 0:
                        r["state"] = "waiting_decode"
                        r["ready_ms"] = finish
                    else:
                        r["state"] = "done"
                        r["completion_time_ms"] = finish
            else:
                consecutive_decode_batches += 1
                for i in tc["idxs"]:
                    r = shadow[i]
                    if int(r["id"]) == target_id and r["first_decode_start_ms"] is None:
                        r["first_decode_start_ms"] = start
                    last = r["last_token_completion_ms"]
                    if last is not None:
                        r["tbt_intervals_ms"].append(finish - float(last))
                    r["last_token_completion_ms"] = finish
                    r["remaining_decode_tokens"] = int(r["remaining_decode_tokens"]) - 1
                    r["decoded_tokens_done"] = int(r["decoded_tokens_done"]) + 1
                    if int(r["remaining_decode_tokens"]) <= 0:
                        r["state"] = "done"
                        r["completion_time_ms"] = finish
                    else:
                        r["state"] = "waiting_decode"
                        r["ready_ms"] = finish

            if self._is_fcfs_strict() and shadow_fcfs_active_request_id is not None:
                live = next((r for r in shadow if int(r["id"]) == shadow_fcfs_active_request_id), None)
                if live is None or not _shadow_is_active(live):
                    shadow_fcfs_active_request_id = None
        else:
            timeout = True

        target = next(r for r in shadow if int(r["id"]) == target_id)
        cand_finish = target["completion_time_ms"]
        if cand_finish is None:
            timeout = True
        cand_finish_f = float(cand_finish) if cand_finish is not None else math.inf
        cand_first_decode_start = target["first_decode_start_ms"]
        cand_prefill_start = target["prefill_start_ms"]
        cand_prefill_end = target["prefill_end_ms"]
        pred_gpu_wait = 0.0
        if cand_prefill_start is not None:
            pred_gpu_wait = max(0.0, float(cand_prefill_start) - self.now_ms)
        pred_pim_wait = 0.0
        if est.uses_pim and cand_prefill_end is not None and cand_first_decode_start is not None:
            pred_pim_wait = max(0.0, float(cand_first_decode_start) - float(cand_prefill_end))
        tbt_vals = [float(v) for v in target["tbt_intervals_ms"]]
        if tbt_vals:
            cand_mean_tbt = float(sum(tbt_vals) / len(tbt_vals))
        else:
            cand_mean_tbt = float(est.decode_e2e_ms)

        existing_ok = True
        for r in shadow:
            if int(r["id"]) == target_id:
                continue
            if bool(r["is_lost"]):
                continue
            d = r["deadline_ms"]
            if d is None:
                continue
            c = r["completion_time_ms"]
            if c is None or float(c) > float(d):
                existing_ok = False
                break
        if existing_ok and self.config.slo_tbt_ms is not None:
            tbt_limit = float(self.config.slo_tbt_ms)
            for r in shadow:
                if int(r["id"]) == target_id:
                    continue
                if bool(r["is_lost"]):
                    continue
                if int(r["remaining_decode_tokens"]) <= 0 and not r["tbt_intervals_ms"]:
                    continue
                svc: RequestServiceEstimate = r["svc"]  # type: ignore
                tbt_hist = [float(v) for v in r["tbt_intervals_ms"]]
                mean_tbt = float(sum(tbt_hist) / len(tbt_hist)) if tbt_hist else float(svc.decode_e2e_ms)
                if mean_tbt > tbt_limit:
                    existing_ok = False
                    break
        return {
            "timeout": timeout,
            "candidate_finish_ms": cand_finish_f,
            "candidate_mean_tbt_ms": cand_mean_tbt,
            "candidate_pred_gpu_wait_ms": pred_gpu_wait,
            "candidate_pred_pim_wait_ms": pred_pim_wait,
            "existing_non_lost_feasible": existing_ok,
        }

    def _admit_waiting_request(self,
                               rr: ReplayRequest,
                               route: str,
                               est: RequestServiceEstimate,
                               cand: RouteCandidate,
                               reason: str,
                               forced_best_effort: bool = False):
        rr.route = route
        rr.admission_route = route
        rr.route_decision_reason = reason
        rr.admission_forced_best_effort = forced_best_effort
        rr.svc = est
        rr.remaining_decode_tokens = rr.svc.decode_tokens
        rr.state = "waiting_prefill"
        rr.ready_ms = self.now_ms
        rr.admission_time_ms = self.now_ms
        rr.admission_queue_wait_ms = self.now_ms - rr.arrival_ms
        rr.admission_last_reject_reason = ""
        rr.admission_last_harmed_count = cand.harmed_count
        rr.admission_last_total_harm_ms = cand.total_harm_ms
        rr.admission_last_single_harm_ms = cand.max_single_harm_ms
        rr.admission_last_own_miss_ms = cand.own_miss_ms
        rr.chosen_predicted_finish_ms = cand.predicted_finish_ms
        rr.latest_predicted_finish_ms = cand.predicted_finish_ms
        rr.chosen_predicted_gpu_wait_ms = cand.predicted_gpu_wait_ms
        rr.chosen_predicted_pim_wait_ms = cand.predicted_pim_wait_ms
        rr.chosen_predicted_incremental_energy_nj = cand.predicted_incremental_energy_nj
        rr.chosen_predicted_incremental_decode_energy_nj = cand.predicted_incremental_decode_energy_nj
        rr.chosen_slack_ms = cand.slack_ms
        if rr.svc.mapping.clipped_lin:
            self.mapping_clipped_lin_count += 1
        if rr.svc.mapping.clipped_lout:
            self.mapping_clipped_lout_count += 1
        self._inc_route_count(rr.route, 1)
        self.admission_queue_wait_samples_ms.append(max(0.0, self.now_ms - rr.arrival_ms))
        if "fallback_gpu" in reason:
            self.route_fallback_count += 1
        self._record_debug_event(
            "admit_route",
            request_id=rr.request_id,
            request_arrival_ms=rr.arrival_ms,
            context_tokens=rr.context_tokens,
            generated_tokens=rr.generated_tokens,
            admission_route=rr.admission_route,
            chosen_route=rr.route,
            decision_reason=rr.route_decision_reason,
            predicted_finish_ms=rr.chosen_predicted_finish_ms,
            predicted_e2e_ms=((rr.chosen_predicted_finish_ms - rr.arrival_ms)
                              if rr.chosen_predicted_finish_ms is not None else None),
            predicted_gpu_wait_ms=rr.chosen_predicted_gpu_wait_ms,
            predicted_pim_wait_ms=rr.chosen_predicted_pim_wait_ms,
            predicted_incremental_energy_nj=rr.chosen_predicted_incremental_energy_nj,
            predicted_incremental_decode_energy_nj=rr.chosen_predicted_incremental_decode_energy_nj,
            is_lost=rr.is_lost,
            forced_best_effort=rr.admission_forced_best_effort,
            admission_attempts=rr.admission_attempts,
            admission_bypass_count=rr.admission_bypass_count,
            admission_queue_wait_ms=max(0.0, self.now_ms - rr.arrival_ms),
        )

    def _process_admission_queue(self):
        while True:
            q = self._waiting_admission_requests()
            if not q:
                return
            self.admission_queue_max_len = max(self.admission_queue_max_len, len(q))
            self._prefetch_request_estimates(q)
            rr = q[0]
            if rr.admission_next_retry_ms > self.now_ms:
                return
            rr.admission_attempts += 1

            route_candidates: List[RouteCandidate] = []
            est_by_route: Dict[str, RequestServiceEstimate] = {}
            own_feasible_flags: List[bool] = []
            for route in self.route_order:
                est = self._estimate_request_cached(route,
                                                    context_tokens=rr.context_tokens,
                                                    generated_tokens=rr.generated_tokens,
                                                    bs=self._predicted_lookup_batch_size(route))
                if est is None:
                    route_candidates.append(RouteCandidate(route=route,
                                                           eligible=False,
                                                           uses_pim=("pim" in route),
                                                           predicted_finish_ms=math.inf,
                                                           predicted_gpu_wait_ms=math.inf,
                                                           predicted_pim_wait_ms=math.inf,
                                                           deadline_ms=rr.deadline_ms,
                                                           slack_ms=None,
                                                           reason="unsupported_length"))
                    continue
                proj = self._project_admission_candidate(rr, route, est)
                if bool(proj["timeout"]):
                    self.admission_shadow_timeout_count += 1
                    route_candidates.append(RouteCandidate(route=route,
                                                           eligible=False,
                                                           uses_pim=est.uses_pim,
                                                           predicted_finish_ms=math.inf,
                                                           predicted_gpu_wait_ms=math.inf,
                                                           predicted_pim_wait_ms=math.inf,
                                                           deadline_ms=rr.deadline_ms,
                                                           slack_ms=None,
                                                           reason="shadow_timeout"))
                    continue

                finish = float(proj["candidate_finish_ms"])
                pred_gpu_wait = float(proj["candidate_pred_gpu_wait_ms"])
                pred_pim_wait = float(proj["candidate_pred_pim_wait_ms"])
                own_e2e_ok = (rr.deadline_ms is None) or (finish <= rr.deadline_ms)
                own_tbt_ok = ((self.config.slo_tbt_ms is None) or
                              (float(proj["candidate_mean_tbt_ms"]) <= float(self.config.slo_tbt_ms)))
                existing_ok = bool(proj["existing_non_lost_feasible"])
                own_feasible_flags.append(own_e2e_ok)
                eligible = own_e2e_ok and own_tbt_ok and existing_ok
                reason = ""
                if not own_e2e_ok:
                    reason = "candidate_e2e_slo_miss"
                elif not own_tbt_ok:
                    reason = "candidate_tbt_slo_miss"
                elif not existing_ok:
                    reason = "existing_slo_violation"
                slack_ms = None if rr.deadline_ms is None else (rr.deadline_ms - finish)
                route_candidates.append(RouteCandidate(route=route,
                                                       eligible=eligible,
                                                       uses_pim=est.uses_pim,
                                                       predicted_finish_ms=finish,
                                                       predicted_gpu_wait_ms=pred_gpu_wait,
                                                       predicted_pim_wait_ms=pred_pim_wait,
                                                       deadline_ms=rr.deadline_ms,
                                                       slack_ms=slack_ms,
                                                       reason=reason))
                est_by_route[route] = est

            decision = self.policy.choose_route(route_candidates,
                                                now_ms=self.now_ms,
                                                deadline_ms=rr.deadline_ms)
            cand_by_route = {c.route: c for c in route_candidates}
            if (not decision.dropped and decision.route is not None and
                    decision.route in est_by_route and
                    cand_by_route.get(decision.route) is not None and
                    cand_by_route[decision.route].eligible):
                self._admit_waiting_request(rr,
                                            route=decision.route,
                                            est=est_by_route[decision.route],
                                            cand=cand_by_route[decision.route],
                                            reason=decision.reason)
                continue

            # Lost-but-run: own deadline infeasible for every route with estimate.
            own_infeasible_all = bool(own_feasible_flags) and not any(own_feasible_flags)
            if own_infeasible_all:
                rr.is_lost = True
                self.admission_lost_count += 1
                admitted = [c for c in route_candidates if c.route in est_by_route]
                if not admitted:
                    rr.state = "dropped"
                    rr.dropped_reason = "no_route_for_lost_request"
                    rr.route_decision_reason = rr.dropped_reason
                    self._record_debug_event(
                        "admit_drop",
                        request_id=rr.request_id,
                        request_arrival_ms=rr.arrival_ms,
                        context_tokens=rr.context_tokens,
                        generated_tokens=rr.generated_tokens,
                        decision_reason=rr.dropped_reason,
                    )
                    continue
                best = min(admitted, key=lambda c: (c.predicted_finish_ms, c.route))
                self._admit_waiting_request(rr,
                                            route=best.route,
                                            est=est_by_route[best.route],
                                            cand=best,
                                            reason="admit_lost_best_effort")
                continue

            rr.admission_last_reject_reason = decision.reason
            retry_dt = max(0.0, float(self.config.admission_retry_interval_ms))
            if retry_dt == 0.0:
                retry_dt = 1e-6
            rr.admission_next_retry_ms = self.now_ms + retry_dt
            self.admission_reject_count += 1
            self._record_debug_event(
                "admission_reject",
                request_id=rr.request_id,
                request_arrival_ms=rr.arrival_ms,
                context_tokens=rr.context_tokens,
                generated_tokens=rr.generated_tokens,
                decision_reason=decision.reason,
                admission_attempts=rr.admission_attempts,
                next_retry_ms=rr.admission_next_retry_ms,
                route_candidates=[self._route_candidate_debug_dict(c) for c in route_candidates],
            )
            return

    def _process_slo_guarded_admission_queue(self):
        while True:
            q = self._waiting_admission_requests()
            if not q:
                return
            self.admission_queue_max_len = max(self.admission_queue_max_len, len(q))
            self._prefetch_request_estimates(q)
            baseline_run = self._incremental_active_forecast_results()
            baseline_signature = self._incremental_active_forecast_cache_key
            baseline_results = baseline_run.requests
            baseline_total_energy_nj = (
                baseline_run.total_prefill_energy_nj + baseline_run.total_decode_energy_nj)
            baseline_total_decode_energy_nj = baseline_run.total_decode_energy_nj
            admitted_any = False

            for idx, rr in enumerate(q):
                if rr.admission_next_retry_ms > self.now_ms:
                    continue
                rr.admission_attempts += 1

                route_candidates: List[RouteCandidate] = []
                est_by_route: Dict[str, RequestServiceEstimate] = {}
                for route in self.route_order:
                    est = self._estimate_request_cached(route,
                                                        context_tokens=rr.context_tokens,
                                                        generated_tokens=rr.generated_tokens,
                                                        bs=self._predicted_lookup_batch_size(route))
                    if est is None:
                        route_candidates.append(RouteCandidate(route=route,
                                                               eligible=False,
                                                               uses_pim=("pim" in route),
                                                               predicted_finish_ms=math.inf,
                                                               predicted_gpu_wait_ms=math.inf,
                                                               predicted_pim_wait_ms=math.inf,
                                                               deadline_ms=rr.deadline_ms,
                                                               slack_ms=None,
                                                               reason="unsupported_length"))
                        continue
                    cand = self._evaluate_slo_guarded_route(rr,
                                                            route,
                                                            est,
                                                            baseline_signature,
                                                            baseline_results,
                                                            baseline_total_energy_nj,
                                                            baseline_total_decode_energy_nj)
                    route_candidates.append(cand)
                    est_by_route[route] = est

                self._record_debug_event(
                    "admission_eval",
                    request_id=rr.request_id,
                    request_arrival_ms=rr.arrival_ms,
                    admission_attempts=rr.admission_attempts,
                    route_candidates=[self._route_candidate_debug_dict(c) for c in route_candidates],
                )

                decision = self.policy.choose_route(route_candidates,
                                                    now_ms=self.now_ms,
                                                    deadline_ms=rr.deadline_ms)
                cand_by_route = {c.route: c for c in route_candidates}
                chosen = cand_by_route.get(decision.route) if decision.route is not None else None
                if (not decision.dropped and decision.route is not None and
                        decision.route in est_by_route and chosen is not None and chosen.eligible):
                    self._admit_waiting_request(rr,
                                                route=decision.route,
                                                est=est_by_route[decision.route],
                                                cand=chosen,
                                                reason=decision.reason)
                    self.admission_admitted_from_queue_count += 1
                    for blocked in q[:idx]:
                        if blocked.state != "waiting_admission":
                            continue
                        blocked.admission_bypass_count += 1
                        self.admission_bypassed_count += 1
                        self._record_debug_event(
                            "admission_bypass",
                            bypassed_request_id=blocked.request_id,
                            admitted_request_id=rr.request_id,
                            bypass_count=blocked.admission_bypass_count,
                        )
                    self._refresh_active_predictions(cause="new_admission")
                    admitted_any = True
                    break

                queue_wait_ms = max(0.0, self.now_ms - rr.arrival_ms)
                force_best_effort = (
                    (self.config.admission_max_wait_ms > 0.0 and queue_wait_ms >= float(self.config.admission_max_wait_ms)) or
                    (self.config.admission_max_bypass_count > 0 and rr.admission_bypass_count >= int(self.config.admission_max_bypass_count))
                )
                if force_best_effort:
                    admitted = [c for c in route_candidates if c.route in est_by_route]
                    if not admitted:
                        rr.state = "dropped"
                        rr.dropped_reason = "no_route_for_best_effort"
                        rr.route_decision_reason = rr.dropped_reason
                        self._record_debug_event(
                            "admit_drop",
                            request_id=rr.request_id,
                            request_arrival_ms=rr.arrival_ms,
                            context_tokens=rr.context_tokens,
                            generated_tokens=rr.generated_tokens,
                            decision_reason=rr.dropped_reason,
                        )
                        admitted_any = True
                        break
                    rr.is_lost = True
                    rr.admission_forced_best_effort = True
                    self.admission_lost_count += 1
                    self.admission_best_effort_count += 1
                    best = min(admitted, key=lambda c: (c.predicted_finish_ms, c.route))
                    self._admit_waiting_request(rr,
                                                route=best.route,
                                                est=est_by_route[best.route],
                                                cand=best,
                                                reason="admit_best_effort_due_to_wait_or_bypass",
                                                forced_best_effort=True)
                    self.admission_admitted_from_queue_count += 1
                    self._record_debug_event(
                        "admission_best_effort",
                        request_id=rr.request_id,
                        queue_wait_ms=queue_wait_ms,
                        bypass_count=rr.admission_bypass_count,
                        chosen_route=best.route,
                    )
                    for blocked in q[:idx]:
                        if blocked.state != "waiting_admission":
                            continue
                        blocked.admission_bypass_count += 1
                        self.admission_bypassed_count += 1
                        self._record_debug_event(
                            "admission_bypass",
                            bypassed_request_id=blocked.request_id,
                            admitted_request_id=rr.request_id,
                            bypass_count=blocked.admission_bypass_count,
                        )
                    self._refresh_active_predictions(cause="new_admission")
                    admitted_any = True
                    break

                best_reason = decision.reason
                if decision.dropped:
                    ranked = [c for c in route_candidates if c.route in est_by_route]
                    if ranked:
                        ranked.sort(key=lambda c: (c.predicted_finish_ms, c.route))
                        best_reason = ranked[0].reason or decision.reason
                        best_cand = ranked[0]
                    else:
                        best_cand = None
                else:
                    best_cand = chosen
                rr.admission_last_reject_reason = best_reason
                rr.admission_last_harmed_count = best_cand.harmed_count if best_cand is not None else 0
                rr.admission_last_total_harm_ms = best_cand.total_harm_ms if best_cand is not None else 0.0
                rr.admission_last_single_harm_ms = best_cand.max_single_harm_ms if best_cand is not None else 0.0
                rr.admission_last_own_miss_ms = best_cand.own_miss_ms if best_cand is not None else math.inf
                retry_dt = max(0.0, float(self.config.admission_retry_interval_ms))
                if retry_dt == 0.0:
                    retry_dt = 1e-6
                rr.admission_next_retry_ms = self.now_ms + retry_dt
                self.admission_held_count += 1
                self.admission_blocked_request_ids.add(rr.request_id)
                estimated_routes = [c for c in route_candidates if c.route in est_by_route]
                if estimated_routes and all(c.reason == "candidate_tbt_slo_miss" for c in estimated_routes):
                    self.admission_tbt_blocked_request_ids.add(rr.request_id)
                self._record_debug_event(
                    "admission_hold",
                    request_id=rr.request_id,
                    queue_wait_ms=queue_wait_ms,
                    admission_attempts=rr.admission_attempts,
                    next_retry_ms=rr.admission_next_retry_ms,
                    reason=best_reason,
                    harmed_count=rr.admission_last_harmed_count,
                    total_harm_ms=rr.admission_last_total_harm_ms,
                    max_single_harm_ms=rr.admission_last_single_harm_ms,
                    own_miss_ms=rr.admission_last_own_miss_ms,
                )

            if not admitted_any:
                return

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
            self._prefetch_request_estimates([src_req])
            baseline_total_decode_energy_nj = None
            baseline_total_energy_nj = None
            if self._uses_latency_guarded_energy_policy():
                baseline_run = self._incremental_active_forecast_results()
                baseline_total_energy_nj = (
                    baseline_run.total_prefill_energy_nj + baseline_run.total_decode_energy_nj)
                baseline_total_decode_energy_nj = baseline_run.total_decode_energy_nj
            for route in self.route_order:
                cand, est = self._predict_route_candidate(src_req,
                                                          route,
                                                          deadline_ms=rr.deadline_ms,
                                                          baseline_total_energy_nj=baseline_total_energy_nj,
                                                          baseline_total_decode_energy_nj=baseline_total_decode_energy_nj)
                candidates.append(cand)
                cand_by_route[route] = cand
                if est is not None:
                    est_by_route[route] = est

            decision = self.policy.choose_route(candidates,
                                               now_ms=self.now_ms,
                                               deadline_ms=rr.deadline_ms)
            cand_debug = [self._route_candidate_debug_dict(c) for c in candidates]
            if decision.dropped or decision.route is None or decision.route not in est_by_route:
                rr.state = "dropped"
                rr.dropped_reason = decision.reason
                rr.route_decision_reason = decision.reason
                self.requests.append(rr)
                self._record_debug_event(
                    "admit_drop",
                    request_id=rr.request_id,
                    request_arrival_ms=rr.arrival_ms,
                    context_tokens=rr.context_tokens,
                    generated_tokens=rr.generated_tokens,
                    decision_reason=decision.reason,
                    route_candidates=cand_debug,
                )
                continue

            rr.route = decision.route
            rr.admission_route = decision.route
            rr.route_decision_reason = decision.reason
            if "fallback_gpu" in decision.reason:
                self.route_fallback_count += 1
            rr.svc = est_by_route[decision.route]
            chosen_cand = cand_by_route.get(decision.route)
            if chosen_cand is not None:
                rr.chosen_predicted_finish_ms = chosen_cand.predicted_finish_ms
                rr.latest_predicted_finish_ms = chosen_cand.predicted_finish_ms
                rr.chosen_predicted_gpu_wait_ms = chosen_cand.predicted_gpu_wait_ms
                rr.chosen_predicted_pim_wait_ms = chosen_cand.predicted_pim_wait_ms
                rr.chosen_predicted_incremental_energy_nj = (
                    chosen_cand.predicted_incremental_energy_nj)
                rr.chosen_predicted_incremental_decode_energy_nj = (
                    chosen_cand.predicted_incremental_decode_energy_nj)
                rr.chosen_queue_pressure_ms = chosen_cand.queue_pressure_ms
                rr.chosen_slack_ms = chosen_cand.slack_ms
            rr.remaining_decode_tokens = rr.svc.decode_tokens
            rr.state = "waiting_prefill"
            rr.ready_ms = rr.arrival_ms
            rr.admission_time_ms = rr.arrival_ms
            rr.admission_queue_wait_ms = 0.0

            if rr.svc.mapping.clipped_lin:
                self.mapping_clipped_lin_count += 1
            if rr.svc.mapping.clipped_lout:
                self.mapping_clipped_lout_count += 1

            self._inc_route_count(rr.route, 1)
            self.requests.append(rr)
            self._record_debug_event(
                "admit_route",
                request_id=rr.request_id,
                request_arrival_ms=rr.arrival_ms,
                context_tokens=rr.context_tokens,
                generated_tokens=rr.generated_tokens,
                admission_route=rr.admission_route,
                chosen_route=rr.route,
                decision_reason=rr.route_decision_reason,
                predicted_finish_ms=rr.chosen_predicted_finish_ms,
                predicted_e2e_ms=((rr.chosen_predicted_finish_ms - rr.arrival_ms)
                                  if rr.chosen_predicted_finish_ms is not None else None),
                predicted_gpu_wait_ms=rr.chosen_predicted_gpu_wait_ms,
                predicted_pim_wait_ms=rr.chosen_predicted_pim_wait_ms,
                predicted_incremental_energy_nj=rr.chosen_predicted_incremental_energy_nj,
                predicted_incremental_decode_energy_nj=rr.chosen_predicted_incremental_decode_energy_nj,
                predicted_queue_pressure_ms=rr.chosen_queue_pressure_ms,
                route_candidates=cand_debug,
            )
            self._refresh_active_predictions(cause="new_admission")

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
                              prefill_energy_nj=rr.svc.prefill_energy_nj,
                              decode_energy_nj=0.0,
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
        return self._estimate_request_cached(route,
                                             context_tokens=batch_lin,
                                             generated_tokens=batch_lout,
                                             bs=len(batch_reqs))

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
                                        prefill_energy_nj=batch_est.prefill_energy_nj,
                                        decode_energy_nj=0.0,
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
                              prefill_energy_nj=0.0,
                              decode_energy_nj=rr.svc.decode_energy_nj,
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
                batch_est = self._estimate_request_cached(route,
                                                          context_tokens=batch_lin,
                                                          generated_tokens=2,
                                                          bs=batch_size)
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
                                            prefill_energy_nj=0.0,
                                            decode_energy_nj=batch_est.decode_energy_nj,
                                            earliest_start_ms=earliest_start,
                                            predicted_finish_ms=earliest_start + batch_est.decode_e2e_ms,
                                            batch_size=batch_size))
                break
        return cands

    def _build_fcfs_prefix_decode_task(self) -> Optional[_TaskCandidate]:
        decode_q = self._fcfs_decode_queue()
        if not decode_q:
            return None

        head = decode_q[0]
        cutoff_ms = max(self.now_ms, head.ready_ms)
        prefix = [head]
        if self.config.enable_decode_batching:
            for rr in decode_q[1:]:
                if len(prefix) >= int(self.config.max_decode_batch_size):
                    break
                if rr.route != head.route:
                    break
                if rr.ready_ms > cutoff_ms:
                    break
                prefix.append(rr)

        if not self.config.enable_decode_batching:
            prefix = [head]

        batch_reqs: List[ReplayRequest] = []
        gpu_ms = pim_ms = e2e_ms = decode_energy_nj = 0.0
        for batch_size in range(len(prefix), 0, -1):
            batch_reqs = prefix[:batch_size]
            if batch_size == 1:
                rr = batch_reqs[0]
                if rr.svc is None:
                    continue
                gpu_ms = rr.svc.decode_gpu_ms
                pim_ms = rr.svc.decode_pim_ms
                e2e_ms = rr.svc.decode_e2e_ms
                decode_energy_nj = rr.svc.decode_energy_nj
                break
            batch_lin = max(self._request_decode_context_tokens(r) for r in batch_reqs)
            batch_est = self._estimate_request_cached(head.route or "unknown",
                                                      context_tokens=batch_lin,
                                                      generated_tokens=2,
                                                      bs=batch_size)
            if batch_est is None:
                continue
            gpu_ms = batch_est.decode_gpu_ms
            pim_ms = batch_est.decode_pim_ms
            e2e_ms = batch_est.decode_e2e_ms
            decode_energy_nj = batch_est.decode_energy_nj
            break

        if not batch_reqs:
            return None

        earliest_start = self._predict_stage_start(cutoff_ms, gpu_ms, pim_ms,
                                                   self.gpu_free_ms, self.pim_free_ms)
        return _TaskCandidate(requests=list(batch_reqs),
                              route=(head.route or "unknown"),
                              phase="decode",
                              ready_ms=cutoff_ms,
                              gpu_ms=gpu_ms,
                              pim_ms=pim_ms,
                              e2e_ms=e2e_ms,
                              prefill_energy_nj=0.0,
                              decode_energy_nj=decode_energy_nj,
                              earliest_start_ms=earliest_start,
                              predicted_finish_ms=earliest_start + e2e_ms,
                              batch_size=len(batch_reqs))

    def _choose_next_task(self) -> Optional[_TaskCandidate]:
        if self._is_fcfs_strict():
            rr = self._fcfs_active_request()
            if rr is None:
                return None
            if rr.state == "waiting_prefill":
                return self._prefill_task_for_request(rr)
            if rr.state == "waiting_decode":
                return self._single_decode_task_for_request(rr)
            self.fcfs_active_request_id = None
            return None

        if self._is_prefill_priority_fcfs_decode():
            rr = self._pick_oldest_prefill_request()
            if rr is not None:
                return self._prefill_task_for_request(rr)
            return self._build_fcfs_prefix_decode_task()

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
            self.prefill_energy_total_nj += max(0.0, tc.prefill_energy_nj)
            self.prefill_energy_by_route_nj[tc.route] = (
                self.prefill_energy_by_route_nj.get(tc.route, 0.0) + max(0.0, tc.prefill_energy_nj))
            rebound_changed = False
            for rr in tc.requests:
                rr.prefill_start_ms = tc.earliest_start_ms
                rr.prefill_end_ms = finish_ms
                rr.first_token_time_ms = finish_ms
                rr.last_token_completion_ms = finish_ms
                if rr.remaining_decode_tokens > 0:
                    self._transition_to_waiting_decode(rr, ready_ms=finish_ms)
                    old_route = rr.route
                    self._maybe_rebind_decode_route(rr, decision_time_ms=finish_ms)
                    rebound_changed = rebound_changed or (old_route != rr.route)
                else:
                    rr.state = "done"
                    rr.completion_time_ms = finish_ms
            if rebound_changed:
                self._refresh_active_predictions(cause="decode_rebind")
        else:
            self.decode_batch_sizes.append(tc.batch_size)
            self.consecutive_decode_batches += 1
            self.decode_energy_total_nj += max(0.0, tc.decode_energy_nj)
            self.decode_energy_by_route_nj[tc.route] = (
                self.decode_energy_by_route_nj.get(tc.route, 0.0) + max(0.0, tc.decode_energy_nj))
            self.decode_token_count += tc.batch_size
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

        self._refresh_fcfs_active_request()

    def run(self) -> ReplaySummary:
        self.now_ms = 0.0
        if self.arrivals:
            self.now_ms = min(0.0, self.arrivals[0].arrival_ms)

        while True:
            if self._uses_waiting_admission_queue():
                self._enqueue_arrivals_up_to_now()
                if self.config.enable_predictive_admission:
                    self._process_admission_queue()
                else:
                    self._process_slo_guarded_admission_queue()
            else:
                self._admit_arrivals_up_to_now()

            task = self._choose_next_task()
            next_arrival_ms = None
            if self._arrival_idx < len(self.arrivals):
                next_arrival_ms = self.arrivals[self._arrival_idx].arrival_ms
            next_retry_ms = self._min_waiting_admission_retry_ms() if self._uses_waiting_admission_queue() else None

            if task is None:
                next_event_ms = None
                for t in (next_arrival_ms, next_retry_ms):
                    if t is None:
                        continue
                    next_event_ms = t if next_event_ms is None else min(next_event_ms, t)
                if next_event_ms is None:
                    break
                self._record_debug_event(
                    "advance_to_event",
                    next_event_ms=next_event_ms,
                    next_arrival_ms=next_arrival_ms,
                    next_retry_ms=next_retry_ms,
                )
                self.now_ms = max(self.now_ms, next_event_ms)
                continue

            # Let earlier arrivals enter before scheduling a later-start task.
            next_event_ms = None
            for t in (next_arrival_ms, next_retry_ms):
                if t is None:
                    continue
                next_event_ms = t if next_event_ms is None else min(next_event_ms, t)
            if next_event_ms is not None and next_event_ms < task.earliest_start_ms:
                self._record_debug_event(
                    "yield_for_event",
                    next_arrival_ms=next_arrival_ms,
                    next_retry_ms=next_retry_ms,
                    task_earliest_start_ms=task.earliest_start_ms,
                    task_phase=task.phase,
                    task_route=task.route,
                    task_batch_size=task.batch_size,
                    task_request_ids=[r.request_id for r in task.requests],
                )
                self.now_ms = max(self.now_ms, next_event_ms)
                continue

            before_states = [{
                "request_id": rr.request_id,
                "state": rr.state,
                "remaining_decode_tokens": rr.remaining_decode_tokens,
                "ready_ms": rr.ready_ms,
            } for rr in task.requests]
            self._record_debug_event(
                "dispatch_task",
                phase=task.phase,
                route=task.route,
                batch_size=task.batch_size,
                request_ids=[r.request_id for r in task.requests],
                task_ready_ms=task.ready_ms,
                task_earliest_start_ms=task.earliest_start_ms,
                task_gpu_ms=task.gpu_ms,
                task_pim_ms=task.pim_ms,
                task_e2e_ms=task.e2e_ms,
                task_prefill_energy_nj=task.prefill_energy_nj,
                task_decode_energy_nj=task.decode_energy_nj,
                task_energy_nj=(task.prefill_energy_nj + task.decode_energy_nj),
                task_predicted_finish_ms=task.predicted_finish_ms,
                request_states_before=before_states,
            )
            self.now_ms = max(self.now_ms, task.earliest_start_ms)
            self._reserve_resources(task.earliest_start_ms, task.gpu_ms, task.pim_ms)
            self._complete_task(task)
            finish_ms = task.earliest_start_ms + task.e2e_ms
            after_states = [{
                "request_id": rr.request_id,
                "state": rr.state,
                "remaining_decode_tokens": rr.remaining_decode_tokens,
                "ready_ms": rr.ready_ms,
                "completion_time_ms": rr.completion_time_ms,
                "prefill_start_ms": rr.prefill_start_ms,
                "prefill_end_ms": rr.prefill_end_ms,
                "last_token_completion_ms": rr.last_token_completion_ms,
            } for rr in task.requests]
            self._record_debug_event(
                "complete_task",
                phase=task.phase,
                route=task.route,
                batch_size=task.batch_size,
                request_ids=[r.request_id for r in task.requests],
                finish_ms=finish_ms,
                request_states_after=after_states,
            )

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
        prefill_energy_nj = {
            "total": float(self.prefill_energy_total_nj),
            "by_route": dict(sorted(
                (route, float(val)) for route, val in self.prefill_energy_by_route_nj.items())),
        }
        decode_energy_nj = {
            "total": float(self.decode_energy_total_nj),
            "by_route": dict(sorted(
                (route, float(val)) for route, val in self.decode_energy_by_route_nj.items())),
        }
        total_energy_by_route: Dict[str, float] = {}
        for route, val in self.prefill_energy_by_route_nj.items():
            total_energy_by_route[route] = total_energy_by_route.get(route, 0.0) + float(val)
        for route, val in self.decode_energy_by_route_nj.items():
            total_energy_by_route[route] = total_energy_by_route.get(route, 0.0) + float(val)
        energy_nj = {
            "total": float(self.prefill_energy_total_nj + self.decode_energy_total_nj),
            "prefill_total": float(self.prefill_energy_total_nj),
            "decode_total": float(self.decode_energy_total_nj),
            "by_route": dict(sorted((route, float(val)) for route, val in total_energy_by_route.items())),
        }
        admission_queue = {
            "max_len": float(self.admission_queue_max_len),
            "mean_wait_ms": (float(sum(self.admission_queue_wait_samples_ms) /
                                   len(self.admission_queue_wait_samples_ms))
                             if self.admission_queue_wait_samples_ms else 0.0),
        }
        admission = {
            "reject_count": float(self.admission_reject_count),
            "lost_count": float(self.admission_lost_count),
            "shadow_timeout_count": float(self.admission_shadow_timeout_count),
            "held_count": float(self.admission_held_count),
            "admitted_from_queue_count": float(self.admission_admitted_from_queue_count),
            "bypassed_count": float(self.admission_bypassed_count),
            "best_effort_count": float(self.admission_best_effort_count),
            "blocked_count": float(len(self.admission_blocked_request_ids)),
            "blocked_due_to_tbt_count": float(len(self.admission_tbt_blocked_request_ids)),
        }
        local_scheduling = {
            "local_scheduling_policy": self.config.local_scheduling_policy,
            "fcfs_decode_prediction_mode": self.config.fcfs_decode_prediction_mode,
            "prefill_guard_ms": (float(self.config.prefill_guard_ms)
                                  if self.config.prefill_guard_ms is not None else math.nan),
            "max_consecutive_decode_batches": float(self.config.max_consecutive_decode_batches),
            "decode_batch_cap_with_prefill": float(self.config.decode_batch_cap_with_prefill),
            "enable_decode_rebind": float(self.config.enable_decode_rebind),
            "decode_rebind_margin_ms": float(self.config.decode_rebind_margin_ms),
            "decode_rebind_attempt_count": float(sum(1 for r in self.requests if r.decode_rebind_attempted)),
            "decode_rebound_count": float(sum(1 for r in self.requests if r.decode_rebound)),
            "enable_predictive_admission": float(self.config.enable_predictive_admission),
            "enable_slo_guarded_admission": float(self.config.enable_slo_guarded_admission),
            "slo_tbt_ms": (float(self.config.slo_tbt_ms)
                            if self.config.slo_tbt_ms is not None else math.nan),
            "admission_shadow_max_steps": float(self.config.admission_shadow_max_steps),
            "admission_retry_interval_ms": float(self.config.admission_retry_interval_ms),
            "admission_max_harmed_requests": float(self.config.admission_max_harmed_requests),
            "admission_max_total_harm_ms": float(self.config.admission_max_total_harm_ms),
            "admission_max_single_harm_ms": float(self.config.admission_max_single_harm_ms),
            "admission_max_own_miss_ms": float(self.config.admission_max_own_miss_ms),
            "admission_max_bypass_count": float(self.config.admission_max_bypass_count),
            "admission_max_wait_ms": float(self.config.admission_max_wait_ms),
            "admission_aging_harm_ms_per_ms": float(self.config.admission_aging_harm_ms_per_ms),
            "admission_aging_harmed_requests_per_ms": float(self.config.admission_aging_harmed_requests_per_ms),
            "prefill_guard_trigger_count": float(self.prefill_guard_trigger_count),
            "decode_limit_trigger_count": float(self.decode_limit_trigger_count),
        }
        cost_model_stats = {}
        try:
            cost_model_stats = {
                str(k): float(v) for k, v in self.cost_model.stats().items()
            }
        except Exception:
            cost_model_stats = {}
        cache_stats = {
            "estimate_cache_hits": float(self.estimate_cache_hits),
            "estimate_cache_misses": float(self.estimate_cache_misses),
            "estimate_prefetch_populated": float(self.estimate_prefetch_populated),
            "incremental_active_forecast_cache_hits": float(self.incremental_active_forecast_cache_hits),
            "incremental_active_forecast_cache_misses": float(self.incremental_active_forecast_cache_misses),
            "projection_cache_hits": float(self.projection_cache_hits),
            "projection_cache_misses": float(self.projection_cache_misses),
        }
        for key, value in cost_model_stats.items():
            cache_stats[f"cost_model_{key}"] = value

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
            prefill_energy_nj=prefill_energy_nj,
            decode_energy_nj=decode_energy_nj,
            energy_nj=energy_nj,
            energy_nj_per_request=(
                float((self.prefill_energy_total_nj + self.decode_energy_total_nj) / len(completed))
                if completed else math.nan),
            decode_energy_nj_per_decode_token=(
                float(self.decode_energy_total_nj / self.decode_token_count)
                if self.decode_token_count > 0 else math.nan),
            admission_queue=admission_queue,
            admission=admission,
            local_scheduling=local_scheduling,
            cache_stats=cache_stats,
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
                "admission_route": r.admission_route,
                "state": r.state,
                "route_decision_reason": r.route_decision_reason,
                "dropped_reason": r.dropped_reason,
                "admission_time_ms": r.admission_time_ms,
                "admission_queue_wait_ms": r.admission_queue_wait_ms,
                "admission_attempts": r.admission_attempts,
                "admission_bypass_count": r.admission_bypass_count,
                "admission_last_reject_reason": r.admission_last_reject_reason,
                "admission_last_harmed_count": r.admission_last_harmed_count,
                "admission_last_total_harm_ms": r.admission_last_total_harm_ms,
                "admission_last_single_harm_ms": r.admission_last_single_harm_ms,
                "admission_last_own_miss_ms": r.admission_last_own_miss_ms,
                "admission_forced_best_effort": r.admission_forced_best_effort,
                "is_lost": r.is_lost,
                "ttft_ms": ttft,
                "prefill_wait_ms": prefill_wait,
                "e2e_ms": e2e,
                "prefill_energy_nj": (r.svc.prefill_energy_nj if r.svc else None),
                "decode_energy_nj": (r.svc.decode_energy_nj if r.svc else None),
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
                "latest_predicted_finish_ms": r.latest_predicted_finish_ms,
                "chosen_predicted_gpu_wait_ms": r.chosen_predicted_gpu_wait_ms,
                "chosen_predicted_pim_wait_ms": r.chosen_predicted_pim_wait_ms,
                "chosen_predicted_incremental_energy_nj":
                    r.chosen_predicted_incremental_energy_nj,
                "chosen_predicted_incremental_decode_energy_nj":
                    r.chosen_predicted_incremental_decode_energy_nj,
                "chosen_queue_pressure_ms": r.chosen_queue_pressure_ms,
                "chosen_slack_ms": r.chosen_slack_ms,
                "decode_bind_time_ms": r.decode_bind_time_ms,
                "decode_rebind_attempted": r.decode_rebind_attempted,
                "decode_rebound": r.decode_rebound,
                "decode_rebind_reason": r.decode_rebind_reason,
                "decode_enqueue_seq": r.decode_enqueue_seq,
                "prediction_refresh_count": r.prediction_refresh_count,
                "decoded_tokens_done": r.decoded_tokens_done,
            })
        return pd.DataFrame(rows)

    def debug_events_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.debug_events)
