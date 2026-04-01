from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

import pandas as pd

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
    admission_check_own_slo: bool = False
    admission_max_bypass_count: int = 0
    admission_max_wait_ms: float = 0.0
    admission_aging_harm_ms_per_ms: float = 0.0
    admission_aging_harmed_requests_per_ms: float = 0.0
    prompt_priority: bool = True
    slo_e2e_ms: Optional[float] = None
    slo_ttft_ms: Optional[float] = None
    share_gpu_prefill_across_routes: bool = False
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
