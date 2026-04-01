from __future__ import annotations

import json
import math
from dataclasses import replace
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from src.azure_trace import AzureTraceRequest
from src.scheduler_policy import QueueAwareFinishTimePolicy, RouteCandidate
from .types import (CostModelProtocol, QueuePressureSnapshot, ReplayConfig, ReplayRequest, ReplaySummary, RequestServiceEstimate, _IncrementalForecastRun, _TaskCandidate, _percentiles)

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
                raise ValueError("energy-aware route policies require prefill_priority_fcfs_decode")
            if not self._use_incremental_prediction_for_fcfs_decode():
                raise ValueError("energy-aware route policies require fcfs_decode_prediction_mode=incremental")
            missing_routes = [route for route in self.route_order
                              if not self.cost_model.supports_decode_energy(route)]
            if missing_routes:
                raise ValueError(
                    "energy-aware route policies require decode_energy_nj support for all configured routes; "
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

    def _route_candidate_debug_dict(self,
                                    cand: RouteCandidate,
                                    est: Optional[RequestServiceEstimate] = None) -> Dict[str, object]:
        row = {
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
            "route_active_requests": cand.route_active_requests,
            "route_waiting_prefill_count": cand.route_waiting_prefill_count,
            "route_waiting_decode_count": cand.route_waiting_decode_count,
            "route_pending_decode_tokens": cand.route_pending_decode_tokens,
            "harmed_count": cand.harmed_count,
            "total_harm_ms": cand.total_harm_ms,
            "max_single_harm_ms": cand.max_single_harm_ms,
            "own_miss_ms": cand.own_miss_ms,
            "reason": cand.reason,
        }
        if est is not None:
            decode_total_energy_nj = float(est.decode_energy_nj) * float(est.decode_tokens)
            row.update({
                "prefill_energy_nj": float(est.prefill_energy_nj),
                "decode_energy_per_token_nj": float(est.decode_energy_nj),
                "decode_tokens": float(est.decode_tokens),
                "decode_total_energy_nj": decode_total_energy_nj,
                "request_total_energy_nj": float(est.prefill_energy_nj) + decode_total_energy_nj,
                "prefill_e2e_ms": float(est.prefill_e2e_ms),
                "decode_e2e_ms_per_token": float(est.decode_e2e_ms),
                "decode_total_e2e_ms": float(est.decode_e2e_ms) * float(est.decode_tokens),
                "mapped_lin": int(est.mapping.mapped_lin),
                "mapped_lout": int(est.mapping.mapped_lout),
                "mapped_bs": int(est.mapping.mapped_bs),
                "clipped_lin": bool(est.mapping.clipped_lin),
                "clipped_lout": bool(est.mapping.clipped_lout),
            })
        return row

    def _route_load_snapshot(self, route: str) -> Dict[str, int]:
        waiting_prefill_count = 0
        waiting_decode_count = 0
        pending_decode_tokens = 0
        for rr in self.requests:
            if rr.route != route:
                continue
            if rr.state == "waiting_prefill":
                waiting_prefill_count += 1
            elif rr.state == "waiting_decode":
                waiting_decode_count += 1
                pending_decode_tokens += int(max(0, rr.remaining_decode_tokens))
        return {
            "route_active_requests": waiting_prefill_count + waiting_decode_count,
            "route_waiting_prefill_count": waiting_prefill_count,
            "route_waiting_decode_count": waiting_decode_count,
            "route_pending_decode_tokens": pending_decode_tokens,
        }

    def _annotate_route_candidate(self, cand: RouteCandidate) -> RouteCandidate:
        snapshot = self._route_load_snapshot(cand.route)
        cand.route_active_requests = int(snapshot["route_active_requests"])
        cand.route_waiting_prefill_count = int(snapshot["route_waiting_prefill_count"])
        cand.route_waiting_decode_count = int(snapshot["route_waiting_decode_count"])
        cand.route_pending_decode_tokens = int(snapshot["route_pending_decode_tokens"])
        return cand

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
            est = self.cost_model.estimate_request(
                route,
                context_tokens=int(context_tokens),
                generated_tokens=int(generated_tokens),
                bs=int(bs),
                unsupported_policy=self.config.unsupported_policy,
                lin_bucket=self.config.lin_bucket,
                lout_bucket=self.config.lout_bucket,
            )
            self._estimate_cache[key] = self._shared_gpu_prefill_estimate(
                route=route,
                context_tokens=int(context_tokens),
                generated_tokens=int(generated_tokens),
                bs=int(bs),
                est=est,
            )
        else:
            self.estimate_cache_hits += 1
        return self._estimate_cache[key]

    def _shared_gpu_prefill_estimate(self,
                                     route: str,
                                     context_tokens: int,
                                     generated_tokens: int,
                                     bs: int,
                                     est: Optional[RequestServiceEstimate]
                                     ) -> Optional[RequestServiceEstimate]:
        if (not self.config.share_gpu_prefill_across_routes or
                est is None or
                route == "gpu_only"):
            return est
        if "gpu_only" not in self.cost_model.routes():
            return est
        gpu_est = self._estimate_request_cached("gpu_only",
                                                context_tokens=context_tokens,
                                                generated_tokens=generated_tokens,
                                                bs=bs)
        if gpu_est is None:
            return est
        return replace(est,
                       prefill_e2e_ms=float(gpu_est.prefill_e2e_ms),
                       prefill_gpu_ms=float(gpu_est.prefill_gpu_ms),
                       prefill_pim_ms=float(gpu_est.prefill_pim_ms),
                       prefill_energy_nj=float(gpu_est.prefill_energy_nj))

    def _prefetch_request_estimates(self,
                                    requests: Iterable[object],
                                    routes: Optional[Iterable[str]] = None):
        route_list = list(routes) if routes is not None else list(self.route_order)
        if not route_list:
            return
        if self.config.share_gpu_prefill_across_routes and "gpu_only" in self.cost_model.routes():
            if "gpu_only" not in route_list:
                route_list.append("gpu_only")
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
            for key, req_spec, est in zip(missing_keys_by_route[route],
                                          missing_by_route[route],
                                          estimates):
                context_tokens, generated_tokens, bs = req_spec
                self._estimate_cache[key] = self._shared_gpu_prefill_estimate(
                    route=route,
                    context_tokens=int(context_tokens),
                    generated_tokens=int(generated_tokens),
                    bs=int(bs),
                    est=est,
                )
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
            "route_policy": getattr(self.policy, "route_policy", ""),
            "energy_latency_guard_ms": float(getattr(self.policy, "energy_latency_guard_ms", math.nan)),
            "route_load_balance_ms_per_active_request": float(
                getattr(self.policy, "route_load_balance_ms_per_active_request", math.nan)),
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
            "admission_check_own_slo": float(self.config.admission_check_own_slo),
            "admission_max_bypass_count": float(self.config.admission_max_bypass_count),
            "admission_max_wait_ms": float(self.config.admission_max_wait_ms),
            "admission_aging_harm_ms_per_ms": float(self.config.admission_aging_harm_ms_per_ms),
            "admission_aging_harmed_requests_per_ms": float(self.config.admission_aging_harmed_requests_per_ms),
            "share_gpu_prefill_across_routes": float(self.config.share_gpu_prefill_across_routes),
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


from .forecast import (
    _shadow_request_from_replay_request,
    _shadow_request_for_candidate,
    _simulate_prefill_priority_fcfs_decode_shadow,
    _project_prefill_priority_fcfs_decode_candidate,
    _incremental_request_from_replay_request,
    _incremental_request_for_candidate,
    _simulate_prefill_priority_fcfs_decode_incremental,
    _project_prefill_priority_fcfs_decode_incremental_candidate,
    _incremental_request_for_waiting_request,
    _incremental_active_forecast_results,
    _effective_slo_guard_budgets,
    _evaluate_slo_guarded_route,
    _heuristic_decode_candidate_from_replay_request,
    _heuristic_decode_step_estimate,
    _simulate_prefill_priority_fcfs_decode_heuristic,
    _project_prefill_priority_fcfs_decode_heuristic_candidate,
    _project_prefill_priority_fcfs_decode_heuristic_decode_candidate,
    _refresh_active_predictions,
    _predict_route_candidate,
    _predict_decode_route_candidate,
    _maybe_rebind_decode_route,
)
from .admission import (
    _enqueue_arrivals_up_to_now,
    _waiting_admission_requests,
    _project_admission_candidate,
    _admit_waiting_request,
    _process_admission_queue,
    _process_slo_guarded_admission_queue,
    _admit_arrivals_up_to_now,
)
from .tasks import (
    _prefill_task_for_request,
    _estimate_prefill_batch,
    _build_prefill_batch_candidates,
    _single_decode_task_for_request,
    _build_decode_batch_candidates,
    _build_fcfs_prefix_decode_task,
    _choose_next_task,
    _reserve_resources,
    _complete_task,
)

TraceReplaySimulator._shadow_request_from_replay_request = _shadow_request_from_replay_request
TraceReplaySimulator._shadow_request_for_candidate = _shadow_request_for_candidate
TraceReplaySimulator._simulate_prefill_priority_fcfs_decode_shadow = _simulate_prefill_priority_fcfs_decode_shadow
TraceReplaySimulator._project_prefill_priority_fcfs_decode_candidate = _project_prefill_priority_fcfs_decode_candidate
TraceReplaySimulator._incremental_request_from_replay_request = _incremental_request_from_replay_request
TraceReplaySimulator._incremental_request_for_candidate = _incremental_request_for_candidate
TraceReplaySimulator._simulate_prefill_priority_fcfs_decode_incremental = _simulate_prefill_priority_fcfs_decode_incremental
TraceReplaySimulator._project_prefill_priority_fcfs_decode_incremental_candidate = _project_prefill_priority_fcfs_decode_incremental_candidate
TraceReplaySimulator._incremental_request_for_waiting_request = _incremental_request_for_waiting_request
TraceReplaySimulator._incremental_active_forecast_results = _incremental_active_forecast_results
TraceReplaySimulator._effective_slo_guard_budgets = _effective_slo_guard_budgets
TraceReplaySimulator._evaluate_slo_guarded_route = _evaluate_slo_guarded_route
TraceReplaySimulator._heuristic_decode_candidate_from_replay_request = _heuristic_decode_candidate_from_replay_request
TraceReplaySimulator._heuristic_decode_step_estimate = _heuristic_decode_step_estimate
TraceReplaySimulator._simulate_prefill_priority_fcfs_decode_heuristic = _simulate_prefill_priority_fcfs_decode_heuristic
TraceReplaySimulator._project_prefill_priority_fcfs_decode_heuristic_candidate = _project_prefill_priority_fcfs_decode_heuristic_candidate
TraceReplaySimulator._project_prefill_priority_fcfs_decode_heuristic_decode_candidate = _project_prefill_priority_fcfs_decode_heuristic_decode_candidate
TraceReplaySimulator._refresh_active_predictions = _refresh_active_predictions
TraceReplaySimulator._predict_route_candidate = _predict_route_candidate
TraceReplaySimulator._predict_decode_route_candidate = _predict_decode_route_candidate
TraceReplaySimulator._maybe_rebind_decode_route = _maybe_rebind_decode_route
TraceReplaySimulator._enqueue_arrivals_up_to_now = _enqueue_arrivals_up_to_now
TraceReplaySimulator._waiting_admission_requests = _waiting_admission_requests
TraceReplaySimulator._project_admission_candidate = _project_admission_candidate
TraceReplaySimulator._admit_waiting_request = _admit_waiting_request
TraceReplaySimulator._process_admission_queue = _process_admission_queue
TraceReplaySimulator._process_slo_guarded_admission_queue = _process_slo_guarded_admission_queue
TraceReplaySimulator._admit_arrivals_up_to_now = _admit_arrivals_up_to_now
TraceReplaySimulator._prefill_task_for_request = _prefill_task_for_request
TraceReplaySimulator._estimate_prefill_batch = _estimate_prefill_batch
TraceReplaySimulator._build_prefill_batch_candidates = _build_prefill_batch_candidates
TraceReplaySimulator._single_decode_task_for_request = _single_decode_task_for_request
TraceReplaySimulator._build_decode_batch_candidates = _build_decode_batch_candidates
TraceReplaySimulator._build_fcfs_prefix_decode_task = _build_fcfs_prefix_decode_task
TraceReplaySimulator._choose_next_task = _choose_next_task
TraceReplaySimulator._reserve_resources = _reserve_resources
TraceReplaySimulator._complete_task = _complete_task
