from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class RouteCandidate:
    route: str
    eligible: bool
    uses_pim: bool
    predicted_finish_ms: float
    predicted_gpu_wait_ms: float
    predicted_pim_wait_ms: float
    predicted_incremental_energy_nj: float = 0.0
    predicted_incremental_decode_energy_nj: float = 0.0
    queue_pressure_ms: float = 0.0
    deadline_ms: Optional[float] = None
    slack_ms: Optional[float] = None
    candidate_mean_tbt_ms: Optional[float] = None
    harmed_count: int = 0
    total_harm_ms: float = 0.0
    max_single_harm_ms: float = 0.0
    own_miss_ms: float = 0.0
    reason: str = ""


@dataclass
class RouteDecision:
    route: Optional[str]
    dropped: bool
    reason: str


class QueueAwareFinishTimePolicy:
    """Admission-time routing policy with optional PIM wait threshold.

    Supported route policies:
    - drop ineligible routes
    - if `pim_wait_threshold_ms` is set, reject PIM routes whose predicted wait
      exceeds the threshold
    - `min_finish`: choose route with smallest predicted finish time
    - `slack_then_finish`: prefer routes that meet deadline; otherwise pick the
      route with the least lateness (largest slack)
    - `latency_guarded_energy`: choose the lowest-energy route within an
      absolute finish-time guard of the fastest predicted route; when no route
      meets the deadline, fall back to `slack_then_finish`
    """

    def __init__(self,
                 pim_wait_threshold_ms: Optional[float] = None,
                 route_policy: str = "min_finish",
                 prefer_gpu_on_tie: bool = True,
                 energy_latency_guard_ms: float = 5.0):
        if route_policy not in {"min_finish", "slack_then_finish", "latency_guarded_energy"}:
            raise ValueError(f"Unsupported route policy '{route_policy}'")
        self.pim_wait_threshold_ms = pim_wait_threshold_ms
        self.route_policy = route_policy
        self.prefer_gpu_on_tie = prefer_gpu_on_tie
        self.energy_latency_guard_ms = max(0.0, float(energy_latency_guard_ms))

    def choose_route(self,
                     candidates: List[RouteCandidate],
                     now_ms: Optional[float] = None,
                     deadline_ms: Optional[float] = None) -> RouteDecision:
        del now_ms

        eligible: List[RouteCandidate] = [c for c in candidates if c.eligible]
        if not eligible:
            return RouteDecision(route=None,
                                 dropped=True,
                                 reason="no_eligible_route")

        filtered: List[RouteCandidate] = []
        for cand in eligible:
            if cand.uses_pim and self.pim_wait_threshold_ms is not None:
                if cand.predicted_pim_wait_ms > self.pim_wait_threshold_ms:
                    continue
            filtered.append(cand)

        if not filtered:
            gpu = next((c for c in eligible if not c.uses_pim), None)
            if gpu is not None:
                return RouteDecision(route=gpu.route,
                                     dropped=False,
                                     reason="fallback_gpu_due_to_pim_threshold")
            # No GPU fallback available.
            return RouteDecision(route=None,
                                 dropped=True,
                                 reason="all_pim_routes_rejected_by_threshold")

        def _finish_sort_key(c: RouteCandidate):
            # Tie-breaker: prefer GPU-only if requested (more predictable tail).
            tie_gpu_bias = 0
            if self.prefer_gpu_on_tie:
                tie_gpu_bias = 1 if c.uses_pim else 0
            return (c.predicted_finish_ms, tie_gpu_bias, c.route)

        if self.route_policy == "min_finish":
            best = min(filtered, key=_finish_sort_key)
            return RouteDecision(route=best.route,
                                 dropped=False,
                                 reason="min_pred_finish")

        def _slack_sort_key(c: RouteCandidate):
            slack = c.slack_ms if c.slack_ms is not None else -math.inf
            return (-slack, *_finish_sort_key(c))

        if self.route_policy == "slack_then_finish":
            if deadline_ms is None:
                best = min(filtered, key=_finish_sort_key)
                return RouteDecision(route=best.route,
                                     dropped=False,
                                     reason="min_pred_finish")

            on_time = [c for c in filtered if c.slack_ms is not None and c.slack_ms >= 0.0]
            if on_time:
                best = min(on_time, key=_finish_sort_key)
                return RouteDecision(route=best.route,
                                     dropped=False,
                                     reason="deadline_feasible_min_finish")

            best = min(filtered, key=_slack_sort_key)
            return RouteDecision(route=best.route,
                                 dropped=False,
                                 reason="max_slack")

        def _energy_sort_key(c: RouteCandidate):
            energy = c.predicted_incremental_energy_nj
            if not math.isfinite(energy):
                energy = c.predicted_incremental_decode_energy_nj
            if not math.isfinite(energy):
                energy = math.inf
            return (energy, *_finish_sort_key(c))

        def _choose_guarded_energy(cands: List[RouteCandidate], reason: str) -> RouteDecision:
            fastest_finish_ms = min(c.predicted_finish_ms for c in cands)
            guarded = [
                c for c in cands
                if c.predicted_finish_ms <= fastest_finish_ms + self.energy_latency_guard_ms + 1e-9
            ]
            best = min(guarded, key=_energy_sort_key)
            return RouteDecision(route=best.route,
                                 dropped=False,
                                 reason=reason)

        if deadline_ms is not None:
            on_time = [c for c in filtered if c.slack_ms is not None and c.slack_ms >= 0.0]
            if on_time:
                return _choose_guarded_energy(on_time, "deadline_feasible_latency_guarded_energy")

            best = min(filtered, key=_slack_sort_key)
            return RouteDecision(route=best.route,
                                 dropped=False,
                                 reason="max_slack")

        return _choose_guarded_energy(filtered, "latency_guarded_energy")
