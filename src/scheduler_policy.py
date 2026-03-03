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
    queue_pressure_ms: float = 0.0
    deadline_ms: Optional[float] = None
    slack_ms: Optional[float] = None
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
    """

    def __init__(self,
                 pim_wait_threshold_ms: Optional[float] = None,
                 route_policy: str = "min_finish",
                 prefer_gpu_on_tie: bool = True):
        if route_policy not in {"min_finish", "slack_then_finish"}:
            raise ValueError(f"Unsupported route policy '{route_policy}'")
        self.pim_wait_threshold_ms = pim_wait_threshold_ms
        self.route_policy = route_policy
        self.prefer_gpu_on_tie = prefer_gpu_on_tie

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

        if self.route_policy != "slack_then_finish" or deadline_ms is None:
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

        def _slack_sort_key(c: RouteCandidate):
            slack = c.slack_ms if c.slack_ms is not None else -math.inf
            return (-slack, *_finish_sort_key(c))

        best = min(filtered, key=_slack_sort_key)
        return RouteDecision(route=best.route,
                             dropped=False,
                             reason="max_slack")
