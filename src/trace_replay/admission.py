from __future__ import annotations

import math
from typing import Dict, List
from src.scheduler_policy import RouteCandidate
from .types import ReplayRequest, RequestServiceEstimate

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
            own_checks_enabled = bool(self.config.admission_check_own_slo)
            own_e2e_ok = ((not own_checks_enabled) or
                          (rr.deadline_ms is None) or
                          (finish <= rr.deadline_ms))
            own_tbt_ok = ((not own_checks_enabled) or
                          (self.config.slo_tbt_ms is None) or
                          (float(proj["candidate_mean_tbt_ms"]) <= float(self.config.slo_tbt_ms)))
            existing_ok = bool(proj["existing_non_lost_feasible"])
            own_feasible_flags.append(own_e2e_ok)
            eligible = own_e2e_ok and own_tbt_ok and existing_ok
            reason = ""
            if own_checks_enabled and not own_e2e_ok:
                reason = "candidate_e2e_slo_miss"
            elif own_checks_enabled and not own_tbt_ok:
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
