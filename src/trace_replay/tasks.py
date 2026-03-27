from __future__ import annotations

from typing import Dict, List, Optional
from .types import ReplayRequest, RequestServiceEstimate, _TaskCandidate

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

