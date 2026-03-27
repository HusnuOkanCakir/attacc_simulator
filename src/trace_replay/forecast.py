from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple
from src.azure_trace import AzureTraceRequest
from src.scheduler_policy import RouteCandidate
from .types import ReplayRequest, RequestServiceEstimate, _IncrementalForecastRequest, _IncrementalForecastRun

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
    own_checks_enabled = bool(self.config.admission_check_own_slo)
    own_ok = ((not own_checks_enabled) or
              (own_miss_ms <= budgets["max_own_miss_ms"]))
    tbt_ok = ((not own_checks_enabled) or
              self.config.slo_tbt_ms is None or
              (candidate_mean_tbt <= float(self.config.slo_tbt_ms)))
    harmed_ok = harmed_count <= budgets["max_harmed_requests"]
    total_harm_ok = total_harm_ms <= budgets["max_total_harm_ms"]
    single_harm_ok = max_single_harm_ms <= budgets["max_single_harm_ms"]
    eligible = own_ok and tbt_ok and harmed_ok and total_harm_ok and single_harm_ok

    if own_checks_enabled and not own_ok:
        reason = "candidate_e2e_slo_miss"
    elif own_checks_enabled and not tbt_ok:
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
