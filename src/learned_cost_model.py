from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from src.trace_replay.types import (LookupResult, RequestServiceEstimate,
                                    RouteServicePoint)


TARGET_NAMES = (
    "prefill_e2e_ms",
    "prefill_gpu_ms",
    "prefill_pim_ms",
    "prefill_energy_nj",
    "decode_e2e_ms",
    "decode_gpu_ms",
    "decode_pim_ms",
    "decode_energy_nj",
)
FEATURE_NAMES = ("Lin", "Lout", "bs")


@dataclass
class LearnedRouteBundle:
    route: str
    feature_names: Sequence[str]
    target_names: Sequence[str]
    bounds: Dict[str, Tuple[float, float]]
    estimators: Dict[str, object]
    source_file: str
    metadata: Dict[str, object]


def _bucket(value: int, bucket: int) -> int:
    if bucket <= 1:
        return int(value)
    return max(1, int(round(value / bucket) * bucket))


def _clip_to_bounds(value: float, bounds: Tuple[float, float]) -> Tuple[float, bool]:
    lo, hi = bounds
    clipped = min(max(value, lo), hi)
    return clipped, (clipped != value)


def _sanitize_predicted_point(route: str,
                              lin: int,
                              lout: int,
                              bs: int,
                              preds: Dict[str, float],
                              source_file: str) -> RouteServicePoint:
    vals = {k: max(0.0, float(preds.get(k, 0.0))) for k in TARGET_NAMES}
    vals["prefill_e2e_ms"] = max(vals["prefill_e2e_ms"],
                                 vals["prefill_gpu_ms"],
                                 vals["prefill_pim_ms"])
    vals["decode_e2e_ms"] = max(vals["decode_e2e_ms"],
                                vals["decode_gpu_ms"],
                                vals["decode_pim_ms"])
    return RouteServicePoint(route=route,
                             lin=int(lin),
                             lout=int(lout),
                             bs=int(bs),
                             prefill_e2e_ms=vals["prefill_e2e_ms"],
                             prefill_gpu_ms=vals["prefill_gpu_ms"],
                             prefill_pim_ms=vals["prefill_pim_ms"],
                             prefill_energy_nj=vals["prefill_energy_nj"],
                             decode_e2e_ms=vals["decode_e2e_ms"],
                             decode_gpu_ms=vals["decode_gpu_ms"],
                             decode_pim_ms=vals["decode_pim_ms"],
                             decode_energy_nj=vals["decode_energy_nj"],
                             source_file=source_file)


class LearnedCostModel:
    """Predict request service costs from lightweight ML regressors.

    The model is intended for interpolation inside a bounded operating
    region. Outside training bounds, configure clipping/drop behavior.
    """

    def __init__(self,
                 route_bundles: Dict[str, LearnedRouteBundle]):
        self.route_bundles = route_bundles
        self._stats: Dict[str, float] = {
            "single_estimate_calls": 0.0,
            "batch_estimate_calls": 0.0,
            "batch_estimate_rows": 0.0,
            "target_predict_calls": 0.0,
            "target_predict_rows": 0.0,
        }

    @classmethod
    def from_pickles(cls,
                     route_to_model: Dict[str, str]) -> "LearnedCostModel":
        bundles: Dict[str, LearnedRouteBundle] = {}
        for route, model_path in route_to_model.items():
            with open(model_path, "rb") as f:
                obj = pickle.load(f)
            bundle = LearnedRouteBundle(route=obj["route"],
                                        feature_names=tuple(obj["feature_names"]),
                                        target_names=tuple(obj["target_names"]),
                                        bounds=dict(obj["bounds"]),
                                        estimators=dict(obj["estimators"]),
                                        source_file=str(model_path),
                                        metadata=dict(obj.get("metadata", {})))
            if bundle.route != route:
                raise ValueError(
                    f"Route mismatch: mapping requested '{route}' but model stores '{bundle.route}'")
            bundles[route] = bundle
        return cls(route_bundles=bundles)

    def routes(self) -> List[str]:
        return sorted(self.route_bundles.keys())

    def _prepare_request(self,
                         bundle: LearnedRouteBundle,
                         context_tokens: int,
                         generated_tokens: int,
                         bs: int,
                         unsupported_policy: str) -> Optional[Dict[str, object]]:
        req_lin = int(context_tokens)
        req_lout = int(generated_tokens)
        req_bs = int(bs)
        if req_lout <= 0:
            return None

        lin = req_lin
        lout = req_lout
        bsize = req_bs

        clipped_lin = False
        clipped_lout = False
        clipped_bs = False

        in_bounds = True
        feature_vals = {
            "Lin": float(lin),
            "Lout": float(lout),
            "bs": float(bsize),
        }
        for name in FEATURE_NAMES:
            bounds = bundle.bounds[name]
            v = feature_vals[name]
            if not (bounds[0] <= v <= bounds[1]):
                in_bounds = False
                break

        if not in_bounds:
            if unsupported_policy == "drop":
                return None
            lin, clipped_lin = _clip_to_bounds(float(lin), bundle.bounds["Lin"])
            lout, clipped_lout = _clip_to_bounds(float(lout), bundle.bounds["Lout"])
            bsize, clipped_bs = _clip_to_bounds(float(bsize), bundle.bounds["bs"])
            lin = int(round(lin))
            lout = int(round(lout))
            bsize = int(round(bsize))

        return {
            "requested_lin": req_lin,
            "requested_lout": req_lout,
            "requested_bs": req_bs,
            "lin": int(lin),
            "lout": int(lout),
            "bs": int(bsize),
            "clipped_lin": clipped_lin,
            "clipped_lout": (clipped_lout or clipped_bs),
        }

    def _estimate_from_bundle_batch(self,
                                    route: str,
                                    bundle: LearnedRouteBundle,
                                    requests: Sequence[Tuple[int, int, int]],
                                    unsupported_policy: str,
                                    lin_bucket: int,
                                    lout_bucket: int) -> List[Optional[RequestServiceEstimate]]:
        del lin_bucket, lout_bucket
        results: List[Optional[RequestServiceEstimate]] = [None] * len(requests)
        prepared_rows: List[Dict[str, object]] = []
        prepared_indexes: List[int] = []
        for idx, (context_tokens, generated_tokens, bs) in enumerate(requests):
            prepared = self._prepare_request(bundle,
                                             context_tokens=context_tokens,
                                             generated_tokens=generated_tokens,
                                             bs=bs,
                                             unsupported_policy=unsupported_policy)
            if prepared is None:
                continue
            prepared_rows.append(prepared)
            prepared_indexes.append(idx)

        if not prepared_rows:
            return results

        feats = pd.DataFrame([
            {
                "Lin": float(row["lin"]),
                "Lout": float(row["lout"]),
                "bs": float(row["bs"]),
            }
            for row in prepared_rows
        ], columns=list(FEATURE_NAMES))

        self._stats["batch_estimate_calls"] += 1.0
        self._stats["batch_estimate_rows"] += float(len(prepared_rows))

        pred_arrays: Dict[str, Sequence[float]] = {}
        for target_name in bundle.target_names:
            model = bundle.estimators[target_name]
            preds = model.predict(feats)
            self._stats["target_predict_calls"] += 1.0
            self._stats["target_predict_rows"] += float(len(prepared_rows))
            pred_arrays[target_name] = preds

        for row_idx, prepared in enumerate(prepared_rows):
            preds: Dict[str, float] = {}
            for target_name in bundle.target_names:
                pred = float(pred_arrays[target_name][row_idx])
                if math.isnan(pred) or math.isinf(pred):
                    pred = 0.0
                preds[target_name] = pred

            point = _sanitize_predicted_point(route=route,
                                              lin=int(prepared["lin"]),
                                              lout=int(prepared["lout"]),
                                              bs=int(prepared["bs"]),
                                              preds=preds,
                                              source_file=bundle.source_file)
            lookup = LookupResult(point=point,
                                  requested_lin=int(prepared["requested_lin"]),
                                  requested_lout=int(prepared["requested_lout"]),
                                  mapped_lin=int(prepared["lin"]),
                                  mapped_lout=int(prepared["lout"]),
                                  requested_bs=int(prepared["requested_bs"]),
                                  mapped_bs=int(prepared["bs"]),
                                  clipped_lin=bool(prepared["clipped_lin"]),
                                  clipped_lout=bool(prepared["clipped_lout"]))
            req_lout = int(prepared["requested_lout"])
            decode_tokens = max(0, req_lout - 1)
            results[prepared_indexes[row_idx]] = RequestServiceEstimate(
                route=route,
                uses_pim=point.uses_pim,
                lin=int(prepared["requested_lin"]),
                lout=req_lout,
                generated_tokens=req_lout,
                decode_tokens=decode_tokens,
                prefill_e2e_ms=point.prefill_e2e_ms,
                prefill_gpu_ms=point.prefill_gpu_ms,
                prefill_pim_ms=point.prefill_pim_ms,
                prefill_energy_nj=point.prefill_energy_nj,
                decode_e2e_ms=point.decode_e2e_ms,
                decode_gpu_ms=point.decode_gpu_ms,
                decode_pim_ms=point.decode_pim_ms,
                decode_energy_nj=point.decode_energy_nj,
                mapping=lookup,
            )
        return results

    def supports_decode_energy(self, route: Optional[str] = None) -> bool:
        if route is not None:
            bundle = self.route_bundles.get(route)
            return bundle is not None and "decode_energy_nj" in bundle.target_names
        return all("decode_energy_nj" in bundle.target_names
                   for bundle in self.route_bundles.values())

    def supports_prefill_energy(self, route: Optional[str] = None) -> bool:
        if route is not None:
            bundle = self.route_bundles.get(route)
            return bundle is not None and "prefill_energy_nj" in bundle.target_names
        return all("prefill_energy_nj" in bundle.target_names
                   for bundle in self.route_bundles.values())

    def estimate_request(self,
                         route: str,
                         context_tokens: int,
                         generated_tokens: int,
                         bs: int,
                         unsupported_policy: str,
                         lin_bucket: int,
                         lout_bucket: int) -> Optional[RequestServiceEstimate]:
        bundle = self.route_bundles.get(route)
        if bundle is None:
            return None
        self._stats["single_estimate_calls"] += 1.0
        return self._estimate_from_bundle_batch(
            route=route,
            bundle=bundle,
            requests=[(context_tokens, generated_tokens, bs)],
            unsupported_policy=unsupported_policy,
            lin_bucket=lin_bucket,
            lout_bucket=lout_bucket,
        )[0]

    def estimate_requests_batch(self,
                                route: str,
                                requests: Sequence[Tuple[int, int, int]],
                                unsupported_policy: str,
                                lin_bucket: int,
                                lout_bucket: int) -> List[Optional[RequestServiceEstimate]]:
        bundle = self.route_bundles.get(route)
        if bundle is None:
            return [None] * len(requests)
        return self._estimate_from_bundle_batch(route=route,
                                                bundle=bundle,
                                                requests=requests,
                                                unsupported_policy=unsupported_policy,
                                                lin_bucket=lin_bucket,
                                                lout_bucket=lout_bucket)

    def stats(self) -> Dict[str, float]:
        return dict(self._stats)
