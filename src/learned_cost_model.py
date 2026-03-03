from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from src.trace_replay_sim import (LookupResult, RequestServiceEstimate,
                                  RouteServicePoint)


TARGET_NAMES = (
    "prefill_e2e_ms",
    "prefill_gpu_ms",
    "prefill_pim_ms",
    "decode_e2e_ms",
    "decode_gpu_ms",
    "decode_pim_ms",
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
                             decode_e2e_ms=vals["decode_e2e_ms"],
                             decode_gpu_ms=vals["decode_gpu_ms"],
                             decode_pim_ms=vals["decode_pim_ms"],
                             source_file=source_file)


class LearnedCostModel:
    """Predict request service costs from lightweight ML regressors.

    The model is intended for interpolation inside a bounded operating
    region. Outside training bounds, configure clipping/drop behavior.
    """

    def __init__(self,
                 route_bundles: Dict[str, LearnedRouteBundle]):
        self.route_bundles = route_bundles

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

    def _estimate_from_bundle(self,
                              route: str,
                              bundle: LearnedRouteBundle,
                              context_tokens: int,
                              generated_tokens: int,
                              bs: int,
                              unsupported_policy: str,
                              lin_bucket: int,
                              lout_bucket: int) -> Optional[RequestServiceEstimate]:
        req_lin = int(context_tokens)
        req_lout = int(generated_tokens)
        req_bs = int(bs)
        if req_lout <= 0:
            return None

        # For learned models we should predict on the raw request shape.
        # Bucketing is useful for sparse table lookup, but it defeats the point
        # of ML interpolation when the request is already inside the trained
        # region. We therefore only clip outside the trained bounds.
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

        feats = pd.DataFrame([{
            "Lin": float(lin),
            "Lout": float(lout),
            "bs": float(bsize),
        }], columns=list(FEATURE_NAMES))
        preds: Dict[str, float] = {}
        for target_name in bundle.target_names:
            model = bundle.estimators[target_name]
            pred = float(model.predict(feats)[0])
            if math.isnan(pred) or math.isinf(pred):
                pred = 0.0
            preds[target_name] = pred

        point = _sanitize_predicted_point(route=route,
                                          lin=lin,
                                          lout=lout,
                                          bs=bsize,
                                          preds=preds,
                                          source_file=bundle.source_file)
        lookup = LookupResult(point=point,
                              requested_lin=req_lin,
                              requested_lout=req_lout,
                              mapped_lin=int(lin),
                              mapped_lout=int(lout),
                              requested_bs=req_bs,
                              mapped_bs=int(bsize),
                              clipped_lin=clipped_lin,
                              clipped_lout=(clipped_lout or clipped_bs))
        decode_tokens = max(0, req_lout - 1)
        return RequestServiceEstimate(route=route,
                                      uses_pim=point.uses_pim,
                                      lin=req_lin,
                                      lout=req_lout,
                                      generated_tokens=req_lout,
                                      decode_tokens=decode_tokens,
                                      prefill_e2e_ms=point.prefill_e2e_ms,
                                      prefill_gpu_ms=point.prefill_gpu_ms,
                                      prefill_pim_ms=point.prefill_pim_ms,
                                      decode_e2e_ms=point.decode_e2e_ms,
                                      decode_gpu_ms=point.decode_gpu_ms,
                                      decode_pim_ms=point.decode_pim_ms,
                                      mapping=lookup)

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
        return self._estimate_from_bundle(route=route,
                                          bundle=bundle,
                                          context_tokens=context_tokens,
                                          generated_tokens=generated_tokens,
                                          bs=bs,
                                          unsupported_policy=unsupported_policy,
                                          lin_bucket=lin_bucket,
                                          lout_bucket=lout_bucket)
