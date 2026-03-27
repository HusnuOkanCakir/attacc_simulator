from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .types import LookupResult, RequestServiceEstimate, RouteServicePoint

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
