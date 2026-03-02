from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd


@dataclass
class AzureTraceRequest:
    request_id: int
    arrival_ms: float
    context_tokens: int
    generated_tokens: int


def load_azure_llm_trace(
    path_or_url: str,
    limit: Optional[int] = None,
    start_offset: int = 0,
    arrival_time_scale: float = 1.0,
    min_context_tokens: int = 1,
    min_generated_tokens: int = 1,
) -> List[AzureTraceRequest]:
    """Load Azure LLM inference trace and normalize timestamps to t=0 in ms.

    `arrival_time_scale` multiplies inter-arrival times (and absolute normalized
    arrival timestamps). Values < 1.0 increase offered load.
    """
    df = pd.read_csv(path_or_url, parse_dates=["TIMESTAMP"])
    required_cols = {"TIMESTAMP", "ContextTokens", "GeneratedTokens"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Missing required Azure trace columns: {sorted(missing)}")

    df = df.sort_values("TIMESTAMP").reset_index(drop=True)
    if start_offset:
        df = df.iloc[start_offset:]
    if limit is not None:
        df = df.iloc[:limit]
    df = df.reset_index(drop=True)
    if df.empty:
        return []

    # Sanitize token counts.
    df["ContextTokens"] = pd.to_numeric(df["ContextTokens"], errors="coerce").fillna(0).astype(int)
    df["GeneratedTokens"] = pd.to_numeric(df["GeneratedTokens"], errors="coerce").fillna(0).astype(int)
    df = df[(df["ContextTokens"] >= min_context_tokens) &
            (df["GeneratedTokens"] >= min_generated_tokens)].copy()
    df = df.reset_index(drop=True)
    if df.empty:
        return []

    t0 = df.loc[0, "TIMESTAMP"]
    arrivals_ms = (df["TIMESTAMP"] - t0).dt.total_seconds() * 1000.0
    arrivals_ms = arrivals_ms * float(arrival_time_scale)

    out: List[AzureTraceRequest] = []
    for idx, row in df.iterrows():
        out.append(
            AzureTraceRequest(
                request_id=int(idx),
                arrival_ms=float(arrivals_ms.iloc[idx]),
                context_tokens=int(row["ContextTokens"]),
                generated_tokens=int(row["GeneratedTokens"]),
            ))
    return out
