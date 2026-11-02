"""Unambiguous aggregation for complete-generation measurements."""
from __future__ import annotations

import math
from typing import Any


def summarize_generation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"requests": 0, "total_generated_tokens": 0, "total_generation_sec": 0.0,
                "aggregate_tokens_per_sec": None, "latency_p50_sec": None, "latency_p95_sec": None}
    for row in rows:
        if (type(row.get("generated_tokens")) is not int or row["generated_tokens"] < 0
                or isinstance(row.get("generation_elapsed_sec"), bool)
                or not isinstance(row.get("generation_elapsed_sec"), (int, float))
                or not math.isfinite(row["generation_elapsed_sec"]) or row["generation_elapsed_sec"] <= 0):
            raise ValueError("Each generation needs a nonnegative integer token count and positive finite elapsed time")
    elapsed = sum(row["generation_elapsed_sec"] for row in rows)
    tokens = sum(row["generated_tokens"] for row in rows)
    latencies = sorted(row["generation_elapsed_sec"] for row in rows)

    def percentile(q: float) -> float:
        index = (len(latencies) - 1) * q
        lower, upper = math.floor(index), math.ceil(index)
        return latencies[lower] + (latencies[upper] - latencies[lower]) * (index - lower)

    return {"requests": len(rows), "total_generated_tokens": tokens, "total_generation_sec": elapsed,
            "aggregate_tokens_per_sec": tokens / elapsed,
            "latency_p50_sec": percentile(0.50), "latency_p95_sec": percentile(0.95),
            "latency_min_sec": min(latencies), "latency_max_sec": max(latencies),
            "timing_scope": "complete_generation_excluding_load_and_tokenization",
            "ttft_measured": False, "decode_only_throughput_measured": False}
