"""Metric aggregation shared by benchmarks; no model or GPU dependencies."""
from __future__ import annotations

import math
from statistics import NormalDist


def percentile(values, p):
    if not values or not 0 <= p <= 1:
        raise ValueError("Nonempty values and p in [0, 1] required")
    values = sorted(float(x) for x in values)
    if not all(math.isfinite(x) for x in values):
        raise ValueError("Non-finite observations")
    index = (len(values) - 1) * p
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def generation_summary(rows):
    if not rows:
        raise ValueError("No measured requests")
    for row in rows:
        if type(row["elapsed_seconds"]) not in (int, float) or not math.isfinite(row["elapsed_seconds"]) or row["elapsed_seconds"] <= 0:
            raise ValueError("Elapsed time must be finite and positive")
        if type(row["output_tokens"]) is not int or row["output_tokens"] < 0:
            raise ValueError("Output token count must be a nonnegative integer")
        ttft = row.get("ttft_seconds")
        if ttft is not None and (type(ttft) not in (int, float) or not math.isfinite(ttft) or ttft < 0):
            raise ValueError("Measured TTFT must be finite and nonnegative")
    elapsed = sum(row["elapsed_seconds"] for row in rows)
    tokens = sum(row["output_tokens"] for row in rows)
    latencies = [row["elapsed_seconds"] for row in rows]
    ttft = [row["ttft_seconds"] for row in rows if row.get("ttft_seconds") is not None]
    return {
        "requests": len(rows), "output_tokens": tokens, "total_generation_seconds": elapsed,
        "aggregate_tokens_per_second": tokens / elapsed,
        "mean_request_tokens_per_second": sum(row["output_tokens"] / row["elapsed_seconds"] for row in rows) / len(rows),
        "latency_seconds": {"p50": percentile(latencies, .5), "p95": percentile(latencies, .95),
                            "min": min(latencies), "max": max(latencies)},
        "ttft_seconds": ({"p50": percentile(ttft, .5), "p95": percentile(ttft, .95)} if ttft else None),
        "ttft_measured_requests": len(ttft),
        "ttft_scope": "Caller must record the timing origin; absent TTFT is never inferred from whole-request time.",
        "aggregation": "sum(actual_output_tokens) / sum(synchronized_generation_wall_seconds)",
    }


def wilson_interval(successes, total, confidence=.95):
    if type(successes) is not int or type(total) is not int or total <= 0 or not 0 <= successes <= total:
        raise ValueError("Require integer counts 0 <= successes <= total and total > 0")
    if not 0 < confidence < 1:
        raise ValueError("Confidence must be between 0 and 1")
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0., center - radius), min(1., center + radius)]


def paired_correctness(reference, candidate):
    """Compare matching item IDs; lost-correct risk cannot be offset by new wins."""
    def indexed(rows):
        result = {}
        for row in rows:
            key = str(row["id"])
            if key in result or type(row["is_correct"]) is not bool:
                raise ValueError("Duplicate ID or non-boolean correctness")
            result[key] = row["is_correct"]
        return result
    ref, cand = indexed(reference), indexed(candidate)
    if not ref or ref.keys() != cand.keys():
        raise ValueError("Paired comparison requires identical nonempty ID sets")
    n = len(ref)
    lost = sum(ref[k] and not cand[k] for k in ref)
    gained = sum(not ref[k] and cand[k] for k in ref)
    discordant = lost + gained
    if discordant:
        # Exact binomial McNemar two-sided p, log-domain to avoid huge integers.
        log_terms = [math.lgamma(discordant + 1) - math.lgamma(k + 1) -
                     math.lgamma(discordant - k + 1) - discordant * math.log(2)
                     for k in range(min(lost, gained) + 1)]
        p_value = min(1., 2 * sum(math.exp(term) for term in log_terms))
    else:
        p_value = 1.
    return {
        "items": n, "reference_correct": sum(ref.values()), "candidate_correct": sum(cand.values()),
        "accuracy_delta": (gained - lost) / n, "lost_correct": lost, "gained_correct": gained,
        "lost_correct_risk": lost / n, "lost_correct_risk_wilson_95": wilson_interval(lost, n),
        "mcnemar_exact_two_sided_p": p_value,
        "scope": "Descriptive paired evaluation; no automatic non-inferiority or novelty claim.",
    }
