"""Run local synthetic CPU experiments. Never downloads or evaluates an LLM."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import time
import tracemalloc
from datetime import datetime, timezone

import numpy as np

from .projection import TernaryResidualProjection


def _positive(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(str(array.shape).encode())
        digest.update(array.dtype.str.encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _timed_calls(function, hidden_batch):
    timings = []
    for hidden in hidden_batch:
        start = time.perf_counter_ns()
        function(hidden)
        timings.append(time.perf_counter_ns() - start)
    return timings


def _timing_summary(samples):
    array = np.asarray(samples, dtype=np.float64)
    seconds = float(np.sum(array) / 1e9)
    return {
        "calls": len(samples),
        "total_seconds": seconds,
        "median_call_ms": float(np.median(array) / 1e6),
        "p95_call_ms": float(np.percentile(array, 95) / 1e6),
        "decisions_per_second": len(samples) / seconds,
    }


def _peak_traced_bytes(function, hidden_batch):
    gc.collect()
    tracemalloc.start()
    try:
        for hidden in hidden_batch:
            function(hidden)
        _, peak = tracemalloc.get_traced_memory()
        return peak
    finally:
        tracemalloc.stop()


def _add_counts(total, current):
    for key, value in current.items():
        total[key] = total.get(key, 0) + value


def _case(name, *, seed, rows, width, planes, samples, repeats):
    rng = np.random.default_rng(seed)
    build_start = time.perf_counter_ns()
    codes = rng.integers(-1, 2, size=(planes, rows, width), dtype=np.int8)
    if name == "small_corrections":
        scales = np.power(0.02, np.arange(planes))
        model = TernaryResidualProjection(codes, scales)
        construction = "Random ternary codes; plane scales 0.02**plane"
    elif name == "substantial_corrections":
        scales = np.power(0.5, np.arange(planes))
        model = TernaryResidualProjection(codes, scales)
        construction = "Random ternary codes; plane scales 0.5**plane"
    elif name == "exact_ties":
        codes[:] = codes[:, :1, :]
        scales = np.power(0.25, np.arange(planes))
        model = TernaryResidualProjection(codes, scales)
        construction = "Every output row identical; strict certificate must defer"
    elif name == "greedy_dense_approximation":
        dense = rng.normal(size=(rows, width))
        model = TernaryResidualProjection.from_dense(dense, nplanes=planes)
        construction = "Synthetic Gaussian dense matrix; greedy absmax residual planes"
    else:
        raise ValueError(name)
    build_seconds = (time.perf_counter_ns() - build_start) / 1e9
    hidden_batch = rng.normal(size=(samples, width))
    full_work, adaptive_work = {}, {}
    stage_histogram = {str(stage): 0 for stage in range(1, planes + 1)}
    mismatches, exits, fallbacks = 0, 0, 0
    verify_start = time.perf_counter_ns()
    for hidden in hidden_batch:
        full = model.full_reference(hidden)
        adaptive = model.predict(hidden)
        mismatches += full.token_id != adaptive.token_id
        exits += adaptive.early_exit
        fallbacks += adaptive.used_full_fallback
        stage_histogram[str(adaptive.planes_executed)] += 1
        _add_counts(full_work, full.logical_work)
        _add_counts(adaptive_work, adaptive.logical_work)
    verification_seconds = (time.perf_counter_ns() - verify_start) / 1e9
    if mismatches:
        raise AssertionError(f"{name}: {mismatches} incorrect certified/fallback argmax values")
    # Validation above also warms the interpreter/caches. Explicit, equal extra
    # warmup calls are reported and excluded from steady-state timings.
    warmup = min(4, samples)
    for hidden in hidden_batch[:warmup]:
        model.full_reference(hidden)
        model.predict(hidden)
    full_times, adaptive_times = [], []
    for repeat in range(repeats):
        calls = (
            [(model.full_reference, full_times), (model.predict, adaptive_times)]
            if repeat % 2 == 0 else
            [(model.predict, adaptive_times), (model.full_reference, full_times)]
        )
        for function, target in calls:
            target.extend(_timed_calls(function, hidden_batch))
    full_timing, adaptive_timing = _timing_summary(full_times), _timing_summary(adaptive_times)
    full_peak = _peak_traced_bytes(model.full_reference, hidden_batch[:min(samples, 16)])
    adaptive_peak = _peak_traced_bytes(model.predict, hidden_batch[:min(samples, 16)])
    return {
        "name": name,
        "seed": seed,
        "classification": "synthetic_cpu_reference_only",
        "construction": construction,
        "shape": {"output_rows": rows, "hidden_size": width, "planes": planes},
        "samples": samples,
        "timing_repeats": repeats,
        "extra_warmup_calls_per_method": warmup,
        "model_build_seconds": build_seconds,
        "verification_seconds": verification_seconds,
        "model_sha256": _sha256(model.codes, model.scales, model.bias),
        "hidden_batch_sha256": _sha256(hidden_batch),
        "argmax_mismatches_against_complete_representation": int(mismatches),
        "early_certified_decisions": int(exits),
        "full_fallback_decisions": int(fallbacks),
        "early_exit_fraction": exits / samples,
        "planes_executed_histogram": stage_histogram,
        "timing": {
            "complete_ordered_float64_reference": full_timing,
            "adaptive_including_all_bound_and_decision_work": adaptive_timing,
            "complete_reference_seconds_divided_by_adaptive_seconds":
                full_timing["total_seconds"] / adaptive_timing["total_seconds"],
            "interpretation": "Local interpreted NumPy timings, not a production-kernel or LLM speedup claim",
        },
        "memory": {
            "persistent_model_arrays_bytes": model.storage_bytes(),
            "supplied_hidden_batch_bytes": hidden_batch.nbytes,
            "full_peak_traced_call_allocations_bytes": full_peak,
            "adaptive_peak_traced_call_allocations_bytes": adaptive_peak,
            "trace_scope": (
                "Peak traced Python/NumPy allocations during up to 16 complete calls; "
                "persistent arrays and supplied inputs allocated before tracing are reported separately. "
                "This is not GPU memory, complete native-library memory, or process RSS."
            ),
        },
        "logical_work_for_one_pass_over_all_samples": {
            "complete_reference": full_work,
            "adaptive": adaptive_work,
            "interpretation": (
                "Defined scalar reference-algorithm primitives, not hardware FLOPs or DRAM traffic. "
                "Counts include bound-table construction, metadata reads, outward rounding and dominance work. "
                "Python control/object costs are included in wall-clock timing, not in these counters."
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--rows", type=_positive, default=128)
    parser.add_argument("--hidden-size", type=_positive, default=64)
    parser.add_argument("--planes", type=_positive, default=4)
    parser.add_argument("--samples", type=_positive, default=256)
    parser.add_argument("--repeats", type=_positive, default=3)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results" / "reference_cpu.json")
    args = parser.parse_args()
    if args.rows < 2:
        parser.error("--rows must be at least 2 for the comparison fixtures")
    wall_start = time.perf_counter_ns()
    cases = [
        _case(name, seed=args.seed + index, rows=args.rows, width=args.hidden_size,
              planes=args.planes, samples=args.samples, repeats=args.repeats)
        for index, name in enumerate((
            "small_corrections", "substantial_corrections", "exact_ties", "greedy_dense_approximation"
        ))
    ]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    source_hashes = {
        name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
        for name in ("projection.py", "run_experiment.py")
    }
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss_bytes = int(usage.ru_maxrss * (1 if platform.system() == "Darwin" else 1024))
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "synthetic_reference_experiment_completed",
        "actual_llm_weights_evaluated": False,
        "external_language_benchmark_evaluated": False,
        "accuracy_or_reasoning_quality_claim": False,
        "new_method_priority_claim": False,
        "production_speedup_demonstrated": False,
        "certificate_scope": (
            "Greedy argmax equality for identical hidden vector and all planes of the same representation, "
            "using the documented ordered float64 reference; no source-model, generation-distribution, "
            "sequence-level, semantic, or optimized-kernel guarantee"
        ),
        "environment": {
            "python": platform.python_version(), "numpy": np.__version__,
            "platform": platform.platform(), "machine": platform.machine(),
            "cpu_logical_count": os.cpu_count(), "device": "CPU only",
            "baseline_git_commit": commit, "source_sha256": source_hashes,
        },
        "total_run_wall_seconds": (time.perf_counter_ns() - wall_start) / 1e9,
        "whole_process_peak_rss_bytes": rss_bytes,
        "energy_measured": False,
        "cost_accounting": (
            "Per-call timing includes validation, projections, norm, bounds, outward rounding, selection "
            "and fallback. Model construction, verification, warmup and full-run time are separate. "
            "All persistent ndarray storage is counted; traced temporary allocations and process RSS "
            "are reported with their scope. No claim about DRAM bytes, GPU occupancy, or energy is made."
        ),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output), "status": report["status"],
        "cases": [{"name": c["name"], "mismatches": c["argmax_mismatches_against_complete_representation"],
                   "early_exit_fraction": c["early_exit_fraction"],
                   "reference_time_ratio": c["timing"]["complete_reference_seconds_divided_by_adaptive_seconds"]}
                  for c in cases],
    }, indent=2))


if __name__ == "__main__":
    main()
