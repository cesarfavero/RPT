"""Numerical and timing probe of the real packed Triton operator, without LLM weights."""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

from validation.environment import collect_environment


def _run_probe_impl(*, full_shapes=False, repeats=20, seed=20261102):
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    report = {"schema_version": "rpt-kernel-probe-v1", "seed": seed,
              "scope": "synthetic operator test; not full-model quality or end-to-end speed",
              "environment": collect_environment(), "cases": []}
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("triton") is None:
        return {**report, "status": "blocked", "reason": "torch and triton are required"}
    import numpy as np
    import torch
    if not torch.cuda.is_available():
        return {**report, "status": "blocked", "reason": "No accelerator visible to torch"}
    runtime_dir = Path(__file__).resolve().parents[1] / "gemma4_theory/runtime/packed_recovered_l78"
    sys.path.insert(0, str(runtime_dir))
    from triton_loader import triton_ternary_linear
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    shapes = [(1, 67, 35), (3, 65, 19), (16, 256, 128), (33, 256, 128)]
    if full_shapes:
        shapes += [(m, k, n) for m in (1, 16, 128) for k, n in ((2304, 9216), (9216, 2304))]

    def synchronized_seconds(fn):
        torch.cuda.synchronize()
        start = time.perf_counter()
        value = fn()
        torch.cuda.synchronize()
        return value, time.perf_counter() - start

    for m, k, n in shapes:
        codes = rng.integers(0, 3, size=n * k, dtype=np.uint8)
        padded = np.pad(codes, (0, (-codes.size) % 4))
        packed_np = (padded[0::4] | (padded[1::4] << 2) |
                     (padded[2::4] << 4) | (padded[3::4] << 6))
        packed = torch.from_numpy(packed_np.copy()).to("cuda")
        scale = .03125
        weights_np = np.where(codes == 1, scale, np.where(codes == 2, -scale, 0)).reshape(n, k)
        weight = torch.from_numpy(weights_np).to(device="cuda", dtype=torch.float16)
        x = torch.randn(m, k, device="cuda", dtype=torch.float16)
        dense = lambda: torch.nn.functional.linear(x, weight)
        packed_op = lambda: triton_ternary_linear(x, packed, out_features=n, in_features=k, scale=scale)
        try:
            actual, first_call = synchronized_seconds(packed_op)
            expected, _ = synchronized_seconds(dense)
            torch.testing.assert_close(actual, expected, rtol=.01, atol=.01)
            max_error = (actual.float() - expected.float()).abs().max().item()
            for _ in range(3):
                dense()
                packed_op()
            torch.cuda.synchronize()
            # Alternate order to reduce systematic warm/cache bias.
            dense_times, packed_times = [], []
            for index in range(repeats):
                ordered = [(dense, dense_times), (packed_op, packed_times)]
                if index % 2:
                    ordered.reverse()
                for fn, times in ordered:
                    _, duration = synchronized_seconds(fn)
                    times.append(duration)
            dense_median = statistics.median(dense_times)
            packed_median = statistics.median(packed_times)
            report["cases"].append({"shape_mkn": [m, k, n], "status": "passed",
                "dtype": "float16", "scale": scale, "max_absolute_error": max_error,
                "tolerance": {"rtol": .01, "atol": .01}, "first_call_seconds": first_call,
                "first_call_scope": "includes compile/cache lookup; disk cache may already exist",
                "warm_repeats": repeats, "dense_median_seconds": dense_median,
                "triton_median_seconds": packed_median, "speedup_vs_dense": dense_median / packed_median,
                "packed_storage_bytes": packed.numel(), "dense_storage_bytes": weight.numel() * weight.element_size()})
        except Exception as error:
            report["cases"].append({"shape_mkn": [m, k, n], "status": "failed",
                                    "error_type": type(error).__name__, "error": str(error)[:2000]})
            report["status"] = "failed"
            return report
    return {**report, "status": "passed", "backend_executed": "triton",
            "full_model_validated": False}


def run_probe(*, full_shapes=False, repeats=20, seed=20261102):
    try:
        return _run_probe_impl(full_shapes=full_shapes, repeats=repeats, seed=seed)
    except Exception as error:
        # Import errors from vendor shared libraries and allocation failures must
        # leave a structured record just like numerical mismatches.
        return {"schema_version": "rpt-kernel-probe-v1", "status": "failed", "seed": seed,
                "stage": "initialization_or_kernel", "error_type": type(error).__name__,
                "error": str(error)[:2000], "environment": collect_environment(probe_torch=False),
                "cases": [], "full_model_validated": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full-shapes", action="store_true")
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args(argv)
    result = run_probe(full_shapes=args.full_shapes, repeats=args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "report": str(args.output), "reason": result.get("reason")}))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
