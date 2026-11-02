"""Backend execution evidence, independent of torch and accelerator libraries."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def summarize_execution(
    layers: Iterable[tuple[str, Mapping[str, int]]],
    *,
    require_triton: bool = False,
    expected_layers: int = 78,
) -> dict[str, Any]:
    rows = [{"name": name, "triton": int(counts.get("triton", 0)),
             "pytorch": int(counts.get("pytorch", 0))} for name, counts in layers]
    totals = {backend: sum(row[backend] for row in rows) for backend in ("triton", "pytorch")}
    unexecuted = [row["name"] for row in rows if row["triton"] + row["pytorch"] == 0]
    observed = [key for key, count in totals.items() if count]
    backend = observed[0] if len(observed) == 1 else "mixed" if observed else "unobserved"
    all_executed = len(rows) == expected_layers and not unexecuted
    packed_triton = all_executed and backend == "triton"
    result = {
        "backend": backend,
        "backend_counts": totals,
        "total_layers": len(rows),
        "expected_layers": expected_layers,
        "layers_executed": len(rows) - len(unexecuted),
        "all_layers_executed": all_executed,
        "unexecuted_layers": unexecuted,
        "fallback_count": totals["pytorch"],
        "per_layer": rows,
        "is_native_ternary_runtime": packed_triton,
        "packed_triton_execution_verified": packed_triton,
        "is_optimized_kernel": None,
        "kernel_implementation": "packed_decode_then_dense_dot",
        "performance_validated": False,
    }
    if require_triton and not packed_triton:
        raise RuntimeError(f"Triton execution not verified: layers={len(rows)}/{expected_layers}, "
                           f"unexecuted={unexecuted}, fallback_count={totals['pytorch']}")
    return result
