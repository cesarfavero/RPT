"""A conservative reference algorithm, not a production inference kernel.

Only greedy argmax of the *same complete representation and hidden vector* is
certified.  The complete reference uses ordered float64 accumulation; an
unrelated BLAS, CUDA, FP16 or fused-weight implementation is not that reference.
See README.md for the bound, assumptions, accounting, and excluded claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


_F64 = np.dtype(np.float64)
_POS_INF = np.float64(np.inf)
_NEG_INF = np.float64(-np.inf)


@dataclass(frozen=True)
class Certificate:
    """Strict interval dominance, relative to the defined full reference."""

    winner_lower: float
    competitors_upper: float | None
    planes_executed: int
    planes_total: int


@dataclass(frozen=True)
class Decision:
    token_id: int
    planes_executed: int
    planes_total: int
    early_exit: bool
    used_full_fallback: bool
    certificate: Certificate | None
    logical_work: dict[str, int]


@dataclass(frozen=True)
class FullProjection:
    logits: np.ndarray
    token_id: int
    logical_work: dict[str, int]


def _new_work(hidden_size: int) -> dict[str, int]:
    # These are documented reference-algorithm primitives, NOT hardware FLOPs
    # or total DRAM traffic. Timings include validation and all auxiliaries.
    return {
        "hidden_values_validated": hidden_size,
        "projection_code_elements_read": 0,
        "projection_sign_products": 0,
        "projection_accumulator_adds": 0,
        "row_scale_products": 0,
        "row_logit_adds": 0,
        "projection_rows_checked_finite": 0,
        "hidden_abs_elements": 0,
        "hidden_max_reduction_elements": 0,
        "bound_table_adds": 0,
        "bound_table_nextafter": 0,
        "bound_count_metadata_reads": 0,
        "bound_scale_metadata_reads": 0,
        "bound_scale_products": 0,
        "bound_product_nextafter": 0,
        "interval_endpoint_adds": 0,
        "interval_endpoint_nextafter": 0,
        "interval_endpoints_checked_finite": 0,
        "dominance_rows_inspected": 0,
        "certificate_checks": 0,
    }


def _check_float_environment() -> None:
    """Reject a clearly unsupported scalar float environment.

    The argument assumes IEEE-754 binary64 with gradual underflow and correctly
    rounded NumPy elementwise +, -, *, and nextafter. This small check is not a
    proof of an arbitrary platform's arithmetic implementation.
    """
    info = np.finfo(np.float64)
    tiny = np.nextafter(np.float64(0), np.float64(1))
    if (
        _F64.itemsize != 8
        or info.nmant != 52
        or tiny <= 0
        or np.multiply(tiny, np.float64(1)) != tiny
        or np.add(tiny, tiny) != np.nextafter(tiny, _POS_INF)
    ):
        raise RuntimeError("This reference requires binary64 with gradual underflow")


class TernaryResidualProjection:
    """A base ternary plane followed by additive ternary correction planes.

    ``codes`` has shape (planes, output_rows, hidden_size), with values -1/0/1.
    ``scales`` has shape (planes, output_rows), or (planes,) to share each
    plane's scale across rows. ``bias`` has shape (output_rows,).

    Codes occupy int8 in this reference. They are not packed to 1.58 or 2 bits.
    Each plane has per-row float64 scales; zero and negative scales are valid.
    All arrays are copied and made read-only on construction.
    """

    def __init__(
        self,
        codes: Any,
        scales: Any,
        bias: Any | None = None,
    ) -> None:
        _check_float_environment()
        raw = np.asarray(codes)
        if raw.ndim != 3 or any(size < 1 for size in raw.shape):
            raise ValueError("codes must have nonempty (planes, rows, hidden) shape")
        if raw.dtype.kind not in "biuf":
            raise ValueError("codes must be numeric ternary values")
        if not np.all(np.isfinite(raw)) or not np.all(np.isin(raw, (-1, 0, 1))):
            raise ValueError("codes must contain only -1, 0, or 1; no truncation is allowed")
        self.codes = np.array(raw, dtype=np.int8, copy=True, order="C")
        self.nplanes, self.nrows, self.hidden_size = self.codes.shape
        scale_array = np.asarray(scales, dtype=np.float64)
        if scale_array.shape == (self.nplanes,):
            scale_array = np.broadcast_to(scale_array[:, None], (self.nplanes, self.nrows))
        if scale_array.shape != (self.nplanes, self.nrows):
            raise ValueError("scales must have shape (planes,) or (planes, rows)")
        if not np.all(np.isfinite(scale_array)):
            raise ValueError("scales must be finite")
        self.scales = np.array(scale_array, dtype=np.float64, copy=True, order="C")
        bias_array = np.zeros(self.nrows, dtype=np.float64) if bias is None else np.asarray(bias, dtype=np.float64)
        if bias_array.shape != (self.nrows,) or not np.all(np.isfinite(bias_array)):
            raise ValueError("bias must be a finite vector with one value per output row")
        self.bias = np.array(bias_array, dtype=np.float64, copy=True, order="C")
        self.nonzero_counts = np.count_nonzero(self.codes, axis=2).astype(np.int64)
        for array in (self.codes, self.scales, self.bias, self.nonzero_counts):
            array.flags.writeable = False

    @classmethod
    def from_dense(
        cls,
        weights: Any,
        nplanes: int = 3,
        bias: Any | None = None,
    ) -> TernaryResidualProjection:
        """Greedy absmax residual ternarization for reference experiments.

        This convenience constructor does not preserve the original dense
        weights exactly. Certificates refer only to its completed ternary
        representation, not to the input matrix or a source language model.
        """
        if isinstance(nplanes, bool) or not isinstance(nplanes, int) or nplanes < 1:
            raise ValueError("nplanes must be a positive integer")
        residual = np.array(weights, dtype=np.float64, copy=True)
        if residual.ndim != 2 or any(size < 1 for size in residual.shape):
            raise ValueError("weights must be a nonempty matrix")
        if not np.all(np.isfinite(residual)):
            raise ValueError("weights must be finite")
        codes, scales = [], []
        for _ in range(nplanes):
            scale = np.max(np.abs(residual), axis=1)
            normalized = np.divide(
                residual,
                scale[:, None],
                out=np.zeros_like(residual),
                where=scale[:, None] != 0,
            )
            code = np.rint(normalized).astype(np.int8)
            residual = np.subtract(residual, np.multiply(scale[:, None], code))
            if not np.all(np.isfinite(residual)):
                raise ArithmeticError("Dense-to-ternary construction overflowed")
            codes.append(code)
            scales.append(scale)
        return cls(np.stack(codes), np.stack(scales), bias)

    def storage_bytes(self) -> dict[str, int]:
        """Exact persistent ndarray storage; excludes Python object overhead."""
        fields = {
            "int8_codes": self.codes.nbytes,
            "float64_row_scales": self.scales.nbytes,
            "float64_bias": self.bias.nbytes,
            "int64_bound_metadata": self.nonzero_counts.nbytes,
        }
        fields["total_persistent_ndarray_bytes"] = sum(fields.values())
        return fields

    def _hidden(self, hidden: Any) -> np.ndarray:
        hidden_array = np.asarray(hidden, dtype=np.float64)
        if hidden_array.shape != (self.hidden_size,):
            raise ValueError("hidden must be one vector of the declared hidden size")
        if not np.all(np.isfinite(hidden_array)):
            raise ValueError("hidden must be finite")
        return hidden_array

    def _add_plane(
        self,
        prefix: np.ndarray,
        hidden: np.ndarray,
        plane: int,
        work: dict[str, int],
    ) -> np.ndarray:
        # Explicit column order avoids implementation-dependent BLAS reduction
        # order. It is intentionally slow and is shared with the full baseline.
        row_sum = np.zeros(self.nrows, dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            for column in range(self.hidden_size):
                term = np.multiply(self.codes[plane, :, column], hidden[column], dtype=np.float64)
                row_sum = np.add(row_sum, term)
            scaled = np.multiply(row_sum, self.scales[plane])
            result = np.add(prefix, scaled)
        elements = self.nrows * self.hidden_size
        for key in (
            "projection_code_elements_read",
            "projection_sign_products",
            "projection_accumulator_adds",
        ):
            work[key] += elements
        work["row_scale_products"] += self.nrows
        work["row_logit_adds"] += self.nrows
        work["projection_rows_checked_finite"] += self.nrows
        if not np.all(np.isfinite(result)):
            raise ArithmeticError("The defined full-reference arithmetic is nonfinite")
        return result

    def full_reference(self, hidden: Any) -> FullProjection:
        """Evaluate every plane in the fixed order and use NumPy's first argmax."""
        hidden_array = self._hidden(hidden)
        work = _new_work(self.hidden_size)
        logits = self.bias.copy()
        for plane in range(self.nplanes):
            logits = self._add_plane(logits, hidden_array, plane, work)
        return FullProjection(logits, int(np.argmax(logits)), work)

    def _sum_bound_table(self, hidden: np.ndarray, work: dict[str, int]) -> np.ndarray:
        # In real arithmetic n*||hidden||_inf is an L1 row bound. This recurrence
        # bounds the actual ordered floating-point sum including cancellation,
        # subnormals and accumulation rounding, instead of assuming exact sums.
        magnitude = np.max(np.abs(hidden))
        work["hidden_abs_elements"] += self.hidden_size
        work["hidden_max_reduction_elements"] += self.hidden_size
        table = np.zeros(self.hidden_size + 1, dtype=np.float64)
        if magnitude == 0:
            return table  # Every signed input and every row sum is exactly zero.
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            for count in range(1, self.hidden_size + 1):
                table[count] = np.nextafter(np.add(table[count - 1], magnitude), _POS_INF)
        work["bound_table_adds"] += self.hidden_size
        work["bound_table_nextafter"] += self.hidden_size
        return table

    def _remaining_interval(
        self,
        prefix: np.ndarray,
        next_plane: int,
        sum_bounds: np.ndarray,
        work: dict[str, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        lower, upper = prefix.copy(), prefix.copy()
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            for plane in range(next_plane, self.nplanes):
                product = np.multiply(sum_bounds[self.nonzero_counts[plane]], np.abs(self.scales[plane]))
                # inf*0 means our coarse bound cannot determine the actual
                # floating row sum. Keep an infinite enclosure, never NaN or 0.
                product = np.where(np.isnan(product), _POS_INF, product)
                radius = np.nextafter(product, _POS_INF)
                lower = np.nextafter(np.subtract(lower, radius), _NEG_INF)
                upper = np.nextafter(np.add(upper, radius), _POS_INF)
                work["bound_count_metadata_reads"] += self.nrows
                work["bound_scale_metadata_reads"] += self.nrows
                work["bound_scale_products"] += self.nrows
                work["bound_product_nextafter"] += self.nrows
                work["interval_endpoint_adds"] += 2 * self.nrows
                work["interval_endpoint_nextafter"] += 2 * self.nrows
        return lower, upper

    def prefix_interval(self, hidden: Any, planes_executed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Diagnostic enclosure; returns (prefix logits, lower, upper).

        Not used for timing in the experiment, because it recomputes a prefix.
        This helper makes the numerical enclosure independently testable.
        """
        if isinstance(planes_executed, bool) or not isinstance(planes_executed, int):
            raise ValueError("planes_executed must be an integer")
        if not 0 <= planes_executed <= self.nplanes:
            raise ValueError("planes_executed is outside the representation")
        hidden_array = self._hidden(hidden)
        work = _new_work(self.hidden_size)
        prefix = self.bias.copy()
        for plane in range(planes_executed):
            prefix = self._add_plane(prefix, hidden_array, plane, work)
        bounds = self._sum_bound_table(hidden_array, work)
        lower, upper = self._remaining_interval(prefix, planes_executed, bounds, work)
        return prefix, lower, upper

    def predict(self, hidden: Any) -> Decision:
        """Return only a greedy token ID, never approximate probabilities.

        A strict dominance check prevents an early exit on ties. If no early
        certificate is possible, evaluate every plane exactly as full_reference.
        Overflow/nonfinite arithmetic is an error, not an uncertified answer.
        """
        hidden_array = self._hidden(hidden)
        work = _new_work(self.hidden_size)
        prefix = self.bias.copy()
        # One-plane projections cannot skip a correction. Avoid building bounds.
        sum_bounds = self._sum_bound_table(hidden_array, work) if self.nplanes > 1 else None
        for plane in range(self.nplanes):
            prefix = self._add_plane(prefix, hidden_array, plane, work)
            executed = plane + 1
            if executed == self.nplanes:
                return Decision(
                    int(np.argmax(prefix)), executed, self.nplanes,
                    False, True, None, work,
                )
            lower, upper = self._remaining_interval(prefix, executed, sum_bounds, work)
            work["certificate_checks"] += 1
            work["interval_endpoints_checked_finite"] += 2 * self.nrows
            work["dominance_rows_inspected"] += self.nrows
            # Infinite bounds may hide an overflowing reference. Defer all such
            # cases rather than certifying merely because the winner is obvious.
            if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
                continue
            winner = int(np.argmax(lower))
            if self.nrows == 1:
                competitor_upper = None
                dominates = True
            else:
                competitor_upper = float(max(
                    np.max(upper[:winner]) if winner else _NEG_INF,
                    np.max(upper[winner + 1:]) if winner + 1 < self.nrows else _NEG_INF,
                ))
                dominates = lower[winner] > competitor_upper
            if dominates:
                certificate = Certificate(
                    float(lower[winner]), competitor_upper, executed, self.nplanes,
                )
                return Decision(
                    winner, executed, self.nplanes, True, False, certificate, work,
                )
        raise AssertionError("Nonempty plane list must return a decision")
