"""Numerical and behavioral guarantees of the synthetic projection reference."""

import itertools
import unittest

import numpy as np

from research.adaptive_output import TernaryResidualProjection


class AdaptiveOutputTests(unittest.TestCase):
    def assert_equivalent(self, model, hidden):
        full = model.full_reference(hidden)
        decision = model.predict(hidden)
        self.assertEqual(decision.token_id, full.token_id)
        self.assertGreaterEqual(decision.planes_executed, 1)
        self.assertLessEqual(decision.planes_executed, model.nplanes)
        if decision.early_exit:
            self.assertFalse(decision.used_full_fallback)
            self.assertIsNotNone(decision.certificate)
            self.assertLess(decision.planes_executed, model.nplanes)
            certificate = decision.certificate
            if certificate.competitors_upper is not None:
                self.assertGreater(certificate.winner_lower, certificate.competitors_upper)
        else:
            self.assertTrue(decision.used_full_fallback)
            self.assertIsNone(decision.certificate)
            self.assertEqual(decision.planes_executed, model.nplanes)
        for prefix in range(model.nplanes + 1):
            _, lower, upper = model.prefix_interval(hidden, prefix)
            self.assertTrue(np.all(lower <= full.logits), (prefix, lower, full.logits))
            self.assertTrue(np.all(full.logits <= upper), (prefix, full.logits, upper))
        return decision

    def test_large_margin_can_certify_after_base_plane(self):
        model = TernaryResidualProjection(
            [[[1, 1], [-1, -1], [0, 1]], [[1, -1], [1, 1], [-1, 0]]],
            [8.0, 0.01],
        )
        decision = self.assert_equivalent(model, [1.0, 1.0])
        self.assertTrue(decision.early_exit)
        self.assertEqual(decision.planes_executed, 1)
        self.assertGreater(decision.logical_work["bound_scale_products"], 0)

    def test_later_plane_can_reverse_the_apparent_winner(self):
        model = TernaryResidualProjection(
            [[[1], [0]], [[-1], [1]]], [1.0, 2.0],
        )
        self.assertEqual(np.argmax(model.prefix_interval([1.0], 1)[0]), 0)
        decision = self.assert_equivalent(model, [1.0])
        self.assertEqual(decision.token_id, 1)
        self.assertTrue(decision.used_full_fallback)

    def test_multiple_refinements_can_exit_before_last_plane(self):
        model = TernaryResidualProjection(
            [[[1], [1]], [[1], [-1]], [[1], [1]], [[-1], [1]]],
            [1.0, 2.0, 0.01, 0.001],
        )
        decision = self.assert_equivalent(model, [1.0])
        self.assertTrue(decision.early_exit)
        self.assertEqual(decision.planes_executed, 2)

    def test_exact_ties_fall_back_to_first_full_argmax(self):
        model = TernaryResidualProjection(np.ones((3, 4, 3)), [1, 0.1, 0.01])
        decision = self.assert_equivalent(model, [0.25, -1.0, 3.0])
        self.assertTrue(decision.used_full_fallback)
        self.assertEqual(decision.token_id, 0)

    def test_one_plane_is_exact_and_does_not_build_bounds(self):
        model = TernaryResidualProjection([[[1, -1], [0, 1]]], [2])
        decision = self.assert_equivalent(model, [1.0, 2.0])
        self.assertTrue(decision.used_full_fallback)
        self.assertEqual(decision.logical_work["bound_table_adds"], 0)

    def test_one_row_and_zero_hidden_are_well_defined(self):
        single = TernaryResidualProjection([[[1, -1]], [[0, 1]]], [1, 0.1])
        self.assert_equivalent(single, [0.0, -0.0])
        biased = TernaryResidualProjection(np.zeros((3, 3, 2)), [0, 0, 0], bias=[-1, 3, 0])
        self.assertEqual(self.assert_equivalent(biased, [0.0, 0.0]).token_id, 1)

    def test_rounding_at_ulp_boundary_cannot_produce_false_certificate(self):
        unit = np.spacing(np.float64(1.0))
        model = TernaryResidualProjection(
            [[[0], [0]], [[1], [-1]], [[0], [1]]],
            [1.0, unit / 2, unit],
            bias=[1.0, np.nextafter(1.0, np.inf)],
        )
        self.assert_equivalent(model, [1.0])

    def test_subnormal_products_and_cancellation(self):
        eta = np.nextafter(np.float64(0), np.float64(1))
        model = TernaryResidualProjection(
            [[[1, -1, 1], [-1, 1, 0]], [[0, 1, -1], [1, 0, 1]]],
            [[0.5, 1], [1, -0.5]],
            bias=[eta, -eta],
        )
        self.assert_equivalent(model, [eta, 2 * eta, 3 * eta])
        cancellation = TernaryResidualProjection(
            [[[1, 1, 1], [1, 0, 1]], [[1, -1, 0], [-1, 1, 0]]],
            [1.0, 1e-20],
        )
        self.assert_equivalent(cancellation, [1e16, 1.0, -1e16])

    def test_coarse_bound_overflow_defers_to_finite_reference(self):
        model = TernaryResidualProjection(
            [[[0, 0], [0, 0]], [[1, -1], [-1, 1]]],
            [1.0, 1e-300], bias=[1, 0],
        )
        decision = self.assert_equivalent(model, [1e308, 1e308])
        self.assertTrue(decision.used_full_fallback)

    def test_nonfinite_reference_is_an_error_not_a_certificate(self):
        model = TernaryResidualProjection(
            [[[0, 0], [0, 0]], [[1, 1], [0, 0]]],
            [1.0, 1.0], bias=[10, 0],
        )
        for operation in (model.predict, model.full_reference):
            with self.assertRaises(ArithmeticError):
                operation([1e308, 1e308])

    def test_exhaustive_small_ternary_patterns(self):
        # Exhaust every 3-plane, 2-row, 1-column code layout (3**6).
        for values in itertools.product((-1, 0, 1), repeat=6):
            model = TernaryResidualProjection(np.array(values).reshape(3, 2, 1), [1.0, -0.5, 0.125])
            for hidden in ([0.0], [0.25], [-2.0]):
                full = model.full_reference(hidden)
                self.assertEqual(model.predict(hidden).token_id, full.token_id)

    def test_randomized_all_prefix_enclosures_and_argmax(self):
        rng = np.random.default_rng(1701)
        for _ in range(160):
            planes = int(rng.integers(1, 6))
            rows = int(rng.integers(1, 15))
            width = int(rng.integers(1, 25))
            codes = rng.integers(-1, 2, size=(planes, rows, width), dtype=np.int8)
            scales = rng.normal(size=(planes, rows)) * np.power(10.0, rng.uniform(-12, 12, size=(planes, rows)))
            hidden = rng.normal(size=width) * np.power(10.0, rng.uniform(-12, 12, size=width))
            bias = rng.normal(size=rows)
            self.assert_equivalent(TernaryResidualProjection(codes, scales, bias), hidden)

    def test_extreme_finite_ranges(self):
        rng = np.random.default_rng(991)
        for exponent in (-300, -150, 0, 150, 300):
            codes = rng.integers(-1, 2, size=(4, 6, 8), dtype=np.int8)
            hidden = rng.uniform(-1, 1, size=8) * (10.0 ** exponent)
            scale = (10.0 ** (-exponent)) * np.array([0.25, 0.125, 0.0625, 0.03125])
            self.assert_equivalent(TernaryResidualProjection(codes, scale), hidden)

    def test_dense_constructor_does_not_claim_dense_equivalence(self):
        weights = np.array([[1.0, 0.3, -0.7], [0.0, 0.0, 0.0], [0.4, -0.9, 0.2]])
        model = TernaryResidualProjection.from_dense(weights, nplanes=3)
        self.assertTrue(np.all(np.isin(model.codes, [-1, 0, 1])))
        self.assert_equivalent(model, [0.2, -0.3, 0.7])
        self.assertEqual(model.storage_bytes()["total_persistent_ndarray_bytes"],
                         sum(a.nbytes for a in (model.codes, model.scales, model.bias, model.nonzero_counts)))

    def test_invalid_inputs_are_rejected_without_silent_truncation(self):
        for codes in ([], np.zeros((0, 2, 3)), [[[0.2]]], [[[2]]], [[[np.nan]]], [[[1 + 1j]]]):
            with self.assertRaises(ValueError):
                TernaryResidualProjection(codes, [1])
        with self.assertRaises(ValueError):
            TernaryResidualProjection([[[1]]], [np.inf])
        with self.assertRaises(ValueError):
            TernaryResidualProjection([[[1]]], [1], bias=[np.nan])
        model = TernaryResidualProjection([[[1]]], [1])
        for hidden in ([np.nan], [np.inf], [1, 2], [[1]]):
            with self.assertRaises(ValueError):
                model.predict(hidden)
        for stage in (-1, 2, True, 1.5):
            with self.assertRaises(ValueError):
                model.prefix_interval([1], stage)
        with self.assertRaises(ValueError):
            TernaryResidualProjection.from_dense([[1]], nplanes=True)

    def test_input_arrays_are_copied_and_storage_is_read_only(self):
        codes, scales, bias = np.ones((2, 2, 2)), np.ones(2), np.zeros(2)
        model = TernaryResidualProjection(codes, scales, bias)
        codes[:] = 99
        scales[:] = np.nan
        bias[:] = np.inf
        self.assert_equivalent(model, [1.0, -0.5])
        with self.assertRaises(ValueError):
            model.codes[0, 0, 0] = 0


if __name__ == "__main__":
    unittest.main()
