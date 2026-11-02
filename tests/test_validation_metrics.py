import unittest
from validation.metrics import generation_summary, paired_correctness, wilson_interval
from validation.environment import classify_backend


class MetricsTests(unittest.TestCase):
    def test_aggregate_does_not_average_rates(self):
        summary = generation_summary([{"output_tokens": 10, "elapsed_seconds": 1.},
                                      {"output_tokens": 10, "elapsed_seconds": 10.}])
        self.assertAlmostEqual(summary["aggregate_tokens_per_second"], 20 / 11)
        self.assertEqual(summary["mean_request_tokens_per_second"], 5.5)
        self.assertIsNone(summary["ttft_seconds"])

    def test_invalid_measurements(self):
        for elapsed in (0, -1, float("nan"), float("inf"), True, "1"):
            with self.assertRaises(ValueError):
                generation_summary([{"output_tokens": 1, "elapsed_seconds": elapsed}])
        with self.assertRaises(ValueError):
            generation_summary([{"output_tokens": True, "elapsed_seconds": 1.}])

    def test_invalid_ttft(self):
        for ttft in (-1., float("nan"), float("inf"), True, "1"):
            with self.assertRaises(ValueError):
                generation_summary([{"output_tokens": 1, "elapsed_seconds": 1., "ttft_seconds": ttft}])

    def test_lost_correct_not_cancelled_by_new_win(self):
        report = paired_correctness([{"id": "a", "is_correct": True}, {"id": "b", "is_correct": False}],
                                    [{"id": "b", "is_correct": True}, {"id": "a", "is_correct": False}])
        self.assertEqual(report["accuracy_delta"], 0)
        self.assertEqual(report["lost_correct_risk"], .5)
        self.assertEqual(report["mcnemar_exact_two_sided_p"], 1)

    def test_pairing_fails_closed(self):
        with self.assertRaises(ValueError):
            paired_correctness([{"id": 1, "is_correct": True}], [{"id": 2, "is_correct": True}])

    def test_zero_observed_errors_is_not_zero_risk_bound(self):
        lo, hi = wilson_interval(0, 100)
        self.assertAlmostEqual(lo, 0.)
        self.assertGreater(hi, .03)

    def test_dtk_is_not_called_nvidia(self):
        info = {"status": "ok", "accelerator_available": True, "cuda_build": "12.1",
                "hip_build": "6", "triton_target": {"backend": "cuda"}}
        self.assertEqual(classify_backend(info, {"nvidia-smi": True})["kind"], "hip_or_dtk")

    def test_kernel_availability_is_not_validation(self):
        info = {"status": "ok", "accelerator_available": True, "cuda_build": "12.1",
                "triton_target": {"backend": "cuda"}}
        result = classify_backend(info, {"nvidia-smi": True})
        self.assertEqual(result["kind"], "nvidia_candidate")
        self.assertTrue(result["kernel_probe_required"])


if __name__ == "__main__":
    unittest.main()
