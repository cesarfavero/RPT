import builtins
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from validation.kernel_probe import run_probe
from validation.run_validation import merge_kernel_report, run_step


class PreflightFailureTests(unittest.TestCase):
    def test_broken_vendor_library_leaves_structured_failure(self):
        original_import = builtins.__import__
        def failing_import(name, *args, **kwargs):
            if name == "torch":
                raise OSError("libamdhip64.so is unavailable")
            return original_import(name, *args, **kwargs)
        with patch("validation.kernel_probe.collect_environment", return_value={}), \
             patch("validation.kernel_probe.importlib.util.find_spec", return_value=object()), \
             patch("builtins.__import__", side_effect=failing_import):
            result = run_probe()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error_type"], "OSError")
        self.assertFalse(result["full_model_validated"])

    def test_subprocess_timeout_is_a_failed_step(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch("validation.run_validation.subprocess.run", side_effect=subprocess.TimeoutExpired("test", 1)):
            result = run_step("hung", [sys.executable, "-V"], Path(directory), timeout_seconds=1)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["reason"], "step_timeout")

    def test_success_report_cannot_override_timeout_or_crash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "kernel.json"
            path.write_text(json.dumps({"status": "passed"}), encoding="utf-8")
            for step in ({"status": "failed", "reason": "step_timeout"},
                         {"status": "failed", "exit_code": -9},
                         {"status": "failed", "exit_code": 2}):
                with self.subTest(step=step):
                    self.assertEqual(merge_kernel_report(step, path)["status"], "failed")

    def test_invalid_or_partial_kernel_report_leaves_structured_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "kernel.json"
            for content in ('{"status":', '[]', '{"status": "unknown"}'):
                with self.subTest(content=content):
                    path.write_text(content, encoding="utf-8")
                    result = merge_kernel_report({"status": "passed", "exit_code": 0}, path)
                    self.assertEqual(result["status"], "failed")
                    self.assertEqual(result["reason"], "invalid_kernel_report")

    def test_missing_kernel_report_cannot_certify_a_successful_process(self):
        with tempfile.TemporaryDirectory() as directory:
            result = merge_kernel_report({"status": "passed", "exit_code": 0},
                                         Path(directory) / "missing.json")
        self.assertEqual(result["status"], "failed")

    def test_missing_accelerator_is_blocked_with_the_expected_exit_code(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "kernel.json"
            path.write_text(json.dumps({"status": "blocked", "reason": "missing_torch"}), encoding="utf-8")
            result = merge_kernel_report({"status": "failed", "exit_code": 2}, path)
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["reason"], "missing_torch")


if __name__ == "__main__":
    unittest.main()
