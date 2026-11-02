"""Exercise validation boundaries with small fakes, without claiming GPU evidence."""

import contextlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from validation import lm_eval_packed as runner


class FakeTask:
    def __init__(self):
        self.eval_docs = [{"question": "An independent science item", "answer": "A"},
                          {"question": "An independent language item", "answer": "B"}]

    def doc_to_text(self, document):
        return "Question: " + document["question"] + "\nAnswer:"

    def doc_to_target(self, document):
        return document["answer"]

    def has_test_docs(self):
        return True

    def dump_config(self):
        return {"task": "fake-task", "dataset_path": "test-fixture", "test_split": "test",
                "dataset_kwargs": {"revision": "immutable-fixture-revision"}}


class HarnessProvenanceTests(unittest.TestCase):
    def test_frozen_record_has_raw_and_rendered_prompt(self):
        records, metadata = runner.task_snapshot({"fake-task": FakeTask()}, limit=1)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["question"], "An independent science item")
        self.assertIn("Question:", records[0]["rendered_prompt"])
        self.assertEqual(metadata["fake-task"]["dataset_revision"], "immutable-fixture-revision")

    def test_scored_document_mutation_is_rejected(self):
        records, _ = runner.task_snapshot({"fake-task": FakeTask()}, limit=1)
        result = {"samples": {"fake-task": [{"doc_id": 0, "doc": {"question": "changed", "answer": "A"}}]}}
        with self.assertRaises(ValueError):
            runner.verify_logged_samples(result, records)

    def test_scored_document_omission_is_rejected(self):
        records, _ = runner.task_snapshot({"fake-task": FakeTask()}, limit=None)
        result = {"samples": {"fake-task": [{"doc_id": 0, "doc": records[0]["document"]}]}}
        with self.assertRaises(ValueError):
            runner.verify_logged_samples(result, records)

    def test_multiple_filters_preserve_one_document_identity(self):
        records, _ = runner.task_snapshot({"fake-task": FakeTask()}, limit=1)
        sample = {"doc_id": 0, "doc": records[0]["document"]}
        self.assertEqual(runner.verify_logged_samples({"samples": {"fake-task": [sample, sample]}}, records), {"fake-task": 1})

    def test_extra_task_is_rejected(self):
        records, _ = runner.task_snapshot({"fake-task": FakeTask()}, limit=1)
        with self.assertRaises(ValueError):
            runner.verify_logged_samples({"samples": {"fake-task": [], "extra": []}}, records)

    def test_artifact_identity_tracks_bytes_not_only_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.bin"
            path.write_bytes(b"original")
            first = runner.fingerprint_directory(directory)
            path.write_bytes(b"modified")
            self.assertNotEqual(first["sha256"], runner.fingerprint_directory(directory)["sha256"])

    def test_nonfinite_results_are_not_serializable_as_success(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                runner._json_safe({"results": {"accuracy": value}})

    def test_offline_is_applied_to_all_huggingface_components(self):
        with patch.dict(os.environ, {}, clear=False):
            runner._configure_offline(True)
            for variable in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
                self.assertEqual(os.environ[variable], "1")

    def test_remote_dense_model_needs_an_immutable_revision(self):
        args = runner.parse_args(["--inspect-artifact", "--backend", "hf-dense", "--model-source", "missing-owner/model"])
        with self.assertRaises(ValueError):
            runner.resolve_model_source(args)

    def test_synchronization_uses_all_model_devices_not_process_default(self):
        cuda = types.SimpleNamespace(is_available=lambda: True, current_device=lambda: 7, synchronize=Mock())
        torch = types.SimpleNamespace(cuda=cuda)
        model = types.SimpleNamespace(
            hf_device_map={'first': 'cuda:1', 'second': 2, 'offloaded': 'cpu', 'disk': 'disk'},
            parameters=lambda: iter([types.SimpleNamespace(device='cuda:1'), types.SimpleNamespace(device='cpu')]),
            buffers=lambda: iter([types.SimpleNamespace(device='cuda:3')]),
        )
        self.assertEqual(runner.synchronize_model_cuda(torch, model, 'cuda:4'), [1, 2, 3, 4])
        self.assertEqual([call.kwargs['device'] for call in cuda.synchronize.call_args_list], [1, 2, 3, 4])
        cuda.synchronize.reset_mock()
        self.assertEqual(runner.synchronize_model_cuda(torch, types.SimpleNamespace(), 'cuda'), [7])
        cuda.synchronize.assert_called_once_with(device=7)

    def test_cpu_only_model_does_not_synchronize_unrelated_cuda_devices(self):
        cuda = types.SimpleNamespace(is_available=lambda: True, synchronize=Mock())
        self.assertEqual(runner.synchronize_model_cuda(types.SimpleNamespace(cuda=cuda), types.SimpleNamespace(), 'cpu'), [])
        cuda.synchronize.assert_not_called()

    def test_failure_output_has_no_metrics_or_partial_json(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            with patch.object(runner, "run_evaluation", return_value={"status": "completed", "metric": math.nan}), contextlib.redirect_stderr(io.StringIO()):
                status = runner.main(["--split-manifest", "unused", "--output", str(output)])
            self.assertEqual(status, 2)
            data = json.loads(output.read_text())
            self.assertEqual(data["status"], "failed")
            self.assertIsNone(data["metrics"])
            self.assertNotIn("metric", data)


class ExactLoadedModelTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.bundle = root / "bundle"
        for name in ("student_pruned", "adapter_recovered_l78", "packed"):
            (self.bundle / name).mkdir(parents=True)
        (self.bundle / "student_pruned/config.json").write_text("{}")
        (self.bundle / "adapter_recovered_l78/adapter_config.json").write_text("{}")
        (self.bundle / "packed/manifest.json").write_text("{}")
        (self.bundle / "packed/mlp_ternary_2bit.bin").write_bytes(b"synthetic-test-fixture-not-model-weights")
        self.runtime_dir = root / "runtime"
        self.runtime_dir.mkdir()
        (self.runtime_dir / "triton_loader.py").write_text("# synthetic test fixture\n")
        self.manifest = root / "split.json"
        self.output = root / "output.json"
        self.task = FakeTask()
        self.manager = object()
        self.tasks = {"fake-task": self.task}
        self.versions = {"lm-eval": "fixture-only"}
        freeze_args = runner.parse_args(["--freeze-only", "--tasks", "fake-task", "--split-manifest", str(self.manifest)])
        with patch.object(runner, "_load_tasks", return_value=(self.manager, self.tasks)), patch.object(runner, "_package_versions", return_value=self.versions):
            declaration = runner.freeze_tasks(freeze_args)
        self.args = runner.parse_args([
            "--tasks", "fake-task", "--split-manifest", str(self.manifest),
            "--expected-split-sha256", declaration["manifest_sha256"],
            "--model-dir", str(self.bundle), "--expected-artifact-sha256", runner.fingerprint_directory(self.bundle)["sha256"],
            "--output", str(self.output),
        ])
        self.model = types.SimpleNamespace(eval=Mock())
        self.tokenizer = object()
        self.runtime = types.SimpleNamespace(
            load_triton_packed_recovered_l78=Mock(return_value=(self.model, self.tokenizer, {"backend": "unobserved"})),
            reset_runtime_execution=Mock(),
            runtime_execution_report=Mock(return_value={"backend": "triton", "fallback_count": 0}),
        )

    def fake_modules(self, *, replace_model=False):
        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(is_available=lambda: True, current_device=lambda: 0, synchronize=Mock(), get_device_name=lambda device: "fixture-GPU")
        torch.device = lambda device: device
        harness = types.ModuleType("lm_eval")
        def evaluate(**kwargs):
            self.harness_arguments = kwargs
            return {"results": {"fake-task": {"acc,none": 0.5}}, "samples": {"fake-task": [
                {"doc_id": index, "doc": document} for index, document in enumerate(self.task.eval_docs)
            ]}}
        harness.simple_evaluate = evaluate
        models = types.ModuleType("lm_eval.models")
        huggingface = types.ModuleType("lm_eval.models.huggingface")
        def wrap(**kwargs):
            self.wrap_arguments = kwargs
            return types.SimpleNamespace(model=object() if replace_model else kwargs["pretrained"])
        huggingface.HFLM = wrap
        return {"torch": torch, "lm_eval": harness, "lm_eval.models": models, "lm_eval.models.huggingface": huggingface}

    def run_fake(self, **options):
        with patch.object(runner, "_load_tasks", return_value=(self.manager, self.tasks)), \
             patch.object(runner, "_package_versions", return_value=self.versions), \
             patch.object(runner, "RUNTIME_DIR", self.runtime_dir), \
             patch.object(runner, "_runtime_module", return_value=self.runtime), \
             patch.dict("sys.modules", self.fake_modules(**options)):
            return runner.run_evaluation(self.args)

    def test_harness_receives_exact_packed_model_with_fallback_disabled(self):
        result = self.run_fake()
        self.assertEqual(result["status"], "completed")
        self.assertIs(self.wrap_arguments["pretrained"], self.model)
        self.assertIs(self.wrap_arguments["tokenizer"], self.tokenizer)
        load_kwargs = self.runtime.load_triton_packed_recovered_l78.call_args.kwargs
        self.assertTrue(load_kwargs["require_triton"])
        self.assertFalse(load_kwargs["fallback_to_pytorch"])
        self.assertTrue(load_kwargs["verify_sha"])
        self.runtime.runtime_execution_report.assert_called_once_with(self.model, require_triton=True)
        self.assertEqual(result["verified_sample_counts"], {"fake-task": 2})
        self.assertFalse(self.harness_arguments["cache_requests"])
        self.assertIsNone(self.harness_arguments["use_cache"])

    def test_model_replacement_by_wrapper_aborts_before_scoring(self):
        with self.assertRaisesRegex(RuntimeError, "replaced"):
            self.run_fake(replace_model=True)
        self.runtime.runtime_execution_report.assert_not_called()

    def test_changed_artifact_aborts_before_model_load(self):
        (self.bundle / "packed/mlp_ternary_2bit.bin").write_bytes(b"different-fixture")
        with self.assertRaisesRegex(ValueError, "artifact differs"):
            self.run_fake()
        self.runtime.load_triton_packed_recovered_l78.assert_not_called()

    def test_changed_experimental_policy_aborts_before_model_load(self):
        self.args.seed += 1
        with self.assertRaisesRegex(ValueError, "experimental policy"):
            self.run_fake()
        self.runtime.load_triton_packed_recovered_l78.assert_not_called()

    def test_invalid_runtime_execution_cannot_publish_a_score(self):
        self.runtime.runtime_execution_report.side_effect = RuntimeError("Some layers used CPU fallback")
        with self.assertRaisesRegex(RuntimeError, "CPU fallback"):
            self.run_fake()


if __name__ == "__main__":
    unittest.main()
