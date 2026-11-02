"""Evaluate the exact packed artifact with lm-evaluation-harness.

Three explicit operations are available: --freeze-only declares the exact task
documents before scoring; --inspect-artifact computes the model identity; a run
requires both expected hashes. hf-dense is a separately requested control, never
a fallback. Importing this module needs only the Python standard library.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from typing import Any, Mapping

from .splits import (
    SplitValidationError, build_split_manifest, canonical_json, load_records,
    sha256_bytes, sha256_file, verify_split_manifest, write_frozen_manifest,
)


RUNNER_VERSION = "rpt-lm-eval-v1"
ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = ROOT / "gemma4_theory/runtime/packed_recovered_l78"
DEFAULT_TASKS = "arc_easy,arc_challenge,hellaswag,truthfulqa_mc1"


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("A result/provenance value is not finite; no score will be published")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if callable(value):
        name = f"{getattr(value, '__module__', '')}.{getattr(value, '__qualname__', type(value).__name__)}"
        try:
            source_sha = sha256_bytes(inspect.getsource(value).encode("utf-8"))
        except (OSError, TypeError):
            source_sha = None
        return {"callable": name, "source_sha256": source_sha}
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    raise TypeError(f"Cannot serialize provenance value of type {type(value).__name__}")


def _configure_offline(enabled: bool) -> None:
    if enabled:
        # Set before importing datasets/transformers/huggingface_hub.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


def synchronize_model_cuda(torch: Any, model: Any, requested_device: str) -> list[int]:
    """Synchronize every CUDA device referenced by model placement or tensors."""
    if not torch.cuda.is_available():
        return []
    indices: set[int] = set()

    def include(value: Any) -> None:
        if type(value) is int:
            if value >= 0:
                indices.add(value)
            return
        match = re.fullmatch(r"cuda(?::(\d+))?", str(value))
        if match:
            indices.add(int(match.group(1)) if match.group(1) is not None else torch.cuda.current_device())

    include(requested_device)
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, Mapping):
        for device in device_map.values():
            include(device)
    for method in ("parameters", "buffers"):
        tensors = getattr(model, method, None)
        if callable(tensors):
            for tensor in tensors():
                include(getattr(tensor, "device", None))
    devices = sorted(indices)
    for device in devices:
        torch.cuda.synchronize(device=device)
    return devices


def fingerprint_directory(directory: str | Path) -> dict[str, Any]:
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Artifact directory is missing: {directory}")
    files = []
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory)
        if any(part in {".git", ".cache", "__pycache__"} for part in relative.parts):
            continue
        if not path.is_file() or path.suffix == ".pyc":
            continue
        files.append({"path": relative.as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    if not files:
        raise ValueError(f"Artifact directory is empty: {directory}")
    return {"sha256": sha256_bytes(canonical_json(files).encode("utf-8")), "files": files,
            "total_bytes": sum(item["bytes"] for item in files)}


def resolve_model_source(args: argparse.Namespace) -> Path:
    _configure_offline(args.offline)
    if args.backend == "packed-triton":
        if not args.model_dir:
            raise ValueError("packed-triton requires --model-dir")
        root = Path(args.model_dir).expanduser().resolve()
        for relative in ("student_pruned", "adapter_recovered_l78", "packed/manifest.json", "packed/mlp_ternary_2bit.bin"):
            if not (root / relative).exists():
                raise FileNotFoundError(f"Missing packed bundle component: {relative}")
        return root
    if not args.model_source:
        raise ValueError("hf-dense requires --model-source")
    source = Path(args.model_source).expanduser()
    if source.is_dir():
        return source.resolve()
    if not args.model_revision or not re.fullmatch(r"[0-9a-fA-F]{40}", args.model_revision):
        raise ValueError("A remote hf-dense source requires --model-revision with a full immutable commit SHA")
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id=args.model_source, revision=args.model_revision,
                                  local_files_only=args.offline, token=os.environ.get("HF_TOKEN"))).resolve()


def _package_versions() -> dict[str, str | None]:
    versions = {}
    for package in ("lm-eval", "torch", "transformers", "peft", "triton", "datasets", "accelerate", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _load_tasks(task_names: list[str]) -> tuple[Any, dict[str, Any]]:
    from lm_eval.tasks import TaskManager
    manager = TaskManager()
    if hasattr(manager, "load"):
        loaded = manager.load(task_names)
        tasks = loaded["tasks"]
    else:
        # Published 0.4.x harness interface. This changes only task discovery;
        # the model object is always the already-loaded, verified instance.
        from lm_eval.tasks import get_task_dict
        nested = get_task_dict(task_names, task_manager=manager)
        tasks = {}
        def collect(value: Mapping[Any, Any]) -> None:
            for key, item in value.items():
                if isinstance(item, Mapping):
                    collect(item)
                elif item is not None:
                    name = item.get_config("task") or str(key)
                    if name in tasks:
                        raise ValueError(f"Duplicate harness task: {name}")
                    tasks[name] = item
        collect(nested)
    if not tasks:
        raise ValueError("No harness tasks were loaded")
    return manager, dict(sorted(tasks.items()))


def task_snapshot(tasks: Mapping[str, Any], *, limit: int | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = []
    metadata = {}
    for task_name, task in tasks.items():
        if hasattr(task, "eval_docs"):
            documents = task.eval_docs
        elif task.has_test_docs():
            documents = task.test_docs()
        elif task.has_validation_docs():
            documents = task.validation_docs()
        else:
            raise ValueError(f"Task has no evaluation split: {task_name}")
        count = 0
        for index, document in enumerate(documents):
            if limit is not None and index >= limit:
                break
            document = _json_safe(document)
            rendered = task.doc_to_text(document)
            if not isinstance(rendered, str) or not rendered.strip():
                rendered = task.doc_to_target(document)
            if not isinstance(rendered, str) or not rendered.strip():
                raise ValueError(f"Task {task_name} has no auditable textual prompt at document {index}")
            question = document.get("question") if isinstance(document, dict) else None
            if not isinstance(question, str) or not question.strip():
                question = rendered
            records.append({"id": str(index), "dataset_id": task_name, "question": question,
                            "rendered_prompt": rendered, "document": document, "task": task_name,
                            "fingerprint_prompt": canonical_json({"prompt": rendered, "document": document})})
            count += 1
        if count == 0:
            raise ValueError(f"Empty harness task: {task_name}")
        config = _json_safe(task.dump_config())
        dataset_info = getattr(documents, "info", None)
        metadata[task_name] = {
            "selected_count": count,
            "source_count": len(documents) if hasattr(documents, "__len__") else None,
            "dataset_fingerprint": getattr(documents, "_fingerprint", None),
            "dataset_version": str(getattr(dataset_info, "version", "unknown")),
            "dataset_path": config.get("dataset_path"),
            "dataset_name": config.get("dataset_name"),
            "dataset_revision": (config.get("dataset_kwargs") or {}).get("revision"),
            "evaluation_split": config.get("test_split") if task.has_test_docs() else config.get("validation_split"),
            "config": config,
            "config_sha256": sha256_bytes(canonical_json(config).encode("utf-8")),
        }
    return records, metadata


def verify_logged_samples(result: Mapping[str, Any], records: list[dict[str, Any]]) -> dict[str, int]:
    """Check that scored documents are exactly the frozen selection, per task."""
    expected: dict[str, dict[str, str]] = {}
    for record in records:
        expected.setdefault(record["task"], {})[str(record["id"])] = sha256_bytes(canonical_json(record["document"]).encode("utf-8"))
    samples = result.get("samples")
    if not isinstance(samples, Mapping) or set(samples) != set(expected):
        raise ValueError("Harness sample logs do not cover exactly the frozen tasks")
    counts = {}
    for task, documents in expected.items():
        observed = {}
        for sample in samples[task]:
            identifier = str(sample["doc_id"])
            digest = sha256_bytes(canonical_json(_json_safe(sample["doc"])).encode("utf-8"))
            if identifier in observed and observed[identifier] != digest:
                raise ValueError(f"Harness logged conflicting documents for {task}:{identifier}")
            observed[identifier] = digest
        if observed != documents:
            raise ValueError(f"Scored document identities differ from frozen selection: {task}")
        counts[task] = len(observed)
    return counts


def _evaluation_policy(args: argparse.Namespace) -> dict[str, Any]:
    return {"tasks": sorted(args.tasks.split(",")), "limit": args.limit, "num_fewshot": args.num_fewshot,
            "apply_chat_template": args.apply_chat_template, "seed": args.seed, "max_length": args.max_length}


def freeze_tasks(args: argparse.Namespace) -> dict[str, Any]:
    _configure_offline(args.offline)
    _, tasks = _load_tasks(args.tasks.split(","))
    records, metadata = task_snapshot(tasks, limit=args.limit)
    known_training_records = []
    sources = []
    manifest_path = Path(args.split_manifest).resolve()
    for path in args.known_training:
        source = Path(path).resolve()
        known_training_records.extend(load_records(source))
        sources.append({"path": os.path.relpath(source, manifest_path.parent), "sha256": sha256_file(source)})
    manifest = build_split_manifest(
        {"test": records}, dataset_id="rpt-external-lm-eval", known_training_records=known_training_records,
        training_data_complete=args.training_data_complete, prompt_field="fingerprint_prompt",
        provenance={"runner_version": RUNNER_VERSION, "lm_eval_version": _package_versions()["lm-eval"],
                    "evaluation_policy": _evaluation_policy(args), "tasks": metadata},
    )
    manifest["sources"] = {"known_training": sources}
    digest = write_frozen_manifest(manifest, manifest_path)
    return {"status": "frozen", "manifest_sha256": digest, "split_manifest": str(manifest_path),
            "task_counts": {name: meta["selected_count"] for name, meta in metadata.items()},
            "training_data_complete": args.training_data_complete, "metrics": None}


def _runtime_module() -> Any:
    sys.path.insert(0, str(RUNTIME_DIR))
    specification = importlib.util.spec_from_file_location("_rpt_verified_triton_loader", RUNTIME_DIR / "triton_loader.py")
    if specification is None or specification.loader is None:
        raise ImportError("Cannot resolve the RPT Triton loader")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    _configure_offline(args.offline)
    if not args.expected_split_sha256 or not args.expected_artifact_sha256:
        raise ValueError("Evaluation requires --expected-split-sha256 and --expected-artifact-sha256")
    # Validate data before loading expensive model weights.
    manager, tasks = _load_tasks(args.tasks.split(","))
    records, metadata = task_snapshot(tasks, limit=args.limit)
    split_report = verify_split_manifest(args.split_manifest, expected_sha256=args.expected_split_sha256,
                                         evaluated_split="test", records=records)
    manifest = json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))
    frozen_provenance = manifest.get("provenance", {})
    versions = _package_versions()
    if frozen_provenance.get("evaluation_policy") != _evaluation_policy(args):
        raise SplitValidationError("Evaluation settings differ from the frozen experimental policy")
    if frozen_provenance.get("lm_eval_version") != versions["lm-eval"]:
        raise SplitValidationError("lm-eval version differs from the frozen task manifest")
    if frozen_provenance.get("tasks") != metadata:
        raise SplitValidationError("Task configuration or dataset fingerprint changed after freezing")

    import torch
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    validation_source = fingerprint_directory(ROOT / "validation")
    harness_source = {
        "huggingface_adapter_sha256": sha256_file(inspect.getfile(HFLM)),
        "evaluator_sha256": sha256_file(inspect.getfile(inspect.unwrap(simple_evaluate))),
    }
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device is unavailable; evaluation will not fall back to CPU")
    if args.backend == "packed-triton" and not args.device.startswith("cuda"):
        raise RuntimeError("packed-triton evaluation requires a CUDA device")
    model_path = resolve_model_source(args)
    artifact_before = fingerprint_directory(model_path)
    if artifact_before["sha256"] != args.expected_artifact_sha256:
        raise ValueError("Model artifact differs from --expected-artifact-sha256")
    runtime_before = fingerprint_directory(RUNTIME_DIR) if args.backend == "packed-triton" else None
    started = time.perf_counter()
    if args.backend == "packed-triton":
        runtime = _runtime_module()
        packed_hash = sha256_file(model_path / "packed/mlp_ternary_2bit.bin")
        model, tokenizer, loader_report = runtime.load_triton_packed_recovered_l78(
            model_path, token=os.environ.get("HF_TOKEN"), device_map=args.device, dtype=args.dtype,
            verify_sha=True, expected_packed_sha256=packed_hash,
            require_triton=True, fallback_to_pytorch=False,
        )
        runtime.reset_runtime_execution(model)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(str(model_path), device_map=args.device,
                                                    torch_dtype=args.dtype, local_files_only=True, trust_remote_code=False)
        loader_report = {"backend": "hf-dense", "model_source": args.model_source,
                         "model_revision": args.model_revision, "local_snapshot": str(model_path)}
    model.eval()
    load_elapsed = time.perf_counter() - started
    wrapped = HFLM(pretrained=model, tokenizer=tokenizer, device=args.device, batch_size=args.batch_size,
                   max_length=args.max_length, trust_remote_code=False, parallelize=False)
    if wrapped.model is not model:
        raise RuntimeError("Harness replaced the loaded model instance; refusing to score a different artifact")
    synchronized_devices = synchronize_model_cuda(torch, model, args.device)
    started = time.perf_counter()
    result = simple_evaluate(
        model=wrapped, tasks=list(tasks.values()), task_manager=manager, batch_size=args.batch_size,
        num_fewshot=args.num_fewshot, limit=args.limit, log_samples=True,
        use_cache=None, cache_requests=False, bootstrap_iters=args.bootstrap_iters,
        apply_chat_template=args.apply_chat_template, random_seed=args.seed,
        numpy_random_seed=args.seed, torch_random_seed=args.seed, fewshot_random_seed=args.seed,
        confirm_run_unsafe_code=args.confirm_run_unsafe_code,
    )
    synchronized_devices = sorted(set(synchronized_devices) | set(synchronize_model_cuda(torch, model, args.device)))
    elapsed = time.perf_counter() - started
    if not isinstance(result, Mapping) or not result.get("results"):
        raise RuntimeError("Harness returned no task results")
    sample_counts = verify_logged_samples(result, records)
    execution_report = runtime.runtime_execution_report(model, require_triton=True) if args.backend == "packed-triton" else {"backend": "hf-dense"}
    artifact_after = fingerprint_directory(model_path)
    if artifact_before != artifact_after:
        raise RuntimeError("Model artifact files changed during evaluation")
    if runtime_before is not None and runtime_before != fingerprint_directory(RUNTIME_DIR):
        raise RuntimeError("Runtime source files changed during evaluation")
    if validation_source != fingerprint_directory(ROOT / "validation"):
        raise RuntimeError("Validation source files changed during evaluation")
    # Repeat the source/data checks so a run cannot complete against a modified declaration.
    verify_split_manifest(args.split_manifest, expected_sha256=args.expected_split_sha256,
                          evaluated_split="test", records=records)
    try:
        git_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                 capture_output=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_sha = None
    return {
        "status": "completed", "runner_version": RUNNER_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_classification": split_report["classification"],
        "generalization_scope": split_report["held_out_scope"],
        "backend": args.backend, "split_verification": split_report,
        "artifact": artifact_before, "runtime_source": runtime_before,
        "validation_source": validation_source, "harness_source": harness_source,
        "loader_report": loader_report, "execution_report": execution_report,
        "task_provenance": metadata, "evaluation_policy": _evaluation_policy(args),
        "verified_sample_counts": sample_counts, "package_versions": versions,
        "python": sys.version, "platform": platform.platform(), "git_sha": git_sha,
        "device": args.device, "dtype": args.dtype,
        "synchronized_cuda_devices": synchronized_devices,
        "gpu": torch.cuda.get_device_name(torch.device(args.device)) if args.device.startswith("cuda") else None,
        "load_elapsed_sec": load_elapsed, "evaluation_elapsed_sec": elapsed,
        "harness_results": _json_safe(result),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument("--freeze-only", action="store_true")
    operation.add_argument("--inspect-artifact", action="store_true")
    parser.add_argument("--backend", choices=["packed-triton", "hf-dense"], default="packed-triton")
    parser.add_argument("--model-dir")
    parser.add_argument("--model-source")
    parser.add_argument("--model-revision")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--tasks", default=DEFAULT_TASKS)
    parser.add_argument("--split-manifest")
    parser.add_argument("--expected-split-sha256")
    parser.add_argument("--expected-artifact-sha256")
    parser.add_argument("--known-training", action="append", default=[])
    parser.add_argument("--training-data-complete", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-iters", type=int, default=10000)
    parser.add_argument("--apply-chat-template", action="store_true")
    parser.add_argument("--confirm-run-unsafe-code", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--output", help="New JSON result path; existing results are never overwritten")
    args = parser.parse_args(argv)
    args.tasks = ",".join(name.strip() for name in args.tasks.split(",") if name.strip())
    if not args.tasks or len(set(args.tasks.split(","))) != len(args.tasks.split(",")):
        parser.error("--tasks must contain distinct task names")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive; omit it for the complete task")
    if min(args.batch_size, args.max_length) <= 0 or min(args.num_fewshot, args.bootstrap_iters) < 0:
        parser.error("Invalid batch/context/fewshot/bootstrap setting")
    if not args.inspect_artifact and not args.split_manifest:
        parser.error("--split-manifest is required")
    if not args.freeze_only and not args.inspect_artifact and not args.output:
        parser.error("Evaluation requires --output")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output and Path(args.output).exists():
        raise FileExistsError(f"Refusing to overwrite an existing result: {args.output}")
    try:
        if args.inspect_artifact:
            model_path = resolve_model_source(args)
            result = {"status": "inspected", "backend": args.backend, "model_path": str(model_path),
                      "artifact": fingerprint_directory(model_path), "metrics": None}
        elif args.freeze_only:
            result = freeze_tasks(args)
        else:
            result = run_evaluation(args)
        # Serialize fully before opening a result file, including successful runs.
        encoded_result = json.dumps(_json_safe(result), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    except Exception as exc:
        result = {"status": "failed", "runner_version": RUNNER_VERSION, "backend": args.backend,
                  "error_type": type(exc).__name__, "error": str(exc), "metrics": None}
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            with Path(args.output).open("x", encoding="utf-8") as handle:
                json.dump(result, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
        print(json.dumps(result, ensure_ascii=False), file=sys.stderr)
        return 2
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.output).open("x", encoding="utf-8") as handle:
            handle.write(encoded_result)
        print(json.dumps({"status": result["status"], "output": args.output}, indent=2))
    else:
        print(encoded_result, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
