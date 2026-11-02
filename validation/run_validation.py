"""One entry point for portable checks and explicit accelerator validation.

No dependencies, models or datasets are installed or downloaded by this entry
point. A missing prerequisite produces `blocked`, never a successful benchmark.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from validation.environment import collect_environment

ROOT = Path(__file__).resolve().parents[1]


def source_inventory():
    files = {}
    for folder in ("validation", "research/adaptive_output", "gemma4_theory/runtime/packed_recovered_l78", "tests"):
        for path in sorted((ROOT / folder).rglob("*.py")):
            files[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True))
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"git_commit": commit, "git_dirty": dirty, "source_sha256": files}


def run_step(name, command, output_dir, timeout_seconds=1800):
    start = time.perf_counter()
    logfile = output_dir / (name + ".log")
    with logfile.open("w", encoding="utf-8") as handle:
        try:
            process = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                     check=False, timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            return {"name": name, "status": "failed", "reason": "step_timeout",
                    "timeout_seconds": timeout_seconds, "elapsed_seconds": time.perf_counter() - start,
                    "log_file": logfile.name}
    return {"name": name, "status": "passed" if process.returncode == 0 else "failed",
            "exit_code": process.returncode, "elapsed_seconds": time.perf_counter() - start,
            "log_file": logfile.name}


def merge_kernel_report(step, kernel_file):
    """A child report cannot turn a crash or timeout into successful validation."""
    step = dict(step)
    if step.get("reason") == "step_timeout":
        return step
    if not kernel_file.is_file():
        return {**step, "status": "failed", "reason": "kernel_report_missing"}
    try:
        kernel = json.loads(kernel_file.read_text(encoding="utf-8"))
        if not isinstance(kernel, dict) or kernel.get("status") not in {"passed", "blocked", "failed"}:
            raise ValueError("Invalid kernel report status")
    except (OSError, UnicodeError, ValueError) as error:
        return {**step, "status": "failed", "reason": "invalid_kernel_report",
                "report_error": str(error)[:1000]}
    # kernel_probe deliberately exits 2 for blocked/failed results. Every other
    # nonzero exit, including a crash after a report was written, remains failed.
    expected_exit = 0 if kernel["status"] == "passed" else 2
    if step.get("exit_code") != expected_exit:
        return {**step, "status": "failed", "reason": "kernel_process_failed",
                "reported_kernel_status": kernel["status"]}
    return {**step, "status": kernel["status"], "reason": kernel.get("reason")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("cpu", "preflight", "kernel", "all"), default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("validation_runs") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    parser.add_argument("--full-shapes", action="store_true")
    parser.add_argument("--step-timeout", type=int, default=1800, help="Seconds allowed for each child process")
    args = parser.parse_args(argv)
    if args.step_timeout <= 0:
        parser.error("--step-timeout must be positive")
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": "rpt-validation-run-v1", "phase": args.phase,
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "source": source_inventory(),
              "environment": collect_environment(), "steps": [], "model_quality_evaluated": False}
    if args.phase in ("cpu", "all"):
        report["steps"].append(run_step("unit_tests", [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], out, args.step_timeout))
    if args.phase in ("kernel", "all"):
        command = [sys.executable, "-m", "validation.kernel_probe", "--output", str(out / "kernel.json")]
        if args.full_shapes:
            command.append("--full-shapes")
        step = run_step("kernel_probe", command, out, args.step_timeout)
        report["steps"].append(merge_kernel_report(step, out / "kernel.json"))
    statuses = [step["status"] for step in report["steps"]]
    report["status"] = "failed" if "failed" in statuses else ("blocked" if "blocked" in statuses else "passed")
    report["status_scope"] = "Only the requested phases; never an implicit full-model or hardware certification."
    report["next_stage"] = "Use validation.lm_eval_packed with frozen datasets, local model artifacts and explicit backend."
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "phase": args.phase, "report": str(out / "report.json"),
                      "model_quality_evaluated": False}, indent=2))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
