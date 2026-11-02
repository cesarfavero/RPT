"""Read-only, credential-free inventory for NVIDIA, CPU and SCNet/DTK hosts."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PACKAGES = ("numpy", "torch", "triton", "pytorch-triton", "pytorch-triton-rocm",
            "transformers", "peft", "accelerate", "safetensors", "datasets", "lm_eval")


def package_versions():
    result = {}
    for name in PACKAGES:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _torch_probe():
    # Isolate driver initialization so a broken driver cannot hang the inventory.
    script = '''
import json
import torch
r = {"version": str(torch.__version__), "cuda_build": torch.version.cuda,
     "hip_build": getattr(torch.version, "hip", None), "accelerator_available": torch.cuda.is_available(), "devices": []}
if r["accelerator_available"]:
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        r["devices"].append({"index": i, "name": p.name, "total_memory_bytes": p.total_memory,
                             "capability": [p.major, p.minor]})
try:
    import triton
    r["triton_version"] = triton.__version__
    if r["accelerator_available"]:
        target = triton.runtime.driver.active.get_current_target()
        r["triton_target"] = {"backend": str(target.backend), "arch": str(target.arch), "warp_size": target.warp_size}
except Exception as e:
    r["triton_probe_error_type"] = type(e).__name__
print(json.dumps(r))
'''
    try:
        proc = subprocess.run([sys.executable, "-c", script], capture_output=True,
                              text=True, timeout=30, check=False)
        if proc.returncode:
            return {"status": "failed", "exit_code": proc.returncode}
        return {"status": "ok", **json.loads(proc.stdout.strip().splitlines()[-1])}
    except subprocess.TimeoutExpired:
        return {"status": "timed_out"}
    except (ValueError, IndexError):
        return {"status": "invalid_probe_output"}


def classify_backend(torch_info, vendor_tools):
    """Availability is only a candidate; an actual numerical kernel test is required."""
    if torch_info.get("status") != "ok":
        return {"kind": "unavailable", "kernel_probe_required": True,
                "reason": "PyTorch or its driver could not be initialized."}
    if not torch_info.get("accelerator_available"):
        return {"kind": "cpu", "kernel_probe_required": True,
                "reason": "No accelerator visible to PyTorch."}
    if torch_info.get("hip_build") or vendor_tools.get("hy-smi"):
        return {"kind": "hip_or_dtk", "kernel_probe_required": True,
                "reason": "Vendor stack detected; NVIDIA validation does not transfer to this backend."}
    target = torch_info.get("triton_target", {}).get("backend")
    if torch_info.get("cuda_build") and target == "cuda" and vendor_tools.get("nvidia-smi"):
        return {"kind": "nvidia_candidate", "kernel_probe_required": True,
                "reason": "CUDA/Triton detected; numerical and performance checks still required."}
    return {"kind": "accelerator_unverified", "kernel_probe_required": True,
            "reason": "A CUDA-compatible API alone does not identify supported hardware."}


def collect_environment(probe_torch=True):
    packages = package_versions()
    vendor_tools = {name: shutil.which(name) is not None
                    for name in ("nvidia-smi", "rocm-smi", "hy-smi", "sinfo", "sbatch")}
    torch_info = (_torch_probe() if probe_torch and packages["torch"] else
                  {"status": "not_installed" if not packages["torch"] else "not_probed"})
    memory = None
    try:
        memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        pass
    return {
        "schema_version": "rpt-environment-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": {"system": platform.system(), "machine": platform.machine(),
                     "python": platform.python_version(), "cpu_count": os.cpu_count(),
                     "physical_memory_bytes": memory},
        "packages": packages, "vendor_tools_present": vendor_tools, "torch": torch_info,
        "backend_assessment": classify_backend(torch_info, vendor_tools),
        "secret_values_collected": False,
        "model_quality_evaluated": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-torch-probe", action="store_true")
    args = parser.parse_args(argv)
    report = collect_environment(not args.no_torch_probe)
    payload = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
