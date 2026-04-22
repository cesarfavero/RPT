from __future__ import annotations

from pathlib import Path
import hashlib
import json
from typing import Any

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


EXPECTED_PACKED_SHA256 = "0b6849fe3ceaab615a4e54f63f5a96e9d9d431ba03b7b791a7ab0638ea174116"
EXPECTED_TENSOR_COUNT = 78


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unpack_codes_from_blob(blob: bytes, n: int) -> np.ndarray:
    raw = np.frombuffer(blob, dtype=np.uint8)
    codes = np.empty(raw.size * 4, dtype=np.uint8)
    codes[0::4] = raw & 0b11
    codes[1::4] = (raw >> 2) & 0b11
    codes[2::4] = (raw >> 4) & 0b11
    codes[3::4] = (raw >> 6) & 0b11
    return codes[:n]


def unpack_tensor_from_blob(
    blob: bytes,
    n: int,
    shape: list[int],
    scale: float,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    codes = unpack_codes_from_blob(blob, n)
    values = np.zeros(n, dtype=np.float32)
    values[codes == 1] = scale
    values[codes == 2] = -scale
    return torch.from_numpy(values).view(tuple(shape)).to(dtype=dtype)


def materialize_packed_mlp_into_model(
    model: torch.nn.Module,
    manifest: dict[str, Any],
    bin_path: Path,
) -> list[str]:
    named_params = dict(model.named_parameters())
    applied: list[str] = []
    missing: list[str] = []
    meta: list[str] = []

    with bin_path.open("rb") as handle:
        for item in manifest["tensors"]:
            name = item["name"]
            param = named_params.get(name)
            if param is None:
                missing.append(name)
                continue
            if getattr(param, "is_meta", False):
                meta.append(name)
                continue

            handle.seek(item["offset"])
            blob = handle.read(item["packed_bytes"])
            reconstructed = unpack_tensor_from_blob(
                blob,
                item["numel"],
                item["shape"],
                item["scale"],
                dtype=torch.float32,
            )
            if tuple(param.shape) != tuple(reconstructed.shape):
                raise RuntimeError(
                    f"Shape mismatch for {name}: "
                    f"param={tuple(param.shape)} reconstructed={tuple(reconstructed.shape)}"
                )

            with torch.no_grad():
                param.copy_(reconstructed.to(device=param.device, dtype=param.dtype))
            applied.append(name)

    if missing or meta:
        raise RuntimeError(
            json.dumps(
                {"missing_params": missing, "meta_params": meta},
                ensure_ascii=False,
                indent=2,
            )
        )
    if len(applied) != EXPECTED_TENSOR_COUNT:
        raise RuntimeError(f"Expected {EXPECTED_TENSOR_COUNT} tensors, got {len(applied)}")
    return applied


def resolve_layout(model_dir: str | Path) -> dict[str, Path]:
    root = Path(model_dir).expanduser().resolve()
    paths = {
        "root": root,
        "student_pruned": root / "student_pruned",
        "adapter": root / "adapter_recovered_l78",
        "packed_manifest": root / "packed" / "manifest.json",
        "packed_bin": root / "packed" / "mlp_ternary_2bit.bin",
    }
    missing = [str(path) for key, path in paths.items() if key != "root" and not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Modelo distribuivel incompleto. Arquivos/pastas ausentes:\n"
            + "\n".join(f"- {item}" for item in missing)
        )
    return paths


def load_packed_recovered_l78(
    model_dir: str | Path,
    *,
    token: str | None = None,
    device_map: str | dict[str, Any] = "auto",
    dtype: str | torch.dtype = "auto",
    verify_sha: bool = True,
) -> tuple[Any, Any, dict[str, Any]]:
    paths = resolve_layout(model_dir)
    packed_sha = sha256_file(paths["packed_bin"])
    if verify_sha and packed_sha != EXPECTED_PACKED_SHA256:
        raise RuntimeError(f"Packed SHA mismatch: expected {EXPECTED_PACKED_SHA256}, got {packed_sha}")

    manifest = json.loads(paths["packed_manifest"].read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(str(paths["student_pruned"]), token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        str(paths["student_pruned"]),
        dtype=dtype,
        device_map=device_map,
        token=token,
    )
    applied = materialize_packed_mlp_into_model(base_model, manifest, paths["packed_bin"])
    model = PeftModel.from_pretrained(
        base_model,
        str(paths["adapter"]),
        token=token,
        is_trainable=False,
    )
    model.eval()

    report = {
        "model_dir": str(paths["root"]),
        "load_order": "student_pruned -> packed MLP materialization -> recovered adapter",
        "packed_sha256": packed_sha,
        "applied_packed_mlp_tensors": len(applied),
        "runtime_kind": "materialized_hybrid",
        "is_native_ternary_runtime": False,
    }
    return model, tokenizer, report
