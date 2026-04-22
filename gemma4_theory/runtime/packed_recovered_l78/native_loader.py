from __future__ import annotations

from pathlib import Path
import gc
import hashlib
import json
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
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


class PackedTernaryLinear(torch.nn.Module):
    """Prototype ternary linear layer backed by the 2-bit packed artifact.

    This is intentionally a structural prototype, not an optimized kernel. It keeps
    packed bytes as a persistent buffer and reconstructs the ternary weight inside
    forward. That proves compatibility without permanently storing the dense MLP
    weight, but it can be slower than regular dense matmul.
    """

    def __init__(
        self,
        *,
        packed_bytes: bytes,
        shape: list[int],
        numel: int,
        scale: float,
        name: str,
        cache_unpacked: bool = False,
    ) -> None:
        super().__init__()
        self.name = name
        self.out_features = int(shape[0])
        self.in_features = int(shape[1])
        self.numel = int(numel)
        self.scale = float(scale)
        self.cache_unpacked = bool(cache_unpacked)
        raw = np.frombuffer(packed_bytes, dtype=np.uint8).copy()
        self.register_buffer("packed", torch.from_numpy(raw), persistent=True)
        self.register_buffer("_cached_weight", torch.empty(0), persistent=False)

    @property
    def bias(self) -> None:
        return None

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"packed_bytes={self.packed.numel()}, cache_unpacked={self.cache_unpacked}"
        )

    def _unpack_weight(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if (
            self.cache_unpacked
            and self._cached_weight.numel() == self.numel
            and self._cached_weight.device == device
            and self._cached_weight.dtype == dtype
        ):
            return self._cached_weight.view(self.out_features, self.in_features)

        raw = self.packed.to(device=device, non_blocking=True)
        codes = torch.empty(raw.numel() * 4, device=device, dtype=torch.uint8)
        codes[0::4] = raw & 0b11
        codes[1::4] = (raw >> 2) & 0b11
        codes[2::4] = (raw >> 4) & 0b11
        codes[3::4] = (raw >> 6) & 0b11
        codes = codes[: self.numel]

        weight = torch.empty(self.numel, device=device, dtype=torch.float32)
        weight.zero_()
        weight[codes == 1] = self.scale
        weight[codes == 2] = -self.scale
        weight = weight.view(self.out_features, self.in_features).to(dtype=dtype)

        if self.cache_unpacked:
            self._cached_weight = weight.reshape(-1).detach()
        return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self._unpack_weight(x.device, x.dtype)
        return F.linear(x, weight, None)


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


def _candidate_roots(model: torch.nn.Module) -> list[torch.nn.Module]:
    roots: list[torch.nn.Module] = [model]
    base_model = getattr(model, "base_model", None)
    if isinstance(base_model, torch.nn.Module):
        roots.append(base_model)
        inner = getattr(base_model, "model", None)
        if isinstance(inner, torch.nn.Module):
            roots.append(inner)
    inner_model = getattr(model, "model", None)
    if isinstance(inner_model, torch.nn.Module):
        roots.append(inner_model)
    deduped: list[torch.nn.Module] = []
    seen: set[int] = set()
    for root in roots:
        if id(root) not in seen:
            deduped.append(root)
            seen.add(id(root))
    return deduped


def _find_module(model: torch.nn.Module, module_path: str) -> tuple[torch.nn.Module, torch.nn.Module]:
    for root in _candidate_roots(model):
        try:
            return root, root.get_submodule(module_path)
        except AttributeError:
            continue
    raise AttributeError(f"Could not resolve module path in model/PEFT wrapper: {module_path}")


def _set_child(parent: torch.nn.Module, child_name: str, value: torch.nn.Module) -> None:
    if isinstance(parent, (torch.nn.ModuleList, torch.nn.Sequential)) and child_name.isdigit():
        parent[int(child_name)] = value
    else:
        setattr(parent, child_name, value)


def _replace_direct_module(root: torch.nn.Module, module_path: str, replacement: torch.nn.Module) -> None:
    parent_path, child_name = module_path.rsplit(".", 1)
    parent = root.get_submodule(parent_path)
    _set_child(parent, child_name, replacement)


def replace_mlp_linears_with_packed(
    model: torch.nn.Module,
    manifest: dict[str, Any],
    bin_path: str | Path,
    *,
    cache_unpacked: bool = False,
) -> dict[str, Any]:
    bin_path = Path(bin_path)
    replaced: list[str] = []
    replaced_lora_base_layers: list[str] = []
    replaced_direct_modules: list[str] = []

    with bin_path.open("rb") as handle:
        for item in manifest["tensors"]:
            param_name = item["name"]
            module_path = param_name.removesuffix(".weight")
            root, target = _find_module(model, module_path)
            handle.seek(item["offset"])
            blob = handle.read(item["packed_bytes"])
            packed_layer = PackedTernaryLinear(
                packed_bytes=blob,
                shape=item["shape"],
                numel=item["numel"],
                scale=item["scale"],
                name=param_name,
                cache_unpacked=cache_unpacked,
            )

            target_device = None
            target_dtype = None
            base_layer = getattr(target, "base_layer", None)
            if isinstance(base_layer, torch.nn.Module):
                for param in base_layer.parameters(recurse=False):
                    target_device = param.device
                    target_dtype = param.dtype
                    break
                if target_device is not None:
                    packed_layer = packed_layer.to(device=target_device)
                target.base_layer = packed_layer
                replaced_lora_base_layers.append(module_path)
            else:
                for param in target.parameters(recurse=False):
                    target_device = param.device
                    target_dtype = param.dtype
                    break
                if target_device is not None:
                    packed_layer = packed_layer.to(device=target_device)
                _replace_direct_module(root, module_path, packed_layer)
                replaced_direct_modules.append(module_path)

            del target
            if target_dtype is not None:
                packed_layer.to(dtype=target_dtype)
            replaced.append(param_name)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if len(replaced) != EXPECTED_TENSOR_COUNT:
        raise RuntimeError(f"Expected {EXPECTED_TENSOR_COUNT} replacements, got {len(replaced)}")
    return {
        "replaced_count": len(replaced),
        "replaced": replaced,
        "replaced_lora_base_layers": replaced_lora_base_layers,
        "replaced_direct_modules": replaced_direct_modules,
        "cache_unpacked": cache_unpacked,
    }


def load_native_packed_recovered_l78(
    model_dir: str | Path,
    *,
    token: str | None = None,
    device_map: str | dict[str, Any] = "auto",
    dtype: str | torch.dtype = "auto",
    verify_sha: bool = True,
    cache_unpacked: bool = False,
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
    model = PeftModel.from_pretrained(
        base_model,
        str(paths["adapter"]),
        token=token,
        is_trainable=False,
    )
    replacement_report = replace_mlp_linears_with_packed(
        model,
        manifest,
        paths["packed_bin"],
        cache_unpacked=cache_unpacked,
    )
    model.eval()
    report = {
        "model_dir": str(paths["root"]),
        "load_order": "student_pruned -> recovered adapter -> replace MLP base layers with PackedTernaryLinear",
        "packed_sha256": packed_sha,
        "runtime_kind": "native_structural_prototype",
        "is_native_ternary_runtime": True,
        "is_optimized_kernel": False,
        **replacement_report,
    }
    return model, tokenizer, report
