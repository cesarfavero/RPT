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

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - handled at runtime in non-CUDA envs
    triton = None
    tl = None


EXPECTED_PACKED_SHA256 = "0b6849fe3ceaab615a4e54f63f5a96e9d9d431ba03b7b791a7ab0638ea174116"
EXPECTED_TENSOR_COUNT = 78


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unpack_weight_pytorch(
    packed: torch.Tensor,
    *,
    numel: int,
    out_features: int,
    in_features: int,
    scale: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    raw = packed.to(device=device, non_blocking=True)
    codes = torch.empty(raw.numel() * 4, device=device, dtype=torch.uint8)
    codes[0::4] = raw & 0b11
    codes[1::4] = (raw >> 2) & 0b11
    codes[2::4] = (raw >> 4) & 0b11
    codes[3::4] = (raw >> 6) & 0b11
    codes = codes[:numel]
    weight = torch.empty(numel, device=device, dtype=torch.float32)
    weight.zero_()
    weight[codes == 1] = scale
    weight[codes == 2] = -scale
    return weight.view(out_features, in_features).to(dtype=dtype)


if triton is not None:

    @triton.jit
    def _ternary_linear_kernel(
        x_ptr,
        packed_ptr,
        out_ptr,
        M: tl.constexpr,
        K: tl.constexpr,
        N: tl.constexpr,
        scale: tl.constexpr,
        stride_xm: tl.constexpr,
        stride_xk: tl.constexpr,
        stride_om: tl.constexpr,
        stride_on: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k0 in range(0, K, BLOCK_K):
            k_idxs = k0 + offs_k
            x = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + k_idxs[None, :] * stride_xk,
                mask=(offs_m[:, None] < M) & (k_idxs[None, :] < K),
                other=0.0,
            )

            linear = offs_n[:, None] * K + k_idxs[None, :]
            byte_idx = linear // 4
            shift = (linear % 4) * 2
            raw = tl.load(
                packed_ptr + byte_idx,
                mask=(offs_n[:, None] < N) & (k_idxs[None, :] < K),
                other=0,
            ).to(tl.uint32)
            code = (raw >> shift) & 3
            w = tl.where(code == 1, scale, tl.where(code == 2, -scale, 0.0)).to(x.dtype)
            acc += tl.dot(x, tl.trans(w), out_dtype=tl.float32)

        tl.store(
            out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            acc,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )


def triton_ternary_linear(
    x: torch.Tensor,
    packed: torch.Tensor,
    *,
    out_features: int,
    in_features: int,
    scale: float,
    block_m: int = 16,
    block_n: int = 16,
    block_k: int = 64,
) -> torch.Tensor:
    if triton is None or not x.is_cuda:
        raise RuntimeError("Triton runtime requires CUDA and triton installed.")
    x2 = x.reshape(-1, in_features).contiguous()
    packed_cuda = packed.to(device=x.device, non_blocking=True)
    out = torch.empty((x2.shape[0], out_features), device=x.device, dtype=torch.float32)
    grid = (triton.cdiv(x2.shape[0], block_m), triton.cdiv(out_features, block_n))
    _ternary_linear_kernel[grid](
        x2,
        packed_cuda,
        out,
        x2.shape[0],
        in_features,
        out_features,
        float(scale),
        x2.stride(0),
        x2.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return out.reshape(*x.shape[:-1], out_features).to(dtype=x.dtype)


class TritonPackedTernaryLinear(torch.nn.Module):
    """Packed ternary linear backed by Triton, with PyTorch fallback."""

    def __init__(
        self,
        *,
        packed_bytes: bytes,
        shape: list[int],
        numel: int,
        scale: float,
        name: str,
        block_m: int = 16,
        block_n: int = 16,
        block_k: int = 64,
        fallback_to_pytorch: bool = True,
    ) -> None:
        super().__init__()
        self.name = name
        self.out_features = int(shape[0])
        self.in_features = int(shape[1])
        self.numel = int(numel)
        self.scale = float(scale)
        self.block_m = int(block_m)
        self.block_n = int(block_n)
        self.block_k = int(block_k)
        self.fallback_to_pytorch = bool(fallback_to_pytorch)
        raw = np.frombuffer(packed_bytes, dtype=np.uint8).copy()
        self.register_buffer("packed", torch.from_numpy(raw), persistent=True)

    @property
    def bias(self) -> None:
        return None

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"packed_bytes={self.packed.numel()}, backend=triton"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda and triton is not None:
            return triton_ternary_linear(
                x,
                self.packed,
                out_features=self.out_features,
                in_features=self.in_features,
                scale=self.scale,
                block_m=self.block_m,
                block_n=self.block_n,
                block_k=self.block_k,
            )
        if not self.fallback_to_pytorch:
            raise RuntimeError("Triton backend unavailable and fallback disabled.")
        weight = _unpack_weight_pytorch(
            self.packed,
            numel=self.numel,
            out_features=self.out_features,
            in_features=self.in_features,
            scale=self.scale,
            device=x.device,
            dtype=x.dtype,
        )
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


def replace_mlp_linears_with_triton(
    model: torch.nn.Module,
    manifest: dict[str, Any],
    bin_path: str | Path,
    *,
    block_m: int = 16,
    block_n: int = 16,
    block_k: int = 64,
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
            packed_layer = TritonPackedTernaryLinear(
                packed_bytes=blob,
                shape=item["shape"],
                numel=item["numel"],
                scale=item["scale"],
                name=param_name,
                block_m=block_m,
                block_n=block_n,
                block_k=block_k,
            )

            target_device = None
            base_layer = getattr(target, "base_layer", None)
            if isinstance(base_layer, torch.nn.Module):
                for param in base_layer.parameters(recurse=False):
                    target_device = param.device
                    break
                if target_device is not None:
                    packed_layer = packed_layer.to(device=target_device)
                target.base_layer = packed_layer
                replaced_lora_base_layers.append(module_path)
            else:
                parent_path, child_name = module_path.rsplit(".", 1)
                parent = root.get_submodule(parent_path)
                for param in target.parameters(recurse=False):
                    target_device = param.device
                    break
                if target_device is not None:
                    packed_layer = packed_layer.to(device=target_device)
                setattr(parent, child_name, packed_layer)
                replaced_direct_modules.append(module_path)

            del target
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
        "backend": "triton",
        "block_m": block_m,
        "block_n": block_n,
        "block_k": block_k,
    }


def load_triton_packed_recovered_l78(
    model_dir: str | Path,
    *,
    token: str | None = None,
    device_map: str | dict[str, Any] = "auto",
    dtype: str | torch.dtype = "auto",
    verify_sha: bool = True,
    block_m: int = 16,
    block_n: int = 16,
    block_k: int = 64,
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
    replacement_report = replace_mlp_linears_with_triton(
        model,
        manifest,
        paths["packed_bin"],
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
    )
    model.eval()
    report = {
        "model_dir": str(paths["root"]),
        "load_order": "student_pruned -> recovered adapter -> replace MLP base layers with TritonPackedTernaryLinear",
        "packed_sha256": packed_sha,
        "runtime_kind": "native_triton_prototype",
        "is_native_ternary_runtime": True,
        "is_optimized_kernel": True,
        "triton_available": triton is not None,
        **replacement_report,
    }
    return model, tokenizer, report
