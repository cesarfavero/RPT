"""Validate RPT artifacts before loading them or replacing an installation.

This module deliberately uses only the Python standard library. Checksums prove
integrity against the supplied manifest, not the identity of its publisher.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path, PurePosixPath
from typing import Any


PACKED_FORMAT = "rpt_hybrid_ternary_mlp_lora_v1"
CHECKSUM_FORMAT = "rpt_artifact_checksums_v1"
LEGACY_PACKED_SHA256 = "0b6849fe3ceaab615a4e54f63f5a96e9d9d431ba03b7b791a7ab0638ea174116"
EXPECTED_TENSOR_COUNT = 78
_SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")
_RESERVED = bytes(int(any(((value >> shift) & 3) == 3 for shift in (0, 2, 4, 6))) for value in range(256))


class ArtifactError(ValueError):
    """An artifact failed structural or integrity validation."""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(value: Any, label: str = "SHA256") -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ArtifactError(f"{label} must be a 64-character SHA256 hex digest")
    return value.lower()


def resolve_expected_packed_sha256(
    production: dict[str, Any], *, override: str | None = None,
) -> str:
    """Resolve a mandatory digest for a verified load or installation.

    Only an *absent* manifest key permits the historical digest (or an explicit
    externally supplied digest). A present null/empty/malformed key is invalid,
    including when an override was supplied. Two supplied digests must agree.
    Disabling verification is an explicit loader option, never a JSON value.
    """
    manifest_hash = None
    if "expected_packed_sha256" in production:
        manifest_hash = require_sha256(production["expected_packed_sha256"], "production.expected_packed_sha256")
    if override is not None:
        supplied = require_sha256(override, "Expected packed SHA256 override")
        if manifest_hash is not None and supplied != manifest_hash:
            raise ArtifactError("Expected packed SHA256 override disagrees with production manifest")
        return supplied
    return manifest_hash if manifest_hash is not None else LEGACY_PACKED_SHA256


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ArtifactError(f"{label} must be an integer >= {minimum}")
    return value


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"Cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArtifactError(f"Expected a JSON object in {path}")
    return value


def validate_packed_codes(blob: bytes, numel: int) -> None:
    """Validate one tensor before constructing a direct packed layer."""
    _integer(numel, 'numel', 1)
    if len(blob) != (numel + 3) // 4:
        raise ArtifactError('Packed byte count does not match numel')
    if blob.translate(_RESERVED).count(b'\x01'):
        raise ArtifactError('Reserved ternary code 3')
    if numel % 4 and blob[-1] >> ((numel % 4) * 2):
        raise ArtifactError('Padding codes must be zero')


def validate_packed_manifest(
    manifest: dict[str, Any],
    bin_path: Path,
    *,
    expected_tensor_count: int | None = None,
    expected_sha256: str | None = None,
    verify_codes: bool = True,
) -> dict[str, Any]:
    """Validate the complete v1 tensor table and stream-check the packed bytes.

    Reject overlapping/gapped/out-of-bounds regions, malformed shapes, reserved
    data codes, nonzero padding and trailing bytes. No weights are materialized.
    """
    if not isinstance(manifest, dict) or manifest.get("format") != PACKED_FORMAT:
        raise ArtifactError(f"Unsupported packed format; expected {PACKED_FORMAT}")
    codebook = manifest.get("codebook")
    if codebook is not None and codebook != {"0": "0", "1": "+scale", "2": "-scale", "3": "reserved"}:
        raise ArtifactError("Packed codebook does not match the v1 runtime")
    tensors = manifest.get("tensors")
    if not isinstance(tensors, list) or not tensors:
        raise ArtifactError("Packed manifest must contain a nonempty tensor list")
    if expected_tensor_count is not None and len(tensors) != expected_tensor_count:
        raise ArtifactError(f"Expected {expected_tensor_count} tensors, got {len(tensors)}")
    names: set[str] = set()
    end = 0
    for index, item in enumerate(tensors):
        if not isinstance(item, dict):
            raise ArtifactError(f"Tensor {index} must be an object")
        name = item.get("name")
        if not isinstance(name, str) or not name.endswith(".weight") or name in names:
            raise ArtifactError(f"Invalid or duplicate tensor name at index {index}: {name!r}")
        names.add(name)
        shape = item.get("shape")
        if not isinstance(shape, list) or len(shape) != 2:
            raise ArtifactError(f"{name}: expected a two-dimensional shape")
        rows, cols = (_integer(n, f"{name} shape", 1) for n in shape)
        numel = _integer(item.get("numel"), f"{name} numel", 1)
        if numel != rows * cols:
            raise ArtifactError(f"{name}: numel does not match shape")
        packed_bytes = _integer(item.get("packed_bytes"), f"{name} packed_bytes", 1)
        if packed_bytes != (numel + 3) // 4:
            raise ArtifactError(f"{name}: packed byte count does not match numel")
        offset = _integer(item.get("offset"), f"{name} offset")
        if offset != end:
            raise ArtifactError(f"{name}: noncontiguous/overlapping offset {offset}, expected {end}")
        end += packed_bytes
        if "pad_codes" in item and _integer(item["pad_codes"], f"{name} pad_codes") != packed_bytes * 4 - numel:
            raise ArtifactError(f"{name}: incorrect padding count")
        scale = item.get("scale")
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale < 0:
            raise ArtifactError(f"{name}: scale must be finite and nonnegative")
    bin_path = Path(bin_path)
    if not bin_path.is_file() or bin_path.stat().st_size != end:
        raise ArtifactError(f"Packed file length does not match tensor table: expected {end} bytes")
    summary = manifest.get("summary", {})
    if not isinstance(summary, dict):
        raise ArtifactError("Manifest summary must be an object")
    for key, expected in (("num_tensors", len(tensors)), ("packed_total_bytes", end)):
        if key in summary and _integer(summary[key], f"summary.{key}") != expected:
            raise ArtifactError(f"Manifest summary.{key} mismatch")
    digest = hashlib.sha256()
    with bin_path.open("rb") as handle:
        for item in tensors:
            remaining = item["packed_bytes"]
            while remaining:
                chunk = handle.read(min(remaining, 1024 * 1024))
                if not chunk:
                    raise ArtifactError("Packed file changed or was truncated while validating")
                digest.update(chunk)
                remaining -= len(chunk)
                if verify_codes:
                    if chunk.translate(_RESERVED).count(b"\x01"):
                        raise ArtifactError(f"{item['name']}: reserved ternary code 3")
                    if remaining == 0 and item["numel"] % 4:
                        used_bits = (item["numel"] % 4) * 2
                        if chunk[-1] >> used_bits:
                            raise ArtifactError(f"{item['name']}: padding codes must be zero")
        if handle.read(1):
            raise ArtifactError("Packed file grew while validating")
    actual_sha = digest.hexdigest()
    if expected_sha256 is not None and actual_sha != require_sha256(expected_sha256, "Expected packed SHA256"):
        raise ArtifactError(f"Packed SHA256 mismatch: expected {expected_sha256}, got {actual_sha}")
    return {"format": PACKED_FORMAT, "num_tensors": len(tensors), "packed_bytes": end,
            "packed_sha256": actual_sha, "codes_validated": verify_codes}


def _relative_file(root: Path, name: str) -> Path:
    pure = PurePosixPath(name)
    if not name or "\\" in name or pure.is_absolute() or any(part in (".", "..") for part in pure.parts):
        raise ArtifactError(f"Invalid checksum path: {name!r}")
    path = root.joinpath(*pure.parts)
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ArtifactError(f"Checksum path is not a regular file inside bundle: {name}")
    return path


def verify_bundle_checksums(root: Path, *, required: bool = False) -> dict[str, Any]:
    """Check all files when a complete checksum manifest is supplied."""
    checksum_path = root / "artifact_checksums.json"
    files = [p for p in root.rglob("*") if p.is_file()]
    if any(p.is_symlink() for p in root.rglob("*")):
        raise ArtifactError("Bundle contents must not contain symbolic links")
    if not checksum_path.exists():
        if required:
            raise ArtifactError("Complete artifact_checksums.json is required")
        return {"complete_checksums": False, "verified_files": 0}
    payload = read_json(checksum_path)
    if payload.get("format") != CHECKSUM_FORMAT or not isinstance(payload.get("files"), dict):
        raise ArtifactError("Unsupported artifact_checksums.json format")
    expected = payload["files"]
    actual = {p.relative_to(root).as_posix() for p in files if p != checksum_path}
    if set(expected) != actual:
        raise ArtifactError(f"Checksum coverage mismatch: missing={sorted(actual - set(expected))}, extra={sorted(set(expected) - actual)}")
    for name, digest in expected.items():
        path = _relative_file(root, name)
        if sha256_file(path) != require_sha256(digest, name):
            raise ArtifactError(f"Checksum mismatch: {name}")
    return {"complete_checksums": True, "verified_files": len(expected)}


def _nonempty_file(root: Path, relative: str) -> Path:
    path = _relative_file(root, relative)
    if path.stat().st_size == 0:
        raise ArtifactError(f"Empty model artifact: {relative}")
    return path


def _validate_weights(root: Path, *, adapter: bool = False) -> dict[str, Any]:
    """Check checkpoint files and shard references without deserializing weights."""
    stems = ("adapter_model.safetensors", "adapter_model.bin") if adapter else ("model.safetensors", "pytorch_model.bin")
    singles = [name for name in stems if (root / name).exists()]
    indices = [] if adapter else [name + ".index.json" for name in stems if (root / (name + ".index.json")).exists()]
    if not singles and not indices:
        raise ArtifactError(f"Missing {'adapter' if adapter else 'model'} weights in {root.name}")
    files = set(singles)
    for name in singles:
        _nonempty_file(root, name)
    for name in indices:
        index = read_json(_nonempty_file(root, name))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ArtifactError(f"{root.name}/{name}: weight_map must be a nonempty object")
        expected_suffix = ".safetensors" if name.startswith(stems[0]) else ".bin"
        for weight_name, shard in weight_map.items():
            if not isinstance(weight_name, str) or not weight_name or not isinstance(shard, str) or not shard.endswith(expected_suffix):
                raise ArtifactError(f"{root.name}/{name}: invalid weight or shard name")
            _nonempty_file(root, shard)
            files.add(shard)
    return {"files": sorted(files), "indices": sorted(indices)}


def validate_model_layout(root: Path) -> dict[str, Any]:
    """Require the local Hugging Face/PEFT files consumed by this runtime.

    This is a minimum layout gate, not a model-load or numerical validation.
    Binary tensor contents and architecture compatibility still require loading
    the model; those checks cannot be certified by file names and checksums.
    """
    student = root / "student_pruned"
    adapter = root / "adapter_recovered_l78"
    config = read_json(_nonempty_file(student, "config.json"))
    if not isinstance(config.get("model_type"), str) or not config["model_type"].strip():
        raise ArtifactError("student_pruned/config.json must declare a model_type")
    read_json(_nonempty_file(student, "tokenizer_config.json"))
    tokenizers = [name for name in ("tokenizer.json", "tokenizer.model") if (student / name).exists()]
    if not tokenizers:
        raise ArtifactError("Missing tokenizer.json or tokenizer.model in student_pruned")
    for name in tokenizers:
        path = _nonempty_file(student, name)
        if name.endswith(".json"):
            payload = read_json(path)
            if not isinstance(payload.get("model"), dict) or not payload["model"]:
                raise ArtifactError("student_pruned/tokenizer.json must contain a tokenizer model")
    adapter_config = read_json(_nonempty_file(adapter, "adapter_config.json"))
    if adapter_config.get("peft_type") != "LORA":
        raise ArtifactError("adapter_config.json must declare peft_type LORA")
    _integer(adapter_config.get("r"), "adapter_config.r", 1)
    return {"validation": "minimum_layout_only", "model_type": config["model_type"],
            "tokenizer_files": tokenizers, "student_weights": _validate_weights(student),
            "adapter_weights": _validate_weights(adapter, adapter=True)}


def validate_bundle(root: Path, *, require_full_checksums: bool = False) -> dict[str, Any]:
    root = Path(root).resolve()
    for name in ("student_pruned", "adapter_recovered_l78"):
        if not (root / name).is_dir():
            raise ArtifactError(f"Missing bundle directory: {name}")
    for name in ("packed/manifest.json", "packed/mlp_ternary_2bit.bin", "production_manifest.json", "prewarm_profile.json"):
        if not (root / name).is_file():
            raise ArtifactError(f"Missing bundle file: {name}")
    checksums = verify_bundle_checksums(root, required=require_full_checksums)
    production = read_json(root / "production_manifest.json")
    read_json(root / "prewarm_profile.json")
    expected_hash = resolve_expected_packed_sha256(production)
    expected_count = production.get("expected_packed_mlp_tensors", EXPECTED_TENSOR_COUNT)
    _integer(expected_count, "expected_packed_mlp_tensors", 1)
    if expected_count != EXPECTED_TENSOR_COUNT:
        raise ArtifactError(f"This runtime requires exactly {EXPECTED_TENSOR_COUNT} packed MLP tensors")
    layout = validate_model_layout(root)
    packed = validate_packed_manifest(read_json(root / "packed/manifest.json"), root / "packed/mlp_ternary_2bit.bin",
                                     expected_tensor_count=EXPECTED_TENSOR_COUNT, expected_sha256=expected_hash)
    return {"packed": packed, "model_layout": layout, **checksums}
