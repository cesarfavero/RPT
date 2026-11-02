"""Freeze data identities and reject train/development/test contamination.

Hashes detect exact normalized prompt reuse, not semantic paraphrases or unknown
foundation-model pretraining. A held-out classification is scoped to the declared
RPT recovery data, never to the foundation model's undisclosed pretraining corpus.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping
import unicodedata


SCHEMA_VERSION = "rpt-splits-v1"
NORMALIZATION_VERSION = "unicode-nfkc-casefold-accentfold-whitespace-punctuation-v1"
SPLIT_NAMES = frozenset({"train", "development", "test"})
HISTORICAL_QUESTIONS = Path(__file__).resolve().parents[1] / "gemma4_theory/data/gemma4_100_questions.json"


class SplitValidationError(ValueError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_prompt(value: Any) -> str:
    if not isinstance(value, str):
        raise SplitValidationError("A prompt must be a nonempty string")
    text = unicodedata.normalize("NFKC", value).casefold()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([?!.,;:])", r"\1", text).rstrip("?!.").strip()
    if not text:
        raise SplitValidationError("A prompt must be a nonempty string")
    return text


def prompt_sha256(value: str) -> str:
    return sha256_bytes(normalize_prompt(value).encode("utf-8"))


def _overlap_hashes(record: Mapping[str, Any], primary_field: str | None = None) -> list[str]:
    fields = {"question", "prompt", "text", "rendered_prompt"}
    if primary_field:
        fields.add(primary_field)
    values = [record[field] for field in fields if isinstance(record.get(field), str) and record[field].strip()]
    if not values:
        raise SplitValidationError("Record has no textual prompt for exposure checking")
    return sorted({prompt_sha256(value) for value in values})


def load_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() == ".jsonl":
            records = [json.loads(line) for line in handle if line.strip()]
        else:
            records = json.load(handle)
    if isinstance(records, dict):
        records = records.get("records", records.get("questions"))
    if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
        raise SplitValidationError(f"Expected a JSON list of records: {path}")
    if not records:
        raise SplitValidationError(f"Empty dataset: {path}")
    return records


def record_fingerprints(
    records: Iterable[Mapping[str, Any]],
    *,
    dataset_id: str,
    prompt_field: str | None = None,
) -> list[dict[str, Any]]:
    result = []
    seen_ids: set[str] = set()
    seen_prompts: set[str] = set()
    for record in records:
        key = prompt_field or next((name for name in ("question", "prompt", "text") if name in record), None)
        if not key or key not in record:
            raise SplitValidationError("Record is missing a question, prompt, or text field")
        fingerprint = prompt_sha256(record[key])
        namespace = str(record.get("dataset_id", dataset_id)).strip().casefold()
        identifier = record.get("id", fingerprint)
        if identifier is None or not str(identifier).strip():
            raise SplitValidationError("Record contains an empty id")
        normalized_id = f"{namespace}:{unicodedata.normalize('NFKC', str(identifier)).strip().casefold()}"
        if normalized_id in seen_ids:
            raise SplitValidationError(f"Duplicate record id: {normalized_id}")
        if fingerprint in seen_prompts:
            raise SplitValidationError(f"Duplicate normalized prompt in dataset: {normalized_id}")
        seen_ids.add(normalized_id)
        seen_prompts.add(fingerprint)
        result.append({
            "id": normalized_id,
            "prompt_sha256": fingerprint,
            "overlap_prompt_sha256": _overlap_hashes(record, key),
            "record_sha256": sha256_bytes(canonical_json(dict(record)).encode("utf-8")),
        })
    return result


def find_overlaps(fingerprints_by_split: Mapping[str, list[Mapping[str, Any]]]) -> list[dict[str, str]]:
    """Report cross-split overlap by normalized prompt OR namespaced record ID."""
    overlaps = []
    names = sorted(fingerprints_by_split)
    for index, left in enumerate(names):
        left_prompts = {digest: row["id"] for row in fingerprints_by_split[left]
                        for digest in row.get("overlap_prompt_sha256", [row["prompt_sha256"]])}
        left_ids = {row["id"] for row in fingerprints_by_split[left]}
        for right in names[index + 1:]:
            for row in fingerprints_by_split[right]:
                shared = set(row.get("overlap_prompt_sha256", [row["prompt_sha256"]])) & set(left_prompts)
                reason = "prompt" if shared else "id" if row["id"] in left_ids else None
                if reason:
                    overlaps.append({"left": left, "right": right, "reason": reason,
                                     "record_id": row["id"], "prompt_sha256": row["prompt_sha256"]})
    return overlaps


def _digest_manifest(manifest: Mapping[str, Any]) -> str:
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    return sha256_bytes(canonical_json(body).encode("utf-8"))


def build_split_manifest(
    records_by_split: Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    dataset_id: str,
    known_training_records: Iterable[Mapping[str, Any]] = (),
    training_data_complete: bool = False,
    prompt_field: str | None = None,
    allow_train_development_overlap: bool = False,
    provenance: Mapping[str, Any] | None = None,
    include_historical_exposure: bool = True,
) -> dict[str, Any]:
    """Create a manifest without writing it; callers may attach hashed sources.

    Default historical exposure is intentional: the 100 QA examples cannot become
    held-out merely by moving them to a new filename or changing their IDs.
    """
    if not dataset_id.strip() or not records_by_split:
        raise SplitValidationError("dataset_id and at least one split are required")
    unknown = set(records_by_split) - SPLIT_NAMES
    if unknown:
        raise SplitValidationError(f"Unknown split names: {sorted(unknown)}")
    splits = {}
    for name, records in records_by_split.items():
        fingerprints = record_fingerprints(records, dataset_id=dataset_id, prompt_field=prompt_field)
        if not fingerprints:
            raise SplitValidationError(f"Empty split: {name}")
        splits[name] = {"count": len(fingerprints), "records": fingerprints}
    # Merge known recovery sources by prompt; repeated exposure is not an error.
    exposure_records = list(known_training_records)
    historical_source = None
    if include_historical_exposure:
        if not HISTORICAL_QUESTIONS.is_file():
            raise SplitValidationError("Historical training exposure file is missing; cannot verify disjointness")
        exposure_records.extend(load_records(HISTORICAL_QUESTIONS))
        historical_source = {"sha256": sha256_file(HISTORICAL_QUESTIONS), "dataset_id": "rpt-historical-100"}
    exposure_prompts = sorted({digest for record in exposure_records for digest in _overlap_hashes(record)})
    overlap_groups = {name: data["records"] for name, data in splits.items()}
    overlaps = find_overlaps(overlap_groups)
    fatal = [item for item in overlaps if not (
        allow_train_development_overlap and {item["left"], item["right"]} == {"train", "development"}
    )]
    exposure_set = set(exposure_prompts)
    exposed_test = [row for row in splits.get("test", {}).get("records", [])
                    if set(row["overlap_prompt_sha256"]) & exposure_set]
    if fatal or exposed_test:
        raise SplitValidationError(canonical_json({"split_overlaps": fatal, "test_seen_in_recovery": exposed_test}))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": dataset_id,
        "prompt_field": prompt_field,
        "training_data_complete": bool(training_data_complete),
        "held_out_scope": "declared RPT recovery data; foundation-model pretraining exposure is unknown",
        "semantic_overlap_checked": False,
        "allow_train_development_overlap": bool(allow_train_development_overlap),
        "splits": splits,
        "known_training_prompt_sha256": exposure_prompts,
        "historical_exposure": historical_source,
        "overlaps": overlaps,
        "provenance": dict(provenance or {}),
        "sources": {},
    }
    manifest["manifest_sha256"] = _digest_manifest(manifest)
    return manifest


def write_frozen_manifest(manifest: Mapping[str, Any], output_path: str | Path) -> str:
    """Create once. Never silently replace an existing experimental declaration."""
    output_path = Path(output_path)
    body = dict(manifest)
    body["manifest_sha256"] = _digest_manifest(body)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(body, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return body["manifest_sha256"]


def freeze_split_manifest(
    split_sources: Mapping[str, str | Path | list[str | Path]],
    output_path: str | Path,
    *,
    dataset_id: str,
    known_training_sources: Iterable[str | Path] = (),
    training_data_complete: bool = False,
    prompt_field: str | None = None,
    allow_train_development_overlap: bool = False,
) -> dict[str, Any]:
    output_path = Path(output_path).resolve()
    records = {}
    sources: dict[str, list[dict[str, str]]] = {}
    for split, paths in split_sources.items():
        paths = [paths] if isinstance(paths, (str, Path)) else paths
        records[split] = []
        sources[split] = []
        for path in paths:
            source = Path(path).resolve()
            records[split].extend(load_records(source))
            sources[split].append({"path": os.path.relpath(source, output_path.parent), "sha256": sha256_file(source)})
    known_training_records = []
    sources["known_training"] = []
    for path in known_training_sources:
        source = Path(path).resolve()
        known_training_records.extend(load_records(source))
        sources["known_training"].append({"path": os.path.relpath(source, output_path.parent), "sha256": sha256_file(source)})
    manifest = build_split_manifest(
        records, dataset_id=dataset_id, known_training_records=known_training_records,
        training_data_complete=training_data_complete, prompt_field=prompt_field,
        allow_train_development_overlap=allow_train_development_overlap,
    )
    manifest["sources"] = sources
    manifest["manifest_sha256"] = _digest_manifest(manifest)
    write_frozen_manifest(manifest, output_path)
    return manifest


def verify_split_manifest(
    manifest_path: str | Path,
    *,
    expected_sha256: str | None = None,
    evaluated_split: str = "test",
    records: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Verify the declaration and optionally the exact records being evaluated."""
    manifest_path = Path(manifest_path).resolve()
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("normalization_version") != NORMALIZATION_VERSION:
        raise SplitValidationError("Unsupported split manifest/normalization version")
    actual_digest = _digest_manifest(manifest)
    if manifest.get("manifest_sha256") != actual_digest:
        raise SplitValidationError("Split manifest content was changed after freezing")
    if expected_sha256 and expected_sha256 != actual_digest:
        raise SplitValidationError("Split manifest does not match the expected SHA256")
    if evaluated_split not in manifest.get("splits", {}):
        raise SplitValidationError(f"Manifest has no {evaluated_split!r} split")
    for split, sources in manifest.get("sources", {}).items():
        loaded = []
        for source in sources:
            source_path = (manifest_path.parent / source["path"]).resolve()
            if not source_path.is_file() or sha256_file(source_path) != source["sha256"]:
                raise SplitValidationError(f"Dataset source changed or is missing: {source_path}")
            loaded.extend(load_records(source_path))
        if split in manifest["splits"] and sources:
            observed = record_fingerprints(loaded, dataset_id=manifest["dataset_id"], prompt_field=manifest.get("prompt_field"))
            if observed != manifest["splits"][split]["records"]:
                raise SplitValidationError(f"Source records differ from frozen {split} split")
    if records is not None:
        observed = record_fingerprints(records, dataset_id=manifest["dataset_id"], prompt_field=manifest.get("prompt_field"))
        if observed != manifest["splits"][evaluated_split]["records"]:
            raise SplitValidationError("Evaluation records differ from the frozen split")
    historical = manifest.get("historical_exposure")
    if historical:
        if not HISTORICAL_QUESTIONS.is_file() or sha256_file(HISTORICAL_QUESTIONS) != historical["sha256"]:
            raise SplitValidationError("Historical recovery exposure record changed or is missing")
    groups = {name: data["records"] for name, data in manifest["splits"].items()}
    overlaps = find_overlaps(groups)
    fatal = [item for item in overlaps if not (
        manifest.get("allow_train_development_overlap") is True
        and {item["left"], item["right"]} == {"train", "development"}
    )]
    exposure = set(manifest.get("known_training_prompt_sha256", []))
    exposed_test = [row for row in groups.get("test", [])
                    if set(row.get("overlap_prompt_sha256", [row["prompt_sha256"]])) & exposure]
    if fatal or exposed_test:
        raise SplitValidationError("Frozen data contains forbidden cross-split or recovery/test overlap")
    classification = "development"
    if evaluated_split == "test":
        classification = "held_out" if manifest.get("training_data_complete") is True else "external_validation"
    return {
        "verified": True,
        "manifest_sha256": actual_digest,
        "evaluated_split": evaluated_split,
        "classification": classification,
        "overlap_count": 0,
        "allowed_development_overlap_count": len(overlaps),
        "sample_count": manifest["splits"][evaluated_split]["count"],
        "training_data_complete": manifest.get("training_data_complete", False),
        "held_out_scope": manifest["held_out_scope"],
        "semantic_overlap_checked": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze", help="Freeze data identities without overwriting a prior declaration")
    freeze.add_argument("--dataset-id", required=True)
    freeze.add_argument("--train", action="append", default=[])
    freeze.add_argument("--development", action="append", default=[])
    freeze.add_argument("--test", action="append", default=[])
    freeze.add_argument("--known-training", action="append", default=[])
    freeze.add_argument("--training-data-complete", action="store_true", help="Declare that all RPT recovery data has been supplied")
    freeze.add_argument("--allow-train-development-overlap", action="store_true")
    freeze.add_argument("--output", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--split-manifest", required=True)
    verify.add_argument("--expected-sha256", required=True)
    verify.add_argument("--split", choices=sorted(SPLIT_NAMES), default="test")
    args = parser.parse_args()
    if args.command == "freeze":
        sources = {name: getattr(args, name) for name in SPLIT_NAMES if getattr(args, name)}
        manifest = freeze_split_manifest(
            sources, args.output, dataset_id=args.dataset_id, known_training_sources=args.known_training,
            training_data_complete=args.training_data_complete,
            allow_train_development_overlap=args.allow_train_development_overlap,
        )
        print(json.dumps({"manifest_sha256": manifest["manifest_sha256"], "output": str(args.output)}, indent=2))
    else:
        print(json.dumps(verify_split_manifest(args.split_manifest, expected_sha256=args.expected_sha256,
                                              evaluated_split=args.split), indent=2))


if __name__ == "__main__":
    main()
