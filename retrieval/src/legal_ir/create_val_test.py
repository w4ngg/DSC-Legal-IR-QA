from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_SEED = 2026
DEFAULT_VALIDATION_SIZE = 700
DEFAULT_TEST_SIZE = 700
SPLIT_FORMAT_VERSION = 1


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_training_data(path: str | Path) -> dict[str, dict[str, Any]]:
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as handle:
            payload = json.load(handle, object_pairs_hook=_reject_duplicate_json_keys)
    except json.JSONDecodeError as exc:
        raise ValueError(f"input is not valid JSON: {source}: {exc}") from exc
    if not isinstance(payload, dict) or not payload:
        raise ValueError("input must be a non-empty JSON object keyed by query ID")

    records: dict[str, dict[str, Any]] = {}
    for query_id, record in payload.items():
        if not isinstance(query_id, str) or not query_id.strip():
            raise ValueError("every query ID must be a non-empty string")
        if not isinstance(record, dict):
            raise ValueError(f"query {query_id} must be a JSON object")
        question = record.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"query {query_id}.question must be a non-empty string")
        answer = record.get("answer")
        if not isinstance(answer, list) or not answer:
            raise ValueError(f"query {query_id}.answer must be a non-empty JSON array")
        if len(answer) > 5:
            raise ValueError(f"query {query_id}.answer contains more than 5 IDs")
        normalized_answer: list[str] = []
        for document_id in answer:
            if not isinstance(document_id, str) or not document_id.strip():
                raise ValueError(
                    f"query {query_id}.answer must contain non-empty string IDs"
                )
            normalized_answer.append(document_id.strip())
        if len(set(normalized_answer)) != len(normalized_answer):
            raise ValueError(f"query {query_id}.answer contains duplicate IDs")
        records[query_id] = record
    return records


def normalize_question_for_grouping(question: str) -> str:
    """Normalize only for duplicate grouping; output records remain unchanged."""

    normalized = unicodedata.normalize("NFC", question).casefold()
    normalized = " ".join(normalized.split())
    if not normalized:
        raise ValueError("question becomes empty after grouping normalization")
    return normalized


@dataclass(frozen=True, slots=True)
class QuestionGroup:
    normalized_question: str
    query_ids: tuple[str, ...]

    @property
    def size(self) -> int:
        return len(self.query_ids)


def _question_groups(records: Mapping[str, Mapping[str, Any]]) -> list[QuestionGroup]:
    grouped: dict[str, list[str]] = {}
    for query_id, record in records.items():
        group_key = normalize_question_for_grouping(str(record["question"]))
        grouped.setdefault(group_key, []).append(query_id)
    return [
        QuestionGroup(group_key, tuple(query_ids))
        for group_key, query_ids in sorted(grouped.items())
    ]


def _take_exact_groups(
    groups: Sequence[QuestionGroup],
    *,
    target_size: int,
    split_name: str,
) -> tuple[list[QuestionGroup], list[QuestionGroup]]:
    selected: list[QuestionGroup] = []
    remaining: list[QuestionGroup] = []
    selected_size = 0
    for group in groups:
        if selected_size + group.size <= target_size:
            selected.append(group)
            selected_size += group.size
        else:
            remaining.append(group)
    if selected_size != target_size:
        raise ValueError(
            f"cannot create exactly {target_size} records for {split_name} without "
            "placing duplicate questions in different splits; choose another size or seed"
        )
    return selected, remaining


def split_records(
    records: Mapping[str, Mapping[str, Any]],
    *,
    validation_size: int,
    test_size: int,
    seed: int,
) -> dict[str, dict[str, Mapping[str, Any]]]:
    """Randomly assign question groups and preserve original record order."""

    total_size = len(records)
    if validation_size <= 0 or test_size <= 0:
        raise ValueError("validation_size and test_size must both be positive")
    if validation_size + test_size >= total_size:
        raise ValueError(
            "validation_size + test_size must leave at least one training record"
        )

    groups = _question_groups(records)
    random.Random(seed).shuffle(groups)
    test_groups, remaining_groups = _take_exact_groups(
        groups,
        target_size=test_size,
        split_name="test",
    )
    validation_groups, training_groups = _take_exact_groups(
        remaining_groups,
        target_size=validation_size,
        split_name="validation",
    )

    split_query_ids = {
        "train": {
            query_id for group in training_groups for query_id in group.query_ids
        },
        "val": {
            query_id for group in validation_groups for query_id in group.query_ids
        },
        "test": {query_id for group in test_groups for query_id in group.query_ids},
    }
    all_query_ids = set(records)
    if set.union(*split_query_ids.values()) != all_query_ids:
        raise RuntimeError("split invariant failed: output union differs from input")
    if any(
        split_query_ids[left] & split_query_ids[right]
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    ):
        raise RuntimeError("split invariant failed: query IDs overlap")

    result: dict[str, dict[str, Mapping[str, Any]]] = {}
    for split_name in ("train", "val", "test"):
        ids = split_query_ids[split_name]
        result[split_name] = {
            query_id: record
            for query_id, record in records.items()
            if query_id in ids
        }
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _query_id_sha256(records: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for query_id in records:
        digest.update(query_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _answer_length_histogram(records: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    histogram = Counter(len(record["answer"]) for record in records.values())
    return {str(length): histogram[length] for length in sorted(histogram)}


def _duplicate_group_statistics(
    records: Mapping[str, Mapping[str, Any]],
) -> dict[str, int]:
    duplicate_groups = [group for group in _question_groups(records) if group.size > 1]
    conflicting_groups = 0
    for group in duplicate_groups:
        answer_sets = {
            tuple(sorted(str(value) for value in records[query_id]["answer"]))
            for query_id in group.query_ids
        }
        conflicting_groups += int(len(answer_sets) > 1)
    return {
        "duplicate_question_group_count": len(duplicate_groups),
        "queries_in_duplicate_question_groups": sum(
            group.size for group in duplicate_groups
        ),
        "conflicting_label_group_count": conflicting_groups,
    }


def _write_json_temporary(destination: Path, payload: Any) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path


def create_val_test(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    validation_size: int = DEFAULT_VALIDATION_SIZE,
    test_size: int = DEFAULT_TEST_SIZE,
    seed: int = DEFAULT_SEED,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create train/validation/test artifacts without modifying the source file."""

    source = Path(input_path)
    destination_dir = Path(output_dir)
    destinations = {
        "train": destination_dir / "train.json",
        "val": destination_dir / "val.json",
        "test": destination_dir / "test.json",
        "manifest": destination_dir / "split_manifest.json",
    }
    source_resolved = source.resolve()
    if any(path.resolve() == source_resolved for path in destinations.values()):
        raise ValueError("output paths must not overwrite the input train.json")
    existing = [str(path) for path in destinations.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "output already exists; pass --overwrite to replace all split artifacts: "
            + ", ".join(existing)
        )

    records = _load_training_data(source)
    splits = split_records(
        records,
        validation_size=validation_size,
        test_size=test_size,
        seed=seed,
    )
    temporary_paths: dict[str, Path] = {}
    try:
        for split_name in ("train", "val", "test"):
            temporary_paths[split_name] = _write_json_temporary(
                destinations[split_name],
                splits[split_name],
            )

        manifest: dict[str, Any] = {
            "format_version": SPLIT_FORMAT_VERSION,
            "artifact_type": "dsc2026_task1_random_grouped_split",
            "source": {
                "path": str(source),
                "sha256": _sha256_file(source),
                "query_count": len(records),
            },
            "sampling": {
                "algorithm": "python_random_seeded_question_groups_v1",
                "seed": seed,
                "group_normalization": "Unicode NFC + casefold + collapse whitespace",
                "assignment_order": ["test", "val", "train_remainder"],
                "validation_target": validation_size,
                "test_target": test_size,
            },
            "duplicate_questions": _duplicate_group_statistics(records),
            "outputs": {},
            "invariants": {
                "query_ids_are_disjoint": True,
                "query_id_union_equals_source": True,
                "normalized_questions_are_disjoint": True,
                "source_records_are_unchanged": True,
            },
        }
        for split_name in ("train", "val", "test"):
            manifest["outputs"][split_name] = {
                "filename": destinations[split_name].name,
                "query_count": len(splits[split_name]),
                "sha256": _sha256_file(temporary_paths[split_name]),
                "query_id_sha256": _query_id_sha256(splits[split_name]),
                "gold_documents_per_query": _answer_length_histogram(
                    splits[split_name]
                ),
            }
        temporary_paths["manifest"] = _write_json_temporary(
            destinations["manifest"],
            manifest,
        )

        for split_name in ("train", "val", "test"):
            temporary_paths[split_name].replace(destinations[split_name])
        temporary_paths["manifest"].replace(destinations["manifest"])
        return manifest
    finally:
        for temporary_path in temporary_paths.values():
            temporary_path.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="legal-ir-create-val-test",
        description="Create deterministic random train/validation/test splits",
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--val-size", type=int, default=DEFAULT_VALIDATION_SIZE)
    parser.add_argument("--test-size", type=int, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        manifest = create_val_test(
            args.input,
            args.output_dir,
            validation_size=args.val_size,
            test_size=args.test_size,
            seed=args.seed,
            overwrite=args.overwrite,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
