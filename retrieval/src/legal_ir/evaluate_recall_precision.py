from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


MAX_DOCUMENTS_PER_QUERY = 5


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json_object(path: str | Path, *, label: str) -> dict[str, Any]:
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as handle:
            payload = json.load(handle, object_pairs_hook=_reject_duplicate_json_keys)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object keyed by query ID")
    return {str(query_id): record for query_id, record in payload.items()}


def _normalize_document_id(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must contain only string document IDs")
    document_id = value.strip()
    if not document_id:
        raise ValueError(f"{field} contains an empty document ID")
    return document_id


def _answer_list(
    record: Any,
    *,
    query_id: str,
    label: str,
    allow_empty: bool,
) -> tuple[str, ...]:
    if not isinstance(record, dict):
        raise ValueError(f"{label} query {query_id} must be a JSON object")
    answer = record.get("answer")
    if not isinstance(answer, list):
        raise ValueError(f"{label} query {query_id}.answer must be a JSON array")
    normalized = tuple(
        _normalize_document_id(
            document_id,
            field=f"{label} query {query_id}.answer",
        )
        for document_id in answer
    )
    if not allow_empty and not normalized:
        raise ValueError(f"{label} query {query_id}.answer must not be empty")
    return normalized


def load_gold_answers(path: str | Path) -> dict[str, tuple[str, ...]]:
    """Load labeled Task 1 data as query ID -> gold document IDs."""

    payload = _read_json_object(path, label="gold file")
    if not payload:
        raise ValueError("gold file must contain at least one query")
    answers: dict[str, tuple[str, ...]] = {}
    for query_id, record in payload.items():
        answer = _answer_list(
            record,
            query_id=query_id,
            label="gold",
            allow_empty=False,
        )
        if len(set(answer)) != len(answer):
            raise ValueError(f"gold query {query_id}.answer contains duplicate IDs")
        answers[query_id] = answer
    return answers


def load_prediction_answers(path: str | Path) -> dict[str, tuple[str, ...]]:
    """Load a Task 1 submission as query ID -> predicted document IDs."""

    payload = _read_json_object(path, label="prediction file")
    return {
        query_id: _answer_list(
            record,
            query_id=query_id,
            label="prediction",
            allow_empty=True,
        )
        for query_id, record in payload.items()
    }


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    summary: dict[str, Any]
    per_query: dict[str, dict[str, Any]]


def evaluate_predictions(
    gold_answers: Mapping[str, Sequence[str]],
    prediction_answers: Mapping[str, Sequence[str]],
    *,
    strict_query_ids: bool = False,
) -> EvaluationResult:
    """Compute the official macro Recall and Precision for DSC LegalIR.

    Missing predictions are scored as empty. Extra prediction query IDs do not
    affect the official means, but are reported. A raw answer list longer than
    five receives zero Recall and Precision exactly as required by Task 1.
    """

    if not gold_answers:
        raise ValueError("gold answers must contain at least one query")

    normalized_gold: dict[str, tuple[str, ...]] = {}
    for raw_query_id, raw_answer in gold_answers.items():
        query_id = str(raw_query_id)
        if query_id in normalized_gold:
            raise ValueError(f"gold query IDs collide after string normalization: {query_id}")
        if isinstance(raw_answer, (str, bytes)) or not isinstance(raw_answer, Sequence):
            raise ValueError(f"gold query {query_id}.answer must be a sequence")
        answer = tuple(
            _normalize_document_id(value, field=f"gold query {query_id}.answer")
            for value in raw_answer
        )
        if not answer:
            raise ValueError(f"gold query {query_id}.answer must not be empty")
        if len(set(answer)) != len(answer):
            raise ValueError(f"gold query {query_id}.answer contains duplicate IDs")
        normalized_gold[query_id] = answer

    normalized_predictions: dict[str, tuple[str, ...]] = {}
    for raw_query_id, raw_answer in prediction_answers.items():
        query_id = str(raw_query_id)
        if query_id in normalized_predictions:
            raise ValueError(
                f"prediction query IDs collide after string normalization: {query_id}"
            )
        if isinstance(raw_answer, (str, bytes)) or not isinstance(raw_answer, Sequence):
            raise ValueError(f"prediction query {query_id}.answer must be a sequence")
        normalized_predictions[query_id] = tuple(
            _normalize_document_id(
                value,
                field=f"prediction query {query_id}.answer",
            )
            for value in raw_answer
        )

    gold_query_ids = set(normalized_gold)
    prediction_query_ids = set(normalized_predictions)
    missing_query_ids = sorted(gold_query_ids - prediction_query_ids)
    extra_query_ids = sorted(prediction_query_ids - gold_query_ids)
    if strict_query_ids and (missing_query_ids or extra_query_ids):
        raise ValueError(
            "query ID mismatch: "
            f"{len(missing_query_ids)} missing, {len(extra_query_ids)} extra"
        )

    recall_sum = 0.0
    precision_sum = 0.0
    returned_document_count = 0
    unique_returned_document_count = 0
    over_limit_count = 0
    duplicate_prediction_count = 0
    empty_prediction_count = 0
    per_query: dict[str, dict[str, Any]] = {}

    for query_id, gold in normalized_gold.items():
        predicted = normalized_predictions.get(query_id, ())
        predicted_unique = tuple(dict.fromkeys(predicted))
        returned_document_count += len(predicted)
        unique_returned_document_count += len(predicted_unique)
        empty_prediction_count += int(not predicted)
        has_duplicates = len(predicted_unique) != len(predicted)
        duplicate_prediction_count += int(has_duplicates)
        over_limit = len(predicted) > MAX_DOCUMENTS_PER_QUERY
        over_limit_count += int(over_limit)

        gold_set = set(gold)
        predicted_set = set(predicted_unique)
        matched = tuple(document_id for document_id in gold if document_id in predicted_set)
        if over_limit:
            recall = 0.0
            precision = 0.0
        else:
            recall = len(matched) / len(gold_set)
            precision = (
                len(matched) / len(predicted_set) if predicted_set else 0.0
            )

        recall_sum += recall
        precision_sum += precision
        per_query[query_id] = {
            "gold_answer": list(gold),
            "predicted_answer": list(predicted),
            "matched_answer": list(matched),
            "recall": recall,
            "precision": precision,
            "prediction_count": len(predicted),
            "unique_prediction_count": len(predicted_unique),
            "missing_prediction": query_id not in normalized_predictions,
            "contains_duplicates": has_duplicates,
            "over_max_documents": over_limit,
        }

    query_count = len(normalized_gold)
    summary: dict[str, Any] = {
        "contract_version": 1,
        "metric": "DSC2026_Task1_LegalIR_official_macro",
        "official_macro_recall": recall_sum / query_count,
        "official_macro_precision": precision_sum / query_count,
        "gold_query_count": query_count,
        "max_documents_per_query": MAX_DOCUMENTS_PER_QUERY,
        "missing_prediction_query_count": len(missing_query_ids),
        "missing_prediction_query_ids": missing_query_ids,
        "extra_prediction_query_count": len(extra_query_ids),
        "extra_prediction_query_ids": extra_query_ids,
        "over_limit_query_count": over_limit_count,
        "duplicate_prediction_query_count": duplicate_prediction_count,
        "empty_prediction_query_count": empty_prediction_count,
        "submission_contract_valid": not (
            missing_query_ids
            or extra_query_ids
            or over_limit_count
            or duplicate_prediction_count
        ),
        "average_returned_documents": returned_document_count / query_count,
        "average_unique_returned_documents": (
            unique_returned_document_count / query_count
        ),
    }
    return EvaluationResult(summary=summary, per_query=per_query)


def _write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="legal-ir-evaluate",
        description="Evaluate DSC 2026 Task 1 submission Recall and Precision",
    )
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="optional path for the aggregate JSON report",
    )
    parser.add_argument(
        "--per-query-output",
        type=Path,
        help="optional path for detailed scores of every gold query",
    )
    parser.add_argument(
        "--strict-query-ids",
        action="store_true",
        help="fail instead of scoring missing queries as empty or ignoring extras",
    )
    parser.add_argument(
        "--strict-submission",
        action="store_true",
        help="return a non-zero exit code when any submission contract check fails",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = evaluate_predictions(
            load_gold_answers(args.gold),
            load_prediction_answers(args.predictions),
            strict_query_ids=args.strict_query_ids,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    if args.output:
        _write_json(args.output, result.summary)
    if args.per_query_output:
        _write_json(
            args.per_query_output,
            {
                "summary": result.summary,
                "queries": result.per_query,
            },
        )
    print(json.dumps(result.summary, ensure_ascii=False, indent=2))
    if args.strict_submission and not result.summary["submission_contract_valid"]:
        print(
            "submission contract is invalid; inspect the JSON report for violations",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
