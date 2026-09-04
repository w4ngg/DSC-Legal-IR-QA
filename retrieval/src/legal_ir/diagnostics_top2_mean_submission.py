from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_TOP_K_DOCUMENTS = 5
TOP2_MEAN_SIZE = 2


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json_object(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as handle:
            payload = json.load(handle, object_pairs_hook=_reject_duplicate_json_keys)
    except json.JSONDecodeError as exc:
        raise ValueError(f"diagnostics is not valid JSON: {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("diagnostics must be a JSON object keyed by query ID")
    return {str(query_id): record for query_id, record in payload.items()}


def _write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _required_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    score = float(value)
    if not math.isfinite(score):
        raise ValueError(f"{field} must be a finite number")
    return score


def _required_document_id(value: Any, *, field: str) -> str:
    if value is None:
        raise ValueError(f"{field} is required")
    document_id = str(value).strip()
    if not document_id:
        raise ValueError(f"{field} must not be empty")
    return document_id


def top2_mean_score(candidate: Mapping[str, Any], *, query_id: str) -> float:
    """Aggregate a candidate's evidence reranker scores with top-2 mean."""

    scores_by_chunk = candidate.get("evidence_rerank_scores")
    if not isinstance(scores_by_chunk, Mapping) or not scores_by_chunk:
        document_id = candidate.get("document_id", "<unknown>")
        raise ValueError(
            f"query {query_id} candidate {document_id} has no "
            "evidence_rerank_scores; run search with reranker diagnostics first"
        )

    scores = sorted(
        (
            _required_number(
                value,
                field=(
                    f"query {query_id} candidate "
                    f"{candidate.get('document_id', '<unknown>')}"
                    f".evidence_rerank_scores[{chunk_id!r}]"
                ),
            )
            for chunk_id, value in scores_by_chunk.items()
        ),
        reverse=True,
    )
    top_scores = scores[:TOP2_MEAN_SIZE]
    return sum(top_scores) / len(top_scores)


def rerank_candidates_by_top2_mean(
    candidates: Sequence[Mapping[str, Any]],
    *,
    query_id: str,
) -> list[str]:
    """Return document IDs sorted by top-2 mean evidence score."""

    scored: list[tuple[float, float, str]] = []
    seen: set[str] = set()
    for position, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, Mapping):
            raise ValueError(
                f"query {query_id}.fused_candidates[{position}] must be an object"
            )
        document_id = _required_document_id(
            candidate.get("document_id"),
            field=f"query {query_id}.fused_candidates[{position}].document_id",
        )
        if document_id in seen:
            continue
        seen.add(document_id)
        top2_score = top2_mean_score(candidate, query_id=query_id)
        fusion_score = _required_number(
            candidate.get("fusion_score", 0.0),
            field=f"query {query_id} candidate {document_id}.fusion_score",
        )
        scored.append((top2_score, fusion_score, document_id))

    scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [document_id for _, _, document_id in scored]


def diagnostics_to_top2_mean_submission(
    diagnostics: Mapping[str, Any],
    *,
    top_k_documents: int = DEFAULT_TOP_K_DOCUMENTS,
) -> dict[str, dict[str, list[str]]]:
    if top_k_documents <= 0 or top_k_documents > DEFAULT_TOP_K_DOCUMENTS:
        raise ValueError(
            f"top_k_documents must be between 1 and {DEFAULT_TOP_K_DOCUMENTS}"
        )

    submission: dict[str, dict[str, list[str]]] = {}
    for query_id, record in diagnostics.items():
        if not isinstance(record, Mapping):
            raise ValueError(f"query {query_id} diagnostics must be an object")
        candidates = record.get("fused_candidates")
        if not isinstance(candidates, list):
            raise ValueError(f"query {query_id}.fused_candidates must be a list")
        document_ids = rerank_candidates_by_top2_mean(candidates, query_id=query_id)
        submission[str(query_id)] = {"answer": document_ids[:top_k_documents]}
    return submission


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="legal-ir-diagnostics-top2-mean",
        description=(
            "Replay a diagnostics.json file and emit a Task 1 submission using "
            "top-2 mean over evidence_rerank_scores instead of final MaxP."
        ),
    )
    parser.add_argument("--diagnostics", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--top-k-documents",
        type=int,
        default=DEFAULT_TOP_K_DOCUMENTS,
        help="number of documents per query; defaults to 5 and cannot exceed 5",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        submission = diagnostics_to_top2_mean_submission(
            _read_json_object(args.diagnostics),
            top_k_documents=args.top_k_documents,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    _write_json(args.output, submission)
    print(
        json.dumps(
            {
                "diagnostics": str(args.diagnostics),
                "output": str(args.output),
                "query_count": len(submission),
                "top_k_documents": args.top_k_documents,
                "aggregation": "top2_mean_evidence_rerank_scores",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
