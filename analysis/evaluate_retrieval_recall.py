from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


FORMAT_VERSION = 1
SUPPORTED_CHANNELS = ("bm25", "dense", "hyde")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json_object(path: str | Path, *, label: str) -> dict[str, Any]:
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as handle:
            payload = json.load(
                handle,
                object_pairs_hook=_reject_duplicate_json_keys,
            )
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {source}")
    return payload


def _normalized_identifier(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field} must not be empty")
    return normalized


def _positive_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def load_gold_answers(path: str | Path) -> dict[str, tuple[str, ...]]:
    """Load labeled DSC Task 1 records as query ID -> gold document IDs."""

    payload = _load_json_object(path, label="gold file")
    if not payload:
        raise ValueError("gold file must contain at least one query")

    answers: dict[str, tuple[str, ...]] = {}
    for raw_query_id, record in payload.items():
        query_id = _normalized_identifier(raw_query_id, field="gold query ID")
        if not isinstance(record, dict):
            raise ValueError(f"gold query {query_id} must be a JSON object")
        raw_answer = record.get("answer")
        if not isinstance(raw_answer, list) or not raw_answer:
            raise ValueError(
                f"gold query {query_id}.answer must be a non-empty JSON array"
            )
        answer = tuple(
            _normalized_identifier(
                document_id,
                field=f"gold query {query_id}.answer",
            )
            for document_id in raw_answer
        )
        if len(set(answer)) != len(answer):
            raise ValueError(f"gold query {query_id}.answer contains duplicate IDs")
        answers[query_id] = answer
    return answers


def load_deep_diagnostics(path: str | Path) -> dict[str, Any]:
    """Load the versioned full pre-fusion diagnostics artifact."""

    payload = _load_json_object(path, label="deep diagnostics")
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            "unsupported deep diagnostics format_version: "
            f"{payload.get('format_version')!r}"
        )
    queries = payload.get("queries")
    if not isinstance(queries, dict):
        raise ValueError("deep diagnostics.queries must be a JSON object")
    return payload


def load_diagnostics(path: str | Path) -> dict[str, Any]:
    """Load regular diagnostics keyed directly by query ID."""

    return _load_json_object(path, label="diagnostics")


def _normalize_query_records(
    records: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for raw_query_id, record in records.items():
        query_id = _normalized_identifier(raw_query_id, field=f"{label} query ID")
        if query_id in normalized:
            raise ValueError(
                f"{label} query IDs collide after normalization: {query_id}"
            )
        if not isinstance(record, dict):
            raise ValueError(f"{label} query {query_id} must be a JSON object")
        normalized[query_id] = record
    return normalized


def _normalize_gold_answers(
    gold_answers: Mapping[str, Sequence[str]],
) -> dict[str, tuple[str, ...]]:
    if not gold_answers:
        raise ValueError("gold answers must contain at least one query")
    normalized: dict[str, tuple[str, ...]] = {}
    for raw_query_id, raw_answer in gold_answers.items():
        query_id = _normalized_identifier(
            str(raw_query_id),
            field="gold query ID",
        )
        if query_id in normalized:
            raise ValueError(
                f"gold query IDs collide after normalization: {query_id}"
            )
        if isinstance(raw_answer, (str, bytes)) or not isinstance(
            raw_answer, Sequence
        ):
            raise ValueError(f"gold query {query_id}.answer must be a sequence")
        answer = tuple(
            _normalized_identifier(
                document_id,
                field=f"gold query {query_id}.answer",
            )
            for document_id in raw_answer
        )
        if not answer:
            raise ValueError(f"gold query {query_id}.answer must not be empty")
        if len(set(answer)) != len(answer):
            raise ValueError(f"gold query {query_id}.answer contains duplicate IDs")
        normalized[query_id] = answer
    return normalized


def _normalize_channel_top_k(
    channel_top_k: Mapping[str, int],
) -> dict[str, int]:
    unknown = set(channel_top_k) - set(SUPPORTED_CHANNELS)
    if unknown:
        raise ValueError(f"unsupported retrieval channels: {sorted(unknown)}")
    if "bm25" not in channel_top_k or "dense" not in channel_top_k:
        raise ValueError("channel top-K must include both bm25 and dense")
    return {
        channel: _positive_integer(
            channel_top_k[channel],
            field=f"{channel} top-K",
        )
        for channel in SUPPORTED_CHANNELS
        if channel in channel_top_k
    }


def _alignment(
    gold_query_ids: set[str],
    observed_query_ids: set[str],
) -> dict[str, Any]:
    missing = sorted(gold_query_ids - observed_query_ids)
    extra = sorted(observed_query_ids - gold_query_ids)
    return {
        "missing_query_count": len(missing),
        "missing_query_ids": missing,
        "extra_query_count": len(extra),
        "extra_query_ids": extra,
        "exact_match": not missing and not extra,
    }


def _channel_documents(
    query_record: Mapping[str, Any],
    *,
    query_id: str,
    channel: str,
    top_k: int,
) -> tuple[str, ...]:
    channels = query_record.get("channels")
    if not isinstance(channels, dict):
        raise ValueError(
            f"deep diagnostics query {query_id}.channels must be a JSON object"
        )
    channel_record = channels.get(channel)
    if not isinstance(channel_record, dict):
        raise ValueError(
            f"deep diagnostics query {query_id} is missing channel {channel!r}"
        )

    requested = _positive_integer(
        channel_record.get("requested_top_k_chunks"),
        field=(
            f"deep diagnostics query {query_id} channel {channel} "
            "requested_top_k_chunks"
        ),
    )
    if top_k > requested:
        raise ValueError(
            f"cannot evaluate {channel} top-{top_k} chunks for query {query_id}; "
            f"deep diagnostics only requested top-{requested}"
        )

    raw_hits = channel_record.get("chunk_hits")
    if not isinstance(raw_hits, list):
        raise ValueError(
            f"deep diagnostics query {query_id} channel {channel}.chunk_hits "
            "must be a JSON array"
        )
    returned_count = channel_record.get("returned_chunk_count")
    if returned_count is not None and returned_count != len(raw_hits):
        raise ValueError(
            f"deep diagnostics query {query_id} channel {channel} returned count "
            "does not match chunk_hits"
        )

    parsed_hits: list[tuple[int, str, str]] = []
    seen_chunks: set[str] = set()
    seen_ranks: set[int] = set()
    for position, hit in enumerate(raw_hits, start=1):
        if not isinstance(hit, dict):
            raise ValueError(
                f"deep diagnostics query {query_id} channel {channel} "
                f"chunk hit {position} must be a JSON object"
            )
        rank = _positive_integer(
            hit.get("rank"),
            field=(
                f"deep diagnostics query {query_id} channel {channel} "
                f"chunk hit {position}.rank"
            ),
        )
        chunk_id = _normalized_identifier(
            hit.get("chunk_id"),
            field=(
                f"deep diagnostics query {query_id} channel {channel} "
                f"chunk hit {position}.chunk_id"
            ),
        )
        document_id = _normalized_identifier(
            hit.get("document_id"),
            field=(
                f"deep diagnostics query {query_id} channel {channel} "
                f"chunk hit {position}.document_id"
            ),
        )
        score = hit.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError(
                f"deep diagnostics query {query_id} channel {channel} "
                f"chunk hit {position}.score must be numeric"
            )
        if not math.isfinite(float(score)):
            raise ValueError(
                f"deep diagnostics query {query_id} channel {channel} "
                f"chunk hit {position}.score must be finite"
            )
        if chunk_id in seen_chunks:
            raise ValueError(
                f"deep diagnostics query {query_id} channel {channel} has "
                f"duplicate chunk_id {chunk_id!r}"
            )
        if rank in seen_ranks:
            raise ValueError(
                f"deep diagnostics query {query_id} channel {channel} has "
                f"duplicate rank {rank}"
            )
        seen_chunks.add(chunk_id)
        seen_ranks.add(rank)
        parsed_hits.append((rank, chunk_id, document_id))

    parsed_hits.sort(key=lambda hit: (hit[0], hit[1]))
    observed_ranks = [rank for rank, _, _ in parsed_hits]
    if observed_ranks != list(range(1, len(parsed_hits) + 1)):
        raise ValueError(
            f"deep diagnostics query {query_id} channel {channel} chunk ranks "
            "must be contiguous and one-based"
        )

    # The cutoff is applied to chunks first. Iterating in canonical channel rank
    # order means the first occurrence of a document is its MaxP/best chunk.
    documents: dict[str, None] = {}
    for _, _, document_id in parsed_hits[:top_k]:
        documents.setdefault(document_id, None)
    return tuple(documents)


def _union_documents(
    documents_by_channel: Mapping[str, Sequence[str]],
    channel_order: Sequence[str],
) -> tuple[str, ...]:
    union: dict[str, None] = {}
    for channel in channel_order:
        for document_id in documents_by_channel[channel]:
            union.setdefault(document_id, None)
    return tuple(union)


def _ranked_diagnostic_documents(
    query_record: Mapping[str, Any],
    *,
    query_id: str,
    field: str,
    cutoff: int,
) -> tuple[str, ...]:
    raw_candidates = query_record.get(field)
    if not isinstance(raw_candidates, list):
        raise ValueError(
            f"diagnostics query {query_id}.{field} must be a JSON array"
        )
    documents: list[str] = []
    seen: set[str] = set()
    for position, candidate in enumerate(raw_candidates, start=1):
        if not isinstance(candidate, dict):
            raise ValueError(
                f"diagnostics query {query_id}.{field}[{position}] must be an object"
            )
        document_id = _normalized_identifier(
            candidate.get("document_id"),
            field=f"diagnostics query {query_id}.{field}[{position}].document_id",
        )
        if document_id in seen:
            raise ValueError(
                f"diagnostics query {query_id}.{field} contains duplicate "
                f"document_id {document_id!r}"
            )
        seen.add(document_id)
        documents.append(document_id)
    return tuple(documents[:cutoff])


def _query_recall(
    gold: Sequence[str],
    documents: Sequence[str],
) -> dict[str, Any]:
    gold_set = set(gold)
    retrieved_set = set(documents)
    matched = tuple(
        document_id for document_id in gold if document_id in retrieved_set
    )
    missed = tuple(
        document_id for document_id in gold if document_id not in retrieved_set
    )
    first_gold_rank = next(
        (
            rank
            for rank, document_id in enumerate(documents, start=1)
            if document_id in gold_set
        ),
        None,
    )
    return {
        "retrieved_document_ids": list(documents),
        "retrieved_document_count": len(documents),
        "matched_gold_document_ids": list(matched),
        "missed_gold_document_ids": list(missed),
        "matched_gold_document_count": len(matched),
        "gold_document_count": len(gold_set),
        "recall": len(matched) / len(gold_set),
        "hit": bool(matched),
        "full_hit": len(matched) == len(gold_set),
        "first_gold_document_rank": first_gold_rank,
    }


def _summarize_stage(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot summarize an empty evaluation stage")
    query_count = len(records)
    total_matched = sum(
        int(record["matched_gold_document_count"]) for record in records
    )
    total_gold = sum(int(record["gold_document_count"]) for record in records)
    return {
        "query_count": query_count,
        "macro_recall": sum(float(record["recall"]) for record in records)
        / query_count,
        "micro_recall": total_matched / total_gold,
        "hit_rate": sum(bool(record["hit"]) for record in records) / query_count,
        "full_hit_rate": sum(bool(record["full_hit"]) for record in records)
        / query_count,
        "average_retrieved_documents": sum(
            int(record["retrieved_document_count"]) for record in records
        )
        / query_count,
        "total_matched_gold_documents": total_matched,
        "total_gold_documents": total_gold,
    }


def _configured_cutoff(
    deep_diagnostics: Mapping[str, Any],
    *,
    section: str,
    field: str,
) -> int | None:
    pipeline_config = deep_diagnostics.get("pipeline_config")
    if not isinstance(pipeline_config, dict):
        return None
    section_config = pipeline_config.get(section)
    if not isinstance(section_config, dict):
        return None
    value = section_config.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _normalize_cutoffs(
    cutoffs: Sequence[int] | None,
    *,
    field: str,
    default: int | None,
) -> tuple[int, ...]:
    if cutoffs is None:
        if default is None:
            raise ValueError(
                f"{field} must be provided because diagnostics config has no depth"
            )
        return (default,)
    normalized = tuple(
        sorted(
            {
                _positive_integer(value, field=field)
                for value in cutoffs
            }
        )
    )
    if not normalized:
        raise ValueError(f"{field} must contain at least one cutoff")
    if default is not None and normalized[-1] > default:
        raise ValueError(
            f"{field} requests top-{normalized[-1]}, but diagnostics only logged "
            f"top-{default}"
        )
    return normalized


@dataclass(frozen=True, slots=True)
class RetrievalRecallEvaluation:
    summary: dict[str, Any]
    per_query: dict[str, Any]


def evaluate_retrieval_recall(
    gold_answers: Mapping[str, Sequence[str]],
    deep_diagnostics: Mapping[str, Any],
    *,
    channel_top_k: Mapping[str, int],
    diagnostics: Mapping[str, Any] | None = None,
    fusion_cutoffs: Sequence[int] | None = None,
    final_cutoffs: Sequence[int] | None = None,
    strict_query_ids: bool = False,
) -> RetrievalRecallEvaluation:
    """Evaluate lane, oracle-union, fusion, and final document recall.

    Lane top-K values always refer to chunk ranks. Each channel is truncated at
    that chunk cutoff before chunks are aggregated to unique documents. This is
    intentionally different from truncating the precomputed ``document_hits``.
    """

    gold = _normalize_gold_answers(gold_answers)
    top_k = _normalize_channel_top_k(channel_top_k)
    channel_order = tuple(top_k)

    if deep_diagnostics.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            "unsupported deep diagnostics format_version: "
            f"{deep_diagnostics.get('format_version')!r}"
        )
    raw_deep_queries = deep_diagnostics.get("queries")
    if not isinstance(raw_deep_queries, dict):
        raise ValueError("deep diagnostics.queries must be a JSON object")
    deep_queries = _normalize_query_records(
        raw_deep_queries,
        label="deep diagnostics",
    )

    gold_query_ids = set(gold)
    deep_alignment = _alignment(gold_query_ids, set(deep_queries))
    diagnostics_alignment: dict[str, Any] | None = None
    diagnostic_queries: dict[str, Any] | None = None
    if diagnostics is not None:
        diagnostic_queries = _normalize_query_records(
            diagnostics,
            label="diagnostics",
        )
        diagnostics_alignment = _alignment(
            gold_query_ids,
            set(diagnostic_queries),
        )

    if strict_query_ids and not deep_alignment["exact_match"]:
        raise ValueError(
            "deep diagnostics query ID mismatch: "
            f"{deep_alignment['missing_query_count']} missing, "
            f"{deep_alignment['extra_query_count']} extra"
        )
    if (
        strict_query_ids
        and diagnostics_alignment is not None
        and not diagnostics_alignment["exact_match"]
    ):
        raise ValueError(
            "diagnostics query ID mismatch: "
            f"{diagnostics_alignment['missing_query_count']} missing, "
            f"{diagnostics_alignment['extra_query_count']} extra"
        )

    resolved_fusion_cutoffs: tuple[int, ...] = ()
    resolved_final_cutoffs: tuple[int, ...] = ()
    if diagnostics is not None:
        resolved_fusion_cutoffs = _normalize_cutoffs(
            fusion_cutoffs,
            field="fusion cutoffs",
            default=_configured_cutoff(
                deep_diagnostics,
                section="fusion",
                field="candidate_documents",
            ),
        )
        resolved_final_cutoffs = _normalize_cutoffs(
            final_cutoffs,
            field="final cutoffs",
            default=_configured_cutoff(
                deep_diagnostics,
                section="reranker",
                field="final_top_k_documents",
            ),
        )
    elif fusion_cutoffs is not None or final_cutoffs is not None:
        raise ValueError("diagnostics is required for fusion/final cutoffs")

    retrieval_stage_records: dict[str, list[dict[str, Any]]] = {
        **{channel: [] for channel in channel_order},
        "oracle_union": [],
    }
    fusion_stage_records: dict[int, list[dict[str, Any]]] = {
        cutoff: [] for cutoff in resolved_fusion_cutoffs
    }
    final_stage_records: dict[int, list[dict[str, Any]]] = {
        cutoff: [] for cutoff in resolved_final_cutoffs
    }
    per_query_records: dict[str, Any] = {}

    for query_id, gold_documents in gold.items():
        deep_record = deep_queries.get(query_id)
        documents_by_channel: dict[str, tuple[str, ...]] = {}
        if deep_record is None:
            documents_by_channel = {channel: () for channel in channel_order}
        else:
            documents_by_channel = {
                channel: _channel_documents(
                    deep_record,
                    query_id=query_id,
                    channel=channel,
                    top_k=top_k[channel],
                )
                for channel in channel_order
            }

        union_documents = _union_documents(
            documents_by_channel,
            channel_order,
        )
        retrieval_documents: dict[str, tuple[str, ...]] = {
            **documents_by_channel,
            "oracle_union": union_documents,
        }
        query_retrieval: dict[str, Any] = {}
        for stage, documents in retrieval_documents.items():
            metrics = _query_recall(gold_documents, documents)
            query_retrieval[stage] = metrics
            retrieval_stage_records[stage].append(metrics)

        query_diagnostics: dict[str, Any] | None = None
        if diagnostic_queries is not None:
            diagnostic_record = diagnostic_queries.get(query_id)
            query_fusion: dict[str, Any] = {}
            query_final: dict[str, Any] = {}
            for cutoff in resolved_fusion_cutoffs:
                documents = (
                    ()
                    if diagnostic_record is None
                    else _ranked_diagnostic_documents(
                        diagnostic_record,
                        query_id=query_id,
                        field="fused_candidates",
                        cutoff=cutoff,
                    )
                )
                metrics = _query_recall(gold_documents, documents)
                query_fusion[str(cutoff)] = metrics
                fusion_stage_records[cutoff].append(metrics)
            for cutoff in resolved_final_cutoffs:
                documents = (
                    ()
                    if diagnostic_record is None
                    else _ranked_diagnostic_documents(
                        diagnostic_record,
                        query_id=query_id,
                        field="results",
                        cutoff=cutoff,
                    )
                )
                metrics = _query_recall(gold_documents, documents)
                query_final[str(cutoff)] = metrics
                final_stage_records[cutoff].append(metrics)
            query_diagnostics = {
                "fusion": query_fusion,
                "final": query_final,
            }

        per_query_record: dict[str, Any] = {
            "gold_document_ids": list(gold_documents),
            "missing_deep_diagnostics": deep_record is None,
            "retrieval": query_retrieval,
        }
        if query_diagnostics is not None:
            per_query_record["missing_diagnostics"] = (
                diagnostic_queries.get(query_id) is None
            )
            per_query_record["diagnostics"] = query_diagnostics
        per_query_records[query_id] = per_query_record

    retrieval_summary: dict[str, Any] = {}
    for stage, records in retrieval_stage_records.items():
        stage_summary = _summarize_stage(records)
        if stage in top_k:
            stage_summary = {
                "top_k_chunks": top_k[stage],
                **stage_summary,
            }
        else:
            stage_summary = {
                "channels": list(channel_order),
                "channel_top_k_chunks": dict(top_k),
                **stage_summary,
            }
        retrieval_summary[stage] = stage_summary

    summary: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "metric": "DSC2026_Task1_retrieval_candidate_recall",
        "aggregation": "truncate_chunk_hits_then_document_maxp",
        "gold_query_count": len(gold),
        "active_channels": list(channel_order),
        "channel_top_k_chunks": dict(top_k),
        "deep_diagnostics_query_alignment": deep_alignment,
        "retrieval": retrieval_summary,
    }
    if diagnostics is not None:
        assert diagnostics_alignment is not None
        summary["diagnostics_query_alignment"] = diagnostics_alignment
        summary["diagnostics"] = {
            "fusion": {
                str(cutoff): _summarize_stage(records)
                for cutoff, records in fusion_stage_records.items()
            },
            "final": {
                str(cutoff): _summarize_stage(records)
                for cutoff, records in final_stage_records.items()
            },
        }

    return RetrievalRecallEvaluation(
        summary=summary,
        per_query={
            "format_version": FORMAT_VERSION,
            "metric": "DSC2026_Task1_retrieval_candidate_recall_per_query",
            "active_channels": list(channel_order),
            "channel_top_k_chunks": dict(top_k),
            "queries": per_query_records,
        },
    )


def _parse_cutoffs(value: str) -> tuple[int, ...]:
    raw_parts = [part.strip() for part in value.split(",")]
    if not raw_parts or any(not part for part in raw_parts):
        raise argparse.ArgumentTypeError(
            "cutoffs must be a comma-separated list of positive integers"
        )
    try:
        parsed = tuple(int(part) for part in raw_parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "cutoffs must be a comma-separated list of positive integers"
        ) from exc
    if any(value <= 0 for value in parsed):
        raise argparse.ArgumentTypeError("cutoffs must be positive")
    return tuple(sorted(set(parsed)))


def _write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate document recall from regular/deep retrieval diagnostics. "
            "Lane top-K values are chunk cutoffs applied before document aggregation."
        )
    )
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--deep-diagnostics", required=True, type=Path)
    parser.add_argument("--diagnostics", type=Path)
    parser.add_argument("--bm25-top-k", required=True, type=int)
    parser.add_argument("--dense-top-k", required=True, type=int)
    parser.add_argument(
        "--hyde-top-k",
        type=int,
        help="include the optional HyDE lane in the oracle union",
    )
    parser.add_argument(
        "--fusion-cutoffs",
        type=_parse_cutoffs,
        help=(
            "comma-separated document cutoffs for diagnostics.fused_candidates; "
            "defaults to the logged fusion candidate depth"
        ),
    )
    parser.add_argument(
        "--final-cutoffs",
        type=_parse_cutoffs,
        help=(
            "comma-separated document cutoffs for diagnostics.results; "
            "defaults to the logged final depth"
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--per-query-output", type=Path)
    parser.add_argument("--strict-query-ids", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.diagnostics is None and (
        args.fusion_cutoffs is not None or args.final_cutoffs is not None
    ):
        parser.error("--diagnostics is required for fusion/final cutoffs")
    if (
        args.output is not None
        and args.per_query_output is not None
        and args.output.expanduser().resolve(strict=False)
        == args.per_query_output.expanduser().resolve(strict=False)
    ):
        parser.error("--output and --per-query-output must be different paths")

    channel_top_k = {
        "bm25": args.bm25_top_k,
        "dense": args.dense_top_k,
    }
    if args.hyde_top_k is not None:
        channel_top_k["hyde"] = args.hyde_top_k

    try:
        evaluation = evaluate_retrieval_recall(
            load_gold_answers(args.gold),
            load_deep_diagnostics(args.deep_diagnostics),
            channel_top_k=channel_top_k,
            diagnostics=(
                load_diagnostics(args.diagnostics)
                if args.diagnostics is not None
                else None
            ),
            fusion_cutoffs=args.fusion_cutoffs,
            final_cutoffs=args.final_cutoffs,
            strict_query_ids=args.strict_query_ids,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    if args.output is not None:
        _write_json(args.output, evaluation.summary)
    if args.per_query_output is not None:
        _write_json(args.per_query_output, evaluation.per_query)
    print(json.dumps(evaluation.summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
