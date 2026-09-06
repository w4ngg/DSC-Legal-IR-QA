from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


FORMAT_VERSION = 1
DEFAULT_TOP_K_VALUES = (100, 150, 200, 250, 300, 350, 400)
CHANNELS = ("bm25", "dense")


def _identifier(value: Any, *, field: str) -> str:
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


def _normalize_cutoffs(values: Sequence[int], *, field: str) -> tuple[int, ...]:
    normalized = tuple(
        sorted({_positive_integer(value, field=field) for value in values})
    )
    if not normalized:
        raise ValueError(f"{field} must contain at least one cutoff")
    return normalized


def _parse_cutoffs(value: str) -> tuple[int, ...]:
    parts = [part.strip() for part in value.split(",")]
    if not parts or any(not part for part in parts):
        raise argparse.ArgumentTypeError(
            "top-K values must be comma-separated positive integers"
        )
    try:
        values = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "top-K values must be comma-separated positive integers"
        ) from exc
    if any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("top-K values must be positive")
    return tuple(sorted(set(values)))


def load_deep_diagnostics(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"deep diagnostics must be a JSON object: {source}")
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            "unsupported deep diagnostics format_version: "
            f"{payload.get('format_version')!r}"
        )
    queries = payload.get("queries")
    if not isinstance(queries, dict) or not queries:
        raise ValueError("deep diagnostics.queries must be a non-empty object")
    return payload


def _ranked_short_chunk_ids(
    query_record: Mapping[str, Any],
    *,
    query_id: str,
    channel: str,
    maximum_top_k: int,
) -> tuple[str, ...]:
    channels = query_record.get("channels")
    if not isinstance(channels, dict):
        raise ValueError(
            f"deep diagnostics query {query_id}.channels must be an object"
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
    if requested < maximum_top_k:
        raise ValueError(
            f"cannot evaluate {channel} top-{maximum_top_k} for query {query_id}; "
            f"deep diagnostics only requested top-{requested}"
        )

    raw_hits = channel_record.get("chunk_hits")
    if not isinstance(raw_hits, list):
        raise ValueError(
            f"deep diagnostics query {query_id} channel {channel}.chunk_hits "
            "must be an array"
        )
    returned_count = channel_record.get("returned_chunk_count")
    if returned_count is not None and returned_count != len(raw_hits):
        raise ValueError(
            f"deep diagnostics query {query_id} channel {channel} returned count "
            "does not match chunk_hits"
        )
    if len(raw_hits) < maximum_top_k:
        raise ValueError(
            f"cannot evaluate {channel} top-{maximum_top_k} for query {query_id}; "
            f"deep diagnostics only contains {len(raw_hits)} chunk hits"
        )

    ranked: list[tuple[int, str]] = []
    seen_ranks: set[int] = set()
    seen_chunks: set[str] = set()
    for position, hit in enumerate(raw_hits, start=1):
        if not isinstance(hit, dict):
            raise ValueError(
                f"deep diagnostics query {query_id} channel {channel} hit "
                f"{position} must be an object"
            )
        rank = _positive_integer(
            hit.get("rank"),
            field=(
                f"deep diagnostics query {query_id} channel {channel} "
                f"hit {position}.rank"
            ),
        )
        short_chunk_id = _identifier(
            hit.get("chunk_id"),
            field=(
                f"deep diagnostics query {query_id} channel {channel} "
                f"hit {position}.chunk_id"
            ),
        )
        if rank in seen_ranks:
            raise ValueError(
                f"deep diagnostics query {query_id} channel {channel} has "
                f"duplicate rank {rank}"
            )
        if short_chunk_id in seen_chunks:
            raise ValueError(
                f"deep diagnostics query {query_id} channel {channel} has "
                f"duplicate chunk_id {short_chunk_id!r}"
            )
        seen_ranks.add(rank)
        seen_chunks.add(short_chunk_id)
        ranked.append((rank, short_chunk_id))

    ranked.sort(key=lambda item: (item[0], item[1]))
    ranks = [rank for rank, _ in ranked]
    if ranks != list(range(1, len(ranked) + 1)):
        raise ValueError(
            f"deep diagnostics query {query_id} channel {channel} ranks must "
            "be contiguous and one-based"
        )
    return tuple(short_chunk_id for _, short_chunk_id in ranked[:maximum_top_k])


@dataclass(frozen=True, slots=True)
class PreparedCandidates:
    query_chunks: dict[str, dict[str, tuple[str, ...]]]
    required_short_chunk_ids: frozenset[str]


def prepare_candidates(
    deep_diagnostics: Mapping[str, Any],
    *,
    maximum_top_k_by_channel: Mapping[str, int],
) -> PreparedCandidates:
    queries = deep_diagnostics.get("queries")
    if not isinstance(queries, dict) or not queries:
        raise ValueError("deep diagnostics.queries must be a non-empty object")

    maxima = {
        channel: _positive_integer(
            maximum_top_k_by_channel[channel],
            field=f"{channel} maximum top-K",
        )
        for channel in CHANNELS
    }
    query_chunks: dict[str, dict[str, tuple[str, ...]]] = {}
    required: set[str] = set()
    for raw_query_id, raw_record in queries.items():
        query_id = _identifier(str(raw_query_id), field="query ID")
        if query_id in query_chunks:
            raise ValueError(f"duplicate normalized query ID: {query_id}")
        if not isinstance(raw_record, dict):
            raise ValueError(f"deep diagnostics query {query_id} must be an object")
        by_channel = {
            channel: _ranked_short_chunk_ids(
                raw_record,
                query_id=query_id,
                channel=channel,
                maximum_top_k=maxima[channel],
            )
            for channel in CHANNELS
        }
        query_chunks[query_id] = by_channel
        for chunk_ids in by_channel.values():
            required.update(chunk_ids)

    return PreparedCandidates(
        query_chunks=query_chunks,
        required_short_chunk_ids=frozenset(required),
    )


@dataclass(frozen=True, slots=True)
class LoadedMappings:
    by_short_chunk_id: dict[str, tuple[str, ...]]
    lines_scanned: int


def load_relevant_short_to_long_mappings(
    path: str | Path,
    required_short_chunk_ids: frozenset[str] | set[str],
    *,
    mapping_mode: str = "all",
    log_every: int = 250_000,
) -> LoadedMappings:
    if mapping_mode not in {"all", "primary"}:
        raise ValueError("mapping_mode must be 'all' or 'primary'")
    if log_every < 0:
        raise ValueError("log_every must be non-negative")

    required = set(required_short_chunk_ids)
    if not required:
        raise ValueError("required_short_chunk_ids must not be empty")
    source = Path(path)
    mappings: dict[str, tuple[str, ...]] = {}
    lines_scanned = 0
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            lines_scanned = line_number
            if log_every and line_number % log_every == 0:
                print(
                    "Mapping progress: "
                    f"{line_number:,} lines, {len(mappings):,}/{len(required):,} "
                    "required short chunks found",
                    file=sys.stderr,
                    flush=True,
                )
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON at {source}:{line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(f"mapping record {line_number} must be an object")
            raw_short_chunk_id = record.get("short_chunk_id")
            if not isinstance(raw_short_chunk_id, str):
                raise ValueError(
                    f"mapping record {line_number}.short_chunk_id must be a string"
                )
            short_chunk_id = raw_short_chunk_id.strip()
            if short_chunk_id not in required:
                continue
            if short_chunk_id in mappings:
                raise ValueError(
                    f"duplicate mapping for short chunk {short_chunk_id!r}"
                )

            if mapping_mode == "primary":
                long_chunk_ids = (
                    _identifier(
                        record.get("primary_long_chunk_id"),
                        field=(
                            f"mapping record {line_number}.primary_long_chunk_id"
                        ),
                    ),
                )
            else:
                raw_long_chunk_ids = record.get("long_chunk_ids")
                if not isinstance(raw_long_chunk_ids, list) or not raw_long_chunk_ids:
                    raise ValueError(
                        f"mapping record {line_number}.long_chunk_ids must be a "
                        "non-empty array"
                    )
                long_chunk_ids = tuple(
                    dict.fromkeys(
                        _identifier(
                            item,
                            field=(
                                f"mapping record {line_number}.long_chunk_ids"
                            ),
                        )
                        for item in raw_long_chunk_ids
                    )
                )
            mappings[short_chunk_id] = long_chunk_ids

    missing = sorted(required - mappings.keys())
    if missing:
        preview = ", ".join(repr(item) for item in missing[:10])
        suffix = "" if len(missing) <= 10 else ", ..."
        raise ValueError(
            f"short-to-long mapping is missing {len(missing):,} required short "
            f"chunks: {preview}{suffix}"
        )
    return LoadedMappings(
        by_short_chunk_id=mappings,
        lines_scanned=lines_scanned,
    )


def _mapped_long_prefixes(
    ranked_short_chunk_ids: Sequence[str],
    mappings: Mapping[str, Sequence[str]],
    cutoffs: Sequence[int],
) -> dict[int, frozenset[str]]:
    result: dict[int, frozenset[str]] = {}
    current: set[str] = set()
    previous = 0
    for cutoff in cutoffs:
        for short_chunk_id in ranked_short_chunk_ids[previous:cutoff]:
            current.update(mappings[short_chunk_id])
        result[cutoff] = frozenset(current)
        previous = cutoff
    return result


def _nearest_rank_percentile(values: Sequence[int], fraction: float) -> int:
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def _summarize_counts(values: Sequence[int]) -> dict[str, int | float]:
    if not values:
        raise ValueError("cannot summarize empty counts")
    return {
        "total": sum(values),
        "mean": sum(values) / len(values),
        "median": statistics.median(values),
        "p95": _nearest_rank_percentile(values, 0.95),
        "minimum": min(values),
        "maximum": max(values),
    }


@dataclass(frozen=True, slots=True)
class CandidateGridEvaluation:
    summary: dict[str, Any]
    per_query: dict[str, Any]


def evaluate_candidate_grid(
    prepared: PreparedCandidates,
    mappings: Mapping[str, Sequence[str]],
    *,
    bm25_top_k_values: Sequence[int] = DEFAULT_TOP_K_VALUES,
    dense_top_k_values: Sequence[int] = DEFAULT_TOP_K_VALUES,
    mapping_mode: str = "all",
    mapping_lines_scanned: int | None = None,
) -> CandidateGridEvaluation:
    bm25_cutoffs = _normalize_cutoffs(
        bm25_top_k_values,
        field="BM25 top-K values",
    )
    dense_cutoffs = _normalize_cutoffs(
        dense_top_k_values,
        field="dense top-K values",
    )
    if mapping_mode not in {"all", "primary"}:
        raise ValueError("mapping_mode must be 'all' or 'primary'")
    if not prepared.query_chunks:
        raise ValueError("prepared candidates must contain at least one query")

    combinations = [
        (bm25_k, dense_k)
        for bm25_k in bm25_cutoffs
        for dense_k in dense_cutoffs
    ]
    short_counts: dict[tuple[int, int], list[int]] = {
        combination: [] for combination in combinations
    }
    long_counts: dict[tuple[int, int], list[int]] = {
        combination: [] for combination in combinations
    }
    per_query_records: dict[str, Any] = {}

    for query_id, by_channel in prepared.query_chunks.items():
        bm25_chunks = by_channel["bm25"]
        dense_chunks = by_channel["dense"]
        if len(bm25_chunks) < bm25_cutoffs[-1]:
            raise ValueError(
                f"query {query_id} only has {len(bm25_chunks)} prepared BM25 hits"
            )
        if len(dense_chunks) < dense_cutoffs[-1]:
            raise ValueError(
                f"query {query_id} only has {len(dense_chunks)} prepared dense hits"
            )

        bm25_short_prefix = {
            cutoff: frozenset(bm25_chunks[:cutoff]) for cutoff in bm25_cutoffs
        }
        dense_short_prefix = {
            cutoff: frozenset(dense_chunks[:cutoff]) for cutoff in dense_cutoffs
        }
        bm25_long_prefix = _mapped_long_prefixes(
            bm25_chunks,
            mappings,
            bm25_cutoffs,
        )
        dense_long_prefix = _mapped_long_prefixes(
            dense_chunks,
            mappings,
            dense_cutoffs,
        )

        grid: dict[str, Any] = {}
        for bm25_k, dense_k in combinations:
            unique_short_count = len(
                bm25_short_prefix[bm25_k] | dense_short_prefix[dense_k]
            )
            unique_long_count = len(
                bm25_long_prefix[bm25_k] | dense_long_prefix[dense_k]
            )
            short_counts[(bm25_k, dense_k)].append(unique_short_count)
            long_counts[(bm25_k, dense_k)].append(unique_long_count)
            grid[f"bm25_{bm25_k}_dense_{dense_k}"] = {
                "bm25_top_k_short_chunks": bm25_k,
                "dense_top_k_short_chunks": dense_k,
                "unique_short_chunks": unique_short_count,
                "unique_long_chunks": unique_long_count,
            }
        per_query_records[query_id] = {"candidate_grid": grid}

    grid_summary: dict[str, Any] = {}
    for combination in combinations:
        bm25_k, dense_k = combination
        short_summary = _summarize_counts(short_counts[combination])
        long_summary = _summarize_counts(long_counts[combination])
        key = f"bm25_{bm25_k}_dense_{dense_k}"
        grid_summary[key] = {
            "bm25_top_k_short_chunks": bm25_k,
            "dense_top_k_short_chunks": dense_k,
            "unique_short_chunks_per_query": short_summary,
            "unique_long_chunks_per_query": long_summary,
            "long_to_short_pair_ratio": (
                long_summary["total"] / short_summary["total"]
            ),
        }

    def matrix(metric: str) -> list[list[int | float]]:
        return [
            [
                grid_summary[f"bm25_{bm25_k}_dense_{dense_k}"][
                    "unique_long_chunks_per_query"
                ][metric]
                for dense_k in dense_cutoffs
            ]
            for bm25_k in bm25_cutoffs
        ]

    summary: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "analysis": "short_chunk_lane_union_to_unique_long_chunk_grid",
        "mapping_mode": mapping_mode,
        "query_count": len(prepared.query_chunks),
        "bm25_top_k_values": list(bm25_cutoffs),
        "dense_top_k_values": list(dense_cutoffs),
        "required_unique_short_chunks": len(
            prepared.required_short_chunk_ids
        ),
        "loaded_short_to_long_mappings": len(mappings),
        "grid": grid_summary,
        "matrices": {
            "rows": {"name": "bm25_top_k", "values": list(bm25_cutoffs)},
            "columns": {
                "name": "dense_top_k",
                "values": list(dense_cutoffs),
            },
            "mean_unique_long_chunks_per_query": matrix("mean"),
            "total_query_long_chunk_pairs": matrix("total"),
            "p95_unique_long_chunks_per_query": matrix("p95"),
        },
    }
    if mapping_lines_scanned is not None:
        summary["mapping_lines_scanned"] = mapping_lines_scanned

    return CandidateGridEvaluation(
        summary=summary,
        per_query={
            "format_version": FORMAT_VERSION,
            "analysis": (
                "short_chunk_lane_union_to_unique_long_chunk_grid_per_query"
            ),
            "mapping_mode": mapping_mode,
            "queries": per_query_records,
        },
    )


def _format_number(value: int | float) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return f"{value:,}"


def _print_matrix(
    *,
    title: str,
    row_values: Sequence[int],
    column_values: Sequence[int],
    values: Sequence[Sequence[int | float]],
) -> None:
    rendered = [[_format_number(item) for item in row] for row in values]
    row_labels = [str(value) for value in row_values]
    column_labels = [str(value) for value in column_values]
    first_width = max(len("BM25\\Dense"), *(len(item) for item in row_labels))
    widths = [
        max(len(label), *(len(row[index]) for row in rendered))
        for index, label in enumerate(column_labels)
    ]
    print(f"\n{title}")
    print(
        "BM25\\Dense".rjust(first_width)
        + "  "
        + "  ".join(
            label.rjust(width) for label, width in zip(column_labels, widths)
        )
    )
    for label, row in zip(row_labels, rendered):
        print(
            label.rjust(first_width)
            + "  "
            + "  ".join(
                item.rjust(width) for item, width in zip(row, widths)
            )
        )


def print_summary(summary: Mapping[str, Any]) -> None:
    matrices = summary["matrices"]
    rows = matrices["rows"]["values"]
    columns = matrices["columns"]["values"]
    print(f"Queries: {summary['query_count']:,}")
    print(f"Mapping mode: {summary['mapping_mode']}")
    print(
        "Relevant short mappings loaded: "
        f"{summary['loaded_short_to_long_mappings']:,}"
    )
    if "mapping_lines_scanned" in summary:
        print(f"Mapping lines scanned: {summary['mapping_lines_scanned']:,}")
    _print_matrix(
        title="Mean unique long chunks per query",
        row_values=rows,
        column_values=columns,
        values=matrices["mean_unique_long_chunks_per_query"],
    )
    _print_matrix(
        title="Total query-long chunk pairs",
        row_values=rows,
        column_values=columns,
        values=matrices["total_query_long_chunk_pairs"],
    )
    _print_matrix(
        title="P95 unique long chunks per query",
        row_values=rows,
        column_values=columns,
        values=matrices["p95_unique_long_chunks_per_query"],
    )


def _write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "For every BM25/dense top-K pair, union ranked short chunks, map "
            "them to long chunks, deduplicate long chunks, and count the "
            "resulting reranker candidates."
        )
    )
    parser.add_argument("--deep-diagnostics", required=True, type=Path)
    parser.add_argument("--short-to-long", required=True, type=Path)
    parser.add_argument(
        "--bm25-top-k-values",
        type=_parse_cutoffs,
        default=DEFAULT_TOP_K_VALUES,
        help="comma-separated chunk cutoffs; default: 100,150,...,400",
    )
    parser.add_argument(
        "--dense-top-k-values",
        type=_parse_cutoffs,
        default=DEFAULT_TOP_K_VALUES,
        help="comma-separated chunk cutoffs; default: 100,150,...,400",
    )
    parser.add_argument(
        "--mapping-mode",
        choices=("all", "primary"),
        default="all",
        help=(
            "all maps through long_chunk_ids; primary maps only through "
            "primary_long_chunk_id"
        ),
    )
    parser.add_argument(
        "--mapping-log-every",
        type=int,
        default=250_000,
        help="progress interval while scanning JSONL; 0 disables progress",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional summary JSON; omitted means no artifact is written",
    )
    parser.add_argument(
        "--per-query-output",
        type=Path,
        help="optional per-query JSON; omitted means no artifact is written",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.mapping_log_every < 0:
        parser.error("--mapping-log-every must be non-negative")
    if (
        args.output is not None
        and args.per_query_output is not None
        and args.output.expanduser().resolve(strict=False)
        == args.per_query_output.expanduser().resolve(strict=False)
    ):
        parser.error("--output and --per-query-output must be different paths")

    try:
        bm25_cutoffs = _normalize_cutoffs(
            args.bm25_top_k_values,
            field="BM25 top-K values",
        )
        dense_cutoffs = _normalize_cutoffs(
            args.dense_top_k_values,
            field="dense top-K values",
        )
        deep = load_deep_diagnostics(args.deep_diagnostics)
        prepared = prepare_candidates(
            deep,
            maximum_top_k_by_channel={
                "bm25": bm25_cutoffs[-1],
                "dense": dense_cutoffs[-1],
            },
        )
        # The diagnostics payload can be large at 2 lanes x 400 hits x all
        # queries. Candidate tuples retain the IDs needed below, so release the
        # original nested hit objects before streaming the mapping dataset.
        del deep
        loaded_mappings = load_relevant_short_to_long_mappings(
            args.short_to_long,
            prepared.required_short_chunk_ids,
            mapping_mode=args.mapping_mode,
            log_every=args.mapping_log_every,
        )
        evaluation = evaluate_candidate_grid(
            prepared,
            loaded_mappings.by_short_chunk_id,
            bm25_top_k_values=bm25_cutoffs,
            dense_top_k_values=dense_cutoffs,
            mapping_mode=args.mapping_mode,
            mapping_lines_scanned=loaded_mappings.lines_scanned,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    if args.output is not None:
        _write_json(args.output, evaluation.summary)
    if args.per_query_output is not None:
        _write_json(args.per_query_output, evaluation.per_query)
    print_summary(evaluation.summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
