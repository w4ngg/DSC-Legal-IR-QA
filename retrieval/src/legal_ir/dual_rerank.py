from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence

from .interfaces import PassageReranker
from .io import ChunkStore
from .schema import Chunk, RankedChunk


def _required_text(value: Any, field_name: str) -> str:
    if value is None:
        raise ValueError(f"{field_name} is required")
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must not be empty")
    return sys.intern(text)


def _dual_chunk_identity(
    chunk_id: str,
    *,
    expected_granularity: str,
) -> tuple[str, str]:
    """Return ``(document_id, namespace)`` from a dual chunk identifier."""

    parts = chunk_id.rsplit(":", 3)
    if (
        len(parts) != 4
        or not parts[0]
        or not parts[1]
        or parts[2] != expected_granularity
        or not parts[3].isdigit()
    ):
        raise ValueError(
            f"invalid {expected_granularity} dual chunk_id: {chunk_id!r}"
        )
    return parts[0], parts[1]


@dataclass(frozen=True, slots=True)
class ShortToLongMapping:
    short_chunk_id: str
    document_id: str
    primary_long_chunk_id: str
    long_chunk_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        short_chunk_id = _required_text(
            self.short_chunk_id, "short_to_long.short_chunk_id"
        )
        document_id = _required_text(
            self.document_id, "short_to_long.document_id"
        )
        primary_long_chunk_id = _required_text(
            self.primary_long_chunk_id,
            "short_to_long.primary_long_chunk_id",
        )
        long_chunk_ids = tuple(
            _required_text(item, "short_to_long.long_chunk_ids[]")
            for item in self.long_chunk_ids
        )
        if not long_chunk_ids:
            raise ValueError("short_to_long.long_chunk_ids must not be empty")
        if len(set(long_chunk_ids)) != len(long_chunk_ids):
            raise ValueError(
                f"mapping for short chunk {short_chunk_id} has duplicate long IDs"
            )
        if primary_long_chunk_id not in long_chunk_ids:
            raise ValueError(
                f"mapping for short chunk {short_chunk_id} does not contain its "
                "primary long chunk"
            )

        short_document_id, namespace = _dual_chunk_identity(
            short_chunk_id,
            expected_granularity="short",
        )
        if short_document_id != document_id:
            raise ValueError(
                f"short chunk {short_chunk_id} belongs to document "
                f"{short_document_id}, not {document_id}"
            )
        for long_chunk_id in long_chunk_ids:
            long_document_id, long_namespace = _dual_chunk_identity(
                long_chunk_id,
                expected_granularity="long",
            )
            if long_document_id != document_id:
                raise ValueError(
                    f"long chunk {long_chunk_id} belongs to document "
                    f"{long_document_id}, not {document_id}"
                )
            if long_namespace != namespace:
                raise ValueError(
                    f"short chunk {short_chunk_id} and long chunk {long_chunk_id} "
                    "use different dual chunk namespaces"
                )

        object.__setattr__(self, "short_chunk_id", short_chunk_id)
        object.__setattr__(self, "document_id", document_id)
        object.__setattr__(
            self, "primary_long_chunk_id", primary_long_chunk_id
        )
        object.__setattr__(self, "long_chunk_ids", long_chunk_ids)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ShortToLongMapping":
        raw_long_ids = value.get("long_chunk_ids")
        if not isinstance(raw_long_ids, (list, tuple)):
            raise ValueError("short_to_long.long_chunk_ids must be an array")
        return cls(
            short_chunk_id=value.get("short_chunk_id"),
            document_id=value.get("document_id"),
            primary_long_chunk_id=value.get("primary_long_chunk_id"),
            long_chunk_ids=tuple(raw_long_ids),
        )


class ShortToLongLookup(Protocol):
    def get(self, short_chunk_id: str) -> ShortToLongMapping: ...


class ShortToLongStore:
    """In-memory lookup for the standalone ``short_to_long.jsonl`` artifact."""

    def __init__(self, mappings: Sequence[ShortToLongMapping]) -> None:
        if not mappings:
            raise ValueError("short-to-long store must contain at least one mapping")
        by_id: dict[str, ShortToLongMapping] = {}
        for mapping in mappings:
            if mapping.short_chunk_id in by_id:
                raise ValueError(
                    f"duplicate short_chunk_id in mapping: {mapping.short_chunk_id}"
                )
            by_id[mapping.short_chunk_id] = mapping
        self._by_id = by_id

    def __len__(self) -> int:
        return len(self._by_id)

    def get(self, short_chunk_id: str) -> ShortToLongMapping:
        try:
            return self._by_id[short_chunk_id]
        except KeyError as exc:
            raise KeyError(
                f"short chunk has no short-to-long mapping: {short_chunk_id}"
            ) from exc

    @classmethod
    def load_jsonl(cls, path: str | Path) -> "ShortToLongStore":
        mappings: list[ShortToLongMapping] = []
        source = Path(path)
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError("record must be a JSON object")
                    mappings.append(ShortToLongMapping.from_dict(value))
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"invalid short-to-long mapping at "
                        f"{source}:{line_number}: {exc}"
                    ) from exc
        return cls(mappings)


class ShortChunkMetadataLookup:
    """Resolve mappings already embedded in indexed short-chunk metadata.

    This avoids loading a second two-million-record mapping dictionary during
    online search. The standalone mapping artifact remains useful for offline
    jobs and for validating that the index and dual dataset belong together.
    """

    def __init__(self, short_chunks: ChunkStore) -> None:
        self.short_chunks = short_chunks

    def get(self, short_chunk_id: str) -> ShortToLongMapping:
        chunk = self.short_chunks.get(short_chunk_id)
        metadata_granularity = chunk.metadata.get("granularity")
        if metadata_granularity not in (None, "short"):
            raise ValueError(
                f"chunk {short_chunk_id} is not marked as a short chunk"
            )
        return ShortToLongMapping.from_dict(
            {
                "short_chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "primary_long_chunk_id": chunk.metadata.get(
                    "primary_long_chunk_id"
                ),
                "long_chunk_ids": chunk.metadata.get("long_chunk_ids"),
            }
        )


@dataclass(frozen=True, slots=True)
class DualChunkAssets:
    mappings: ShortToLongLookup
    long_chunks: ChunkStore

    @classmethod
    def load(
        cls,
        *,
        short_to_long_path: str | Path,
        long_chunks_path: str | Path,
    ) -> "DualChunkAssets":
        return cls(
            mappings=ShortToLongStore.load_jsonl(short_to_long_path),
            long_chunks=ChunkStore.load_jsonl(long_chunks_path),
        )

    @classmethod
    def from_indexed_short_chunks(
        cls,
        *,
        short_chunks: ChunkStore,
        long_chunks_path: str | Path,
    ) -> "DualChunkAssets":
        return cls(
            mappings=ShortChunkMetadataLookup(short_chunks),
            long_chunks=ChunkStore.load_jsonl(
                long_chunks_path,
                compact_for_search=True,
            ),
        )

    def validate_mapping(
        self,
        mapping: ShortToLongMapping,
        *,
        expected_document_id: str,
    ) -> tuple[Chunk, ...]:
        if mapping.document_id != expected_document_id:
            raise ValueError(
                f"retrieved short chunk {mapping.short_chunk_id} has document_id "
                f"{expected_document_id}, but its mapping has "
                f"{mapping.document_id}"
            )
        long_chunks: list[Chunk] = []
        for long_chunk_id in mapping.long_chunk_ids:
            try:
                long_chunk = self.long_chunks.get(long_chunk_id)
            except KeyError as exc:
                raise KeyError(
                    f"mapping for short chunk {mapping.short_chunk_id} references "
                    f"unknown long chunk {long_chunk_id}"
                ) from exc
            if long_chunk.document_id != mapping.document_id:
                raise ValueError(
                    f"long chunk {long_chunk_id} has document_id "
                    f"{long_chunk.document_id}, expected {mapping.document_id}"
                )
            metadata_granularity = long_chunk.metadata.get("granularity")
            if metadata_granularity not in (None, "long"):
                raise ValueError(
                    f"chunk {long_chunk_id} is not marked as a long chunk"
                )
            # The mapping dataclass already checks the short/long namespace.
            # Parse the loaded long record as well so a forged store key cannot
            # silently bypass that invariant.
            _dual_chunk_identity(
                long_chunk.chunk_id,
                expected_granularity="long",
            )
            long_chunks.append(long_chunk)
        return tuple(long_chunks)


@dataclass(frozen=True, slots=True)
class ShortCandidateSupport:
    short_chunk_id: str
    document_id: str
    channel_ranks: Mapping[str, int]
    channel_scores: Mapping[str, float]
    channel_contributions: Mapping[str, float]
    total_contribution: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "short_chunk_id": self.short_chunk_id,
            "document_id": self.document_id,
            "channel_ranks": dict(self.channel_ranks),
            "channel_scores": dict(self.channel_scores),
            "channel_contributions": dict(self.channel_contributions),
            "total_contribution": self.total_contribution,
        }


@dataclass(frozen=True, slots=True)
class LongCandidate:
    long_chunk_id: str
    document_id: str
    retrieval_support_score: float
    retrieval_rank: int
    supporting_short_chunks: tuple[ShortCandidateSupport, ...]
    primary_supporting_short_chunk_ids: tuple[str, ...]
    channel_best_ranks: Mapping[str, int]
    channel_contributions: Mapping[str, float]

    @property
    def supporting_short_chunk_ids(self) -> tuple[str, ...]:
        return tuple(item.short_chunk_id for item in self.supporting_short_chunks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "long_chunk_id": self.long_chunk_id,
            "document_id": self.document_id,
            "retrieval_support_score": self.retrieval_support_score,
            "retrieval_rank": self.retrieval_rank,
            "supporting_short_chunk_ids": list(
                self.supporting_short_chunk_ids
            ),
            "primary_supporting_short_chunk_ids": list(
                self.primary_supporting_short_chunk_ids
            ),
            "channel_best_ranks": dict(self.channel_best_ranks),
            "channel_contributions": dict(self.channel_contributions),
            "supporting_short_chunks": [
                item.to_dict() for item in self.supporting_short_chunks
            ],
        }


@dataclass(frozen=True, slots=True)
class LongCandidatePool:
    candidates: tuple[LongCandidate, ...]
    lane_short_chunk_counts: Mapping[str, int]
    unique_short_chunk_count: int
    mapped_long_occurrence_count: int
    unique_long_chunk_count: int
    long_candidate_limit: int | None

    @property
    def selected_long_chunk_count(self) -> int:
        return len(self.candidates)

    def counts_dict(self) -> dict[str, Any]:
        return {
            "lane_short_chunks": dict(self.lane_short_chunk_counts),
            "unique_short_chunks": self.unique_short_chunk_count,
            "mapped_long_occurrences": self.mapped_long_occurrence_count,
            "unique_long_chunks_before_cutoff": self.unique_long_chunk_count,
            "selected_long_chunks": self.selected_long_chunk_count,
            "long_candidate_limit": self.long_candidate_limit,
        }


@dataclass(frozen=True, slots=True)
class RerankedLongCandidate:
    candidate: LongCandidate
    reranker_score: float
    reranker_rank: int

    @property
    def long_chunk_id(self) -> str:
        return self.candidate.long_chunk_id

    @property
    def document_id(self) -> str:
        return self.candidate.document_id

    def to_dict(self) -> dict[str, Any]:
        value = self.candidate.to_dict()
        value.update(
            {
                "reranker_score": self.reranker_score,
                "reranker_rank": self.reranker_rank,
            }
        )
        return value


@dataclass(frozen=True, slots=True)
class LongDocumentResult:
    document_id: str
    score: float
    rank: int
    best_long_chunk_id: str
    best_long_chunk_rank: int
    contributing_long_chunks: tuple[RerankedLongCandidate, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "score": self.score,
            "rank": self.rank,
            "best_long_chunk_id": self.best_long_chunk_id,
            "best_long_chunk_rank": self.best_long_chunk_rank,
            "contributing_long_chunk_ids": [
                item.long_chunk_id for item in self.contributing_long_chunks
            ],
            "long_chunk_scores": {
                item.long_chunk_id: item.reranker_score
                for item in self.contributing_long_chunks
            },
        }


@dataclass(frozen=True, slots=True)
class LongRerankOutcome:
    pool: LongCandidatePool
    reranked_long_candidates: tuple[RerankedLongCandidate, ...]
    post_reranker_long_top_k: int
    document_candidates: tuple[LongDocumentResult, ...]
    results: tuple[LongDocumentResult, ...]

    def to_diagnostics_dict(self) -> dict[str, Any]:
        return {
            "candidate_counts": self.pool.counts_dict(),
            "post_reranker_long_top_k": self.post_reranker_long_top_k,
            # Keep every selected long candidate, not only the post-reranker
            # top-k, so aggregations and cutoffs can be replayed offline.
            "long_candidates": [
                item.to_dict() for item in self.reranked_long_candidates
            ],
            "top_long_candidates": [
                item.to_dict()
                for item in self.reranked_long_candidates[
                    : self.post_reranker_long_top_k
                ]
            ],
            "document_candidates": [
                item.to_dict() for item in self.document_candidates
            ],
            "results": [item.to_dict() for item in self.results],
        }


@dataclass(slots=True)
class _MutableShortSupport:
    document_id: str
    channel_ranks: dict[str, int]
    channel_scores: dict[str, float]
    channel_contributions: dict[str, float]


def _short_supports(
    ranked_channels: Mapping[str, Sequence[RankedChunk]],
    *,
    channel_weights: Mapping[str, float],
    rrf_k: int,
) -> tuple[dict[str, ShortCandidateSupport], dict[str, int]]:
    if isinstance(rrf_k, bool) or not isinstance(rrf_k, int) or rrf_k <= 0:
        raise ValueError("rrf_k must be a positive integer")

    supports: dict[str, _MutableShortSupport] = {}
    lane_counts: dict[str, int] = {}
    for channel in sorted(ranked_channels):
        try:
            weight = float(channel_weights.get(channel, 0.0))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"channel weight for {channel!r} must be numeric"
            ) from exc
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(
                f"channel weight for {channel!r} must be finite and non-negative"
            )
        if weight == 0:
            continue

        best_by_short: dict[str, RankedChunk] = {}
        for hit in ranked_channels[channel]:
            if hit.channel != channel:
                raise ValueError(
                    f"channel {channel!r} contains a hit labeled {hit.channel!r}"
                )
            if (
                isinstance(hit.rank, bool)
                or not isinstance(hit.rank, int)
                or hit.rank <= 0
            ):
                raise ValueError(
                    f"short chunk {hit.chunk_id} has invalid rank {hit.rank}"
                )
            try:
                numeric_score = float(hit.score)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"short chunk {hit.chunk_id} has a non-numeric score"
                ) from exc
            if not math.isfinite(numeric_score):
                raise ValueError(
                    f"short chunk {hit.chunk_id} has a non-finite score"
                )
            previous = best_by_short.get(hit.chunk_id)
            if previous is None or (hit.rank, -numeric_score, hit.chunk_id) < (
                previous.rank,
                -float(previous.score),
                previous.chunk_id,
            ):
                best_by_short[hit.chunk_id] = hit

        lane_counts[channel] = len(best_by_short)
        for short_chunk_id, hit in sorted(
            best_by_short.items(),
            key=lambda item: (item[1].rank, item[0]),
        ):
            _dual_chunk_identity(
                short_chunk_id,
                expected_granularity="short",
            )
            contribution = weight / (rrf_k + hit.rank)
            support = supports.get(short_chunk_id)
            if support is None:
                support = _MutableShortSupport(
                    document_id=hit.document_id,
                    channel_ranks={},
                    channel_scores={},
                    channel_contributions={},
                )
                supports[short_chunk_id] = support
            elif support.document_id != hit.document_id:
                raise ValueError(
                    f"short chunk {short_chunk_id} has inconsistent document IDs "
                    f"{support.document_id} and {hit.document_id}"
                )
            support.channel_ranks[channel] = hit.rank
            support.channel_scores[channel] = float(hit.score)
            support.channel_contributions[channel] = contribution

    frozen = {
        short_chunk_id: ShortCandidateSupport(
            short_chunk_id=short_chunk_id,
            document_id=support.document_id,
            channel_ranks=MappingProxyType(dict(support.channel_ranks)),
            channel_scores=MappingProxyType(dict(support.channel_scores)),
            channel_contributions=MappingProxyType(
                dict(support.channel_contributions)
            ),
            total_contribution=math.fsum(
                support.channel_contributions.values()
            ),
        )
        for short_chunk_id, support in supports.items()
    }
    return frozen, lane_counts


def build_long_candidate_pool(
    ranked_channels: Mapping[str, Sequence[RankedChunk]],
    *,
    assets: DualChunkAssets,
    channel_weights: Mapping[str, float],
    rrf_k: int,
    long_candidate_limit: int | None,
) -> LongCandidatePool:
    """Union short hits, map all memberships, and rank unique long chunks.

    ``long_candidate_limit=None`` is full mode. A positive integer applies a
    deterministic cutoff after mapping and long-ID deduplication.
    """

    if long_candidate_limit is not None and (
        isinstance(long_candidate_limit, bool)
        or not isinstance(long_candidate_limit, int)
        or long_candidate_limit <= 0
    ):
        raise ValueError("long_candidate_limit must be null or a positive integer")

    short_supports, lane_counts = _short_supports(
        ranked_channels,
        channel_weights=channel_weights,
        rrf_k=rrf_k,
    )

    long_supports: dict[str, list[ShortCandidateSupport]] = {}
    primary_long_supports: dict[str, set[str]] = {}
    long_chunks: dict[str, Chunk] = {}
    mapped_occurrences = 0
    for short_chunk_id in sorted(short_supports):
        short_support = short_supports[short_chunk_id]
        mapping = assets.mappings.get(short_chunk_id)
        mapped = assets.validate_mapping(
            mapping,
            expected_document_id=short_support.document_id,
        )
        mapped_occurrences += len(mapped)
        for long_chunk in mapped:
            long_chunks[long_chunk.chunk_id] = long_chunk
            long_supports.setdefault(long_chunk.chunk_id, []).append(
                short_support
            )
            if long_chunk.chunk_id == mapping.primary_long_chunk_id:
                primary_long_supports.setdefault(long_chunk.chunk_id, set()).add(
                    short_chunk_id
                )

    pre_ranked: list[
        tuple[
            str,
            Chunk,
            tuple[ShortCandidateSupport, ...],
            tuple[str, ...],
            float,
            dict[str, int],
            dict[str, float],
        ]
    ] = []
    for long_chunk_id, raw_supports in long_supports.items():
        supports = tuple(
            sorted(raw_supports, key=lambda item: item.short_chunk_id)
        )
        channel_best_ranks: dict[str, int] = {}
        channel_contribution_values: dict[str, list[float]] = {}
        for support in supports:
            for channel, rank in support.channel_ranks.items():
                current = channel_best_ranks.get(channel)
                if current is None or rank < current:
                    channel_best_ranks[channel] = rank
            for channel, contribution in support.channel_contributions.items():
                channel_contribution_values.setdefault(channel, []).append(
                    contribution
                )
        channel_contributions = {
            channel: math.fsum(values)
            for channel, values in channel_contribution_values.items()
        }
        total = math.fsum(channel_contributions.values())
        pre_ranked.append(
            (
                long_chunk_id,
                long_chunks[long_chunk_id],
                supports,
                tuple(sorted(primary_long_supports.get(long_chunk_id, set()))),
                total,
                channel_best_ranks,
                channel_contributions,
            )
        )

    pre_ranked.sort(key=lambda item: (-item[4], item[0]))
    unique_long_count = len(pre_ranked)
    candidates = tuple(
        LongCandidate(
            long_chunk_id=long_chunk_id,
            document_id=long_chunk.document_id,
            retrieval_support_score=total,
            retrieval_rank=rank,
            supporting_short_chunks=supports,
            primary_supporting_short_chunk_ids=primary_supports,
            channel_best_ranks=MappingProxyType(dict(channel_best_ranks)),
            channel_contributions=MappingProxyType(
                dict(channel_contributions)
            ),
        )
        for rank, (
            long_chunk_id,
            long_chunk,
            supports,
            primary_supports,
            total,
            channel_best_ranks,
            channel_contributions,
        ) in enumerate(pre_ranked, start=1)
    )
    if long_candidate_limit is not None:
        candidates = candidates[:long_candidate_limit]

    return LongCandidatePool(
        candidates=candidates,
        lane_short_chunk_counts=MappingProxyType(dict(lane_counts)),
        unique_short_chunk_count=len(short_supports),
        mapped_long_occurrence_count=mapped_occurrences,
        unique_long_chunk_count=unique_long_count,
        long_candidate_limit=long_candidate_limit,
    )


def score_long_candidates(
    query: str,
    pool: LongCandidatePool,
    *,
    long_chunks: ChunkStore,
    reranker: PassageReranker,
) -> tuple[RerankedLongCandidate, ...]:
    """Score every selected long candidate with the original user query."""

    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("query must not be empty")
    passages = [
        long_chunks.get(candidate.long_chunk_id).index_text
        for candidate in pool.candidates
    ]
    raw_scores = reranker.score(normalized_query, passages)
    if len(raw_scores) != len(pool.candidates):
        raise ValueError(
            f"reranker returned {len(raw_scores)} scores for "
            f"{len(pool.candidates)} long candidates"
        )

    scored: list[tuple[LongCandidate, float]] = []
    for candidate, raw_score in zip(pool.candidates, raw_scores, strict=True):
        score = float(raw_score)
        if not math.isfinite(score):
            raise ValueError(
                f"reranker returned a non-finite score for long chunk "
                f"{candidate.long_chunk_id}"
            )
        scored.append((candidate, score))
    scored.sort(
        key=lambda item: (
            -item[1],
            item[0].retrieval_rank,
            item[0].long_chunk_id,
        )
    )
    return tuple(
        RerankedLongCandidate(
            candidate=candidate,
            reranker_score=score,
            reranker_rank=rank,
        )
        for rank, (candidate, score) in enumerate(scored, start=1)
    )


def aggregate_long_maxp(
    reranked: Sequence[RerankedLongCandidate],
    *,
    post_reranker_long_top_k: int = 20,
    final_top_k_documents: int = 5,
) -> tuple[tuple[LongDocumentResult, ...], tuple[LongDocumentResult, ...]]:
    """Take reranker top-k long chunks, then aggregate documents with MaxP."""

    for name, value in {
        "post_reranker_long_top_k": post_reranker_long_top_k,
        "final_top_k_documents": final_top_k_documents,
    }.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if final_top_k_documents > 5:
        raise ValueError("final_top_k_documents must be <= 5")

    top_long = tuple(reranked[:post_reranker_long_top_k])
    by_document: dict[str, list[RerankedLongCandidate]] = {}
    for item in top_long:
        by_document.setdefault(item.document_id, []).append(item)

    ordered_groups = sorted(
        by_document.items(),
        key=lambda item: (
            -item[1][0].reranker_score,
            item[1][0].reranker_rank,
            item[0],
        ),
    )
    document_candidates = tuple(
        LongDocumentResult(
            document_id=document_id,
            score=long_candidates[0].reranker_score,
            rank=rank,
            best_long_chunk_id=long_candidates[0].long_chunk_id,
            best_long_chunk_rank=long_candidates[0].reranker_rank,
            contributing_long_chunks=tuple(long_candidates),
        )
        for rank, (document_id, long_candidates) in enumerate(
            ordered_groups,
            start=1,
        )
    )
    return (
        document_candidates,
        document_candidates[:final_top_k_documents],
    )


def rerank_long_candidate_pool(
    query: str,
    pool: LongCandidatePool,
    *,
    long_chunks: ChunkStore,
    reranker: PassageReranker,
    post_reranker_long_top_k: int = 20,
    final_top_k_documents: int = 5,
) -> LongRerankOutcome:
    reranked = score_long_candidates(
        query,
        pool,
        long_chunks=long_chunks,
        reranker=reranker,
    )
    document_candidates, results = aggregate_long_maxp(
        reranked,
        post_reranker_long_top_k=post_reranker_long_top_k,
        final_top_k_documents=final_top_k_documents,
    )
    return LongRerankOutcome(
        pool=pool,
        reranked_long_candidates=reranked,
        post_reranker_long_top_k=post_reranker_long_top_k,
        document_candidates=document_candidates,
        results=results,
    )
