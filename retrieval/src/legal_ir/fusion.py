from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence

from .schema import DocumentCandidate, RankedChunk, ScoredChunk


@dataclass(frozen=True, slots=True)
class _RankedDocument:
    document_id: str
    score: float
    rank: int


def rank_chunks(channel: str, hits: Sequence[ScoredChunk]) -> list[RankedChunk]:
    """Deduplicate a channel and assign deterministic, one-based chunk ranks."""

    best_by_chunk: dict[str, ScoredChunk] = {}
    for hit in hits:
        if not math.isfinite(hit.score):
            continue
        previous = best_by_chunk.get(hit.chunk_id)
        if previous is None or hit.score > previous.score:
            best_by_chunk[hit.chunk_id] = hit
    ordered = sorted(
        best_by_chunk.values(),
        key=lambda hit: (-hit.score, hit.chunk_id),
    )
    return [
        RankedChunk(
            chunk_id=hit.chunk_id,
            document_id=hit.document_id,
            score=hit.score,
            rank=rank,
            channel=channel,
        )
        for rank, hit in enumerate(ordered, start=1)
    ]


def _rank_documents(hits: Sequence[RankedChunk]) -> list[_RankedDocument]:
    """MaxP aggregation inside one channel, before rank fusion."""

    max_score: dict[str, float] = {}
    for hit in hits:
        current = max_score.get(hit.document_id)
        if current is None or hit.score > current:
            max_score[hit.document_id] = hit.score
    ordered = sorted(max_score.items(), key=lambda item: (-item[1], item[0]))
    return [
        _RankedDocument(document_id=document_id, score=score, rank=rank)
        for rank, (document_id, score) in enumerate(ordered, start=1)
    ]


def weighted_rrf(
    channel_hits: Mapping[str, Sequence[ScoredChunk]],
    *,
    channel_weights: Mapping[str, float],
    rrf_k: int,
    top_k_documents: int,
    evidence_chunks_per_document: int,
    grounded_channels: frozenset[str] = frozenset({"bm25", "dense"}),
) -> list[DocumentCandidate]:
    """Fuse incomparable scores by ranks after chunk-to-document aggregation."""

    ranked_chunks = {
        channel: rank_chunks(channel, hits) for channel, hits in channel_hits.items()
    }
    candidates: dict[str, DocumentCandidate] = {}

    for channel, hits in ranked_chunks.items():
        weight = float(channel_weights.get(channel, 0.0))
        if weight <= 0:
            continue
        for document in _rank_documents(hits):
            candidate = candidates.setdefault(
                document.document_id,
                DocumentCandidate(document_id=document.document_id, fusion_score=0.0),
            )
            candidate.fusion_score += weight / (rrf_k + document.rank)
            candidate.channel_ranks[channel] = document.rank
            candidate.channel_scores[channel] = document.score

    ordered = sorted(
        candidates.values(),
        key=lambda candidate: (-candidate.fusion_score, candidate.document_id),
    )[:top_k_documents]
    retained_documents = {candidate.document_id for candidate in ordered}

    # A second rank-based fusion selects real chunks to show the reranker.
    evidence_scores: dict[str, dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    grounded_evidence_scores: dict[str, dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for channel, hits in ranked_chunks.items():
        weight = float(channel_weights.get(channel, 0.0))
        if weight <= 0:
            continue
        for hit in hits:
            if hit.document_id in retained_documents:
                contribution = weight / (rrf_k + hit.rank)
                evidence_scores[hit.document_id][hit.chunk_id] += contribution
                if channel in grounded_channels:
                    grounded_evidence_scores[hit.document_id][
                        hit.chunk_id
                    ] += contribution

    for candidate in ordered:
        ranked_evidence = sorted(
            evidence_scores[candidate.document_id].items(),
            key=lambda item: (-item[1], item[0]),
        )
        selected: list[str] = []
        # Reserve one slot for evidence retrieved from the real query when available.
        ranked_grounded = sorted(
            grounded_evidence_scores[candidate.document_id].items(),
            key=lambda item: (-item[1], item[0]),
        )
        if ranked_grounded:
            selected.append(ranked_grounded[0][0])
        for chunk_id, _ in ranked_evidence:
            if len(selected) >= evidence_chunks_per_document:
                break
            if chunk_id not in selected:
                selected.append(chunk_id)
        candidate.evidence_chunk_ids = selected
    return ordered
