from __future__ import annotations

import math
from collections.abc import Sequence

from .config import PipelineConfig
from .fusion import weighted_rrf
from .hyde import normalize_hyde_text
from .interfaces import (
    ChunkRetriever,
    HypotheticalDocumentGenerator,
    PassageReranker,
)
from .io import ChunkStore
from .schema import DocumentCandidate, ScoredChunk, SearchResponse, SearchResult


class RetrievalPipeline:
    """BM25 + Vietnamese dense/HyDE, document RRF, then cross-encoder reranking."""

    def __init__(
        self,
        *,
        chunks: ChunkStore,
        bm25: ChunkRetriever,
        dense: ChunkRetriever,
        config: PipelineConfig,
        hyde_generator: HypotheticalDocumentGenerator | None = None,
        reranker: PassageReranker | None = None,
    ) -> None:
        if config.hyde.enabled and hyde_generator is None:
            raise ValueError("HyDE is enabled but no generator was provided")
        if config.reranker.enabled and reranker is None:
            raise ValueError("reranking is enabled but no reranker was provided")
        self.chunks = chunks
        self.bm25 = bm25
        self.dense = dense
        self.config = config
        self.hyde_generator = hyde_generator
        self.reranker = reranker

    def _validate_hits(
        self, channel: str, hits: Sequence[ScoredChunk]
    ) -> list[ScoredChunk]:
        validated: list[ScoredChunk] = []
        for hit in hits:
            chunk = self.chunks.get(hit.chunk_id)
            if hit.document_id != chunk.document_id:
                raise ValueError(
                    f"{channel} returned chunk {hit.chunk_id} with document_id "
                    f"{hit.document_id}, expected {chunk.document_id}"
                )
            validated.append(hit)
        return validated

    def _rerank(
        self, query: str, candidates: list[DocumentCandidate]
    ) -> list[DocumentCandidate]:
        if not self.config.reranker.enabled:
            return candidates
        assert self.reranker is not None

        references: list[tuple[DocumentCandidate, str]] = []
        passages: list[str] = []
        for candidate in candidates:
            for chunk_id in candidate.evidence_chunk_ids:
                chunk = self.chunks.get(chunk_id)
                references.append((candidate, chunk_id))
                # retrieval_text is expected to contain compact title/hierarchy metadata.
                passages.append(chunk.index_text)

        scores = self.reranker.score(query, passages)
        if len(scores) != len(references):
            raise ValueError(
                f"reranker returned {len(scores)} scores for {len(references)} passages"
            )
        for (candidate, chunk_id), score in zip(references, scores):
            numeric_score = float(score)
            if not math.isfinite(numeric_score):
                raise ValueError(
                    f"reranker returned a non-finite score for chunk {chunk_id}"
                )
            candidate.evidence_rerank_scores[chunk_id] = numeric_score
            if candidate.rerank_score is None or numeric_score > candidate.rerank_score:
                candidate.rerank_score = numeric_score

        return sorted(
            candidates,
            key=lambda candidate: (
                -(candidate.rerank_score if candidate.rerank_score is not None else -float("inf")),
                -candidate.fusion_score,
                candidate.document_id,
            ),
        )

    @staticmethod
    def _to_result(candidate: DocumentCandidate) -> SearchResult:
        return SearchResult(
            document_id=candidate.document_id,
            score=(
                candidate.rerank_score
                if candidate.rerank_score is not None
                else candidate.fusion_score
            ),
            fusion_score=candidate.fusion_score,
            rerank_score=candidate.rerank_score,
            evidence_chunk_ids=tuple(candidate.evidence_chunk_ids),
            channel_ranks=dict(candidate.channel_ranks),
            channel_scores=dict(candidate.channel_scores),
            evidence_rerank_scores=dict(candidate.evidence_rerank_scores),
        )

    def search(self, query: str) -> SearchResponse:
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")

        bm25_hits = self._validate_hits(
            "bm25", self.bm25.search(query, self.config.bm25.top_k_chunks)
        )
        dense_hits = self._validate_hits(
            "dense", self.dense.search(query, self.config.dense.top_k_chunks)
        )
        channels: dict[str, list[ScoredChunk]] = {
            "bm25": bm25_hits,
            "dense": dense_hits,
        }

        hypothesis: str | None = None
        if self.config.hyde.enabled:
            assert self.hyde_generator is not None
            hypothesis = normalize_hyde_text(self.hyde_generator.generate(query))
            if not hypothesis:
                raise ValueError("HyDE generator returned an empty document")
            # HyDE is intentionally a dense-only lane: never BM25 the hallucination.
            channels["hyde"] = self._validate_hits(
                "hyde",
                self.dense.search(hypothesis, self.config.hyde.top_k_chunks),
            )

        candidates = weighted_rrf(
            channels,
            channel_weights=self.config.fusion.channel_weights,
            rrf_k=self.config.fusion.rrf_k,
            top_k_documents=self.config.fusion.candidate_documents,
            evidence_chunks_per_document=self.config.fusion.evidence_chunks_per_document,
        )
        fused_candidates = list(candidates)
        candidates = self._rerank(query, candidates)
        candidates = candidates[: self.config.reranker.final_top_k_documents]

        results = tuple(self._to_result(candidate) for candidate in candidates)
        return SearchResponse(
            query=query,
            results=results,
            hypothetical_document=hypothesis,
            fused_candidates=tuple(
                self._to_result(candidate) for candidate in fused_candidates
            ),
        )
