from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from .config import PipelineConfig
from .dual_rerank import (
    DualChunkAssets,
    LongCandidatePool,
    LongDocumentResult,
    build_long_candidate_pool,
    rerank_long_candidate_pool,
)
from .fusion import fuse_ranked_channels, rank_channels, rank_documents
from .hyde import normalize_hyde_text
from .interfaces import (
    ChunkRetriever,
    DenseChunkRetriever,
    HypotheticalDocumentGenerator,
    PassageReranker,
)
from .io import ChunkStore
from .schema import (
    DeepQueryDiagnostics,
    DocumentCandidate,
    RankedChunk,
    RetrievalChannelDiagnostics,
    ScoredChunk,
    SearchResponse,
    SearchResult,
)


class RetrievalPipeline:
    """BM25 + Vietnamese dense/HyDE, document RRF, then cross-encoder reranking."""

    def __init__(
        self,
        *,
        chunks: ChunkStore,
        bm25: ChunkRetriever,
        dense: DenseChunkRetriever,
        config: PipelineConfig,
        hyde_generator: HypotheticalDocumentGenerator | None = None,
        reranker: PassageReranker | None = None,
        dual_chunk_assets: DualChunkAssets | None = None,
    ) -> None:
        if config.hyde.enabled and hyde_generator is None:
            raise ValueError("HyDE is enabled but no generator was provided")
        if config.reranker.enabled and reranker is None:
            raise ValueError("reranking is enabled but no reranker was provided")
        if config.long_context.enabled and dual_chunk_assets is None:
            raise ValueError(
                "long-context reranking is enabled but no dual chunk assets "
                "were provided"
            )
        if config.long_context.enabled and not config.reranker.enabled:
            raise ValueError("long-context reranking requires an enabled reranker")
        self.chunks = chunks
        self.bm25 = bm25
        self.dense = dense
        self.config = config
        self.hyde_generator = hyde_generator
        self.reranker = reranker
        self.dual_chunk_assets = dual_chunk_assets

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

    @staticmethod
    def _long_pool_to_fused_results(
        pool: LongCandidatePool,
    ) -> tuple[SearchResult, ...]:
        """Expose deterministic pre-reranker MaxP document candidates.

        Long candidates are already ordered by mapped short-chunk RRF support,
        so the first occurrence of a document is its strongest retrieval-side
        long chunk.
        """

        seen_documents: set[str] = set()
        results: list[SearchResult] = []
        for candidate in pool.candidates:
            if candidate.document_id in seen_documents:
                continue
            seen_documents.add(candidate.document_id)
            results.append(
                SearchResult(
                    document_id=candidate.document_id,
                    score=candidate.retrieval_support_score,
                    fusion_score=candidate.retrieval_support_score,
                    rerank_score=None,
                    evidence_chunk_ids=(candidate.long_chunk_id,),
                    channel_ranks=dict(candidate.channel_best_ranks),
                    channel_scores=dict(candidate.channel_contributions),
                )
            )
        return tuple(results)

    @staticmethod
    def _long_document_to_result(
        document: LongDocumentResult,
    ) -> SearchResult:
        best = document.contributing_long_chunks[0]
        return SearchResult(
            document_id=document.document_id,
            score=document.score,
            fusion_score=best.candidate.retrieval_support_score,
            rerank_score=document.score,
            evidence_chunk_ids=tuple(
                item.long_chunk_id
                for item in document.contributing_long_chunks
            ),
            channel_ranks=dict(best.candidate.channel_best_ranks),
            channel_scores=dict(best.candidate.channel_contributions),
            evidence_rerank_scores={
                item.long_chunk_id: item.reranker_score
                for item in document.contributing_long_chunks
            },
        )

    def _search_long_context(
        self,
        query: str,
        ranked_channels: Mapping[str, Sequence[RankedChunk]],
        *,
        hypothesis: str | None,
    ) -> SearchResponse:
        assert self.dual_chunk_assets is not None
        assert self.reranker is not None

        long_config = self.config.long_context
        long_candidate_limit = (
            None
            if long_config.candidate_mode == "full"
            else long_config.candidate_top_k
        )
        pool = build_long_candidate_pool(
            ranked_channels,
            assets=self.dual_chunk_assets,
            channel_weights=self.config.fusion.channel_weights,
            rrf_k=self.config.fusion.rrf_k,
            long_candidate_limit=long_candidate_limit,
        )
        outcome = rerank_long_candidate_pool(
            query,
            pool,
            long_chunks=self.dual_chunk_assets.long_chunks,
            reranker=self.reranker,
            post_reranker_long_top_k=long_config.rerank_top_k_chunks,
            final_top_k_documents=self.config.reranker.final_top_k_documents,
        )
        long_diagnostics = outcome.to_diagnostics_dict()
        long_diagnostics["candidate_mode"] = long_config.candidate_mode
        long_diagnostics["stores_all_scored_long_candidates"] = bool(
            long_config.diagnostics_store_all_candidates
        )
        if not long_config.diagnostics_store_all_candidates:
            long_diagnostics.pop("long_candidates", None)

        return SearchResponse(
            query=query,
            results=tuple(
                self._long_document_to_result(document)
                for document in outcome.results
            ),
            hypothetical_document=hypothesis,
            fused_candidates=self._long_pool_to_fused_results(pool),
            long_context_diagnostics=long_diagnostics,
        )

    def _search(
        self, query: str, *, capture_deep_diagnostics: bool
    ) -> tuple[SearchResponse, DeepQueryDiagnostics | None]:
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
        channel_search: dict[str, tuple[str, str, int]] = {
            "bm25": ("query", query, self.config.bm25.top_k_chunks),
            "dense": ("query", query, self.config.dense.top_k_chunks),
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
                self.dense.search(
                    hypothesis,
                    self.config.hyde.top_k_chunks,
                    input_type="document",
                ),
            )
            channel_search["hyde"] = (
                "hypothetical_document",
                hypothesis,
                self.config.hyde.top_k_chunks,
            )

        ranked_channels = rank_channels(channels)
        deep_diagnostics: DeepQueryDiagnostics | None = None
        if capture_deep_diagnostics:
            deep_diagnostics = DeepQueryDiagnostics(
                query=query,
                hypothetical_document=hypothesis,
                channels={
                    channel: RetrievalChannelDiagnostics(
                        search_text=channel_search[channel][1],
                        search_text_source=channel_search[channel][0],
                        requested_top_k_chunks=channel_search[channel][2],
                        chunk_hits=tuple(hits),
                        document_hits=tuple(rank_documents(hits)),
                    )
                    for channel, hits in ranked_channels.items()
                },
            )

        if self.config.long_context.enabled:
            return (
                self._search_long_context(
                    query,
                    ranked_channels,
                    hypothesis=hypothesis,
                ),
                deep_diagnostics,
            )

        candidates = fuse_ranked_channels(
            ranked_channels,
            channel_weights=self.config.fusion.channel_weights,
            rrf_k=self.config.fusion.rrf_k,
            top_k_documents=self.config.fusion.candidate_documents,
            evidence_chunks_per_document=self.config.fusion.evidence_chunks_per_document,
        )
        fused_candidates = list(candidates)
        candidates = self._rerank(query, candidates)
        candidates = candidates[: self.config.reranker.final_top_k_documents]

        results = tuple(self._to_result(candidate) for candidate in candidates)
        return (
            SearchResponse(
                query=query,
                results=results,
                hypothetical_document=hypothesis,
                fused_candidates=tuple(
                    self._to_result(candidate) for candidate in fused_candidates
                ),
            ),
            deep_diagnostics,
        )

    def close(self) -> None:
        """Release persistent model workers owned by runtime adapters."""

        if self.reranker is not None:
            close = getattr(self.reranker, "close", None)
            if callable(close):
                close()

    def search(self, query: str) -> SearchResponse:
        """Run retrieval without retaining the full pre-fusion trace."""

        response, _ = self._search(query, capture_deep_diagnostics=False)
        return response

    def search_with_deep_diagnostics(
        self, query: str
    ) -> tuple[SearchResponse, DeepQueryDiagnostics]:
        """Run retrieval once and also return every canonical channel hit."""

        response, diagnostics = self._search(
            query, capture_deep_diagnostics=True
        )
        assert diagnostics is not None
        return response, diagnostics
