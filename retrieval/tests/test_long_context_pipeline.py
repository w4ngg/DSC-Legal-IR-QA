from __future__ import annotations

import unittest
from dataclasses import replace
from typing import Sequence

from legal_ir.config import LongContextConfig, PipelineConfig
from legal_ir.dual_rerank import (
    DualChunkAssets,
    ShortToLongMapping,
    ShortToLongStore,
)
from legal_ir.io import ChunkStore
from legal_ir.pipeline import RetrievalPipeline
from legal_ir.schema import Chunk, ScoredChunk


NAMESPACE = "dual_char_v1_pipeline_test"


def _short_id(document_id: str, index: int) -> str:
    return f"{document_id}:{NAMESPACE}:short:{index:06d}"


def _long_id(document_id: str, index: int) -> str:
    return f"{document_id}:{NAMESPACE}:long:{index:06d}"


class StaticRetriever:
    def __init__(self, results: dict[str, list[ScoredChunk]]) -> None:
        self.results = results
        self.calls: list[tuple[str, int, str]] = []

    def search(
        self,
        query: str,
        top_k: int,
        *,
        input_type: str = "query",
    ) -> list[ScoredChunk]:
        self.calls.append((query, top_k, input_type))
        return self.results.get(query, [])[:top_k]


class RecordingReranker:
    def __init__(self, scores_by_passage: dict[str, float]) -> None:
        self.scores_by_passage = scores_by_passage
        self.queries: list[str] = []
        self.passage_batches: list[list[str]] = []

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        self.queries.append(query)
        self.passage_batches.append(list(passages))
        return [self.scores_by_passage[passage] for passage in passages]


def _long_config(
    *,
    candidate_mode: str = "full",
    candidate_top_k: int | None = None,
    rerank_top_k_chunks: int = 20,
    final_top_k_documents: int = 5,
) -> PipelineConfig:
    default = PipelineConfig()
    return replace(
        default,
        bm25=replace(default.bm25, top_k_chunks=50),
        dense=replace(default.dense, top_k_chunks=100),
        hyde=replace(default.hyde, enabled=False),
        reranker=replace(
            default.reranker,
            enabled=True,
            final_top_k_documents=final_top_k_documents,
        ),
        long_context=LongContextConfig(
            enabled=True,
            candidate_mode=candidate_mode,
            candidate_top_k=candidate_top_k,
            rerank_top_k_chunks=rerank_top_k_chunks,
            document_aggregation="maxp",
            diagnostics_store_all_candidates=True,
        ),
    )


def _dual_fixture(
    document_ids: Sequence[str],
) -> tuple[ChunkStore, DualChunkAssets, list[str]]:
    short_chunks: list[Chunk] = []
    long_chunks: list[Chunk] = []
    mappings: list[ShortToLongMapping] = []
    long_passages: list[str] = []
    for index, document_id in enumerate(document_ids):
        short_chunk_id = _short_id(document_id, index)
        long_chunk_id = _long_id(document_id, index)
        long_index_text = f"LONG INDEX TEXT {index}"
        short_chunks.append(
            Chunk(
                short_chunk_id,
                document_id,
                f"short passage {index}",
                f"SHORT INDEX TEXT {index}",
                {
                    "granularity": "short",
                    "primary_long_chunk_id": long_chunk_id,
                    "long_chunk_ids": [long_chunk_id],
                },
            )
        )
        long_chunks.append(
            Chunk(
                long_chunk_id,
                document_id,
                f"long passage {index}",
                long_index_text,
                {"granularity": "long"},
            )
        )
        mappings.append(
            ShortToLongMapping(
                short_chunk_id=short_chunk_id,
                document_id=document_id,
                primary_long_chunk_id=long_chunk_id,
                long_chunk_ids=(long_chunk_id,),
            )
        )
        long_passages.append(long_index_text)
    return (
        ChunkStore(short_chunks),
        DualChunkAssets(
            mappings=ShortToLongStore(mappings),
            long_chunks=ChunkStore(long_chunks),
        ),
        long_passages,
    )


class LongContextPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.query = "Điều kiện cấp giấy phép là gì?"
        # Reranker top 20 contains only A/B. C/D are ranks 21/22, which
        # distinguishes top-long-then-MaxP from MaxP-over-all-long-candidates.
        document_ids = ["A"] * 10 + ["B"] * 10 + ["C", "D"]
        self.short_chunks, self.assets, long_passages = _dual_fixture(document_ids)
        bm25_indices = list(range(12))
        dense_indices = [0] + list(range(12, 22))
        self.bm25_hits = [
            ScoredChunk(
                _short_id(document_ids[index], index),
                document_ids[index],
                float(100 - rank),
            )
            for rank, index in enumerate(bm25_indices, start=1)
        ]
        self.dense_hits = [
            ScoredChunk(
                _short_id(document_ids[index], index),
                document_ids[index],
                1.0 - rank / 100.0,
            )
            for rank, index in enumerate(dense_indices, start=1)
        ]
        self.reranker_scores = {
            passage: float(100 - index)
            for index, passage in enumerate(long_passages)
        }

    def _pipeline(
        self,
        config: PipelineConfig,
        reranker: RecordingReranker,
    ) -> tuple[RetrievalPipeline, StaticRetriever, StaticRetriever]:
        bm25 = StaticRetriever({self.query: self.bm25_hits})
        dense = StaticRetriever({self.query: self.dense_hits})
        pipeline = RetrievalPipeline(
            chunks=self.short_chunks,
            bm25=bm25,
            dense=dense,
            config=config,
            reranker=reranker,
            dual_chunk_assets=self.assets,
        )
        return pipeline, bm25, dense

    def test_full_mode_scores_every_long_and_top20_precedes_maxp(self) -> None:
        reranker = RecordingReranker(self.reranker_scores)
        pipeline, bm25, dense = self._pipeline(_long_config(), reranker)

        response, deep = pipeline.search_with_deep_diagnostics(
            f"  {self.query}  "
        )
        payload = response.to_dict()
        diagnostics = payload["long_context"]

        self.assertEqual(bm25.calls, [(self.query, 50, "query")])
        self.assertEqual(dense.calls, [(self.query, 100, "query")])
        self.assertEqual(set(deep.channels), {"bm25", "dense"})
        self.assertEqual(reranker.queries, [self.query])
        self.assertEqual(len(reranker.passage_batches[0]), 22)
        self.assertTrue(
            all(
                passage.startswith("LONG INDEX TEXT")
                for passage in reranker.passage_batches[0]
            )
        )
        self.assertTrue(
            all(
                not passage.startswith("SHORT INDEX TEXT")
                for passage in reranker.passage_batches[0]
            )
        )

        counts = diagnostics["candidate_counts"]
        self.assertEqual(counts["lane_short_chunks"], {"bm25": 12, "dense": 11})
        self.assertEqual(counts["unique_short_chunks"], 22)
        self.assertEqual(counts["unique_long_chunks_before_cutoff"], 22)
        self.assertEqual(counts["selected_long_chunks"], 22)
        self.assertIsNone(counts["long_candidate_limit"])
        self.assertEqual(len(diagnostics["long_candidates"]), 22)
        self.assertEqual(len(diagnostics["top_long_candidates"]), 20)
        self.assertEqual(
            [item["reranker_rank"] for item in diagnostics["long_candidates"]],
            list(range(1, 23)),
        )
        self.assertTrue(
            all(
                item["supporting_short_chunks"]
                for item in diagnostics["long_candidates"]
            )
        )

        # C/D would appear if MaxP happened over all 22 chunks. Correct behavior
        # first cuts the long ranking at 20, leaving only A and B to aggregate.
        self.assertEqual(response.document_ids, ["A", "B"])
        self.assertEqual(
            [item["document_id"] for item in diagnostics["results"]],
            ["A", "B"],
        )

    def test_cutoff_mode_applies_prerank_limit_before_reranker(self) -> None:
        reranker = RecordingReranker(self.reranker_scores)
        config = _long_config(
            candidate_mode="cutoff",
            candidate_top_k=3,
            rerank_top_k_chunks=3,
        )
        pipeline, _, _ = self._pipeline(config, reranker)

        response = pipeline.search(self.query)
        diagnostics = response.to_dict()["long_context"]

        self.assertEqual(len(reranker.passage_batches), 1)
        self.assertEqual(len(reranker.passage_batches[0]), 3)
        counts = diagnostics["candidate_counts"]
        self.assertEqual(counts["unique_long_chunks_before_cutoff"], 22)
        self.assertEqual(counts["selected_long_chunks"], 3)
        self.assertEqual(counts["long_candidate_limit"], 3)
        self.assertEqual(len(diagnostics["long_candidates"]), 3)
        self.assertEqual(
            {item["long_chunk_id"] for item in diagnostics["long_candidates"]},
            {
                item["long_chunk_id"]
                for item in diagnostics["top_long_candidates"]
            },
        )

    def test_final_result_never_exceeds_five_documents(self) -> None:
        document_ids = [f"DOC-{index % 7}" for index in range(21)]
        short_chunks, assets, long_passages = _dual_fixture(document_ids)
        hits = [
            ScoredChunk(
                _short_id(document_id, index),
                document_id,
                float(100 - index),
            )
            for index, document_id in enumerate(document_ids)
        ]
        reranker = RecordingReranker(
            {
                passage: float(100 - index)
                for index, passage in enumerate(long_passages)
            }
        )
        pipeline = RetrievalPipeline(
            chunks=short_chunks,
            bm25=StaticRetriever({self.query: hits}),
            dense=StaticRetriever({self.query: []}),
            config=_long_config(),
            reranker=reranker,
            dual_chunk_assets=assets,
        )

        response = pipeline.search(self.query)

        self.assertEqual(len(response.results), 5)
        self.assertEqual(len(set(response.document_ids)), 5)

    def test_enabled_long_mode_requires_dual_assets(self) -> None:
        with self.assertRaisesRegex(ValueError, "dual.*assets|assets.*dual"):
            RetrievalPipeline(
                chunks=self.short_chunks,
                bm25=StaticRetriever({self.query: self.bm25_hits}),
                dense=StaticRetriever({self.query: self.dense_hits}),
                config=_long_config(),
                reranker=RecordingReranker(self.reranker_scores),
            )

    def test_legacy_mode_does_not_require_or_emit_long_context(self) -> None:
        query = "legacy query"
        short_chunks = ChunkStore(
            [
                Chunk("legacy-a", "A", "short A", "SHORT A"),
                Chunk("legacy-b", "B", "short B", "SHORT B"),
            ]
        )
        bm25 = StaticRetriever(
            {query: [ScoredChunk("legacy-a", "A", 2.0)]}
        )
        dense = StaticRetriever(
            {query: [ScoredChunk("legacy-b", "B", 1.0)]}
        )
        reranker = RecordingReranker({"SHORT A": 1.0, "SHORT B": 2.0})
        default = PipelineConfig()
        config = replace(
            default,
            hyde=replace(default.hyde, enabled=False),
            long_context=replace(default.long_context, enabled=False),
            reranker=replace(default.reranker, enabled=True),
        )
        pipeline = RetrievalPipeline(
            chunks=short_chunks,
            bm25=bm25,
            dense=dense,
            config=config,
            reranker=reranker,
        )

        response = pipeline.search(query)

        self.assertEqual(response.document_ids, ["B", "A"])
        self.assertEqual(reranker.queries, [query])
        self.assertTrue(
            all(
                passage.startswith("SHORT")
                for passage in reranker.passage_batches[0]
            )
        )
        self.assertNotIn("long_context", response.to_dict())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
