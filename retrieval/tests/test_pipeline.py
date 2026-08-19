from __future__ import annotations

import unittest
from dataclasses import replace
from typing import Sequence

from legal_ir.config import PipelineConfig
from legal_ir.io import ChunkStore
from legal_ir.pipeline import RetrievalPipeline
from legal_ir.schema import Chunk, ScoredChunk


class StaticRetriever:
    def __init__(self, results: dict[str, list[ScoredChunk]]) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int) -> list[ScoredChunk]:
        self.calls.append((query, top_k))
        return self.results.get(query, [])[:top_k]


class StaticHyDE:
    def __init__(self, hypothesis: str) -> None:
        self.hypothesis = hypothesis
        self.calls: list[str] = []

    def generate(self, query: str) -> str:
        self.calls.append(query)
        return self.hypothesis


class RecordingReranker:
    def __init__(self, passage_scores: dict[str, float]) -> None:
        self.passage_scores = passage_scores
        self.query: str | None = None
        self.passages: list[str] = []

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        self.query = query
        self.passages = list(passages)
        return [self.passage_scores[passage] for passage in passages]


def pipeline_config(*, hyde: bool = True, reranker: bool = True) -> PipelineConfig:
    default = PipelineConfig()
    return replace(
        default,
        bm25=replace(default.bm25, top_k_chunks=10),
        dense=replace(default.dense, top_k_chunks=10),
        hyde=replace(default.hyde, enabled=hyde, top_k_chunks=10),
        fusion=replace(
            default.fusion,
            candidate_documents=10,
            evidence_chunks_per_document=1,
        ),
        reranker=replace(
            default.reranker,
            enabled=reranker,
            final_top_k_documents=3,
        ),
    )


class RetrievalPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.query = "Điều kiện cấp giấy phép là gì?"
        self.hypothesis = "Đoạn quy phạm giả định do SLM sinh ra"
        self.chunks = ChunkStore(
            [
                Chunk("a-1", "A", "thật A", "Luật A | thật A"),
                Chunk("a-2", "A", "thật A phụ", "Luật A | thật A phụ"),
                Chunk("b-1", "B", "thật B", "Luật B | thật B"),
                Chunk("c-1", "C", "thật C", "Luật C | thật C"),
            ]
        )

    def test_hyde_is_dense_only_and_reranker_uses_original_query(self) -> None:
        bm25 = StaticRetriever(
            {
                self.query: [
                    ScoredChunk("a-1", "A", 20.0),
                    ScoredChunk("b-1", "B", 10.0),
                    ScoredChunk("a-2", "A", 9.0),
                ]
            }
        )
        dense = StaticRetriever(
            {
                self.query: [
                    ScoredChunk("b-1", "B", 0.9),
                    ScoredChunk("c-1", "C", 0.8),
                ],
                self.hypothesis: [
                    ScoredChunk("c-1", "C", 0.95),
                    ScoredChunk("a-1", "A", 0.7),
                ],
            }
        )
        hyde = StaticHyDE(self.hypothesis)
        reranker = RecordingReranker(
            {"Luật A | thật A": 2.0, "Luật B | thật B": 1.0, "Luật C | thật C": 3.0}
        )
        pipeline = RetrievalPipeline(
            chunks=self.chunks,
            bm25=bm25,
            dense=dense,
            hyde_generator=hyde,
            reranker=reranker,
            config=pipeline_config(),
        )

        response = pipeline.search(self.query)

        self.assertEqual(response.document_ids, ["C", "A", "B"])
        self.assertEqual(response.fused_candidate_document_ids, ("B", "A", "C"))
        self.assertEqual(bm25.calls, [(self.query, 10)])
        self.assertEqual(
            dense.calls,
            [(self.query, 10), (self.hypothesis, 10)],
        )
        self.assertEqual(hyde.calls, [self.query])
        self.assertEqual(reranker.query, self.query)
        self.assertNotIn(self.hypothesis, reranker.passages)
        self.assertEqual(response.hypothetical_document, self.hypothesis)

    def test_ablation_can_disable_hyde_and_reranker(self) -> None:
        bm25 = StaticRetriever({self.query: [ScoredChunk("a-1", "A", 10.0)]})
        dense = StaticRetriever({self.query: [ScoredChunk("b-1", "B", 0.9)]})
        pipeline = RetrievalPipeline(
            chunks=self.chunks,
            bm25=bm25,
            dense=dense,
            config=pipeline_config(hyde=False, reranker=False),
        )

        response = pipeline.search(self.query)

        self.assertEqual(response.document_ids, ["A", "B"])
        self.assertIsNone(response.hypothetical_document)
        self.assertTrue(all(result.rerank_score is None for result in response.results))

    def test_backend_cannot_break_chunk_document_mapping(self) -> None:
        bm25 = StaticRetriever({self.query: [ScoredChunk("a-1", "WRONG", 10.0)]})
        dense = StaticRetriever({self.query: []})
        pipeline = RetrievalPipeline(
            chunks=self.chunks,
            bm25=bm25,
            dense=dense,
            config=pipeline_config(hyde=False, reranker=False),
        )

        with self.assertRaisesRegex(ValueError, "expected A"):
            pipeline.search(self.query)


if __name__ == "__main__":
    unittest.main()
