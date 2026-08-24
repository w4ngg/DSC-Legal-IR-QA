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
        self.input_types: list[str] = []

    def search(
        self, query: str, top_k: int, *, input_type: str = "query"
    ) -> list[ScoredChunk]:
        self.calls.append((query, top_k))
        self.input_types.append(input_type)
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
        self.assertEqual(dense.input_types, ["query", "document"])
        self.assertEqual(hyde.calls, [self.query])
        self.assertEqual(reranker.query, self.query)
        self.assertNotIn(self.hypothesis, reranker.passages)
        self.assertEqual(response.hypothetical_document, self.hypothesis)

    def test_hyde_output_is_normalized_before_dense_search(self) -> None:
        normalized = "Đoạn pháp luật giả định"
        raw = "  Đoạn pha\u0301p\r\nluật\\n giả định  "
        bm25 = StaticRetriever({self.query: []})
        dense = StaticRetriever({self.query: [], normalized: []})
        pipeline = RetrievalPipeline(
            chunks=self.chunks,
            bm25=bm25,
            dense=dense,
            hyde_generator=StaticHyDE(raw),
            config=pipeline_config(reranker=False),
        )

        response = pipeline.search(self.query)

        self.assertEqual(dense.calls, [(self.query, 10), (normalized, 10)])
        self.assertEqual(response.hypothetical_document, normalized)

    def test_deep_diagnostics_keeps_every_channel_before_fusion_cutoff(self) -> None:
        bm25 = StaticRetriever(
            {
                self.query: [
                    ScoredChunk("a-2", "A", 9.0),
                    ScoredChunk("b-1", "B", 10.0),
                    ScoredChunk("a-1", "A", 20.0),
                    ScoredChunk("a-1", "A", 19.0),
                    ScoredChunk("c-1", "C", float("nan")),
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
        config = pipeline_config(reranker=False)
        config = replace(
            config,
            fusion=replace(config.fusion, candidate_documents=1),
            reranker=replace(config.reranker, final_top_k_documents=1),
        )
        pipeline = RetrievalPipeline(
            chunks=self.chunks,
            bm25=bm25,
            dense=dense,
            hyde_generator=StaticHyDE(self.hypothesis),
            config=config,
        )

        response, diagnostics = pipeline.search_with_deep_diagnostics(self.query)
        payload = diagnostics.to_dict()

        self.assertEqual(len(response.fused_candidates), 1)
        self.assertEqual(set(payload["channels"]), {"bm25", "dense", "hyde"})
        self.assertEqual(
            [hit["chunk_id"] for hit in payload["channels"]["bm25"]["chunk_hits"]],
            ["a-1", "b-1", "a-2"],
        )
        self.assertEqual(
            payload["channels"]["bm25"]["document_hits"],
            [
                {
                    "rank": 1,
                    "document_id": "A",
                    "score": 20.0,
                    "best_chunk_id": "a-1",
                    "best_chunk_rank": 1,
                },
                {
                    "rank": 2,
                    "document_id": "B",
                    "score": 10.0,
                    "best_chunk_id": "b-1",
                    "best_chunk_rank": 2,
                },
            ],
        )
        self.assertEqual(payload["channels"]["dense"]["search_text"], self.query)
        self.assertEqual(
            payload["channels"]["hyde"]["search_text"], self.hypothesis
        )
        self.assertEqual(
            payload["channels"]["hyde"]["search_text_source"],
            "hypothetical_document",
        )
        self.assertEqual(bm25.calls, [(self.query, 10)])
        self.assertEqual(
            dense.calls,
            [(self.query, 10), (self.hypothesis, 10)],
        )
        self.assertNotIn("channels", response.to_dict())

    def test_deep_diagnostics_omits_disabled_hyde_channel(self) -> None:
        pipeline = RetrievalPipeline(
            chunks=self.chunks,
            bm25=StaticRetriever({self.query: []}),
            dense=StaticRetriever({self.query: []}),
            config=pipeline_config(hyde=False, reranker=False),
        )

        _, diagnostics = pipeline.search_with_deep_diagnostics(self.query)

        self.assertEqual(set(diagnostics.channels), {"bm25", "dense"})
        self.assertIsNone(diagnostics.hypothetical_document)

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
