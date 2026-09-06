from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from typing import Sequence

from legal_ir.dual_rerank import (
    DualChunkAssets,
    ShortChunkMetadataLookup,
    ShortToLongMapping,
    ShortToLongStore,
    aggregate_long_maxp,
    build_long_candidate_pool,
    rerank_long_candidate_pool,
    score_long_candidates,
)
from legal_ir.io import ChunkStore
from legal_ir.schema import Chunk, RankedChunk


NAMESPACE = "dual_char_v1_test"


def short_id(document_id: str, index: int) -> str:
    return f"{document_id}:{NAMESPACE}:short:{index:06d}"


def long_id(document_id: str, index: int) -> str:
    return f"{document_id}:{NAMESPACE}:long:{index:06d}"


def mapping(
    document_id: str,
    short_index: int,
    *long_indices: int,
) -> ShortToLongMapping:
    ids = tuple(long_id(document_id, index) for index in long_indices)
    return ShortToLongMapping(
        short_chunk_id=short_id(document_id, short_index),
        document_id=document_id,
        primary_long_chunk_id=ids[0],
        long_chunk_ids=ids,
    )


class RecordingReranker:
    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.query: str | None = None
        self.passages: list[str] = []

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        self.query = query
        self.passages = list(passages)
        return [self.scores[passage] for passage in passages]


class DualRerankTest(unittest.TestCase):
    def setUp(self) -> None:
        self.long_chunks = ChunkStore(
            [
                Chunk(
                    long_id("A", 0),
                    "A",
                    "passage A0",
                    "title A | passage A0",
                    {"granularity": "long"},
                ),
                Chunk(
                    long_id("A", 1),
                    "A",
                    "passage A1",
                    "title A | passage A1",
                    {"granularity": "long"},
                ),
                Chunk(
                    long_id("B", 0),
                    "B",
                    "passage B0",
                    "title B | passage B0",
                    {"granularity": "long"},
                ),
                Chunk(
                    long_id("C", 0),
                    "C",
                    "passage C0",
                    "title C | passage C0",
                    {"granularity": "long"},
                ),
            ]
        )
        self.mapping_store = ShortToLongStore(
            [
                mapping("A", 0, 0, 1),
                mapping("A", 1, 1),
                mapping("B", 0, 0),
                mapping("C", 0, 0),
            ]
        )
        self.assets = DualChunkAssets(
            mappings=self.mapping_store,
            long_chunks=self.long_chunks,
        )
        self.channels = {
            "bm25": [
                RankedChunk(short_id("A", 0), "A", 9.0, 1, "bm25"),
                RankedChunk(short_id("B", 0), "B", 8.0, 2, "bm25"),
            ],
            "dense": [
                RankedChunk(short_id("A", 1), "A", 0.9, 1, "dense"),
                RankedChunk(short_id("A", 0), "A", 0.8, 2, "dense"),
                RankedChunk(short_id("C", 0), "C", 0.7, 3, "dense"),
            ],
        }

    def test_union_mapping_rrf_provenance_and_cutoff(self) -> None:
        pool = build_long_candidate_pool(
            self.channels,
            assets=self.assets,
            channel_weights={"bm25": 1.0, "dense": 2.0},
            rrf_k=10,
            long_candidate_limit=2,
        )

        self.assertEqual(pool.lane_short_chunk_counts, {"bm25": 2, "dense": 3})
        self.assertEqual(pool.unique_short_chunk_count, 4)
        self.assertEqual(pool.mapped_long_occurrence_count, 5)
        self.assertEqual(pool.unique_long_chunk_count, 4)
        self.assertEqual(pool.selected_long_chunk_count, 2)
        self.assertEqual(
            [item.long_chunk_id for item in pool.candidates],
            [long_id("A", 1), long_id("A", 0)],
        )

        first = pool.candidates[0]
        expected_bm25 = 1.0 / 11
        expected_dense = 2.0 / 11 + 2.0 / 12
        self.assertTrue(
            math.isclose(
                first.retrieval_support_score,
                expected_bm25 + expected_dense,
            )
        )
        self.assertEqual(first.channel_best_ranks, {"bm25": 1, "dense": 1})
        self.assertTrue(
            math.isclose(first.channel_contributions["bm25"], expected_bm25)
        )
        self.assertTrue(
            math.isclose(first.channel_contributions["dense"], expected_dense)
        )
        self.assertEqual(
            first.supporting_short_chunk_ids,
            (short_id("A", 0), short_id("A", 1)),
        )
        shared_short = first.supporting_short_chunks[0]
        self.assertEqual(shared_short.channel_ranks, {"bm25": 1, "dense": 2})
        self.assertEqual(shared_short.channel_scores, {"bm25": 9.0, "dense": 0.8})

    def test_full_mode_keeps_all_and_ties_use_long_chunk_id(self) -> None:
        channels = {
            "bm25": [
                RankedChunk(short_id("A", 0), "A", 1.0, 1, "bm25"),
            ]
        }
        pool = build_long_candidate_pool(
            channels,
            assets=self.assets,
            channel_weights={"bm25": 1.0},
            rrf_k=60,
            long_candidate_limit=None,
        )

        self.assertIsNone(pool.long_candidate_limit)
        self.assertEqual(
            [item.long_chunk_id for item in pool.candidates],
            [long_id("A", 0), long_id("A", 1)],
        )
        self.assertEqual([item.retrieval_rank for item in pool.candidates], [1, 2])

    def test_reranks_every_long_then_top_k_long_maxp_to_documents(self) -> None:
        pool = build_long_candidate_pool(
            self.channels,
            assets=self.assets,
            channel_weights={"bm25": 1.0, "dense": 1.0},
            rrf_k=60,
            long_candidate_limit=None,
        )
        reranker = RecordingReranker(
            {
                "title A | passage A0": 0.80,
                "title A | passage A1": 0.90,
                "title B | passage B0": 0.85,
                "title C | passage C0": 0.70,
            }
        )

        outcome = rerank_long_candidate_pool(
            "  original query  ",
            pool,
            long_chunks=self.long_chunks,
            reranker=reranker,
            post_reranker_long_top_k=3,
            final_top_k_documents=2,
        )

        self.assertEqual(reranker.query, "original query")
        self.assertEqual(len(reranker.passages), 4)
        self.assertEqual(
            [item.long_chunk_id for item in outcome.reranked_long_candidates],
            [long_id("A", 1), long_id("B", 0), long_id("A", 0), long_id("C", 0)],
        )
        self.assertEqual(
            [item.document_id for item in outcome.document_candidates],
            ["A", "B"],
        )
        self.assertEqual([item.document_id for item in outcome.results], ["A", "B"])
        self.assertEqual(
            [
                item.long_chunk_id
                for item in outcome.document_candidates[0].contributing_long_chunks
            ],
            [long_id("A", 1), long_id("A", 0)],
        )

        diagnostics = outcome.to_diagnostics_dict()
        self.assertEqual(len(diagnostics["long_candidates"]), 4)
        self.assertEqual(len(diagnostics["top_long_candidates"]), 3)
        self.assertEqual(diagnostics["candidate_counts"]["selected_long_chunks"], 4)
        self.assertIn(
            "supporting_short_chunks",
            diagnostics["long_candidates"][0],
        )

    def test_reranker_score_validation(self) -> None:
        pool = build_long_candidate_pool(
            {"bm25": self.channels["bm25"][:1]},
            assets=self.assets,
            channel_weights={"bm25": 1.0},
            rrf_k=60,
            long_candidate_limit=None,
        )

        class TooShort:
            def score(self, query: str, passages: Sequence[str]) -> list[float]:
                return [1.0]

        with self.assertRaisesRegex(ValueError, "1 scores for 2"):
            score_long_candidates(
                "query",
                pool,
                long_chunks=self.long_chunks,
                reranker=TooShort(),
            )

        class NonFinite:
            def score(self, query: str, passages: Sequence[str]) -> list[float]:
                return [float("nan")] * len(passages)

        with self.assertRaisesRegex(ValueError, "non-finite"):
            score_long_candidates(
                "query",
                pool,
                long_chunks=self.long_chunks,
                reranker=NonFinite(),
            )

    def test_mapping_and_loaded_long_document_must_match(self) -> None:
        with self.assertRaisesRegex(ValueError, "different dual chunk namespaces"):
            ShortToLongMapping(
                short_chunk_id=short_id("A", 0),
                document_id="A",
                primary_long_chunk_id="A:other:long:000000",
                long_chunk_ids=("A:other:long:000000",),
            )

        bad_assets = DualChunkAssets(
            mappings=ShortToLongStore([mapping("A", 0, 0)]),
            long_chunks=ChunkStore(
                [
                    Chunk(
                        long_id("A", 0),
                        "WRONG",
                        "bad",
                        metadata={"granularity": "long"},
                    )
                ]
            ),
        )
        with self.assertRaisesRegex(ValueError, "has document_id WRONG"):
            build_long_candidate_pool(
                {"bm25": self.channels["bm25"][:1]},
                assets=bad_assets,
                channel_weights={"bm25": 1.0},
                rrf_k=60,
                long_candidate_limit=None,
            )

    def test_load_standalone_mapping_and_use_indexed_short_metadata(self) -> None:
        short = Chunk(
            short_id("A", 0),
            "A",
            "short passage",
            metadata={
                "granularity": "short",
                "primary_long_chunk_id": long_id("A", 0),
                "long_chunk_ids": [long_id("A", 0), long_id("A", 1)],
            },
        )
        metadata_lookup = ShortChunkMetadataLookup(ChunkStore([short]))
        self.assertEqual(
            metadata_lookup.get(short.chunk_id).long_chunk_ids,
            (long_id("A", 0), long_id("A", 1)),
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "short_to_long.jsonl"
            record = {
                "short_chunk_id": short.chunk_id,
                "document_id": "A",
                "primary_long_chunk_id": long_id("A", 0),
                "long_chunk_ids": [long_id("A", 0), long_id("A", 1)],
            }
            path.write_text(
                json.dumps(record, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            loaded = ShortToLongStore.load_jsonl(path)

        self.assertEqual(len(loaded), 1)
        self.assertEqual(
            loaded.get(short.chunk_id).long_chunk_ids,
            tuple(record["long_chunk_ids"]),
        )

    def test_maxp_validates_positive_cutoffs(self) -> None:
        with self.assertRaisesRegex(ValueError, "post_reranker_long_top_k"):
            aggregate_long_maxp([], post_reranker_long_top_k=0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
