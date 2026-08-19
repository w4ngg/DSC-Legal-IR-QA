from __future__ import annotations

import unittest

from legal_ir.fusion import weighted_rrf
from legal_ir.schema import ScoredChunk


class WeightedRRFTest(unittest.TestCase):
    def test_chunk_hits_are_aggregated_before_document_ranking(self) -> None:
        candidates = weighted_rrf(
            {
                "bm25": [
                    ScoredChunk("a-1", "A", 10.0),
                    ScoredChunk("a-2", "A", 9.0),
                    ScoredChunk("b-1", "B", 8.0),
                ]
            },
            channel_weights={"bm25": 1.0},
            rrf_k=60,
            top_k_documents=10,
            evidence_chunks_per_document=2,
        )

        self.assertEqual([candidate.document_id for candidate in candidates], ["A", "B"])
        self.assertEqual(candidates[0].channel_ranks["bm25"], 1)
        self.assertEqual(candidates[1].channel_ranks["bm25"], 2)
        self.assertEqual(candidates[0].evidence_chunk_ids, ["a-1", "a-2"])

    def test_raw_scores_from_different_channels_are_not_compared(self) -> None:
        candidates = weighted_rrf(
            {
                "bm25": [ScoredChunk("a", "A", 1000.0)],
                "dense": [ScoredChunk("b", "B", 0.99)],
            },
            channel_weights={"bm25": 1.0, "dense": 1.0},
            rrf_k=60,
            top_k_documents=10,
            evidence_chunks_per_document=1,
        )

        self.assertAlmostEqual(candidates[0].fusion_score, candidates[1].fusion_score)
        self.assertEqual({candidate.document_id for candidate in candidates}, {"A", "B"})

    def test_rerank_evidence_reserves_a_real_query_chunk(self) -> None:
        candidates = weighted_rrf(
            {
                "bm25": [ScoredChunk("grounded", "A", 1.0)],
                "dense": [],
                "hyde": [
                    ScoredChunk("hyde-1", "A", 0.99),
                    ScoredChunk("hyde-2", "A", 0.98),
                ],
            },
            channel_weights={"bm25": 0.1, "dense": 1.0, "hyde": 10.0},
            rrf_k=60,
            top_k_documents=10,
            evidence_chunks_per_document=2,
        )

        self.assertIn("grounded", candidates[0].evidence_chunk_ids)
        self.assertEqual(len(candidates[0].evidence_chunk_ids), 2)

    def test_grounded_reservation_respects_one_chunk_limit(self) -> None:
        candidates = weighted_rrf(
            {
                "bm25": [ScoredChunk("grounded", "A", 1.0)],
                "hyde": [ScoredChunk("hyde", "A", 0.99)],
            },
            channel_weights={"bm25": 0.1, "hyde": 10.0},
            rrf_k=60,
            top_k_documents=10,
            evidence_chunks_per_document=1,
        )

        self.assertEqual(candidates[0].evidence_chunk_ids, ["grounded"])


if __name__ == "__main__":
    unittest.main()
