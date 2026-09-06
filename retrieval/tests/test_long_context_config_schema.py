from __future__ import annotations

import unittest
from pathlib import Path

from legal_ir.config import PipelineConfig
from legal_ir.schema import SearchResponse


class LongContextConfigTest(unittest.TestCase):
    def test_defaults_preserve_legacy_search_mode(self) -> None:
        config = PipelineConfig()

        self.assertFalse(config.long_context.enabled)
        self.assertEqual(config.long_context.candidate_mode, "full")
        self.assertIsNone(config.long_context.candidate_top_k)
        self.assertFalse(config.reranker.multi_gpu)

    def test_full_and_cutoff_candidate_modes_are_unambiguous(self) -> None:
        full = PipelineConfig.from_mapping(
            {
                "long_context": {
                    "enabled": True,
                    "candidate_mode": "full",
                    "candidate_top_k": None,
                    "rerank_top_k_chunks": 20,
                }
            }
        )
        self.assertTrue(full.long_context.enabled)
        self.assertIsNone(full.long_context.candidate_top_k)

        cutoff = PipelineConfig.from_mapping(
            {
                "long_context": {
                    "enabled": True,
                    "candidate_mode": "cutoff",
                    "candidate_top_k": 75,
                    "rerank_top_k_chunks": 20,
                }
            }
        )
        self.assertEqual(cutoff.long_context.candidate_top_k, 75)

    def test_invalid_long_candidate_configs_are_rejected(self) -> None:
        invalid_configs = (
            (
                {"candidate_mode": "unknown"},
                "candidate_mode must be 'full' or 'cutoff'",
            ),
            (
                {"candidate_mode": "full", "candidate_top_k": 100},
                "candidate_top_k must be null",
            ),
            (
                {"candidate_mode": "cutoff", "candidate_top_k": None},
                "candidate_top_k must be a positive integer",
            ),
            (
                {"candidate_mode": "cutoff", "candidate_top_k": True},
                "candidate_top_k must be a positive integer",
            ),
            (
                {
                    "candidate_mode": "cutoff",
                    "candidate_top_k": 10,
                    "rerank_top_k_chunks": 20,
                },
                "rerank_top_k_chunks must be <= candidate_top_k",
            ),
            (
                {"rerank_top_k_chunks": 0},
                "rerank_top_k_chunks must be positive",
            ),
            (
                {"document_aggregation": "mean"},
                "document_aggregation must be 'maxp'",
            ),
            (
                {"diagnostics_store_all_candidates": "yes"},
                "diagnostics_store_all_candidates must be a boolean",
            ),
        )
        for values, message in invalid_configs:
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, message):
                    PipelineConfig.from_mapping({"long_context": values})

    def test_reranker_multi_gpu_controls_are_validated(self) -> None:
        config = PipelineConfig.from_mapping(
            {
                "reranker": {
                    "multi_gpu": True,
                    "multi_gpu_stall_timeout_seconds": 900,
                }
            }
        )
        self.assertTrue(config.reranker.multi_gpu)
        self.assertEqual(config.reranker.multi_gpu_stall_timeout_seconds, 900)

        with self.assertRaisesRegex(ValueError, "multi_gpu must be a boolean"):
            PipelineConfig.from_mapping({"reranker": {"multi_gpu": "auto"}})
        for invalid in (0, -1, 1.5, True):
            with self.subTest(timeout=invalid):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    PipelineConfig.from_mapping(
                        {
                            "reranker": {
                                "multi_gpu_stall_timeout_seconds": invalid
                            }
                        }
                    )

    def test_long_context_requires_the_reranker(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "long_context.enabled requires reranker.enabled",
        ):
            PipelineConfig.from_mapping(
                {
                    "reranker": {"enabled": False},
                    "long_context": {"enabled": True},
                }
            )

    def test_checked_in_dual_long_preset_matches_selected_ablation(self) -> None:
        config_path = (
            Path(__file__).parents[1]
            / "configs"
            / "vietnamese_embedding_dual_long_rerank.yaml"
        )
        yaml_text = config_path.read_text(encoding="utf-8")
        sections = {
            name: yaml_text.split(f"{name}:\n", maxsplit=1)[1].split(
                "\n\n", maxsplit=1
            )[0]
            for name in ("bm25", "dense", "hyde", "reranker", "long_context")
        }

        self.assertIn("  top_k_chunks: 50", sections["bm25"])
        self.assertIn("  top_k_chunks: 100", sections["dense"])
        self.assertIn("  enabled: false", sections["hyde"])
        self.assertIn("  enabled: true", sections["reranker"])
        self.assertIn("  multi_gpu: true", sections["reranker"])
        self.assertIn("  final_top_k_documents: 5", sections["reranker"])
        self.assertIn("  enabled: true", sections["long_context"])
        self.assertIn("  candidate_mode: full", sections["long_context"])
        self.assertIn("  candidate_top_k: null", sections["long_context"])
        self.assertIn("  rerank_top_k_chunks: 20", sections["long_context"])
        self.assertIn("  document_aggregation: maxp", sections["long_context"])
        self.assertIn(
            "  diagnostics_store_all_candidates: true",
            sections["long_context"],
        )


class LongContextDiagnosticsSchemaTest(unittest.TestCase):
    def test_legacy_response_omits_long_context_diagnostics(self) -> None:
        payload = SearchResponse(query="Câu hỏi?", results=()).to_dict()

        self.assertNotIn("long_context", payload)

    def test_serializes_every_scored_candidate_and_audit_count(self) -> None:
        candidates = [
            {
                "long_chunk_id": "long-2",
                "document_id": "doc-2",
                "retrieval_rank": 2,
                "retrieval_support_score": 0.02,
                "reranker_score": 3.5,
                "reranker_rank": 1,
                "supporting_short_chunk_ids": ["short-b", "short-c"],
                "channel_best_ranks": {"bm25": 8, "dense": 3},
            },
            {
                "long_chunk_id": "long-1",
                "document_id": "doc-1",
                "retrieval_rank": 1,
                "retrieval_support_score": 0.03,
                "reranker_score": -1.25,
                "reranker_rank": 2,
                "supporting_short_chunk_ids": ["short-a"],
                "channel_best_ranks": {"bm25": 1},
            },
        ]
        diagnostics = {
            "candidate_counts": {
                "unique_short_chunks": 3,
                "mapped_long_occurrences": 5,
                "unique_long_chunks_before_cutoff": 2,
                "selected_long_chunks": 2,
                "long_candidate_limit": None,
            },
            "post_reranker_long_top_k": 20,
            "long_candidates": candidates,
            "top_long_candidates": candidates,
            "document_candidates": [],
            "results": [],
        }

        payload = SearchResponse(
            query="Câu hỏi?",
            results=(),
            long_context_diagnostics=diagnostics,
        ).to_dict()
        long_context = payload["long_context"]

        counts = long_context["candidate_counts"]
        self.assertEqual(counts["unique_short_chunks"], 3)
        self.assertEqual(counts["mapped_long_occurrences"], 5)
        self.assertEqual(counts["unique_long_chunks_before_cutoff"], 2)
        self.assertEqual(counts["selected_long_chunks"], 2)
        self.assertEqual(len(long_context["long_candidates"]), 2)
        self.assertEqual(
            long_context["long_candidates"][0]["supporting_short_chunk_ids"],
            ["short-b", "short-c"],
        )


if __name__ == "__main__":
    unittest.main()
