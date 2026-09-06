from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from evaluate_retrieval_recall import (
    evaluate_retrieval_recall,
    main,
)


def _hit(rank: int, chunk_id: str, document_id: str, score: float) -> dict:
    return {
        "rank": rank,
        "chunk_id": chunk_id,
        "document_id": document_id,
        "score": score,
    }


def _channel(requested: int, hits: list[dict]) -> dict:
    return {
        "requested_top_k_chunks": requested,
        "returned_chunk_count": len(hits),
        "chunk_hits": hits,
        # Deliberately bogus: the evaluator must rebuild documents from chunk_hits.
        "document_hits": [{"document_id": "must-not-be-used"}],
    }


def _deep_payload() -> dict:
    return {
        "format_version": 1,
        "pipeline_config": {
            "fusion": {"candidate_documents": 3},
            "reranker": {"final_top_k_documents": 2},
        },
        "queries": {
            "q1": {
                "channels": {
                    "bm25": _channel(
                        3,
                        [
                            _hit(1, "b1", "X", 10.0),
                            _hit(2, "b2", "G1", 9.0),
                            _hit(3, "b3", "G2", 8.0),
                        ],
                    ),
                    "dense": _channel(
                        2,
                        [
                            _hit(1, "d1", "G2", 0.9),
                            _hit(2, "d2", "Y", 0.8),
                        ],
                    ),
                    "hyde": _channel(
                        1,
                        [_hit(1, "h1", "G1", 0.7)],
                    ),
                }
            },
            "q2": {
                "channels": {
                    "bm25": _channel(3, [_hit(1, "b4", "Z", 7.0)]),
                    "dense": _channel(2, [_hit(1, "d3", "N", 0.6)]),
                    "hyde": _channel(1, [_hit(1, "h2", "N", 0.5)]),
                }
            },
        },
    }


def _diagnostics_payload() -> dict:
    return {
        "q1": {
            "fused_candidates": [
                {"document_id": "X"},
                {"document_id": "G1"},
                {"document_id": "G2"},
            ],
            "results": [
                {"document_id": "G2"},
                {"document_id": "X"},
            ],
        },
        "q2": {
            "fused_candidates": [
                {"document_id": "N"},
                {"document_id": "Z"},
            ],
            "results": [
                {"document_id": "N"},
                {"document_id": "Z"},
            ],
        },
    }


class EvaluateRetrievalRecallTest(unittest.TestCase):
    def setUp(self) -> None:
        self.gold = {
            "q1": ("G1", "G2"),
            "q2": ("Z",),
        }

    def test_truncates_chunks_before_document_aggregation_and_unions_lanes(self) -> None:
        evaluation = evaluate_retrieval_recall(
            self.gold,
            _deep_payload(),
            channel_top_k={"bm25": 2, "dense": 1},
        )

        retrieval = evaluation.summary["retrieval"]
        self.assertAlmostEqual(retrieval["bm25"]["macro_recall"], 0.75)
        self.assertAlmostEqual(retrieval["dense"]["macro_recall"], 0.25)
        self.assertEqual(retrieval["oracle_union"]["macro_recall"], 1.0)
        q1 = evaluation.per_query["queries"]["q1"]["retrieval"]
        self.assertEqual(q1["bm25"]["retrieved_document_ids"], ["X", "G1"])
        self.assertEqual(
            q1["oracle_union"]["retrieved_document_ids"],
            ["X", "G1", "G2"],
        )
        self.assertNotIn("hyde", retrieval)

    def test_optional_hyde_is_included_only_when_requested(self) -> None:
        evaluation = evaluate_retrieval_recall(
            self.gold,
            _deep_payload(),
            channel_top_k={"bm25": 1, "dense": 1, "hyde": 1},
        )

        self.assertIn("hyde", evaluation.summary["retrieval"])
        q1 = evaluation.per_query["queries"]["q1"]["retrieval"]
        self.assertEqual(q1["hyde"]["retrieved_document_ids"], ["G1"])
        self.assertEqual(q1["oracle_union"]["recall"], 1.0)

    def test_regular_diagnostics_add_fusion_and_final_recall(self) -> None:
        evaluation = evaluate_retrieval_recall(
            self.gold,
            _deep_payload(),
            channel_top_k={"bm25": 2, "dense": 1},
            diagnostics=_diagnostics_payload(),
            fusion_cutoffs=(1, 2, 3),
            final_cutoffs=(1, 2),
            strict_query_ids=True,
        )

        diagnostics = evaluation.summary["diagnostics"]
        self.assertEqual(diagnostics["fusion"]["1"]["macro_recall"], 0.0)
        self.assertAlmostEqual(
            diagnostics["fusion"]["2"]["macro_recall"],
            0.75,
        )
        self.assertEqual(diagnostics["fusion"]["3"]["macro_recall"], 1.0)
        self.assertAlmostEqual(
            diagnostics["final"]["1"]["macro_recall"],
            0.25,
        )
        self.assertAlmostEqual(
            diagnostics["final"]["2"]["macro_recall"],
            0.75,
        )

    def test_rejects_cutoff_deeper_than_logged_chunk_hits(self) -> None:
        with self.assertRaisesRegex(ValueError, "only requested top-3"):
            evaluate_retrieval_recall(
                self.gold,
                _deep_payload(),
                channel_top_k={"bm25": 4, "dense": 1},
            )

    def test_missing_query_scores_empty_and_strict_mode_rejects_it(self) -> None:
        deep = _deep_payload()
        del deep["queries"]["q2"]

        evaluation = evaluate_retrieval_recall(
            self.gold,
            deep,
            channel_top_k={"bm25": 2, "dense": 1},
        )
        self.assertEqual(
            evaluation.summary["deep_diagnostics_query_alignment"][
                "missing_query_ids"
            ],
            ["q2"],
        )
        self.assertAlmostEqual(
            evaluation.summary["retrieval"]["oracle_union"]["macro_recall"],
            0.5,
        )

        with self.assertRaisesRegex(ValueError, "query ID mismatch"):
            evaluate_retrieval_recall(
                self.gold,
                deep,
                channel_top_k={"bm25": 2, "dense": 1},
                strict_query_ids=True,
            )

    def test_cli_writes_summary_and_per_query_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gold_path = root / "gold.json"
            deep_path = root / "deep.json"
            diagnostics_path = root / "diagnostics.json"
            summary_path = root / "summary.json"
            per_query_path = root / "per_query.json"
            gold_path.write_text(
                json.dumps(
                    {
                        "q1": {"question": "Q1?", "answer": ["G1", "G2"]},
                        "q2": {"question": "Q2?", "answer": ["Z"]},
                    }
                ),
                encoding="utf-8",
            )
            deep_path.write_text(json.dumps(_deep_payload()), encoding="utf-8")
            diagnostics_path.write_text(
                json.dumps(_diagnostics_payload()),
                encoding="utf-8",
            )

            with redirect_stdout(StringIO()):
                exit_code = main(
                    [
                        "--gold",
                        str(gold_path),
                        "--deep-diagnostics",
                        str(deep_path),
                        "--diagnostics",
                        str(diagnostics_path),
                        "--bm25-top-k",
                        "2",
                        "--dense-top-k",
                        "1",
                        "--fusion-cutoffs",
                        "1,2,3",
                        "--final-cutoffs",
                        "1,2",
                        "--output",
                        str(summary_path),
                        "--per-query-output",
                        str(per_query_path),
                        "--strict-query-ids",
                    ]
                )

            self.assertEqual(exit_code, 0)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            per_query = json.loads(per_query_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["retrieval"]["oracle_union"]["macro_recall"], 1.0)
            self.assertIn("q1", per_query["queries"])


if __name__ == "__main__":
    unittest.main()
