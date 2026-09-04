from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from legal_ir.diagnostics_top2_mean_submission import (
    diagnostics_to_top2_mean_submission,
    main,
    top2_mean_score,
)


class DiagnosticsTop2MeanSubmissionTest(unittest.TestCase):
    def test_top2_mean_uses_two_highest_evidence_scores(self) -> None:
        candidate = {
            "document_id": "A",
            "evidence_rerank_scores": {
                "a-1": 10.0,
                "a-2": 4.0,
                "a-3": -2.0,
            },
        }

        self.assertEqual(top2_mean_score(candidate, query_id="q1"), 7.0)

    def test_single_evidence_score_is_allowed(self) -> None:
        candidate = {
            "document_id": "A",
            "evidence_rerank_scores": {"a-1": 3.5},
        }

        self.assertEqual(top2_mean_score(candidate, query_id="q1"), 3.5)

    def test_submission_is_sorted_by_top2_mean_not_maxp(self) -> None:
        diagnostics = {
            "q1": {
                "results": [
                    {
                        "document_id": "A",
                        "rerank_score": 10.0,
                    }
                ],
                "fused_candidates": [
                    {
                        "document_id": "A",
                        "fusion_score": 0.5,
                        "evidence_rerank_scores": {
                            "a-1": 10.0,
                            "a-2": -10.0,
                        },
                    },
                    {
                        "document_id": "B",
                        "fusion_score": 0.4,
                        "evidence_rerank_scores": {
                            "b-1": 5.0,
                            "b-2": 4.0,
                        },
                    },
                    {
                        "document_id": "C",
                        "fusion_score": 0.3,
                        "evidence_rerank_scores": {"c-1": 3.0},
                    },
                ],
            }
        }

        submission = diagnostics_to_top2_mean_submission(diagnostics)

        self.assertEqual(submission, {"q1": {"answer": ["B", "C", "A"]}})

    def test_cli_writes_official_submission_json(self) -> None:
        diagnostics = {
            "q1": {
                "fused_candidates": [
                    {
                        "document_id": str(index),
                        "fusion_score": 1.0 / index,
                        "evidence_rerank_scores": {f"c-{index}": float(index)},
                    }
                    for index in range(1, 8)
                ]
            }
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            diagnostics_path = root / "diagnostics.json"
            output_path = root / "submission.json"
            diagnostics_path.write_text(
                json.dumps(diagnostics, ensure_ascii=False),
                encoding="utf-8",
            )

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "--diagnostics",
                        str(diagnostics_path),
                        "--output",
                        str(output_path),
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertIn("top2_mean_evidence_rerank_scores", stdout.getvalue())
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(
                payload,
                {"q1": {"answer": ["7", "6", "5", "4", "3"]}},
            )

    def test_missing_evidence_scores_are_rejected(self) -> None:
        diagnostics = {
            "q1": {
                "fused_candidates": [
                    {
                        "document_id": "A",
                        "fusion_score": 0.5,
                        "evidence_rerank_scores": {},
                    }
                ]
            }
        }

        with self.assertRaisesRegex(ValueError, "has no evidence_rerank_scores"):
            diagnostics_to_top2_mean_submission(diagnostics)


if __name__ == "__main__":
    unittest.main()
