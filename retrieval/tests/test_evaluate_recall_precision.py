from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from legal_ir.evaluate_recall_precision import (
    evaluate_predictions,
    load_gold_answers,
    load_prediction_answers,
    main,
)


class EvaluateRecallPrecisionTest(unittest.TestCase):
    def test_official_macro_scores_include_missing_queries(self) -> None:
        result = evaluate_predictions(
            {
                "q1": ("A", "B"),
                "q2": ("C",),
                "q3": ("D",),
            },
            {
                "q1": ("A", "X"),
                "q2": ("C",),
            },
        )

        self.assertAlmostEqual(result.summary["official_macro_recall"], 0.5)
        self.assertAlmostEqual(result.summary["official_macro_precision"], 0.5)
        self.assertEqual(result.summary["missing_prediction_query_ids"], ["q3"])
        self.assertEqual(result.per_query["q3"]["recall"], 0.0)
        self.assertEqual(result.per_query["q3"]["precision"], 0.0)
        self.assertFalse(result.summary["submission_contract_valid"])

    def test_more_than_five_predictions_forces_zero_without_truncation(self) -> None:
        result = evaluate_predictions(
            {"q1": ("A",)},
            {"q1": ("A", "B", "C", "D", "E", "F")},
        )

        self.assertEqual(result.summary["official_macro_recall"], 0.0)
        self.assertEqual(result.summary["official_macro_precision"], 0.0)
        self.assertEqual(result.summary["over_limit_query_count"], 1)
        self.assertEqual(result.per_query["q1"]["prediction_count"], 6)

    def test_exactly_five_and_explicit_empty_predictions_are_valid(self) -> None:
        result = evaluate_predictions(
            {"q1": ("A",), "q2": ("Z",)},
            {"q1": ("A", "B", "C", "D", "E"), "q2": ()},
        )

        self.assertAlmostEqual(result.summary["official_macro_recall"], 0.5)
        self.assertAlmostEqual(result.summary["official_macro_precision"], 0.1)
        self.assertEqual(result.summary["empty_prediction_query_count"], 1)
        self.assertTrue(result.summary["submission_contract_valid"])

    def test_duplicate_predictions_use_set_formula_but_invalidate_contract(self) -> None:
        result = evaluate_predictions(
            {"q1": ("A",)},
            {"q1": ("A", "A")},
        )

        self.assertEqual(result.summary["official_macro_recall"], 1.0)
        self.assertEqual(result.summary["official_macro_precision"], 1.0)
        self.assertEqual(result.summary["duplicate_prediction_query_count"], 1)
        self.assertFalse(result.summary["submission_contract_valid"])

    def test_extra_query_is_reported_and_does_not_change_denominator(self) -> None:
        result = evaluate_predictions(
            {"q1": ("A",)},
            {"q1": ("A",), "extra": ("X",)},
        )

        self.assertEqual(result.summary["official_macro_recall"], 1.0)
        self.assertEqual(result.summary["extra_prediction_query_ids"], ["extra"])
        with self.assertRaisesRegex(ValueError, "query ID mismatch"):
            evaluate_predictions(
                {"q1": ("A",)},
                {"q1": ("A",), "extra": ("X",)},
                strict_query_ids=True,
            )

    def test_gold_null_and_non_string_prediction_id_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gold_path = root / "gold.json"
            prediction_path = root / "predictions.json"
            gold_path.write_text(
                json.dumps({"q1": {"question": "Câu hỏi?", "answer": None}}),
                encoding="utf-8",
            )
            prediction_path.write_text(
                json.dumps({"q1": {"answer": [123]}}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "must be a JSON array"):
                load_gold_answers(gold_path)
            with self.assertRaisesRegex(ValueError, "string document IDs"):
                load_prediction_answers(prediction_path)

    def test_duplicate_json_keys_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gold.json"
            path.write_text(
                '{"q1":{"answer":["A"]},"q1":{"answer":["B"]}}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate JSON key: q1"):
                load_gold_answers(path)

    def test_strict_submission_returns_nonzero_and_still_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gold_path = root / "gold.json"
            prediction_path = root / "predictions.json"
            report_path = root / "report.json"
            gold_path.write_text(
                json.dumps({"q1": {"question": "Câu hỏi?", "answer": ["A"]}}),
                encoding="utf-8",
            )
            prediction_path.write_text(json.dumps({}), encoding="utf-8")

            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                exit_code = main(
                    [
                        "--gold",
                        str(gold_path),
                        "--predictions",
                        str(prediction_path),
                        "--output",
                        str(report_path),
                        "--strict-submission",
                    ]
                )

            self.assertEqual(exit_code, 1)
            self.assertTrue(report_path.is_file())
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertFalse(report["submission_contract_valid"])


if __name__ == "__main__":
    unittest.main()
