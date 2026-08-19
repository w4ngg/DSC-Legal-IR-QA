from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from legal_ir.create_val_test import create_val_test, split_records


def _records(count: int = 12) -> dict[str, dict[str, object]]:
    result = {
        f"q{index}": {
            "question": f"Câu hỏi số {index}?",
            "answer": [f"doc-{index}"],
        }
        for index in range(count)
    }
    result["q1"]["question"] = "  THỦ TỤC   cấp phép? "
    result["q2"]["question"] = "thủ tục cấp phép?"
    return result


class CreateValTestTest(unittest.TestCase):
    def test_split_is_deterministic_disjoint_and_keeps_duplicate_questions_together(
        self,
    ) -> None:
        records = _records()
        first = split_records(
            records,
            validation_size=3,
            test_size=3,
            seed=2026,
        )
        second = split_records(
            records,
            validation_size=3,
            test_size=3,
            seed=2026,
        )

        self.assertEqual(first, second)
        self.assertEqual({name: len(value) for name, value in first.items()}, {
            "train": 6,
            "val": 3,
            "test": 3,
        })
        id_sets = {name: set(value) for name, value in first.items()}
        self.assertFalse(id_sets["train"] & id_sets["val"])
        self.assertFalse(id_sets["train"] & id_sets["test"])
        self.assertFalse(id_sets["val"] & id_sets["test"])
        self.assertEqual(set.union(*id_sets.values()), set(records))
        duplicate_locations = [name for name, ids in id_sets.items() if "q1" in ids or "q2" in ids]
        self.assertEqual(len(duplicate_locations), 1)
        self.assertTrue({"q1", "q2"} <= id_sets[duplicate_locations[0]])

    def test_create_writes_three_splits_and_manifest_without_changing_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "source_train.json"
            output_dir = root / "splits"
            source_payload = _records()
            input_path.write_text(
                json.dumps(source_payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            original_bytes = input_path.read_bytes()

            manifest = create_val_test(
                input_path,
                output_dir,
                validation_size=3,
                test_size=3,
                seed=7,
            )

            self.assertEqual(input_path.read_bytes(), original_bytes)
            for filename in ("train.json", "val.json", "test.json", "split_manifest.json"):
                self.assertTrue((output_dir / filename).is_file())
            self.assertEqual(manifest["outputs"]["train"]["query_count"], 6)
            self.assertEqual(manifest["outputs"]["val"]["query_count"], 3)
            self.assertEqual(manifest["outputs"]["test"]["query_count"], 3)
            self.assertEqual(
                manifest["duplicate_questions"]["duplicate_question_group_count"],
                1,
            )
            with self.assertRaises(FileExistsError):
                create_val_test(
                    input_path,
                    output_dir,
                    validation_size=3,
                    test_size=3,
                    seed=7,
                )

    def test_split_sizes_must_leave_training_records(self) -> None:
        with self.assertRaisesRegex(ValueError, "leave at least one"):
            split_records(
                _records(6),
                validation_size=3,
                test_size=3,
                seed=1,
            )


if __name__ == "__main__":
    unittest.main()
