from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from legal_ir.config import PipelineConfig
from legal_ir.io import ChunkStore, load_questions, write_submission
from legal_ir.schema import Chunk, SearchResponse, SearchResult


class IOAndConfigTest(unittest.TestCase):
    def test_chunk_ids_and_document_ids_are_normalized_to_strings(self) -> None:
        chunk = Chunk.from_dict(
            {
                "chunk_id": 21,
                "document_id": 740,
                "passage": "Nội dung khoản 1.",
                "metadata": {"article": "Điều 3"},
            }
        )
        self.assertEqual(chunk.chunk_id, "21")
        self.assertEqual(chunk.document_id, "740")
        self.assertEqual(chunk.index_text, "Nội dung khoản 1.")

    def test_official_question_and_submission_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            query_path = root / "queries.json"
            query_path.write_text(
                json.dumps(
                    {"q1": {"question": "Câu hỏi?", "answer": None}},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            self.assertEqual(load_questions(query_path), {"q1": "Câu hỏi?"})

            response = SearchResponse(
                query="Câu hỏi?",
                results=tuple(
                    SearchResult(
                        document_id=str(index),
                        score=1.0,
                        fusion_score=0.1,
                        rerank_score=1.0,
                        evidence_chunk_ids=(f"c-{index}",),
                        channel_ranks={"bm25": index},
                    )
                    for index in range(1, 8)
                ),
            )
            output_path = root / "submission.json"
            write_submission({"q1": response}, output_path)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload, {"q1": {"answer": ["1", "2", "3", "4", "5"]}})

    def test_duplicate_chunk_id_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate chunk_id"):
            ChunkStore(
                [
                    Chunk("same", "1", "a"),
                    Chunk("same", "2", "b"),
                ]
            )

    def test_task_one_never_allows_more_than_five_documents(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be <= 5"):
            PipelineConfig.from_mapping(
                {"reranker": {"final_top_k_documents": 6}}
            )

    def test_invalid_or_all_zero_fusion_weights_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown fusion channels"):
            PipelineConfig.from_mapping(
                {"fusion": {"channel_weights": {"dens": 1.0}}}
            )
        with self.assertRaisesRegex(ValueError, "positive weight"):
            PipelineConfig.from_mapping(
                {
                    "hyde": {"enabled": False},
                    "fusion": {
                        "channel_weights": {
                            "bm25": 0.0,
                            "dense": 0.0,
                            "hyde": 1.0,
                        }
                    },
                }
            )


if __name__ == "__main__":
    unittest.main()
