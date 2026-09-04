from __future__ import annotations

import hashlib
import json
import re
import tempfile
import unittest
from pathlib import Path

from legal_ir.chunk_dual_granularity import (
    DualGranularityChunkingConfig,
    build_document_dual_chunks,
    build_dual_granularity_chunks,
    normalize_legal_text_with_lines,
)
from legal_ir.io import ChunkStore


class DualGranularityChunkingTest(unittest.TestCase):
    def test_normalization_preserves_non_empty_line_boundaries(self) -> None:
        self.assertEqual(
            normalize_legal_text_with_lines(
                "  Điều 1.\r\n\r\n  Nội\t dung  \u200b thứ nhất\n\nKhoản 2.  "
            ),
            "Điều 1.\nNội dung thứ nhất\nKhoản 2.",
        )

    def test_both_granularities_and_mapping_share_one_coordinate_system(self) -> None:
        document = {
            "id": 21,
            "name": "Luật kiểm thử",
            "link": "https://example.test/21",
            "passage": (
                "Lời mở đầu có nhiều thông tin tổng quát.\n"
                "Điều 1. Quy định chung cho tất cả tổ chức và cá nhân.\n"
                "1. Khoản thứ nhất có nội dung tương đối dài để phải chia nhỏ; "
                "câu tiếp theo vẫn thuộc cùng khoản này.\n"
                "a) Điểm a có quy định riêng.\n"
                "b) Điểm b có quy định riêng."
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            config = DualGranularityChunkingConfig(
                input_dir=Path(directory),
                output_dir=Path(directory) / "out",
                short_max_characters=70,
                long_max_characters=170,
                long_overlap_short_chunks=1,
            )
            normalized = normalize_legal_text_with_lines(document["passage"])
            short, long, mapping = build_document_dual_chunks(
                document,
                Path(directory) / "context_21.json",
                config,
                normalized_text=normalized,
            )

        self.assertGreater(len(short), 4)
        self.assertGreater(len(long), 1)
        self.assertEqual(len(mapping), len(short))
        self.assertTrue(all(len(item["passage"]) <= 70 for item in short))
        self.assertTrue(all(len(item["passage"]) <= 170 for item in long))
        collapse = lambda value: re.sub(r"\s+", " ", value).strip()
        self.assertEqual(
            collapse(" ".join(item["passage"] for item in short)),
            collapse(normalized),
        )

        short_ids = {item["chunk_id"] for item in short}
        long_by_id = {item["chunk_id"]: item for item in long}
        previous_end = -1
        for short_record, mapping_record in zip(short, mapping, strict=True):
            metadata = short_record["metadata"]
            self.assertGreaterEqual(metadata["normalized_character_start"], previous_end)
            previous_end = metadata["normalized_character_end"]
            self.assertEqual(mapping_record["short_chunk_id"], short_record["chunk_id"])
            self.assertIn(
                mapping_record["primary_long_chunk_id"],
                mapping_record["long_chunk_ids"],
            )
            for long_id in mapping_record["long_chunk_ids"]:
                long_metadata = long_by_id[long_id]["metadata"]
                self.assertLessEqual(
                    long_metadata["normalized_character_start"],
                    metadata["normalized_character_start"],
                )
                self.assertGreaterEqual(
                    long_metadata["normalized_character_end"],
                    metadata["normalized_character_end"],
                )

        referenced_short_ids = {
            short_id
            for long_record in long
            for short_id in long_record["metadata"]["short_chunk_ids"]
        }
        self.assertEqual(referenced_short_ids, short_ids)

    def test_one_pass_builder_writes_index_compatible_short_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "selected-contexts"
            output_dir = root / "dual"
            input_dir.mkdir()
            sources = {
                "context_2.json": {
                    "id": 2,
                    "name": "Văn bản 2",
                    "passage": "Điều 1. Nội dung.\n1. Khoản một.\n2. Khoản hai.",
                },
                "context_10.json": {
                    "id": 10,
                    "passage": "   \r\n\t",
                },
            }
            for name, payload in sources.items():
                (input_dir / name).write_text(
                    json.dumps(payload, ensure_ascii=False),
                    encoding="utf-8",
                )

            config = DualGranularityChunkingConfig(
                input_dir=input_dir,
                output_dir=output_dir,
                short_max_characters=30,
                long_max_characters=80,
            )
            manifest = build_dual_granularity_chunks(config)

            short_store = ChunkStore.load_jsonl(config.short_chunks_path)
            long_store = ChunkStore.load_jsonl(config.long_chunks_path)
            mappings = [
                json.loads(line)
                for line in config.mapping_path.read_text(encoding="utf-8").splitlines()
            ]

            self.assertGreater(len(short_store), 0)
            self.assertGreater(len(long_store), 0)
            self.assertEqual(len(mappings), len(short_store))
            self.assertEqual(manifest["source_passes"], 1)
            self.assertEqual(manifest["statistics"]["documents_skipped_empty"], 1)
            self.assertEqual(manifest["skipped_empty_document_ids"], ["10"])
            self.assertEqual(
                manifest["outputs"]["long_chunks"]["purpose"],
                "reranker_context_only_not_retrieval_index",
            )
            self.assertEqual(
                hashlib.sha256(config.short_chunks_path.read_bytes()).hexdigest(),
                manifest["outputs"]["short_chunks"]["sha256"],
            )
            with self.assertRaisesRegex(FileExistsError, "--overwrite"):
                build_dual_granularity_chunks(config)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
