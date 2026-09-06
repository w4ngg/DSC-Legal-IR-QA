from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from legal_ir.cli import _load_dual_chunk_assets, build_parser
from legal_ir.config import PipelineConfig
from legal_ir.io import (
    ChunkStore,
    DeepDiagnosticsWriter,
    load_questions,
    write_submission,
)
from legal_ir.schema import (
    Chunk,
    DeepQueryDiagnostics,
    RankedChunk,
    RankedDocument,
    RetrievalChannelDiagnostics,
    SearchResponse,
    SearchResult,
)


class IOAndConfigTest(unittest.TestCase):
    @staticmethod
    def _deep_query() -> DeepQueryDiagnostics:
        return DeepQueryDiagnostics(
            query="Điều kiện cấp phép?",
            hypothetical_document="Quy định giả định",
            channels={
                "bm25": RetrievalChannelDiagnostics(
                    search_text="Điều kiện cấp phép?",
                    search_text_source="query",
                    requested_top_k_chunks=300,
                    chunk_hits=(
                        RankedChunk("c-1", "21", 12.5, 1, "bm25"),
                    ),
                    document_hits=(
                        RankedDocument("21", 12.5, 1, "c-1", 1),
                    ),
                )
            },
        )

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

    def test_dense_multi_gpu_config_is_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "multi_gpu must be a boolean"):
            PipelineConfig.from_mapping({"dense": {"multi_gpu": "auto"}})
        for invalid in (0, -1, 1.5, True):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    PipelineConfig.from_mapping(
                        {"dense": {"multi_process_chunk_size": invalid}}
                    )
        for invalid in (0, -1, 1.5, True):
            with self.subTest(timeout=invalid):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    PipelineConfig.from_mapping(
                        {
                            "dense": {
                                "multi_gpu_stall_timeout_seconds": invalid
                            }
                        }
                    )

    def test_deep_diagnostics_writer_streams_valid_unicode_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "nested" / "deep_diag.json"
            with DeepDiagnosticsWriter(
                destination,
                pipeline_config={"hyde": {"enabled": True}},
            ) as writer:
                writer.write(84238, self._deep_query())

            payload = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(payload["format_version"], 1)
            self.assertTrue(payload["pipeline_config"]["hyde"]["enabled"])
            query = payload["queries"]["84238"]
            self.assertEqual(query["query"], "Điều kiện cấp phép?")
            self.assertEqual(
                query["channels"]["bm25"]["chunk_hits"][0]["chunk_id"],
                "c-1",
            )
            self.assertTrue(destination.read_bytes().endswith(b"\n"))

    def test_deep_diagnostics_failure_preserves_previous_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "deep_diag.json"
            destination.write_text("previous output\n", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                with DeepDiagnosticsWriter(destination) as writer:
                    writer.write("q1", self._deep_query())
                    raise RuntimeError("interrupted")

            self.assertEqual(
                destination.read_text(encoding="utf-8"), "previous output\n"
            )
            self.assertEqual(list(destination.parent.glob(".deep_diag.json.*.tmp")), [])

    def test_search_parser_accepts_separate_deep_diagnostics_output(self) -> None:
        args = build_parser().parse_args(
            [
                "search",
                "--queries",
                "queries.json",
                "--index-dir",
                "index",
                "--output",
                "submission.json",
                "--deep-diagnostics",
                "deep_diag.json",
                "--dual-chunks-dir",
                "dual_v1",
            ]
        )

        self.assertEqual(args.deep_diagnostics, Path("deep_diag.json"))
        self.assertEqual(args.dual_chunks_dir, Path("dual_v1"))
        self.assertIsNone(args.diagnostics)

    def test_dual_runtime_assets_match_indexed_short_chunk_manifest(self) -> None:
        namespace = "dual_char_v1_test"
        short_id = f"DOC:{namespace}:short:000000"
        long_id = f"DOC:{namespace}:long:000000"
        short_chunks = ChunkStore(
            [
                Chunk(
                    short_id,
                    "DOC",
                    "short",
                    metadata={
                        "granularity": "short",
                        "primary_long_chunk_id": long_id,
                        "long_chunk_ids": [long_id],
                    },
                )
            ]
        )
        config = PipelineConfig.from_mapping(
            {
                "hyde": {"enabled": False},
                "long_context": {"enabled": True},
            }
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            long_chunks_path = root / "long_chunks.jsonl"
            mapping_path = root / "short_to_long.jsonl"
            ChunkStore(
                [
                    Chunk(
                        long_id,
                        "DOC",
                        "long",
                        metadata={"granularity": "long"},
                    )
                ]
            ).save_jsonl(long_chunks_path)
            mapping_path.write_text(
                json.dumps(
                    {
                        "short_chunk_id": short_id,
                        "document_id": "DOC",
                        "primary_long_chunk_id": long_id,
                        "long_chunk_ids": [long_id],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "format_version": 1,
                        "artifact_type": "legal_ir_dual_granularity_chunks",
                        "outputs": {
                            "long_chunks": {
                                "records": 1,
                                "sha256": hashlib.sha256(
                                    long_chunks_path.read_bytes()
                                ).hexdigest(),
                            },
                            "short_to_long": {
                                "records": 1,
                                "sha256": hashlib.sha256(
                                    mapping_path.read_bytes()
                                ).hexdigest(),
                            },
                        },
                        "statistics": {
                            "short_chunks_written": 1,
                            "long_chunks_written": 1,
                            "mappings_written": 1,
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            assets = _load_dual_chunk_assets(
                SimpleNamespace(dual_chunks_dir=root),
                config,
                short_chunks=short_chunks,
            )
            mapping_path.write_text(
                mapping_path.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                _load_dual_chunk_assets(
                    SimpleNamespace(dual_chunks_dir=root),
                    config,
                    short_chunks=short_chunks,
                )

        assert assets is not None
        self.assertEqual(len(assets.long_chunks), 1)
        self.assertEqual(
            assets.mappings.get(short_id).long_chunk_ids,
            (long_id,),
        )

    def test_dual_runtime_assets_are_required_only_when_mode_is_enabled(self) -> None:
        chunks = ChunkStore([Chunk("legacy", "DOC", "short")])
        legacy_assets = _load_dual_chunk_assets(
            SimpleNamespace(dual_chunks_dir=None),
            PipelineConfig(),
            short_chunks=chunks,
        )
        self.assertIsNone(legacy_assets)

        config = PipelineConfig.from_mapping(
            {
                "hyde": {"enabled": False},
                "long_context": {"enabled": True},
            }
        )
        with self.assertRaisesRegex(ValueError, "--dual-chunks-dir is required"):
            _load_dual_chunk_assets(
                SimpleNamespace(dual_chunks_dir=None),
                config,
                short_chunks=chunks,
            )


if __name__ == "__main__":
    unittest.main()
