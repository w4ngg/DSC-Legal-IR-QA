from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from legal_ir.config import PipelineConfig
from legal_ir.indexing import _chunk_records_hash, load_indexes
from legal_ir.io import ChunkStore
from legal_ir.schema import Chunk


def _manifest(config: PipelineConfig, chunks: ChunkStore) -> dict:
    return {
        "format_version": 1,
        "chunk_count": len(chunks),
        "chunk_records_sha256": _chunk_records_hash(chunks),
        "dense": {
            "model_name": config.dense.model_name,
            "revision": config.dense.revision,
            "max_length": config.dense.max_length,
            "dtype": config.dense.dtype,
            "normalize_embeddings": config.dense.normalize_embeddings,
            "index_type": config.dense.index_type,
            "hnsw_m": config.dense.hnsw_m,
            "hnsw_ef_construction": config.dense.hnsw_ef_construction,
        },
        "bm25": {
            "method": config.bm25.method,
            "k1": config.bm25.k1,
            "b": config.bm25.b,
        },
    }


class IndexManifestTest(unittest.TestCase):
    def test_changed_dense_checkpoint_is_rejected_before_loading_backends(self) -> None:
        built_config = PipelineConfig()
        chunks = ChunkStore([Chunk("c1", "21", "nội dung")])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chunks.save_jsonl(root / "chunks.jsonl")
            (root / "manifest.json").write_text(
                json.dumps(_manifest(built_config, chunks)), encoding="utf-8"
            )

            for dense_config in (
                replace(built_config.dense, model_name="another/embedding"),
                replace(built_config.dense, revision="another-revision"),
            ):
                with self.subTest(dense_config=dense_config):
                    query_config = replace(built_config, dense=dense_config)
                    with self.assertRaisesRegex(ValueError, "dense config differs"):
                        load_indexes(root, query_config)

    def test_changed_bm25_build_config_is_rejected_before_loading_backends(self) -> None:
        config = PipelineConfig()
        chunks = ChunkStore([Chunk("c1", "21", "nội dung")])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chunks.save_jsonl(root / "chunks.jsonl")
            manifest = _manifest(config, chunks)
            manifest["bm25"]["k1"] = 9.9
            (root / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "BM25 config differs"):
                load_indexes(root, config)

    def test_changed_index_text_is_rejected(self) -> None:
        config = PipelineConfig()
        original = ChunkStore([Chunk("c1", "21", "nội dung cũ")])
        changed = ChunkStore([Chunk("c1", "21", "nội dung mới")])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            changed.save_jsonl(root / "chunks.jsonl")
            (root / "manifest.json").write_text(
                json.dumps(_manifest(config, original)), encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "records or ordering"):
                load_indexes(root, config)

    def test_runtime_batch_and_multi_gpu_settings_do_not_invalidate_index(self) -> None:
        built_config = PipelineConfig()
        runtime_config = replace(
            built_config,
            dense=replace(
                built_config.dense,
                batch_size=17,
                multi_gpu=False,
                multi_process_chunk_size=511,
                multi_gpu_stall_timeout_seconds=73,
            ),
        )
        chunks = ChunkStore([Chunk("c1", "21", "nội dung")])
        bm25_backend = object()
        dense_backend = object()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chunks.save_jsonl(root / "chunks.jsonl")
            (root / "manifest.json").write_text(
                json.dumps(_manifest(built_config, chunks)), encoding="utf-8"
            )

            with (
                patch(
                    "legal_ir.indexing.BM25Index.load",
                    return_value=bm25_backend,
                ),
                patch(
                    "legal_ir.indexing.FaissDenseIndex.load",
                    return_value=dense_backend,
                ),
            ):
                bundle = load_indexes(root, runtime_config)

        self.assertIs(bundle.bm25, bm25_backend)
        self.assertIs(bundle.dense, dense_backend)


if __name__ == "__main__":
    unittest.main()
