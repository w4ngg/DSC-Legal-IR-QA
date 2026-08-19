from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .bm25 import BM25Index
from .config import PipelineConfig
from .dense import FaissDenseIndex
from .io import ChunkStore


INDEX_FORMAT_VERSION = 1


def _chunk_records_hash(chunks: ChunkStore) -> str:
    digest = hashlib.sha256()
    for chunk in chunks.chunks:
        canonical = json.dumps(
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "index_text": chunk.index_text,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest.update(canonical.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class IndexBundle:
    chunks: ChunkStore
    bm25: BM25Index
    dense: FaissDenseIndex


def build_indexes(
    chunks_path: str | Path,
    index_dir: str | Path,
    config: PipelineConfig,
    *,
    overwrite: bool = False,
) -> IndexBundle:
    destination = Path(index_dir)
    manifest_path = destination / "manifest.json"
    if destination.exists() and not destination.is_dir():
        raise NotADirectoryError(f"index path is not a directory: {destination}")
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(
            f"index directory is not empty: {destination}; pass --overwrite explicitly"
        )
    destination.mkdir(parents=True, exist_ok=True)
    # --overwrite only clears artifacts owned by this package; unrelated files survive.
    if overwrite:
        bm25_path = destination / "bm25"
        if bm25_path.exists():
            shutil.rmtree(bm25_path)
        for artifact in (
            destination / "chunks.jsonl",
            destination / "dense.faiss",
            destination / "manifest.json",
            destination / "manifest.json.tmp",
        ):
            artifact.unlink(missing_ok=True)

    chunks = ChunkStore.load_jsonl(chunks_path)
    chunks.save_jsonl(destination / "chunks.jsonl")
    bm25 = BM25Index.build(chunks, destination / "bm25", config.bm25)
    dense = FaissDenseIndex.build(chunks, destination / "dense.faiss", config.dense)

    manifest: dict[str, Any] = {
        "format_version": INDEX_FORMAT_VERSION,
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
    temporary_manifest = destination / "manifest.json.tmp"
    with temporary_manifest.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary_manifest.replace(manifest_path)
    return IndexBundle(chunks=chunks, bm25=bm25, dense=dense)


def load_indexes(index_dir: str | Path, config: PipelineConfig) -> IndexBundle:
    source = Path(index_dir)
    with (source / "manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("format_version") != INDEX_FORMAT_VERSION:
        raise ValueError(
            f"unsupported index format version: {manifest.get('format_version')}"
        )

    chunks = ChunkStore.load_jsonl(source / "chunks.jsonl")
    if int(manifest.get("chunk_count", -1)) != len(chunks):
        raise ValueError("index manifest and chunks.jsonl have different row counts")
    if manifest.get("chunk_records_sha256") != _chunk_records_hash(chunks):
        raise ValueError("chunk records or ordering do not match the index manifest")

    expected_dense = {
        "model_name": config.dense.model_name,
        "revision": config.dense.revision,
        "max_length": config.dense.max_length,
        "dtype": config.dense.dtype,
        "normalize_embeddings": config.dense.normalize_embeddings,
        "index_type": config.dense.index_type,
        "hnsw_m": config.dense.hnsw_m,
        "hnsw_ef_construction": config.dense.hnsw_ef_construction,
    }
    if manifest.get("dense") != expected_dense:
        raise ValueError(
            "dense config differs from the built index; rebuild it or use the original config"
        )
    expected_bm25 = {
        "method": config.bm25.method,
        "k1": config.bm25.k1,
        "b": config.bm25.b,
    }
    if manifest.get("bm25") != expected_bm25:
        raise ValueError(
            "BM25 config differs from the built index; rebuild it or use the original config"
        )
    bm25 = BM25Index.load(
        chunks,
        source / "bm25",
        mmap=config.bm25.mmap,
    )
    dense = FaissDenseIndex.load(chunks, source / "dense.faiss", config.dense)
    return IndexBundle(chunks=chunks, bm25=bm25, dense=dense)
