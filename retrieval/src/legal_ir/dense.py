from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import DenseConfig
from .io import ChunkStore
from .runtime import inference_torch_dtype, runtime_device
from .schema import ScoredChunk


class VietnameseEmbeddingEncoder:
    """Lazy adapter for the normalized Vietnamese_Embedding_v2 checkpoint."""

    def __init__(self, config: DenseConfig) -> None:
        self.config = config
        self._model: Any | None = None

    def _load(self) -> Any:
        if self._model is None:
            try:
                import torch
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - dependency error path
                raise RuntimeError(
                    "sentence-transformers is required for dense retrieval"
                ) from exc
            self._model = SentenceTransformer(
                self.config.model_name,
                device=runtime_device(self.config.device),
                revision=self.config.revision,
                model_kwargs={
                    "torch_dtype": inference_torch_dtype(
                        torch, self.config.dtype, self.config.device
                    )
                },
            )
            self._model.max_seq_length = self.config.max_length
        return self._model

    def encode(self, texts: list[str], *, show_progress: bool = False) -> Any:
        model = self._load()
        return model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize_embeddings,
        )


class FaissDenseIndex:
    """Exact or HNSW inner-product index over normalized dense vectors."""

    def __init__(
        self,
        index: Any,
        encoder: VietnameseEmbeddingEncoder,
        chunks: ChunkStore,
        config: DenseConfig,
    ) -> None:
        self._index = index
        self._encoder = encoder
        self._chunks = chunks
        self._config = config
        if int(index.ntotal) != len(chunks):
            raise ValueError(
                f"dense index has {index.ntotal} rows but chunk store has {len(chunks)}"
            )
        if config.index_type == "hnsw" and hasattr(index, "hnsw"):
            index.hnsw.efSearch = config.hnsw_ef_search

    @staticmethod
    def _libraries() -> tuple[Any, Any]:
        try:
            import faiss
            import numpy as np
        except ImportError as exc:  # pragma: no cover - dependency error path
            raise RuntimeError(
                "faiss-cpu and numpy are required for the dense index"
            ) from exc
        return faiss, np

    @classmethod
    def build(
        cls,
        chunks: ChunkStore,
        index_path: str | Path,
        config: DenseConfig,
    ) -> "FaissDenseIndex":
        faiss, np = cls._libraries()
        encoder = VietnameseEmbeddingEncoder(config)
        vectors = encoder.encode(
            [chunk.index_text for chunk in chunks.chunks],
            show_progress=True,
        )
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
            raise ValueError(f"unexpected embedding shape: {vectors.shape}")

        dimension = int(vectors.shape[1])
        if config.index_type == "flat":
            index = faiss.IndexFlatIP(dimension)
        else:
            index = faiss.IndexHNSWFlat(
                dimension,
                config.hnsw_m,
                faiss.METRIC_INNER_PRODUCT,
            )
            index.hnsw.efConstruction = config.hnsw_ef_construction
            index.hnsw.efSearch = config.hnsw_ef_search
        index.add(vectors)

        destination = Path(index_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(destination))
        return cls(index, encoder, chunks, config)

    @classmethod
    def load(
        cls,
        chunks: ChunkStore,
        index_path: str | Path,
        config: DenseConfig,
    ) -> "FaissDenseIndex":
        faiss, _ = cls._libraries()
        # Only load index artifacts created locally; faiss does not validate files.
        index = faiss.read_index(str(Path(index_path)))
        return cls(index, VietnameseEmbeddingEncoder(config), chunks, config)

    def search(self, query: str, top_k: int) -> list[ScoredChunk]:
        query = query.strip()
        if not query or top_k <= 0:
            return []
        _, np = self._libraries()
        vector = self._encoder.encode([query], show_progress=False)
        vector = np.ascontiguousarray(vector, dtype=np.float32)
        k = min(top_k, len(self._chunks))
        scores, positions = self._index.search(vector, k)

        hits: list[ScoredChunk] = []
        for raw_position, raw_score in zip(positions[0], scores[0]):
            position = int(raw_position)
            if position < 0 or position >= len(self._chunks):
                continue
            chunk = self._chunks[position]
            hits.append(
                ScoredChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    score=float(raw_score),
                )
            )
        return hits


# Compatibility alias for code written against the first BGE-M3 skeleton.
BGEM3Encoder = VietnameseEmbeddingEncoder
