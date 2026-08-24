from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

from .config import DenseConfig
from .io import ChunkStore
from .runtime import detected_torch_device, inference_torch_dtype, runtime_device
from .schema import ScoredChunk


LOGGER = logging.getLogger("legal_ir")


class VietnameseEmbeddingEncoder:
    """Role-aware SentenceTransformer adapter for Vietnamese legal retrieval."""

    def __init__(self, config: DenseConfig) -> None:
        self.config = config
        self._model: Any | None = None

    @staticmethod
    def _torch() -> Any:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - dependency error path
            raise RuntimeError("torch is required for dense retrieval") from exc
        return torch

    def _load(self, *, initial_device: str | None = None) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - dependency error path
            raise RuntimeError(
                "sentence-transformers is required for dense retrieval"
            ) from exc
        torch = self._torch()
        self._model = SentenceTransformer(
            self.config.model_name,
            device=(
                initial_device
                if initial_device is not None
                else runtime_device(self.config.device)
            ),
            revision=self.config.revision,
            model_kwargs={
                "torch_dtype": inference_torch_dtype(
                    torch, self.config.dtype, self.config.device
                )
            },
        )
        self._model.max_seq_length = self.config.max_length
        return self._model

    def _single_device(self) -> str:
        return detected_torch_device(self._torch(), self.config.device)

    def _document_devices(self, *, use_multi_gpu: bool) -> str | list[str]:
        torch = self._torch()
        single_device = detected_torch_device(torch, self.config.device)
        if (
            use_multi_gpu
            and self.config.multi_gpu
            and self.config.device in {"auto", "cuda"}
            and single_device.startswith("cuda")
            and int(torch.cuda.device_count()) > 1
        ):
            return [f"cuda:{index}" for index in range(torch.cuda.device_count())]
        return single_device

    def _encode_kwargs(self, *, show_progress: bool) -> dict[str, Any]:
        return {
            "batch_size": self.config.batch_size,
            "show_progress_bar": show_progress,
            "convert_to_numpy": True,
            "normalize_embeddings": self.config.normalize_embeddings,
        }

    def encode_queries(
        self, texts: list[str], *, show_progress: bool = False
    ) -> Any:
        """Encode real questions with a model's saved query instruction."""

        model = self._load()
        return model.encode_query(
            texts,
            device=self._single_device(),
            **self._encode_kwargs(show_progress=show_progress),
        )

    def encode_documents(
        self,
        texts: list[str],
        *,
        show_progress: bool = False,
        use_multi_gpu: bool = False,
    ) -> Any:
        """Encode passage-like text; corpus builds may use all visible CUDA GPUs."""

        devices = self._document_devices(use_multi_gpu=use_multi_gpu)
        multi_gpu = isinstance(devices, list)
        # SentenceTransformers spawns one process per GPU. Loading the shared
        # parent on CPU avoids reserving CUDA:0 memory before workers start.
        if multi_gpu:
            # This also handles the less common query-then-corpus call order:
            # discard any accelerator-loaded parent before creating workers.
            self._model = None
        model = self._load(initial_device="cpu" if multi_gpu else None)
        if multi_gpu:
            LOGGER.info(
                "Encoding %d dense documents with multi-GPU devices %s",
                len(texts),
                devices,
            )
        try:
            return model.encode_document(
                texts,
                device=devices,
                chunk_size=self.config.multi_process_chunk_size,
                **self._encode_kwargs(show_progress=show_progress),
            )
        finally:
            if multi_gpu:
                # The temporary pool leaves the parent model on shared CPU memory.
                # A later one-query search should lazy-load a clean accelerator copy.
                self._model = None

    def encode(self, texts: list[str], *, show_progress: bool = False) -> Any:
        """Compatibility alias: unlabelled bulk inputs are treated as documents."""

        return self.encode_documents(texts, show_progress=show_progress)


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
        vectors = encoder.encode_documents(
            [chunk.index_text for chunk in chunks.chunks],
            show_progress=True,
            use_multi_gpu=True,
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

    def search(
        self,
        text: str,
        top_k: int,
        *,
        input_type: Literal["query", "document"] = "query",
    ) -> list[ScoredChunk]:
        text = text.strip()
        if not text or top_k <= 0:
            return []
        if input_type not in {"query", "document"}:
            raise ValueError(f"unsupported dense input_type: {input_type}")
        _, np = self._libraries()
        if input_type == "query":
            vector = self._encoder.encode_queries([text], show_progress=False)
        else:
            # A HyDE output resembles a corpus passage and must not receive
            # Harrier's legal-question instruction.
            vector = self._encoder.encode_documents([text], show_progress=False)
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
