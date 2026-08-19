from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .config import BM25Config
from .io import ChunkStore
from .schema import ScoredChunk


class BM25Index:
    """bm25s adapter that preserves the shared chunk row ordering."""

    def __init__(self, model: Any, chunks: ChunkStore) -> None:
        self._model = model
        self._chunks = chunks
        indexed_rows = int(model.scores["num_docs"])
        if indexed_rows != len(chunks):
            raise ValueError(
                f"BM25 index has {indexed_rows} rows but chunk store has {len(chunks)}"
            )

    @staticmethod
    def _library() -> Any:
        try:
            import bm25s
        except ImportError as exc:  # pragma: no cover - dependency error path
            raise RuntimeError(
                "bm25s is required; install the retrieval project dependencies"
            ) from exc
        return bm25s

    @classmethod
    def build(
        cls,
        chunks: ChunkStore,
        index_dir: str | Path,
        config: BM25Config,
    ) -> "BM25Index":
        bm25s = cls._library()
        texts = [chunk.index_text for chunk in chunks.chunks]
        # Vietnamese stemming and stopword removal are deliberately disabled.
        tokens = bm25s.tokenize(
            texts,
            stopwords=None,
            stemmer=None,
            show_progress=True,
        )
        model = bm25s.BM25(method=config.method, k1=config.k1, b=config.b)
        model.index(tokens, show_progress=True)
        destination = Path(index_dir)
        destination.mkdir(parents=True, exist_ok=True)
        # Chunk metadata is persisted once by ChunkStore, not duplicated here.
        model.save(destination, show_progress=True)
        return cls(model, chunks)

    @classmethod
    def load(
        cls,
        chunks: ChunkStore,
        index_dir: str | Path,
        *,
        mmap: bool = True,
    ) -> "BM25Index":
        bm25s = cls._library()
        model = bm25s.BM25.load(
            Path(index_dir),
            load_corpus=False,
            mmap=mmap,
        )
        return cls(model, chunks)

    def search(self, query: str, top_k: int) -> list[ScoredChunk]:
        query = query.strip()
        if not query or top_k <= 0:
            return []
        bm25s = self._library()
        k = min(top_k, len(self._chunks))
        query_tokens = bm25s.tokenize(
            [query],
            stopwords=None,
            stemmer=None,
            return_ids=False,
            show_progress=False,
        )
        retrieved = self._model.retrieve(query_tokens, k=k, show_progress=False)
        # bm25s 0.3 returns a Results object; older compatible releases return a tuple.
        if hasattr(retrieved, "documents") and hasattr(retrieved, "scores"):
            positions = retrieved.documents[0]
            scores = retrieved.scores[0]
        else:  # pragma: no cover - compatibility path
            positions, scores = retrieved
            positions, scores = positions[0], scores[0]

        hits: list[ScoredChunk] = []
        for raw_position, raw_score in zip(positions, scores):
            position = int(raw_position)
            score = float(raw_score)
            if position < 0 or position >= len(self._chunks):
                continue
            # bm25s may pad top-k with non-matching zero-score rows.
            if not math.isfinite(score) or score <= 0.0:
                continue
            chunk = self._chunks[position]
            hits.append(
                ScoredChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    score=score,
                )
            )
        return hits
