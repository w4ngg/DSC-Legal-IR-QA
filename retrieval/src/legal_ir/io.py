from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .schema import Chunk, SearchResponse


class ChunkStore:
    """Stable positional mapping shared by BM25 and the dense index."""

    def __init__(self, chunks: Iterable[Chunk]) -> None:
        self.chunks = tuple(chunks)
        if not self.chunks:
            raise ValueError("chunk store must contain at least one chunk")
        self._by_id: dict[str, Chunk] = {}
        for chunk in self.chunks:
            if chunk.chunk_id in self._by_id:
                raise ValueError(f"duplicate chunk_id: {chunk.chunk_id}")
            self._by_id[chunk.chunk_id] = chunk

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, position: int) -> Chunk:
        return self.chunks[position]

    def get(self, chunk_id: str) -> Chunk:
        try:
            return self._by_id[chunk_id]
        except KeyError as exc:
            raise KeyError(f"unknown chunk_id: {chunk_id}") from exc

    @classmethod
    def load_jsonl(cls, path: str | Path) -> "ChunkStore":
        chunks: list[Chunk] = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError("record must be a JSON object")
                    chunks.append(Chunk.from_dict(value))
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    raise ValueError(f"invalid chunk at {path}:{line_number}: {exc}") from exc
        return cls(chunks)

    def save_jsonl(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            for chunk in self.chunks:
                handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")


def load_questions(path: str | Path) -> dict[str, str]:
    """Load the official JSON object and return query_id -> question."""

    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("query file must be a JSON object keyed by query ID")
    result: dict[str, str] = {}
    for raw_query_id, record in value.items():
        query_id = str(raw_query_id)
        if not isinstance(record, dict):
            raise ValueError(f"query {query_id} must be a JSON object")
        question = str(record.get("question") or "").strip()
        if not question:
            raise ValueError(f"query {query_id} has an empty question")
        result[query_id] = question
    return result


def write_submission(
    responses: Mapping[str, SearchResponse], path: str | Path
) -> None:
    payload = {
        str(query_id): {"answer": response.document_ids[:5]}
        for query_id, response in responses.items()
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_diagnostics(
    responses: Mapping[str, SearchResponse], path: str | Path
) -> None:
    payload: dict[str, Any] = {
        str(query_id): response.to_dict() for query_id, response in responses.items()
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

