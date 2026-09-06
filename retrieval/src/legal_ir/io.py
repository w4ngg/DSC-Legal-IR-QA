from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, TextIO

from .schema import Chunk, DeepQueryDiagnostics, SearchResponse


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
    def load_jsonl(
        cls,
        path: str | Path,
        *,
        compact_for_search: bool = False,
        retain_dual_mapping: bool = False,
    ) -> "ChunkStore":
        if retain_dual_mapping and not compact_for_search:
            raise ValueError(
                "retain_dual_mapping requires compact_for_search"
            )
        chunks: list[Chunk] = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError("record must be a JSON object")
                    if not compact_for_search:
                        chunks.append(Chunk.from_dict(value))
                        continue

                    passage = value.get("passage", value.get("text"))
                    retrieval_text = value.get("retrieval_text")
                    normalized_retrieval_text = (
                        str(retrieval_text).strip()
                        if retrieval_text is not None
                        else ""
                    )
                    index_text = normalized_retrieval_text or passage
                    raw_metadata = value.get("metadata") or {}
                    if not isinstance(raw_metadata, dict):
                        raise ValueError("metadata must be an object")
                    metadata: dict[str, Any] = {}
                    granularity = raw_metadata.get("granularity")
                    if granularity is not None:
                        metadata["granularity"] = str(granularity)
                    if retain_dual_mapping:
                        primary_long_id = raw_metadata.get(
                            "primary_long_chunk_id"
                        )
                        raw_long_ids = raw_metadata.get("long_chunk_ids")
                        if (
                            not isinstance(primary_long_id, str)
                            or not primary_long_id.strip()
                        ):
                            raise ValueError(
                                "primary_long_chunk_id must be a non-empty string"
                            )
                        if (
                            not isinstance(raw_long_ids, (list, tuple))
                            or not raw_long_ids
                            or any(
                                not isinstance(long_chunk_id, str)
                                or not long_chunk_id.strip()
                                for long_chunk_id in raw_long_ids
                            )
                        ):
                            raise ValueError(
                                "long_chunk_ids must be a non-empty string array"
                            )
                        interned_primary_long_id = sys.intern(primary_long_id)
                        interned_long_ids = tuple(
                            sys.intern(str(long_chunk_id))
                            for long_chunk_id in raw_long_ids
                        )
                        if (
                            len(set(interned_long_ids)) != len(interned_long_ids)
                            or interned_primary_long_id not in interned_long_ids
                        ):
                            raise ValueError(
                                "dual mapping IDs must be unique and contain "
                                "primary_long_chunk_id"
                            )
                        metadata[
                            "primary_long_chunk_id"
                        ] = interned_primary_long_id
                        metadata["long_chunk_ids"] = interned_long_ids

                    raw_document_id = value.get(
                        "document_id", value.get("doc_id")
                    )
                    if raw_document_id is None:
                        raise ValueError("document_id is required")
                    raw_chunk_id = value.get("chunk_id")
                    if raw_chunk_id is None:
                        raise ValueError("chunk_id is required")
                    document_id = sys.intern(str(raw_document_id).strip())
                    chunk_id = str(raw_chunk_id).strip()
                    if granularity == "long":
                        # Long IDs are shared by many short mapping tuples. Once
                        # the short store has interned them, reuse those objects.
                        chunk_id = sys.intern(chunk_id)
                    chunks.append(
                        Chunk(
                            chunk_id=chunk_id,
                            document_id=document_id,
                            passage=index_text,
                            metadata=metadata,
                        )
                    )
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


class DeepDiagnosticsWriter:
    """Stream full retrieval traces and atomically publish one JSON object."""

    FORMAT_VERSION = 1

    def __init__(
        self,
        path: str | Path,
        *,
        pipeline_config: Mapping[str, Any] | None = None,
    ) -> None:
        self.destination = Path(path)
        self.pipeline_config = dict(pipeline_config or {})
        self._handle: TextIO | None = None
        self._temporary_path: Path | None = None
        self._query_ids: set[str] = set()
        self._first_query = True

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )

    def __enter__(self) -> "DeepDiagnosticsWriter":
        if self._handle is not None:
            raise RuntimeError("deep diagnostics writer is already open")
        header = {
            "format_version": self.FORMAT_VERSION,
            "pipeline_config": self.pipeline_config,
        }
        serialized_header = self._json(header)
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=self.destination.parent,
            prefix=f".{self.destination.name}.",
            suffix=".tmp",
            delete=False,
        )
        self._handle = handle
        self._temporary_path = Path(handle.name)
        self._query_ids.clear()
        self._first_query = True
        try:
            # Remove the closing brace so query records can be streamed one at a time.
            handle.write(serialized_header[:-1])
            handle.write(',"queries":{')
        except BaseException:
            self._abort()
            raise
        return self

    def write(self, query_id: str, diagnostics: DeepQueryDiagnostics) -> None:
        if self._handle is None:
            raise RuntimeError("deep diagnostics writer is not open")
        normalized_query_id = str(query_id)
        if normalized_query_id in self._query_ids:
            raise ValueError(f"duplicate deep diagnostics query_id: {normalized_query_id}")

        # Serialize before touching the stream so invalid values cannot leave a
        # half-written query record. Canonical ranking has already removed NaN/Inf.
        record = (
            self._json(normalized_query_id)
            + ":"
            + self._json(diagnostics.to_dict())
        )
        if not self._first_query:
            self._handle.write(",")
        self._handle.write(record)
        self._query_ids.add(normalized_query_id)
        self._first_query = False

    def _abort(self) -> None:
        if self._handle is not None and not self._handle.closed:
            self._handle.close()
        if self._temporary_path is not None:
            self._temporary_path.unlink(missing_ok=True)
        self._handle = None
        self._temporary_path = None

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        if self._handle is None or self._temporary_path is None:
            return False
        if exc_type is not None:
            self._abort()
            return False

        try:
            self._handle.write("}}\n")
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()
            os.replace(self._temporary_path, self.destination)
        except BaseException:
            self._abort()
            raise
        self._handle = None
        self._temporary_path = None
        return False
