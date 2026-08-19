from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


def _required_text(value: Any, field_name: str) -> str:
    if value is None:
        raise ValueError(f"{field_name} is required")
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must not be empty")
    return text


@dataclass(frozen=True, slots=True)
class Chunk:
    """One future chunk and its mapping back to a competition document."""

    chunk_id: str
    document_id: str
    passage: str
    retrieval_text: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "chunk_id", _required_text(self.chunk_id, "chunk_id"))
        object.__setattr__(
            self, "document_id", _required_text(self.document_id, "document_id")
        )
        object.__setattr__(self, "passage", _required_text(self.passage, "passage"))
        if self.retrieval_text is not None:
            normalized = str(self.retrieval_text).strip()
            object.__setattr__(self, "retrieval_text", normalized or None)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def index_text(self) -> str:
        """Text consumed by BM25, dense retrieval, and the cross-encoder."""

        return self.retrieval_text or self.passage

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Chunk":
        passage = value.get("passage", value.get("text"))
        return cls(
            chunk_id=value.get("chunk_id"),
            document_id=value.get("document_id", value.get("doc_id")),
            passage=passage,
            retrieval_text=value.get("retrieval_text"),
            metadata=value.get("metadata") or {},
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "passage": self.passage,
            "metadata": dict(self.metadata),
        }
        if self.retrieval_text is not None:
            result["retrieval_text"] = self.retrieval_text
        return result


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    """A chunk hit returned by one retrieval backend."""

    chunk_id: str
    document_id: str
    score: float


@dataclass(frozen=True, slots=True)
class RankedChunk:
    chunk_id: str
    document_id: str
    score: float
    rank: int
    channel: str


@dataclass(slots=True)
class DocumentCandidate:
    document_id: str
    fusion_score: float
    channel_ranks: dict[str, int] = field(default_factory=dict)
    channel_scores: dict[str, float] = field(default_factory=dict)
    evidence_chunk_ids: list[str] = field(default_factory=list)
    rerank_score: float | None = None
    evidence_rerank_scores: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SearchResult:
    document_id: str
    score: float
    fusion_score: float
    rerank_score: float | None
    evidence_chunk_ids: tuple[str, ...]
    channel_ranks: Mapping[str, int]
    channel_scores: Mapping[str, float] = field(default_factory=dict)
    evidence_rerank_scores: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "score": self.score,
            "fusion_score": self.fusion_score,
            "rerank_score": self.rerank_score,
            "evidence_chunk_ids": list(self.evidence_chunk_ids),
            "channel_ranks": dict(self.channel_ranks),
            "channel_scores": dict(self.channel_scores),
            "evidence_rerank_scores": dict(self.evidence_rerank_scores),
        }


@dataclass(frozen=True, slots=True)
class SearchResponse:
    query: str
    results: tuple[SearchResult, ...]
    hypothetical_document: str | None = None
    fused_candidates: tuple[SearchResult, ...] = ()

    @property
    def document_ids(self) -> list[str]:
        return [result.document_id for result in self.results]

    @property
    def fused_candidate_document_ids(self) -> tuple[str, ...]:
        return tuple(result.document_id for result in self.fused_candidates)

    def to_dict(self, *, include_hypothesis: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "query": self.query,
            "results": [result.to_dict() for result in self.results],
            "fused_candidates": [
                result.to_dict() for result in self.fused_candidates
            ],
        }
        if include_hypothesis:
            value["hypothetical_document"] = self.hypothetical_document
        return value
