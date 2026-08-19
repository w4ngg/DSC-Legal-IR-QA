from __future__ import annotations

from typing import Protocol, Sequence

from .schema import ScoredChunk


class ChunkRetriever(Protocol):
    def search(self, query: str, top_k: int) -> list[ScoredChunk]: ...


class HypotheticalDocumentGenerator(Protocol):
    def generate(self, query: str) -> str: ...


class PassageReranker(Protocol):
    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...

