"""DSC 2026 Vietnamese LegalIR retrieval pipeline."""

from .config import PipelineConfig
from .pipeline import RetrievalPipeline
from .schema import (
    Chunk,
    DeepQueryDiagnostics,
    ScoredChunk,
    SearchResponse,
    SearchResult,
)

__all__ = [
    "Chunk",
    "DeepQueryDiagnostics",
    "PipelineConfig",
    "RetrievalPipeline",
    "ScoredChunk",
    "SearchResponse",
    "SearchResult",
]

__version__ = "0.1.0"
