"""DSC 2026 Vietnamese LegalIR retrieval pipeline."""

from .config import PipelineConfig
from .pipeline import RetrievalPipeline
from .schema import Chunk, ScoredChunk, SearchResponse, SearchResult

__all__ = [
    "Chunk",
    "PipelineConfig",
    "RetrievalPipeline",
    "ScoredChunk",
    "SearchResponse",
    "SearchResult",
]

__version__ = "0.1.0"

