from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class BM25Config:
    method: str = "lucene"
    k1: float = 1.2
    b: float = 0.75
    top_k_chunks: int = 300
    mmap: bool = True


@dataclass(frozen=True, slots=True)
class DenseConfig:
    model_name: str = "AITeamVN/Vietnamese_Embedding_v2"
    revision: str | None = "18b44161e041bf1d3a333ab5144b5b7b93f914d2"
    top_k_chunks: int = 200
    batch_size: int = 8
    max_length: int = 2048
    device: str = "auto"
    dtype: str = "auto"
    normalize_embeddings: bool = True
    index_type: str = "flat"
    hnsw_m: int = 32
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 128


@dataclass(frozen=True, slots=True)
class HyDEConfig:
    enabled: bool = True
    model_name: str = "AITeamVN/Vi-Qwen2-3B-RAG"
    revision: str | None = "eaf427c24d86066a2b35828c499b7db3af321227"
    top_k_chunks: int = 200
    max_new_tokens: int = 192
    do_sample: bool = False
    temperature: float = 0.7
    top_p: float = 0.8
    device: str = "auto"
    dtype: str = "auto"
    trust_remote_code: bool = False


@dataclass(frozen=True, slots=True)
class FusionConfig:
    rrf_k: int = 60
    candidate_documents: int = 50
    evidence_chunks_per_document: int = 2
    channel_weights: Mapping[str, float] = field(
        default_factory=lambda: {"bm25": 1.0, "dense": 1.0, "hyde": 0.5}
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "channel_weights", dict(self.channel_weights))


@dataclass(frozen=True, slots=True)
class RerankerConfig:
    enabled: bool = True
    model_name: str = "AITeamVN/Vietnamese_Reranker"
    revision: str | None = "f536976248403314225d7fdfdbc87f0e9516a54e"
    batch_size: int = 4
    max_length: int = 2304
    device: str = "auto"
    dtype: str = "auto"
    final_top_k_documents: int = 5


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    bm25: BM25Config = field(default_factory=BM25Config)
    dense: DenseConfig = field(default_factory=DenseConfig)
    hyde: HyDEConfig = field(default_factory=HyDEConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)

    def __post_init__(self) -> None:
        positive_values = {
            "bm25.top_k_chunks": self.bm25.top_k_chunks,
            "dense.top_k_chunks": self.dense.top_k_chunks,
            "hyde.top_k_chunks": self.hyde.top_k_chunks,
            "fusion.rrf_k": self.fusion.rrf_k,
            "fusion.candidate_documents": self.fusion.candidate_documents,
            "fusion.evidence_chunks_per_document": self.fusion.evidence_chunks_per_document,
            "reranker.final_top_k_documents": self.reranker.final_top_k_documents,
            "bm25.k1": self.bm25.k1,
            "dense.batch_size": self.dense.batch_size,
            "dense.max_length": self.dense.max_length,
            "dense.hnsw_m": self.dense.hnsw_m,
            "dense.hnsw_ef_construction": self.dense.hnsw_ef_construction,
            "dense.hnsw_ef_search": self.dense.hnsw_ef_search,
            "hyde.max_new_tokens": self.hyde.max_new_tokens,
            "reranker.batch_size": self.reranker.batch_size,
            "reranker.max_length": self.reranker.max_length,
        }
        for name, value in positive_values.items():
            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be numeric, got {value!r}") from exc
            if not math.isfinite(numeric_value) or numeric_value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.reranker.final_top_k_documents > 5:
            raise ValueError(
                "reranker.final_top_k_documents must be <= 5 for the Task 1 format"
            )
        if self.fusion.candidate_documents < self.reranker.final_top_k_documents:
            raise ValueError(
                "fusion.candidate_documents must be >= final_top_k_documents"
            )
        if self.dense.index_type not in {"flat", "hnsw"}:
            raise ValueError("dense.index_type must be 'flat' or 'hnsw'")
        if self.bm25.method not in {"robertson", "atire", "bm25l", "bm25+", "lucene"}:
            raise ValueError(f"unsupported BM25 method: {self.bm25.method}")
        if not 0.0 <= self.bm25.b <= 1.0:
            raise ValueError("bm25.b must be between 0 and 1")
        allowed_dtypes = {"auto", "float16", "bfloat16", "float32"}
        if self.dense.dtype not in allowed_dtypes:
            raise ValueError(f"unsupported dense dtype: {self.dense.dtype}")
        if self.hyde.dtype not in allowed_dtypes:
            raise ValueError(f"unsupported HyDE dtype: {self.hyde.dtype}")
        if self.reranker.dtype not in allowed_dtypes:
            raise ValueError(f"unsupported reranker dtype: {self.reranker.dtype}")
        if self.hyde.do_sample and (
            not math.isfinite(self.hyde.temperature) or self.hyde.temperature <= 0
        ):
            raise ValueError("hyde.temperature must be positive when sampling")
        if not 0.0 < self.hyde.top_p <= 1.0:
            raise ValueError("hyde.top_p must be in (0, 1]")
        allowed_channels = {"bm25", "dense", "hyde"}
        unknown_channels = set(self.fusion.channel_weights) - allowed_channels
        if unknown_channels:
            raise ValueError(
                f"unknown fusion channels: {sorted(unknown_channels)}"
            )
        for channel, weight in self.fusion.channel_weights.items():
            try:
                numeric_weight = float(weight)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"fusion weight for {channel!r} must be numeric"
                ) from exc
            if not math.isfinite(numeric_weight) or numeric_weight < 0:
                raise ValueError(f"fusion weight for {channel!r} must be non-negative")
        active_channels = {"bm25", "dense"}
        if self.hyde.enabled:
            active_channels.add("hyde")
        active_weight = sum(
            float(self.fusion.channel_weights.get(channel, 0.0))
            for channel in active_channels
        )
        if active_weight <= 0:
            raise ValueError("at least one enabled retrieval channel needs positive weight")
        for name, revision in {
            "dense.revision": self.dense.revision,
            "hyde.revision": self.hyde.revision,
            "reranker.revision": self.reranker.revision,
        }.items():
            if revision is not None and not str(revision).strip():
                raise ValueError(f"{name} must be null or a non-empty revision")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PipelineConfig":
        allowed = {"bm25", "dense", "hyde", "fusion", "reranker"}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown config sections: {sorted(unknown)}")
        return cls(
            bm25=BM25Config(**dict(value.get("bm25") or {})),
            dense=DenseConfig(**dict(value.get("dense") or {})),
            hyde=HyDEConfig(**dict(value.get("hyde") or {})),
            fusion=FusionConfig(**dict(value.get("fusion") or {})),
            reranker=RerankerConfig(**dict(value.get("reranker") or {})),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - dependency error path
            raise RuntimeError("PyYAML is required to read the pipeline config") from exc
        with Path(path).open("r", encoding="utf-8") as handle:
            value = yaml.safe_load(handle) or {}
        if not isinstance(value, dict):
            raise ValueError("the YAML config root must be a mapping")
        return cls.from_mapping(value)
