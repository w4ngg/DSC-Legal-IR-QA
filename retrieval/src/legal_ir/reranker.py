from __future__ import annotations

from typing import Any, Sequence

from .config import RerankerConfig
from .runtime import inference_torch_dtype, runtime_device


class VietnameseCrossEncoderReranker:
    """Vietnamese_Reranker adapter returning raw one-label logits."""

    def __init__(self, config: RerankerConfig) -> None:
        self.config = config
        self._model: Any | None = None

    def _load(self) -> Any:
        if self._model is None:
            try:
                import torch
                from sentence_transformers import CrossEncoder
            except ImportError as exc:  # pragma: no cover - dependency error path
                raise RuntimeError(
                    "sentence-transformers and torch are required for reranking"
                ) from exc
            self._model = CrossEncoder(
                self.config.model_name,
                device=runtime_device(self.config.device),
                revision=self.config.revision,
                model_kwargs={
                    "torch_dtype": inference_torch_dtype(
                        torch, self.config.dtype, self.config.device
                    )
                },
                max_length=self.config.max_length,
                activation_fn=torch.nn.Identity(),
            )
        return self._model

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - dependency error path
            raise RuntimeError("numpy is required for reranking") from exc
        model = self._load()
        pairs = [(query, passage) for passage in passages]
        scores = model.predict(
            pairs,
            batch_size=self.config.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [float(value) for value in np.asarray(scores).reshape(-1)]


# Compatibility alias for code written against the first BGE reranker skeleton.
BGECrossEncoderReranker = VietnameseCrossEncoderReranker
