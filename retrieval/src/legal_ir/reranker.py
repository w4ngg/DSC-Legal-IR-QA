from __future__ import annotations

import weakref
from typing import Any, Sequence

from .config import RerankerConfig
from .reranker_multi_gpu import RerankerMultiGPUProcessPool
from .runtime import detected_torch_device, inference_torch_dtype, runtime_device


class VietnameseCrossEncoderReranker:
    """Vietnamese_Reranker adapter returning raw one-label logits."""

    def __init__(self, config: RerankerConfig) -> None:
        self.config = config
        self._model: Any | None = None
        self._multi_gpu_pool: RerankerMultiGPUProcessPool | None = None
        self._pool_finalizer: weakref.finalize | None = None
        self._multi_gpu_checked = False
        self._multi_gpu_devices: tuple[str, ...] = ()
        self._closed = False

    @staticmethod
    def _torch() -> Any:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - dependency error path
            raise RuntimeError("torch is required for reranking") from exc
        return torch

    def _detect_multi_gpu_devices(self) -> tuple[str, ...]:
        if self._multi_gpu_checked:
            return self._multi_gpu_devices
        self._multi_gpu_checked = True
        if not bool(getattr(self.config, "multi_gpu", False)):
            return ()
        if self.config.device not in {"auto", "cuda"}:
            return ()

        torch = self._torch()
        resolved_device = detected_torch_device(torch, self.config.device)
        if not resolved_device.startswith("cuda") or not torch.cuda.is_available():
            return ()
        device_count = int(torch.cuda.device_count())
        if device_count < 2:
            return ()
        self._multi_gpu_devices = tuple(
            f"cuda:{index}" for index in range(device_count)
        )
        return self._multi_gpu_devices

    def _multi_gpu_scorer(self) -> RerankerMultiGPUProcessPool | None:
        devices = self._detect_multi_gpu_devices()
        if not devices:
            return None
        if self._multi_gpu_pool is None:
            # CUDA models live only in isolated workers in multi-GPU mode.
            self._model = None
            pool = RerankerMultiGPUProcessPool(self.config, devices)
            self._multi_gpu_pool = pool
            self._pool_finalizer = weakref.finalize(self, pool.close)
        return self._multi_gpu_pool

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:  # pragma: no cover - dependency error path
                raise RuntimeError(
                    "sentence-transformers and torch are required for reranking"
                ) from exc
            torch = self._torch()
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
        if self._closed:
            raise RuntimeError("reranker is closed")
        multi_gpu = self._multi_gpu_scorer()
        if multi_gpu is not None:
            return multi_gpu.score(query, passages)
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

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._pool_finalizer is not None and self._pool_finalizer.alive:
            self._pool_finalizer()
        elif self._multi_gpu_pool is not None:
            self._multi_gpu_pool.close()
        self._multi_gpu_pool = None

    def __enter__(self) -> "VietnameseCrossEncoderReranker":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        del exc_type, exc, traceback
        self.close()
        return False


# Compatibility alias for code written against the first BGE reranker skeleton.
BGECrossEncoderReranker = VietnameseCrossEncoderReranker
