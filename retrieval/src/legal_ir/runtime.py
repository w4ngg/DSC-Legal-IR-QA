from __future__ import annotations

from typing import Any


def runtime_device(device: str) -> str | None:
    return None if device == "auto" else device


def detected_torch_device(torch: Any, device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def inference_torch_dtype(torch: Any, dtype: str, device: str) -> Any:
    """Prefer BF16 on capable CUDA, FP16 on other accelerators, FP32 on CPU."""

    if dtype != "auto":
        try:
            return getattr(torch, dtype)
        except AttributeError as exc:
            raise ValueError(f"unsupported torch dtype: {dtype}") from exc
    resolved_device = detected_torch_device(torch, device)
    if resolved_device.startswith("cuda"):
        supports_bf16 = getattr(torch.cuda, "is_bf16_supported", None)
        if callable(supports_bf16) and supports_bf16():
            return torch.bfloat16
        return torch.float16
    if resolved_device == "mps":
        return torch.float16
    return torch.float32
