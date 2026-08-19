from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .config import HyDEConfig
from .interfaces import HypotheticalDocumentGenerator
from .runtime import detected_torch_device, inference_torch_dtype


SYSTEM_PROMPT = """Bạn là mô-đun HyDE cho hệ thống tìm kiếm pháp luật Việt Nam.
Đây là tác vụ sinh văn bản truy vấn tổng hợp, không phải tư vấn pháp luật và không cần ngữ cảnh truy xuất.
Hãy tạo đúng một đoạn mô tả pháp lý giả định có khả năng liên quan trực tiếp đến câu hỏi.
Đoạn văn phải nêu rõ chủ thể, hành vi, điều kiện, ngoại lệ, thủ tục hoặc chế tài liên quan.
Không bịa số hiệu văn bản, số Điều, số Khoản, ngày ban hành hoặc mức tiền nếu câu hỏi không cung cấp.
Không giải thích cách làm, không dùng Markdown và không nói rằng đây là văn bản giả định."""

USER_PROMPT_TEMPLATE = """Câu hỏi về pháp luật:
{query}

Viết đoạn văn liên quan dài khoảng 80–180 từ."""


def hyde_cache_namespace(config: HyDEConfig) -> str:
    """Fingerprint every input that may affect a cached hypothesis."""

    generation_config = {
        "model_name": config.model_name,
        "revision": config.revision,
        "max_new_tokens": config.max_new_tokens,
        "do_sample": config.do_sample,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "dtype": config.dtype,
        "device": config.device,
        "trust_remote_code": config.trust_remote_code,
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt_template": USER_PROMPT_TEMPLATE,
    }
    canonical = json.dumps(
        generation_config,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    revision = config.revision or "unpinned"
    return f"{config.model_name}@{revision}:{fingerprint}"


class QwenHyDEGenerator:
    """Generate one reproducible Vietnamese hypothetical legal passage."""

    def __init__(self, config: HyDEConfig) -> None:
        self.config = config
        self._tokenizer: Any | None = None
        self._model: Any | None = None

    def _load(self) -> tuple[Any, Any, Any]:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - dependency error path
            raise RuntimeError(
                "torch and transformers are required for Qwen HyDE"
            ) from exc

        if self._model is None or self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                revision=self.config.revision,
                trust_remote_code=self.config.trust_remote_code,
            )
            resolved_device = detected_torch_device(torch, self.config.device)
            torch_dtype = inference_torch_dtype(
                torch, self.config.dtype, self.config.device
            )
            model_kwargs: dict[str, Any] = {
                "torch_dtype": torch_dtype,
                "trust_remote_code": self.config.trust_remote_code,
                # The checkpoint config disables it, but autoregressive HyDE
                # generation should reuse KV states for tractable latency.
                "use_cache": True,
            }
            if self.config.device == "auto" and resolved_device == "cuda":
                model_kwargs["device_map"] = "auto"
            self._model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                revision=self.config.revision,
                **model_kwargs,
            )
            if "device_map" not in model_kwargs:
                self._model.to(resolved_device)
            self._model.eval()
        return torch, self._tokenizer, self._model

    def generate(self, query: str) -> str:
        query = query.strip()
        if not query:
            raise ValueError("HyDE query must not be empty")
        torch, tokenizer, model = self._load()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(query=query),
            },
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.do_sample,
            "pad_token_id": (
                tokenizer.pad_token_id
                if tokenizer.pad_token_id is not None
                else tokenizer.eos_token_id
            ),
            "use_cache": True,
        }
        if self.config.do_sample:
            generation_kwargs.update(
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )
        with torch.inference_mode():
            output = model.generate(**inputs, **generation_kwargs)
        input_length = int(inputs["input_ids"].shape[-1])
        hypothesis = tokenizer.decode(
            output[0, input_length:],
            skip_special_tokens=True,
        ).strip()
        if not hypothesis:
            raise RuntimeError("HyDE model returned an empty hypothetical document")
        return hypothesis


class JsonlCachedHyDEGenerator:
    """Append-only local cache that avoids regenerating HyDE for batch runs."""

    def __init__(
        self,
        generator: HypotheticalDocumentGenerator,
        cache_path: str | Path,
        *,
        namespace: str,
    ) -> None:
        self.generator = generator
        self.cache_path = Path(cache_path)
        self.namespace = namespace
        self._cache: dict[str, str] = {}
        if self.cache_path.exists():
            with self.cache_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        value = json.loads(line)
                        if value.get("namespace") == namespace:
                            self._cache[str(value["key"])] = str(value["hypothesis"])
                    except (json.JSONDecodeError, KeyError, TypeError):
                        # A cache is disposable; tolerate a partial final append after a crash.
                        continue

    def _key(self, query: str) -> str:
        payload = f"{self.namespace}\0{query.strip()}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def generate(self, query: str) -> str:
        key = self._key(query)
        if key in self._cache:
            return self._cache[key]
        hypothesis = self.generator.generate(query)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "namespace": self.namespace,
                        "key": key,
                        "query": query,
                        "hypothesis": hypothesis,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        self._cache[key] = hypothesis
        return hypothesis
