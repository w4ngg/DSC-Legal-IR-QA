from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from .config import HyDEConfig
from .interfaces import HypotheticalDocumentGenerator
from .runtime import detected_torch_device, inference_torch_dtype


SYSTEM_PROMPT = """Bạn đang tạo một đoạn văn giả định chỉ phục vụ truy xuất thông tin
trong kho văn bản pháp luật tiếng Việt.

Từ câu hỏi của người dùng, hãy viết một đoạn văn ngắn mô phỏng
cách nội dung liên quan có thể được diễn đạt trong một văn bản pháp luật.

Yêu cầu:
- Giữ nguyên các thực thể, điều kiện và phạm vi xuất hiện trong câu hỏi.
- Có thể mở rộng bằng các thuật ngữ pháp lý đồng nghĩa hoặc liên quan trực tiếp.
- KHÔNG tự tạo số hiệu văn bản, Điều, Khoản, Điểm nếu câu hỏi không cung cấp.
- KHÔNG tự đoán con số, thời hạn, mức tiền, tỷ lệ hoặc ngưỡng.
- KHÔNG tự tạo cơ quan, đối tượng hoặc ngoại lệ chưa xuất hiện trong câu hỏi.
- Không trả lời câu hỏi.
- Mục tiêu là tạo văn bản giàu từ khóa và ngữ nghĩa để retrieval, không phải tạo câu trả lời chính xác.
Chỉ trả về đoạn văn giả định, không giải thích."""

USER_PROMPT_TEMPLATE = """Câu hỏi về pháp luật:
{query}

Viết đoạn văn liên quan dài khoảng 80–180 từ."""

HYDE_NORMALIZATION_VERSION = "hyde_nfc_ws_v1"

_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_ESCAPED_WHITESPACE = re.compile(r"\\(?:r\\n|[nrtfv])", re.IGNORECASE)
_HTML_SPACE_ENTITY = re.compile(r"&(?:nbsp|#0*160|#x0*a0);", re.IGNORECASE)
_MARKDOWN_CODE_FENCE = re.compile(
    r"```(?:text|plaintext|markdown|vietnamese|vi)?",
    re.IGNORECASE,
)
_WHITESPACE = re.compile(r"\s+")
_INVISIBLE_TRANSLATION = str.maketrans(
    {
        "\u00ad": None,  # soft hyphen
        "\u200b": " ",  # zero-width space must not join adjacent words
        "\u200c": None,  # zero-width non-joiner
        "\u200d": None,  # zero-width joiner
        "\u200e": None,  # left-to-right mark
        "\u200f": None,  # right-to-left mark
        "\u2060": None,  # word joiner
        "\ufeff": None,  # byte-order mark
        "\u2028": " ",  # Unicode line separator
        "\u2029": " ",  # Unicode paragraph separator
    }
)


def normalize_hyde_text(value: str) -> str:
    """Remove generation artifacts without changing Vietnamese legal semantics.

    Both real line breaks and literal escaped whitespace such as ``\\n`` are
    collapsed. Accents, case, punctuation, numbers, and legal citations remain.
    """

    if not isinstance(value, str):
        raise TypeError("HyDE output must be a string")
    text = unicodedata.normalize("NFC", value)
    text = _HTML_SPACE_ENTITY.sub(" ", text)
    text = _ESCAPED_WHITESPACE.sub(" ", text)
    text = _MARKDOWN_CODE_FENCE.sub(" ", text)
    text = text.replace("```", " ")
    text = text.translate(_INVISIBLE_TRANSLATION)
    text = text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    text = _CONTROL_CHARACTERS.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


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
        "output_normalization_version": HYDE_NORMALIZATION_VERSION,
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
        raw_hypothesis = tokenizer.decode(
            output[0, input_length:],
            skip_special_tokens=True,
        )
        hypothesis = normalize_hyde_text(raw_hypothesis)
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
                    except json.JSONDecodeError:
                        # A cache is disposable; tolerate a partial final append after a crash.
                        continue
                    if not isinstance(value, dict):
                        continue
                    if value.get("namespace") != namespace:
                        continue
                    key = value.get("key")
                    hypothesis = value.get("hypothesis")
                    if not isinstance(key, str) or not isinstance(hypothesis, str):
                        continue
                    hypothesis = normalize_hyde_text(hypothesis)
                    if hypothesis:
                        self._cache[key] = hypothesis

    def _key(self, query: str) -> str:
        payload = f"{self.namespace}\0{query.strip()}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def generate(self, query: str) -> str:
        key = self._key(query)
        if key in self._cache:
            return self._cache[key]
        hypothesis = normalize_hyde_text(self.generator.generate(query))
        if not hypothesis:
            raise RuntimeError("HyDE generator returned an empty hypothetical document")
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
