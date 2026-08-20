from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from legal_ir.config import DenseConfig, HyDEConfig, PipelineConfig, RerankerConfig
from legal_ir.dense import VietnameseEmbeddingEncoder
from legal_ir.hyde import QwenHyDEGenerator, SYSTEM_PROMPT, normalize_hyde_text
from legal_ir.reranker import VietnameseCrossEncoderReranker


EMBEDDING_REVISION = "18b44161e041bf1d3a333ab5144b5b7b93f914d2"
HYDE_REVISION = "eaf427c24d86066a2b35828c499b7db3af321227"
RERANKER_REVISION = "f536976248403314225d7fdfdbc87f0e9516a54e"


class AITeamVNConfigTest(unittest.TestCase):
    def test_python_defaults_pin_the_requested_checkpoints(self) -> None:
        config = PipelineConfig()

        self.assertEqual(
            (config.dense.model_name, config.dense.revision),
            ("AITeamVN/Vietnamese_Embedding_v2", EMBEDDING_REVISION),
        )
        self.assertEqual(
            (config.hyde.model_name, config.hyde.revision),
            ("AITeamVN/Vi-Qwen2-3B-RAG", HYDE_REVISION),
        )
        self.assertEqual(
            (config.reranker.model_name, config.reranker.revision),
            ("AITeamVN/Vietnamese_Reranker", RERANKER_REVISION),
        )

    def test_checked_in_yaml_matches_python_model_defaults(self) -> None:
        python_config = PipelineConfig()
        yaml_path = Path(__file__).parents[1] / "configs" / "default.yaml"
        yaml_text = yaml_path.read_text(encoding="utf-8")

        for component in ("dense", "hyde", "reranker"):
            with self.subTest(component=component):
                expected = getattr(python_config, component)
                section = yaml_text.split(f"{component}:\n", maxsplit=1)[1]
                section = section.split("\n\n", maxsplit=1)[0]
                self.assertIn(f"  model_name: {expected.model_name}\n", section)
                self.assertIn(f"  revision: {expected.revision}\n", section)


class AITeamVNEmbeddingTest(unittest.TestCase):
    def test_encode_requests_unit_normalized_embeddings(self) -> None:
        model = Mock()
        model.encode.return_value = [[1.0, 0.0]]
        config = DenseConfig(
            model_name="AITeamVN/Vietnamese_Embedding_v2",
            revision=EMBEDDING_REVISION,
            normalize_embeddings=True,
        )
        encoder = VietnameseEmbeddingEncoder(config)
        encoder._model = model

        encoded = encoder.encode(["Điều kiện cấp giấy phép là gì?"])

        self.assertEqual(encoded, [[1.0, 0.0]])
        model.encode.assert_called_once_with(
            ["Điều kiện cấp giấy phép là gì?"],
            batch_size=config.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )


class AITeamVNRerankerTest(unittest.TestCase):
    def test_loader_disables_cross_encoder_score_activation(self) -> None:
        calls: list[tuple[str, dict[str, object]]] = []

        class _Identity:
            pass

        class _CrossEncoder:
            def __init__(self, model_name: str, **kwargs: object) -> None:
                calls.append((model_name, kwargs))

        fake_torch = ModuleType("torch")
        fake_torch.nn = SimpleNamespace(Identity=_Identity)  # type: ignore[attr-defined]
        fake_sentence_transformers = ModuleType("sentence_transformers")
        fake_sentence_transformers.CrossEncoder = _CrossEncoder  # type: ignore[attr-defined]
        config = RerankerConfig()

        with (
            patch.dict(
                sys.modules,
                {
                    "torch": fake_torch,
                    "sentence_transformers": fake_sentence_transformers,
                },
            ),
            patch("legal_ir.reranker.runtime_device", return_value="cpu"),
            patch("legal_ir.reranker.inference_torch_dtype", return_value="float32"),
        ):
            VietnameseCrossEncoderReranker(config)._load()

        self.assertEqual(len(calls), 1)
        model_name, kwargs = calls[0]
        self.assertEqual(model_name, "AITeamVN/Vietnamese_Reranker")
        self.assertEqual(kwargs["revision"], RERANKER_REVISION)
        self.assertIsInstance(kwargs["activation_fn"], _Identity)

    def test_score_preserves_raw_scalar_logits(self) -> None:
        model = Mock()
        # A negative logit must stay negative: applying sigmoid/softmax here would
        # silently change the model's native ranking score semantics.
        model.predict.return_value = [[-2.25], [1.5]]
        config = RerankerConfig(
            model_name="AITeamVN/Vietnamese_Reranker",
            revision=RERANKER_REVISION,
        )
        reranker = VietnameseCrossEncoderReranker(config)
        reranker._model = model

        fake_numpy = ModuleType("numpy")

        class _FakeArray:
            def __init__(self, values: list[list[float]]) -> None:
                self.values = values

            def reshape(self, _: int) -> list[float]:
                return [value for row in self.values for value in row]

        fake_numpy.asarray = _FakeArray  # type: ignore[attr-defined]
        with patch.dict(sys.modules, {"numpy": fake_numpy}):
            scores = reranker.score("Ai phải xin giấy phép?", ["đoạn A", "đoạn B"])

        self.assertEqual(scores, [-2.25, 1.5])
        model.predict.assert_called_once_with(
            [
                ("Ai phải xin giấy phép?", "đoạn A"),
                ("Ai phải xin giấy phép?", "đoạn B"),
            ],
            batch_size=config.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )


class _FakeInputs(dict):
    def __init__(self) -> None:
        input_ids = Mock()
        input_ids.shape = (1, 4)
        super().__init__(input_ids=input_ids, attention_mask="attention-mask")
        self.moved_to: str | None = None

    def to(self, device: str) -> "_FakeInputs":
        self.moved_to = device
        return self


class _FakeTokenizer:
    eos_token_id = 42
    pad_token_id = None

    def __init__(self) -> None:
        self.chat_calls: list[tuple[list[dict[str, str]], dict[str, object]]] = []
        self.decode_calls: list[tuple[list[int], dict[str, object]]] = []
        self.inputs = _FakeInputs()

    def apply_chat_template(
        self, messages: list[dict[str, str]], **kwargs: object
    ) -> _FakeInputs:
        self.chat_calls.append((messages, kwargs))
        return self.inputs

    def decode(self, tokens: list[int], **kwargs: object) -> str:
        self.decode_calls.append((tokens, kwargs))
        return "  đoạn\r\nluật\\n giả định  "


class _FakeModel:
    device = "cpu"

    def __init__(self) -> None:
        self.generate_calls: list[dict[str, object]] = []

    def generate(self, **kwargs: object) -> "_FakeGeneratedTokens":
        self.generate_calls.append(kwargs)
        return _FakeGeneratedTokens()


class _FakeGeneratedTokens:
    values = [[101, 102, 103, 104, 501, 502]]

    def __getitem__(self, key: tuple[int, slice]) -> list[int]:
        row, column_slice = key
        return self.values[row][column_slice]


class _FakeTorch:
    @staticmethod
    def inference_mode() -> object:
        return nullcontext()


class AITeamVNHyDETest(unittest.TestCase):
    def test_hyde_normalizer_removes_generation_artifacts_only(self) -> None:
        raw = (
            "\ufeff```text\r\n"
            "  Điều\u200b1.\t Áp dụng pha\u0301p luật và 50% mức phạt\\r\\n"
            "  theo Khoản 2.&nbsp;\x00```"
        )

        normalized = normalize_hyde_text(raw)

        self.assertEqual(
            normalized,
            "Điều 1. Áp dụng pháp luật và 50% mức phạt theo Khoản 2.",
        )
        self.assertEqual(normalize_hyde_text(normalized), normalized)

    def test_qwen2_prompt_is_deterministic_and_generation_uses_kv_cache(self) -> None:
        config = HyDEConfig(
            model_name="AITeamVN/Vi-Qwen2-3B-RAG",
            revision=HYDE_REVISION,
            max_new_tokens=123,
            do_sample=False,
        )
        tokenizer = _FakeTokenizer()
        model = _FakeModel()
        generator = QwenHyDEGenerator(config)
        generator._load = lambda: (_FakeTorch(), tokenizer, model)  # type: ignore[method-assign]

        hypothesis = generator.generate("  Điều kiện cấp giấy phép là gì?  ")

        self.assertEqual(hypothesis, "đoạn luật giả định")
        self.assertEqual(len(tokenizer.chat_calls), 1)
        messages, chat_kwargs = tokenizer.chat_calls[0]
        self.assertEqual(messages[0], {"role": "system", "content": SYSTEM_PROMPT})
        self.assertIn("Điều kiện cấp giấy phép là gì?", messages[1]["content"])
        self.assertNotIn("  Điều kiện cấp giấy phép", messages[1]["content"])
        self.assertEqual(
            chat_kwargs,
            {
                "tokenize": True,
                "add_generation_prompt": True,
                "return_dict": True,
                "return_tensors": "pt",
            },
        )
        self.assertNotIn("enable_thinking", chat_kwargs)
        self.assertEqual(tokenizer.inputs.moved_to, "cpu")

        generation_kwargs = model.generate_calls[0]
        self.assertEqual(generation_kwargs["max_new_tokens"], 123)
        self.assertIs(generation_kwargs["do_sample"], False)
        self.assertIs(generation_kwargs["use_cache"], True)
        self.assertEqual(generation_kwargs["pad_token_id"], 42)
        self.assertNotIn("temperature", generation_kwargs)
        self.assertNotIn("top_p", generation_kwargs)
        self.assertEqual(tokenizer.decode_calls, [([501, 502], {"skip_special_tokens": True})])


if __name__ == "__main__":
    unittest.main()
