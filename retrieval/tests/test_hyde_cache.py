from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from legal_ir.config import HyDEConfig
from legal_ir.hyde import JsonlCachedHyDEGenerator, hyde_cache_namespace


class _StaticGenerator:
    def __init__(self, output: str) -> None:
        self.output = output
        self.calls = 0

    def generate(self, query: str) -> str:
        self.calls += 1
        return self.output


class HyDECacheTest(unittest.TestCase):
    def test_generation_changes_invalidate_namespace(self) -> None:
        base = HyDEConfig()
        self.assertNotEqual(
            hyde_cache_namespace(base),
            hyde_cache_namespace(replace(base, max_new_tokens=base.max_new_tokens + 1)),
        )
        self.assertNotEqual(
            hyde_cache_namespace(base),
            hyde_cache_namespace(replace(base, revision="another-revision")),
        )

    def test_normalizer_version_change_invalidates_namespace(self) -> None:
        config = HyDEConfig()
        current = hyde_cache_namespace(config)

        with patch(
            "legal_ir.hyde.HYDE_NORMALIZATION_VERSION",
            "hyde_nfc_ws_future",
        ):
            changed = hyde_cache_namespace(config)

        self.assertNotEqual(current, changed)

    def test_cache_miss_and_hit_return_the_same_normalized_text(self) -> None:
        with TemporaryDirectory() as directory:
            cache_path = Path(directory) / "hyde.jsonl"
            namespace = hyde_cache_namespace(HyDEConfig())
            first_generator = _StaticGenerator("  Văn bản\r\npha\u0301p luật\\n liên quan  ")
            first = JsonlCachedHyDEGenerator(
                first_generator,
                cache_path,
                namespace=namespace,
            )

            self.assertEqual(first.generate("câu hỏi"), "Văn bản pháp luật liên quan")
            self.assertEqual(first_generator.calls, 1)

            second_generator = _StaticGenerator("không được gọi")
            second = JsonlCachedHyDEGenerator(
                second_generator,
                cache_path,
                namespace=namespace,
            )
            self.assertEqual(second.generate("câu hỏi"), "Văn bản pháp luật liên quan")
            self.assertEqual(second_generator.calls, 0)

    def test_cache_rejects_empty_generated_text(self) -> None:
        with TemporaryDirectory() as directory:
            cached = JsonlCachedHyDEGenerator(
                _StaticGenerator(" \r\n\\n\t\x00 "),
                Path(directory) / "hyde.jsonl",
                namespace="test",
            )

            with self.assertRaisesRegex(RuntimeError, "empty"):
                cached.generate("câu hỏi")


if __name__ == "__main__":
    unittest.main()
