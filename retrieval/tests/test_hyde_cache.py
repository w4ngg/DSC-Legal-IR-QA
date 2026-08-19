from __future__ import annotations

import unittest
from dataclasses import replace

from legal_ir.config import HyDEConfig
from legal_ir.hyde import hyde_cache_namespace


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


if __name__ == "__main__":
    unittest.main()
