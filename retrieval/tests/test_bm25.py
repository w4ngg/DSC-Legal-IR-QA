from __future__ import annotations

import unittest
from types import SimpleNamespace

from legal_ir.bm25 import BM25Index
from legal_ir.io import ChunkStore
from legal_ir.schema import Chunk


class _FakeBM25Model:
    scores = {"num_docs": 2}

    def retrieve(self, *_args, **_kwargs):
        return SimpleNamespace(
            documents=[[0, 1]],
            scores=[[0.0, 2.5]],
        )


class _FakeBM25Library:
    @staticmethod
    def tokenize(*_args, **_kwargs):
        return [["query"]]


class _TestableBM25Index(BM25Index):
    @staticmethod
    def _library():
        return _FakeBM25Library


class BM25IndexTest(unittest.TestCase):
    def test_zero_score_padding_is_not_fused(self) -> None:
        chunks = ChunkStore(
            [Chunk("zero", "A", "a"), Chunk("match", "B", "b")]
        )
        index = _TestableBM25Index(_FakeBM25Model(), chunks)

        hits = index.search("query", top_k=2)

        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].chunk_id, "match")
        self.assertEqual(hits[0].score, 2.5)


if __name__ == "__main__":
    unittest.main()

