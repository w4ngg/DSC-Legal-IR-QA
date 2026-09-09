from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from legal_ir.config import PipelineConfig
from legal_ir.io import ChunkStore
from legal_ir.mine_reranker_stage1 import (
    DenseLongHit,
    GoldQuery,
    LongDenseMiningResult,
    NegativeDocument,
    _verify_training_split,
    mine_stage1_dataset,
    select_stage1_negatives,
)
from legal_ir.schema import Chunk, ScoredChunk
from legal_ir.train_reranker_stage1 import (
    GroupCollator,
    TrainingGroup,
    _load_training_groups,
)


def _negative(
    document_id: str,
    score: float,
    *sources: str,
) -> NegativeDocument:
    return NegativeDocument(
        chunk_id=f"{document_id}:dual:long:000000",
        document_id=document_id,
        teacher_score=score,
        sources=tuple(sources),
        mapped_retrieval_rank=1 if "mapped_short_pool" in sources else None,
        retrieval_support_score=(
            0.1 if "mapped_short_pool" in sources else None
        ),
        long_dense_rank=1 if "long_dense" in sources else None,
        long_dense_score=0.8 if "long_dense" in sources else None,
    )


class RerankerStage1Test(unittest.TestCase):
    def test_negative_mining_prioritizes_hardness_and_distinct_documents(self) -> None:
        candidates = [
            _negative("A", 12.0, "mapped_short_pool"),
            _negative("B", 11.0, "long_dense"),
            _negative("C", 9.5, "mapped_short_pool"),
            _negative("D", 8.5, "long_dense"),
            _negative("E", 7.0, "mapped_short_pool"),
        ]
        selected = select_stage1_negatives(
            candidates,
            positive_score=10.0,
            count=4,
            near_margin=1.0,
            max_violating=1,
            max_near=1,
            min_direct_dense=1,
        )

        self.assertEqual(len(selected), 4)
        self.assertEqual(selected[0][0].document_id, "A")
        self.assertEqual(selected[0][1], "violating")
        self.assertEqual(selected[1][0].document_id, "C")
        self.assertEqual(selected[1][1], "near_margin")
        self.assertEqual(selected[2][0].document_id, "B")
        self.assertEqual(selected[2][1], "long_dense")
        self.assertEqual(
            len({candidate.document_id for candidate, _ in selected}),
            4,
        )

    def test_training_loader_rejects_any_gold_document_as_negative(self) -> None:
        row = {
            "format_version": 1,
            "query_id": "q1",
            "query": "Câu hỏi",
            "gold_document_ids": ["G1", "G2"],
            "positive": {
                "chunk_id": "p",
                "document_id": "G1",
                "text": "positive",
            },
            "negatives": [
                {
                    "chunk_id": "n",
                    "document_id": "G2",
                    "text": "false negative",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stage1_train.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "negative belongs to a gold"):
                _load_training_groups(path, expected_negative_count=1)

    def test_group_collator_keeps_positive_first_and_uses_pair_tokenization(self) -> None:
        calls = []

        class FakeTokenizer:
            def __call__(self, queries, passages, **kwargs):
                calls.append((queries, passages, kwargs))
                return {"input_ids": [[1], [2]]}

        collator = GroupCollator(FakeTokenizer(), max_length=123)
        class FakeLabels(list):
            def tolist(self):
                return list(self)

        fake_torch = SimpleNamespace(
            long=object(),
            zeros=lambda count, dtype: FakeLabels([0] * count),
        )
        with patch.dict(sys.modules, {"torch": fake_torch}):
            batch = collator(
                [
                    TrainingGroup(
                        query_id="q1",
                        query="query",
                        passages=("positive", "negative"),
                        chunk_ids=("p", "n"),
                        document_ids=("G", "N"),
                    )
                ]
            )

        self.assertEqual(calls[0][0], ["query", "query"])
        self.assertEqual(calls[0][1], ["positive", "negative"])
        self.assertEqual(calls[0][2]["max_length"], 123)
        self.assertEqual(batch["group_count"], 1)
        self.assertEqual(batch["group_size"], 2)
        self.assertEqual(batch["labels"].tolist(), [0])

    def test_training_split_verification_accepts_only_manifest_train_hash(self) -> None:
        payload = {
            "q1": {"question": "Câu hỏi", "answer": ["G1"]},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "train.json"
            train_path.write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )
            digest = hashlib.sha256(train_path.read_bytes()).hexdigest()
            manifest_path = root / "split_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "outputs": {
                            "train": {
                                "filename": "train.json",
                                "query_count": 1,
                                "sha256": digest,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            records = [GoldQuery("q1", "Câu hỏi", ("G1",))]
            result = _verify_training_split(
                train_path,
                records,
                split_manifest_path=None,
                allow_unverified=False,
            )

        self.assertTrue(result["verified"])
        self.assertEqual(result["sha256"], digest)

    def test_miner_writes_one_valid_group_from_existing_index_interfaces(self) -> None:
        namespace = "dual_char_v1_test"
        document_ids = ["G", *[f"N{index}" for index in range(7)]]
        long_chunks = ChunkStore(
            [
                Chunk(
                    f"{document_id}:{namespace}:long:000000",
                    document_id,
                    f"passage {20 - index}",
                    metadata={"granularity": "long"},
                )
                for index, document_id in enumerate(document_ids)
            ]
        )
        short_chunks = ChunkStore(
            [
                Chunk(
                    f"{document_id}:{namespace}:short:000000",
                    document_id,
                    f"short {document_id}",
                    metadata={
                        "granularity": "short",
                        "primary_long_chunk_id": (
                            f"{document_id}:{namespace}:long:000000"
                        ),
                        "long_chunk_ids": [
                            f"{document_id}:{namespace}:long:000000"
                        ],
                    },
                )
                for document_id in document_ids
            ]
        )

        class FakeSearch:
            def search(self, query, top_k):
                del query
                return [
                    ScoredChunk(chunk.chunk_id, chunk.document_id, top_k - rank)
                    for rank, chunk in enumerate(short_chunks.chunks)
                ]

        short_bundle = SimpleNamespace(
            chunks=short_chunks,
            bm25=FakeSearch(),
            dense=FakeSearch(),
        )

        class FakeReranker:
            def __init__(self, config):
                del config

            def __enter__(self):
                return self

            def __exit__(self, *args):
                del args

            def score(self, query, passages):
                del query
                return [float(passage.rsplit(" ", 1)[1]) for passage in passages]

        default = PipelineConfig()
        config = PipelineConfig.from_mapping(
            {
                "bm25": {"top_k_chunks": 50},
                "dense": {"top_k_chunks": 100},
                "hyde": {"enabled": False},
                "reranker": {"multi_gpu": False},
                "long_context": {"enabled": True},
            }
        )
        del default

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gold_path = root / "train.json"
            gold_path.write_text(
                json.dumps({"q1": {"question": "query", "answer": ["G"]}}),
                encoding="utf-8",
            )
            digest = hashlib.sha256(gold_path.read_bytes()).hexdigest()
            (root / "split_manifest.json").write_text(
                json.dumps(
                    {
                        "outputs": {
                            "train": {
                                "filename": "train.json",
                                "query_count": 1,
                                "sha256": digest,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            short_index = root / "short_index"
            long_index = root / "long_index"
            short_index.mkdir()
            long_index.mkdir()
            (short_index / "manifest.json").write_text("{}", encoding="utf-8")
            (long_index / "manifest.json").write_text("{}", encoding="utf-8")
            config_path = root / "config.yaml"
            config_path.write_text("test: true\n", encoding="utf-8")
            output_dir = root / "output"
            positive_hit = DenseLongHit(
                long_chunks[0].chunk_id,
                "G",
                0.9,
                1,
            )
            direct_hits = tuple(
                DenseLongHit(chunk.chunk_id, chunk.document_id, 0.8, rank)
                for rank, chunk in enumerate(long_chunks.chunks[1:], start=1)
            )
            long_dense_result = {
                "q1": LongDenseMiningResult(
                    positive_shortlists={"G": (positive_hit,)},
                    global_hits=direct_hits,
                )
            }
            args = argparse.Namespace(
                gold=str(gold_path),
                split_manifest=None,
                short_index_dir=str(short_index),
                long_index_dir=str(long_index),
                config=str(config_path),
                output_dir=str(output_dir),
                positive_dense_top_k=8,
                direct_long_top_k=60,
                negatives_per_positive=7,
                near_margin=2.0,
                max_violating_negatives=3,
                max_near_negatives=2,
                min_direct_dense_negatives=1,
                dense_query_batch_size=16,
                log_every=1,
                max_queries=None,
                allow_unverified_training_data=False,
                overwrite=False,
            )
            with (
                patch(
                    "legal_ir.mine_reranker_stage1.PipelineConfig.from_yaml",
                    return_value=config,
                ),
                patch(
                    "legal_ir.mine_reranker_stage1._load_long_dense_index",
                    return_value=(long_chunks, object(), {"chunk_count": 8}),
                ),
                patch(
                    "legal_ir.mine_reranker_stage1._precompute_long_dense_candidates",
                    return_value=long_dense_result,
                ),
                patch(
                    "legal_ir.mine_reranker_stage1.load_indexes",
                    return_value=short_bundle,
                ),
                patch(
                    "legal_ir.mine_reranker_stage1.VietnameseCrossEncoderReranker",
                    FakeReranker,
                ),
                patch("legal_ir.mine_reranker_stage1._release_cuda_cache"),
            ):
                manifest = mine_stage1_dataset(args)

            row = json.loads(
                (output_dir / "stage1_train.jsonl").read_text(encoding="utf-8")
            )

        self.assertEqual(manifest["dataset"]["group_count"], 1)
        self.assertEqual(row["positive"]["document_id"], "G")
        self.assertEqual(len(row["negatives"]), 7)
        self.assertNotIn("G", {item["document_id"] for item in row["negatives"]})


if __name__ == "__main__":
    unittest.main()
