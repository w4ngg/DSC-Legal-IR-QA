from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from count_long_chunk_candidate_grid import (
    evaluate_candidate_grid,
    load_relevant_short_to_long_mappings,
    prepare_candidates,
)


def _hit(rank: int, chunk_id: str) -> dict:
    return {
        "rank": rank,
        "chunk_id": chunk_id,
        "document_id": chunk_id.split("-")[0],
        "score": 1.0 / rank,
    }


def _channel(chunk_ids: list[str]) -> dict:
    return {
        "requested_top_k_chunks": len(chunk_ids),
        "returned_chunk_count": len(chunk_ids),
        "chunk_hits": [
            _hit(rank, chunk_id)
            for rank, chunk_id in enumerate(chunk_ids, start=1)
        ],
    }


def _deep_payload() -> dict:
    return {
        "format_version": 1,
        "queries": {
            "q1": {
                "channels": {
                    "bm25": _channel(["a-s1", "a-s2"]),
                    "dense": _channel(["a-s1", "b-s3"]),
                }
            },
            "q2": {
                "channels": {
                    "bm25": _channel(["c-s4", "d-s5"]),
                    "dense": _channel(["e-s6", "c-s4"]),
                }
            },
        },
    }


def _mapping_records() -> list[dict]:
    return [
        {
            "short_chunk_id": "a-s1",
            "primary_long_chunk_id": "a-l1",
            "long_chunk_ids": ["a-l1", "a-l2"],
        },
        {
            "short_chunk_id": "a-s2",
            "primary_long_chunk_id": "a-l2",
            "long_chunk_ids": ["a-l2"],
        },
        {
            "short_chunk_id": "b-s3",
            "primary_long_chunk_id": "b-l1",
            "long_chunk_ids": ["b-l1", "b-l2"],
        },
        {
            "short_chunk_id": "c-s4",
            "primary_long_chunk_id": "c-l1",
            "long_chunk_ids": ["c-l1"],
        },
        {
            "short_chunk_id": "d-s5",
            "primary_long_chunk_id": "d-l1",
            "long_chunk_ids": ["d-l1"],
        },
        {
            "short_chunk_id": "e-s6",
            "primary_long_chunk_id": "e-l1",
            "long_chunk_ids": ["e-l1", "e-l2"],
        },
    ]


class CountLongChunkCandidateGridTest(unittest.TestCase):
    def _write_mappings(self, root: Path, records: list[dict] | None = None) -> Path:
        path = root / "short_to_long.jsonl"
        selected = _mapping_records() if records is None else records
        path.write_text(
            "".join(json.dumps(record) + "\n" for record in selected),
            encoding="utf-8",
        )
        return path

    def test_unions_short_chunks_then_deduplicates_all_mapped_longs(self) -> None:
        prepared = prepare_candidates(
            _deep_payload(),
            maximum_top_k_by_channel={"bm25": 2, "dense": 2},
        )
        with tempfile.TemporaryDirectory() as directory:
            loaded = load_relevant_short_to_long_mappings(
                self._write_mappings(Path(directory)),
                prepared.required_short_chunk_ids,
                mapping_mode="all",
                log_every=0,
            )
        evaluation = evaluate_candidate_grid(
            prepared,
            loaded.by_short_chunk_id,
            bm25_top_k_values=(1, 2),
            dense_top_k_values=(1, 2),
            mapping_mode="all",
        )

        q1 = evaluation.per_query["queries"]["q1"]["candidate_grid"]
        self.assertEqual(q1["bm25_1_dense_1"]["unique_short_chunks"], 1)
        self.assertEqual(q1["bm25_1_dense_1"]["unique_long_chunks"], 2)
        self.assertEqual(q1["bm25_1_dense_2"]["unique_short_chunks"], 2)
        self.assertEqual(q1["bm25_1_dense_2"]["unique_long_chunks"], 4)
        self.assertEqual(q1["bm25_2_dense_1"]["unique_long_chunks"], 2)
        self.assertEqual(q1["bm25_2_dense_2"]["unique_long_chunks"], 4)

        summary = evaluation.summary["grid"]["bm25_1_dense_1"]
        self.assertEqual(summary["unique_long_chunks_per_query"]["total"], 5)
        self.assertEqual(summary["unique_long_chunks_per_query"]["mean"], 2.5)

    def test_primary_mode_maps_each_short_chunk_to_only_one_long(self) -> None:
        prepared = prepare_candidates(
            _deep_payload(),
            maximum_top_k_by_channel={"bm25": 2, "dense": 2},
        )
        with tempfile.TemporaryDirectory() as directory:
            loaded = load_relevant_short_to_long_mappings(
                self._write_mappings(Path(directory)),
                prepared.required_short_chunk_ids,
                mapping_mode="primary",
                log_every=0,
            )
        evaluation = evaluate_candidate_grid(
            prepared,
            loaded.by_short_chunk_id,
            bm25_top_k_values=(2,),
            dense_top_k_values=(2,),
            mapping_mode="primary",
        )

        q1 = evaluation.per_query["queries"]["q1"]["candidate_grid"]
        self.assertEqual(q1["bm25_2_dense_2"]["unique_long_chunks"], 3)

    def test_rejects_missing_required_mapping(self) -> None:
        prepared = prepare_candidates(
            _deep_payload(),
            maximum_top_k_by_channel={"bm25": 2, "dense": 2},
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "missing 1 required"):
                load_relevant_short_to_long_mappings(
                    self._write_mappings(Path(directory), _mapping_records()[:-1]),
                    prepared.required_short_chunk_ids,
                    log_every=0,
                )

    def test_rejects_deep_diagnostics_shallower_than_requested_grid(self) -> None:
        with self.assertRaisesRegex(ValueError, "only requested top-2"):
            prepare_candidates(
                _deep_payload(),
                maximum_top_k_by_channel={"bm25": 3, "dense": 2},
            )


if __name__ == "__main__":
    unittest.main()
