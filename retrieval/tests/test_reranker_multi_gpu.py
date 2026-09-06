from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from legal_ir.reranker import VietnameseCrossEncoderReranker
from legal_ir.reranker_multi_gpu import (
    RerankerMultiGPUProcessPool,
    partition_batch_aligned_ranges,
)


def _config(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "model_name": "AITeamVN/Vietnamese_Reranker",
        "revision": "immutable-revision",
        "batch_size": 4,
        "max_length": 2304,
        "device": "auto",
        "dtype": "float16",
        "multi_gpu": True,
        "multi_gpu_stall_timeout_seconds": 60,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _FakeCommandStream:
    def __init__(
        self,
        process: "_FakeProcess",
        spec: dict[str, object],
    ) -> None:
        self.process = process
        self.spec = spec
        self.commands: list[dict[str, object]] = []
        self.closed = False

    def write(self, raw: bytes) -> int:
        command = json.loads(raw.decode("utf-8"))
        self.commands.append(command)
        if command["command"] == "shutdown":
            self.process.return_code = 0
            return len(raw)

        task = json.loads(
            Path(command["input_path"]).read_text(encoding="utf-8")
        )
        rank = int(self.spec["rank"])
        scores = [float(passage) + 100.0 * rank for passage in task["passages"]]
        Path(command["output_path"]).write_text(
            json.dumps(
                {
                    "format_version": 1,
                    "task_id": command["task_id"],
                    "scores": scores,
                }
            ),
            encoding="utf-8",
        )
        Path(self.spec["status_path"]).write_text(
            json.dumps(
                {
                    "state": "ready",
                    "task_id": command["task_id"],
                    "completed": len(scores),
                }
            ),
            encoding="utf-8",
        )
        return len(raw)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class _FakeProcess:
    def __init__(self, spec: dict[str, object]) -> None:
        self.return_code: int | None = None
        self.stdin = _FakeCommandStream(self, spec)
        Path(spec["status_path"]).write_text(
            json.dumps({"state": "ready", "completed": 0}),
            encoding="utf-8",
        )

    def poll(self) -> int | None:
        return self.return_code

    def wait(self, timeout: int | None = None) -> int:
        del timeout
        if self.return_code is None:
            self.return_code = 0
        return self.return_code

    def terminate(self) -> None:
        self.return_code = -15

    def kill(self) -> None:
        self.return_code = -9


class RerankerMultiGPUProcessPoolTest(unittest.TestCase):
    def test_partition_preserves_original_inference_batch_boundaries(self) -> None:
        self.assertEqual(
            partition_batch_aligned_ranges(11, 2, 4),
            [(0, 8), (8, 11)],
        )
        self.assertEqual(
            partition_batch_aligned_ranges(3, 2, 4),
            [(0, 3)],
        )

    def test_persistent_workers_restore_original_passage_order(self) -> None:
        processes: list[_FakeProcess] = []

        def fake_popen(command: list[str], **_: object) -> _FakeProcess:
            spec = json.loads(Path(command[-1]).read_text(encoding="utf-8"))
            process = _FakeProcess(spec)
            processes.append(process)
            return process

        with (
            patch(
                "legal_ir.reranker_multi_gpu._resolve_model_source",
                return_value="/cached/reranker",
            ),
            patch(
                "legal_ir.reranker_multi_gpu.subprocess.Popen",
                side_effect=fake_popen,
            ),
        ):
            pool = RerankerMultiGPUProcessPool(
                _config(batch_size=2),  # type: ignore[arg-type]
                ["cuda:0", "cuda:1"],
            )
            first = pool.score("query one", ["0", "1", "2", "3", "4"])
            second = pool.score("query two", ["5", "6", "7", "8"])
            pool.close()

        # Whole batch boundaries stay intact: [0:4] + [4:5], then [0:2] + [2:4].
        self.assertEqual(first, [0.0, 1.0, 2.0, 3.0, 104.0])
        self.assertEqual(second, [5.0, 6.0, 107.0, 108.0])
        self.assertEqual(len(processes), 2)
        for process in processes:
            actions = [command["command"] for command in process.stdin.commands]
            self.assertEqual(actions, ["score", "score", "shutdown"])
            self.assertTrue(process.stdin.closed)

    def test_pool_rejects_score_after_close(self) -> None:
        def fake_popen(command: list[str], **_: object) -> _FakeProcess:
            spec = json.loads(Path(command[-1]).read_text(encoding="utf-8"))
            return _FakeProcess(spec)

        with (
            patch(
                "legal_ir.reranker_multi_gpu._resolve_model_source",
                return_value="/cached/reranker",
            ),
            patch(
                "legal_ir.reranker_multi_gpu.subprocess.Popen",
                side_effect=fake_popen,
            ),
        ):
            pool = RerankerMultiGPUProcessPool(
                _config(),  # type: ignore[arg-type]
                ["cuda:0", "cuda:1"],
            )
            pool.close()
            with self.assertRaisesRegex(RuntimeError, "pool is closed"):
                pool.score("query", ["passage"])


class VietnameseRerankerMultiGPURoutingTest(unittest.TestCase):
    def test_score_lazily_uses_all_visible_gpus_and_close_retires_pool(self) -> None:
        fake_torch = Mock()
        fake_torch.cuda.is_available.return_value = True
        fake_torch.cuda.device_count.return_value = 2
        pool = Mock()
        pool.score.return_value = [-1.25, 3.5]

        reranker = VietnameseCrossEncoderReranker(
            _config()  # type: ignore[arg-type]
        )
        with (
            patch.object(reranker, "_torch", return_value=fake_torch),
            patch(
                "legal_ir.reranker.detected_torch_device",
                return_value="cuda",
            ),
            patch(
                "legal_ir.reranker.RerankerMultiGPUProcessPool",
                return_value=pool,
            ) as pool_class,
        ):
            scores = reranker.score("query", ["A", "B"])
            repeated = reranker.score("query 2", ["C", "D"])
            reranker.close()

        self.assertEqual(scores, [-1.25, 3.5])
        self.assertEqual(repeated, [-1.25, 3.5])
        pool_class.assert_called_once_with(
            reranker.config,
            ("cuda:0", "cuda:1"),
        )
        self.assertEqual(pool.score.call_count, 2)
        pool.close.assert_called_once_with()

    def test_explicit_single_device_does_not_enable_multi_gpu(self) -> None:
        reranker = VietnameseCrossEncoderReranker(
            _config(device="cuda:0")  # type: ignore[arg-type]
        )
        self.assertEqual(reranker._detect_multi_gpu_devices(), ())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
