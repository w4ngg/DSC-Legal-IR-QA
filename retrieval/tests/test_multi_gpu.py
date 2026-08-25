from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

from legal_ir.config import DenseConfig
from legal_ir.dense_worker import _text_blocks
from legal_ir.multi_gpu import (
    encode_documents_multi_gpu,
    partition_ranges,
    worker_environment,
)


class _FakeArray:
    def __init__(self, rows: list[list[float]]) -> None:
        self.rows = rows
        self.ndim = 2
        width = len(rows[0]) if rows else 0
        self.shape = (len(rows), width)


class _FakeNumpy(ModuleType):
    float32 = "float32"

    def load(self, path: Path, **_: object) -> _FakeArray:
        rows = json.loads(Path(path).read_text(encoding="utf-8"))
        return _FakeArray(rows)

    def concatenate(
        self,
        arrays: list[_FakeArray],
        *,
        axis: int,
        dtype: object,
    ) -> _FakeArray:
        if axis != 0 or dtype != self.float32:
            raise AssertionError("unexpected concatenate arguments")
        return _FakeArray([row for array in arrays for row in array.rows])


class _FinishedProcess:
    def __init__(self, return_code: int = 0) -> None:
        self.return_code: int | None = return_code
        self.terminated = False

    def poll(self) -> int | None:
        return self.return_code

    def terminate(self) -> None:
        self.terminated = True
        self.return_code = -15

    def kill(self) -> None:
        self.return_code = -9

    def wait(self, timeout: int | None = None) -> int:
        del timeout
        if self.return_code is None:
            self.return_code = 0
        return self.return_code


class MultiGPUOrchestrationTest(unittest.TestCase):
    def test_partition_ranges_are_balanced_and_ordered(self) -> None:
        self.assertEqual(partition_ranges(5, 2), [(0, 3), (3, 5)])
        self.assertEqual(partition_ranges(2, 4), [(0, 1), (1, 2)])
        self.assertEqual(partition_ranges(0, 2), [])

    def test_worker_environment_isolates_one_visible_gpu_and_caps_threads(
        self,
    ) -> None:
        with (
            patch.dict(
                os.environ,
                {
                    "CUDA_VISIBLE_DEVICES": "4,7",
                    "OMP_NUM_THREADS": "64",
                    "PYTHONPATH": "",
                },
                clear=False,
            ),
            patch("legal_ir.multi_gpu._available_cpu_count", return_value=4),
        ):
            environment = worker_environment("cuda:1", 2)

        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "7")
        self.assertEqual(environment["OMP_NUM_THREADS"], "2")
        self.assertEqual(environment["TOKENIZERS_PARALLELISM"], "false")
        source_root = str(Path(__file__).parents[1] / "src")
        self.assertIn(source_root, environment["PYTHONPATH"].split(os.pathsep))

    def test_independent_workers_restore_original_corpus_order(self) -> None:
        launches: list[tuple[list[str], dict[str, str]]] = []

        def fake_popen(command: list[str], **kwargs: object) -> _FinishedProcess:
            spec = json.loads(Path(command[-1]).read_text(encoding="utf-8"))
            texts = [
                json.loads(line)
                for line in Path(spec["input_path"])
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            rank = int(spec["rank"])
            rows = [[float(text), float(rank)] for text in texts]
            Path(spec["output_path"]).write_text(
                json.dumps(rows),
                encoding="utf-8",
            )
            Path(spec["status_path"]).write_text(
                json.dumps(
                    {
                        "state": "complete",
                        "completed": len(texts),
                        "total": len(texts),
                    }
                ),
                encoding="utf-8",
            )
            launches.append((command, kwargs["env"]))  # type: ignore[arg-type]
            return _FinishedProcess()

        fake_numpy = _FakeNumpy("numpy")
        config = DenseConfig(
            multi_gpu=True,
            multi_process_chunk_size=2,
            multi_gpu_stall_timeout_seconds=60,
        )
        with (
            patch(
                "legal_ir.multi_gpu._resolve_model_source",
                return_value="/cached/model",
            ),
            patch("legal_ir.multi_gpu.subprocess.Popen", side_effect=fake_popen),
            patch.dict(sys.modules, {"numpy": fake_numpy}),
        ):
            vectors = encode_documents_multi_gpu(
                ["0", "1", "2", "3", "4"],
                devices=["cuda:0", "cuda:1"],
                config=config,
                show_progress=False,
            )

        self.assertEqual(
            vectors.rows,
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 1.0],
                [4.0, 1.0],
            ],
        )
        self.assertEqual(len(launches), 2)
        for command, environment in launches:
            self.assertEqual(command[1:4], ["-m", "legal_ir.dense_worker", "--spec"])
            self.assertIn(environment["CUDA_VISIBLE_DEVICES"], {"0", "1"})

    def test_failed_worker_terminates_its_peer_and_surfaces_error(self) -> None:
        processes: list[_FinishedProcess] = []

        def fake_popen(command: list[str], **_: object) -> _FinishedProcess:
            spec = json.loads(Path(command[-1]).read_text(encoding="utf-8"))
            rank = int(spec["rank"])
            if rank == 0:
                Path(spec["status_path"]).write_text(
                    json.dumps({"state": "error", "error": "CUDA OOM"}),
                    encoding="utf-8",
                )
                process = _FinishedProcess(return_code=7)
            else:
                process = _FinishedProcess(return_code=0)
                process.return_code = None
            processes.append(process)
            return process

        with (
            patch(
                "legal_ir.multi_gpu._resolve_model_source",
                return_value="/cached/model",
            ),
            patch("legal_ir.multi_gpu.subprocess.Popen", side_effect=fake_popen),
        ):
            with self.assertRaisesRegex(RuntimeError, "CUDA OOM"):
                encode_documents_multi_gpu(
                    ["a", "b"],
                    devices=["cuda:0", "cuda:1"],
                    config=DenseConfig(multi_gpu_stall_timeout_seconds=60),
                    show_progress=False,
                )

        self.assertTrue(processes[1].terminated)

    def test_worker_text_blocks_preserve_json_strings_and_order(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            path = Path(directory) / "input.jsonl"
            values = ["Điều 1\nKhoản 2", "có dấu tiếng Việt", "cuối"]
            path.write_text(
                "".join(
                    json.dumps(value, ensure_ascii=False) + "\n"
                    for value in values
                ),
                encoding="utf-8",
            )
            self.assertEqual(list(_text_blocks(path, 2)), [values[:2], values[2:]])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
