from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Sequence

from .config import RerankerConfig
from .multi_gpu import partition_ranges, worker_environment


LOGGER = logging.getLogger("legal_ir")
POLL_INTERVAL_SECONDS = 0.05
PROTOCOL_VERSION = 1


def partition_batch_aligned_ranges(
    total: int,
    parts: int,
    batch_size: int,
) -> list[tuple[int, int]]:
    """Balance whole inference batches without changing batch boundaries."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if total == 0:
        return []
    batch_count = (total + batch_size - 1) // batch_size
    return [
        (batch_start * batch_size, min(total, batch_end * batch_size))
        for batch_start, batch_end in partition_ranges(batch_count, parts)
    ]


def _resolve_model_source(config: RerankerConfig) -> str:
    """Resolve one immutable snapshot before either GPU worker starts."""

    local_path = Path(config.model_name).expanduser()
    if local_path.exists():
        return str(local_path.resolve())
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - transitive dependency failure
        raise RuntimeError(
            "huggingface-hub is required to prepare multi-GPU reranker weights"
        ) from exc

    LOGGER.info(
        "Preparing reranker model snapshot %s at revision %s",
        config.model_name,
        config.revision or "main",
    )
    return str(
        snapshot_download(
            repo_id=config.model_name,
            revision=config.revision,
        )
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def _log_tail(path: Path, limit: int = 8000) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - limit))
            return handle.read().decode("utf-8", errors="replace").strip()
    except OSError:
        return ""


class RerankerMultiGPUProcessPool:
    """Persistent isolated workers that score each query on all visible GPUs.

    The parent process exchanges only JSON files and small command messages with
    workers. Models and CUDA tensors never cross a process boundary. Passage
    ranges are contiguous and results are restored to the caller's exact order.
    """

    def __init__(
        self,
        config: RerankerConfig,
        devices: Sequence[str],
    ) -> None:
        if len(devices) < 2:
            raise ValueError("multi-GPU reranking requires at least two devices")
        self.config = config
        self.devices = tuple(devices)
        self._lock = threading.Lock()
        self._task_id = 0
        self._closed = False
        self._workers: list[dict[str, Any]] = []
        self._temporary: tempfile.TemporaryDirectory[str] | None = None
        self._root: Path | None = None

        model_source = _resolve_model_source(config)
        temporary_root = os.environ.get(
            "LEGAL_IR_RERANKER_MULTI_GPU_TMPDIR",
            os.environ.get("LEGAL_IR_MULTI_GPU_TMPDIR"),
        )
        self._temporary = tempfile.TemporaryDirectory(
            prefix="legal-ir-reranker-",
            dir=temporary_root,
        )
        self._root = Path(self._temporary.name)

        LOGGER.info(
            "Starting %d isolated reranker GPU workers %s",
            len(self.devices),
            list(self.devices),
        )
        try:
            self._launch_workers(model_source)
            self._wait_until_ready()
        except BaseException:
            self._shutdown_workers()
            self._cleanup_temporary_directory()
            self._closed = True
            raise

    @property
    def worker_count(self) -> int:
        return len(self._workers)

    def _timeout_seconds(self) -> int:
        value = getattr(
            self.config,
            "multi_gpu_stall_timeout_seconds",
            1800,
        )
        return int(value)

    def _launch_workers(self, model_source: str) -> None:
        assert self._root is not None
        worker_count = len(self.devices)
        for rank, device in enumerate(self.devices):
            spec_path = self._root / f"worker-{rank}.spec.json"
            status_path = self._root / f"worker-{rank}.status.json"
            log_path = self._root / f"worker-{rank}.log"
            _write_json(
                spec_path,
                {
                    "format_version": PROTOCOL_VERSION,
                    "rank": rank,
                    "model_source": model_source,
                    "model_name": self.config.model_name,
                    "revision": self.config.revision,
                    "batch_size": self.config.batch_size,
                    "max_length": self.config.max_length,
                    "dtype": self.config.dtype,
                    "status_path": str(status_path),
                },
            )

            log_handle = log_path.open("wb")
            try:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "legal_ir.reranker_worker",
                        "--spec",
                        str(spec_path),
                    ],
                    stdin=subprocess.PIPE,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    env=worker_environment(device, worker_count),
                )
            except BaseException:
                log_handle.close()
                raise
            self._workers.append(
                {
                    "rank": rank,
                    "device": device,
                    "process": process,
                    "log_handle": log_handle,
                    "log_path": log_path,
                    "status_path": status_path,
                }
            )

    def _failure_message(self, worker: dict[str, Any], reason: str) -> str:
        status = _read_json(worker["status_path"]) or {}
        details = [f"reranker GPU worker {worker['rank']} {reason}"]
        if status.get("error"):
            details.append(f"worker error: {status['error']}")
        if status.get("traceback"):
            details.append(str(status["traceback"]).strip())
        log_tail = _log_tail(worker["log_path"])
        if log_tail:
            details.append(f"worker log tail:\n{log_tail}")
        return "\n".join(details)

    def _wait_until_ready(self) -> None:
        last_activity = {
            worker["rank"]: time.monotonic() for worker in self._workers
        }
        signatures: dict[int, str | None] = {
            worker["rank"]: None for worker in self._workers
        }
        while True:
            now = time.monotonic()
            ready_count = 0
            for worker in self._workers:
                process = worker["process"]
                return_code = process.poll()
                status = _read_json(worker["status_path"]) or {}
                signature = (
                    json.dumps(status, ensure_ascii=False, sort_keys=True)
                    if status
                    else None
                )
                rank = worker["rank"]
                if signature != signatures[rank]:
                    signatures[rank] = signature
                    last_activity[rank] = now
                state = status.get("state")
                if return_code is not None:
                    raise RuntimeError(
                        self._failure_message(
                            worker,
                            f"exited with code {return_code} during startup",
                        )
                    )
                if state == "error":
                    raise RuntimeError(
                        self._failure_message(worker, "reported a startup error")
                    )
                if state == "ready":
                    ready_count += 1
                    continue
                if now - last_activity[rank] > self._timeout_seconds():
                    raise RuntimeError(
                        self._failure_message(
                            worker,
                            "made no startup progress for "
                            f"{self._timeout_seconds()} seconds",
                        )
                    )
            if ready_count == len(self._workers):
                return
            time.sleep(POLL_INTERVAL_SECONDS)

    @staticmethod
    def _send_command(worker: dict[str, Any], payload: dict[str, Any]) -> None:
        process = worker["process"]
        stream = process.stdin
        if stream is None:
            raise RuntimeError(
                f"reranker GPU worker {worker['rank']} has no command stream"
            )
        try:
            stream.write(
                (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
            )
            stream.flush()
        except (BrokenPipeError, OSError, ValueError) as exc:
            raise RuntimeError(
                f"failed to send command to reranker GPU worker "
                f"{worker['rank']}"
            ) from exc

    def _wait_for_outputs(self, tasks: Sequence[dict[str, Any]]) -> None:
        last_activity = {
            task["worker"]["rank"]: time.monotonic() for task in tasks
        }
        signatures: dict[int, str | None] = {
            task["worker"]["rank"]: None for task in tasks
        }
        while True:
            now = time.monotonic()
            complete = 0
            for task in tasks:
                worker = task["worker"]
                rank = worker["rank"]
                return_code = worker["process"].poll()
                status = _read_json(worker["status_path"]) or {}
                signature = (
                    json.dumps(status, ensure_ascii=False, sort_keys=True)
                    if status
                    else None
                )
                if signature != signatures[rank]:
                    signatures[rank] = signature
                    last_activity[rank] = now
                if return_code is not None:
                    raise RuntimeError(
                        self._failure_message(
                            worker,
                            f"exited with code {return_code} while scoring",
                        )
                    )
                if status.get("state") == "error":
                    raise RuntimeError(
                        self._failure_message(worker, "reported a scoring error")
                    )
                if task["output_path"].is_file():
                    complete += 1
                    continue
                if now - last_activity[rank] > self._timeout_seconds():
                    raise RuntimeError(
                        self._failure_message(
                            worker,
                            "made no scoring progress for "
                            f"{self._timeout_seconds()} seconds",
                        )
                    )
            if complete == len(tasks):
                return
            time.sleep(POLL_INTERVAL_SECONDS)

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        with self._lock:
            if self._closed:
                raise RuntimeError("reranker multi-GPU pool is closed")
            self._task_id += 1
            task_id = self._task_id
            # Keep the exact single-GPU dynamic-padding batches intact. This
            # avoids score drift caused only by changing padding companions.
            ranges = partition_batch_aligned_ranges(
                len(passages),
                len(self._workers),
                int(self.config.batch_size),
            )
            assert self._root is not None
            tasks: list[dict[str, Any]] = []

            try:
                for worker, (start, end) in zip(
                    self._workers, ranges, strict=False
                ):
                    rank = worker["rank"]
                    input_path = (
                        self._root / f"task-{task_id}-worker-{rank}.input.json"
                    )
                    output_path = (
                        self._root / f"task-{task_id}-worker-{rank}.output.json"
                    )
                    _write_json(
                        input_path,
                        {
                            "format_version": PROTOCOL_VERSION,
                            "query": query,
                            "passages": list(passages[start:end]),
                        },
                    )
                    task = {
                        "worker": worker,
                        "start": start,
                        "end": end,
                        "input_path": input_path,
                        "output_path": output_path,
                    }
                    tasks.append(task)
                    self._send_command(
                        worker,
                        {
                            "command": "score",
                            "task_id": task_id,
                            "input_path": str(input_path),
                            "output_path": str(output_path),
                        },
                    )

                self._wait_for_outputs(tasks)
                ordered_scores = [0.0] * len(passages)
                for task in tasks:
                    payload = _read_json(task["output_path"])
                    if payload is None:
                        raise RuntimeError(
                            f"reranker GPU worker {task['worker']['rank']} "
                            "produced an unreadable score file"
                        )
                    if payload.get("format_version") != PROTOCOL_VERSION:
                        raise RuntimeError("unsupported reranker result version")
                    if payload.get("task_id") != task_id:
                        raise RuntimeError(
                            "reranker worker result belongs to a different task"
                        )
                    raw_scores = payload.get("scores")
                    expected_count = task["end"] - task["start"]
                    if (
                        not isinstance(raw_scores, list)
                        or len(raw_scores) != expected_count
                    ):
                        raise RuntimeError(
                            f"reranker GPU worker {task['worker']['rank']} returned "
                            f"an invalid score count; expected {expected_count}"
                        )
                    ordered_scores[task["start"] : task["end"]] = [
                        float(score) for score in raw_scores
                    ]
                return ordered_scores
            except BaseException:
                # A failed task can leave a worker blocked or its protocol out of
                # sync. Retire the complete pool rather than returning later scores
                # from the wrong task.
                self._shutdown_workers()
                self._closed = True
                self._cleanup_temporary_directory()
                raise
            finally:
                for task in tasks:
                    task["input_path"].unlink(missing_ok=True)
                    task["output_path"].unlink(missing_ok=True)

    def _shutdown_workers(self) -> None:
        for worker in self._workers:
            process = worker["process"]
            if process.poll() is None:
                try:
                    self._send_command(worker, {"command": "shutdown"})
                except RuntimeError:
                    pass
            stream = process.stdin
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
        for worker in self._workers:
            process = worker["process"]
            if process.poll() is None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
            worker["log_handle"].close()

    def _cleanup_temporary_directory(self) -> None:
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None
            self._root = None

    def close(self) -> None:
        with self._lock:
            if self._closed:
                self._cleanup_temporary_directory()
                return
            self._closed = True
            self._shutdown_workers()
            self._cleanup_temporary_directory()

    def __enter__(self) -> "RerankerMultiGPUProcessPool":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        del exc_type, exc, traceback
        self.close()
        return False
