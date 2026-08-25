from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Sequence

from .config import DenseConfig


LOGGER = logging.getLogger("legal_ir")
DEFAULT_WORK_CHUNK_SIZE = 256
POLL_INTERVAL_SECONDS = 1.0


def _available_cpu_count() -> int:
    get_affinity = getattr(os, "sched_getaffinity", None)
    if callable(get_affinity):
        try:
            return max(1, len(get_affinity(0)))
        except OSError:
            pass
    return max(1, os.cpu_count() or 1)


def partition_ranges(total: int, parts: int) -> list[tuple[int, int]]:
    """Split ``range(total)`` into contiguous, balanced, non-empty ranges."""

    if total < 0:
        raise ValueError("total must be non-negative")
    if parts <= 0:
        raise ValueError("parts must be positive")
    if total == 0:
        return []

    worker_count = min(total, parts)
    base, remainder = divmod(total, worker_count)
    ranges: list[tuple[int, int]] = []
    start = 0
    for rank in range(worker_count):
        width = base + (1 if rank < remainder else 0)
        end = start + width
        ranges.append((start, end))
        start = end
    return ranges


def _logical_cuda_index(device: str) -> int:
    prefix, separator, raw_index = device.partition(":")
    if prefix != "cuda" or separator != ":" or not raw_index.isdigit():
        raise ValueError(f"multi-GPU worker requires an indexed CUDA device: {device}")
    return int(raw_index)


def worker_environment(device: str, worker_count: int) -> dict[str, str]:
    """Give a worker one visible GPU and cap its share of host CPU threads."""

    if worker_count <= 0:
        raise ValueError("worker_count must be positive")
    logical_index = _logical_cuda_index(device)
    environment = os.environ.copy()

    parent_visible = environment.get("CUDA_VISIBLE_DEVICES")
    if parent_visible:
        visible_tokens = [token.strip() for token in parent_visible.split(",")]
        visible_tokens = [token for token in visible_tokens if token]
        if logical_index >= len(visible_tokens):
            raise RuntimeError(
                f"{device} is not present in CUDA_VISIBLE_DEVICES={parent_visible!r}"
            )
        worker_visible_device = visible_tokens[logical_index]
    else:
        worker_visible_device = str(logical_index)

    environment["CUDA_VISIBLE_DEVICES"] = worker_visible_device
    environment["TOKENIZERS_PARALLELISM"] = "false"
    environment["PYTHONUNBUFFERED"] = "1"
    # A notebook may have inserted retrieval/src into sys.path without installing
    # the project. Preserve that importability in the fresh worker interpreter.
    source_root = str(Path(__file__).resolve().parents[1])
    python_path = environment.get("PYTHONPATH")
    python_path_entries = python_path.split(os.pathsep) if python_path else []
    if source_root not in python_path_entries:
        environment["PYTHONPATH"] = os.pathsep.join(
            [source_root, *python_path_entries]
        )
    # Two concurrent Transformers loaders can oversubscribe Kaggle's four CPU
    # cores. These defaults remain user-overridable before the parent starts.
    environment.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
    environment.setdefault("HF_ENABLE_PARALLEL_LOADING", "false")

    cpu_threads = max(1, _available_cpu_count() // worker_count)
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        try:
            configured = int(environment.get(variable, cpu_threads))
        except (TypeError, ValueError):
            configured = cpu_threads
        environment[variable] = str(max(1, min(configured, cpu_threads)))
    return environment


def _resolve_model_source(config: DenseConfig) -> str:
    """Resolve/download one immutable snapshot before launching GPU workers."""

    local_path = Path(config.model_name).expanduser()
    if local_path.exists():
        return str(local_path.resolve())
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - transitive dependency failure
        raise RuntimeError(
            "huggingface-hub is required to prepare multi-GPU model weights"
        ) from exc

    LOGGER.info(
        "Preparing dense model snapshot %s at revision %s",
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
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _write_text_shard(path: Path, texts: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for text in texts:
            handle.write(json.dumps(text, ensure_ascii=False))
            handle.write("\n")


def _read_status(path: Path) -> dict[str, Any] | None:
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


def _terminate_processes(workers: Sequence[dict[str, Any]]) -> None:
    for worker in workers:
        process = worker["process"]
        if process.poll() is None:
            process.terminate()
    for worker in workers:
        process = worker["process"]
        if process.poll() is not None:
            continue
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def _failure_message(worker: dict[str, Any], reason: str) -> str:
    status = _read_status(worker["status_path"]) or {}
    error = status.get("error")
    traceback_text = status.get("traceback")
    log_tail = _log_tail(worker["log_path"])
    details = [f"dense GPU worker {worker['rank']} {reason}"]
    if error:
        details.append(f"worker error: {error}")
    if traceback_text:
        details.append(str(traceback_text).strip())
    if log_tail:
        details.append(f"worker log tail:\n{log_tail}")
    return "\n".join(details)


def _wait_for_workers(
    workers: Sequence[dict[str, Any]],
    *,
    total_texts: int,
    stall_timeout_seconds: int,
    show_progress: bool,
) -> None:
    last_activity = {worker["rank"]: time.monotonic() for worker in workers}
    status_signatures: dict[int, str | None] = {
        worker["rank"]: None for worker in workers
    }
    last_report: tuple[tuple[int, str, int], ...] | None = None

    while True:
        now = time.monotonic()
        all_finished = True
        report: list[tuple[int, str, int]] = []

        for worker in workers:
            rank = worker["rank"]
            process = worker["process"]
            return_code = process.poll()
            if return_code is None:
                all_finished = False

            status = _read_status(worker["status_path"]) or {}
            signature = (
                json.dumps(status, ensure_ascii=False, sort_keys=True)
                if status
                else None
            )
            if signature != status_signatures[rank]:
                status_signatures[rank] = signature
                last_activity[rank] = now

            state = str(status.get("state", "starting"))
            completed = int(status.get("completed", 0))
            report.append((rank, state, completed))

            if return_code is not None and return_code != 0:
                raise RuntimeError(
                    _failure_message(worker, f"exited with code {return_code}")
                )
            if state == "error":
                raise RuntimeError(_failure_message(worker, "reported an error"))
            if return_code == 0 and state != "complete":
                raise RuntimeError(
                    _failure_message(worker, "exited without a complete status")
                )
            if (
                return_code is None
                and now - last_activity[rank] > stall_timeout_seconds
            ):
                raise RuntimeError(
                    _failure_message(
                        worker,
                        f"made no progress for {stall_timeout_seconds} seconds",
                    )
                )

        report_tuple = tuple(report)
        if show_progress and report_tuple != last_report:
            encoded = sum(item[2] for item in report)
            worker_summary = ", ".join(
                f"gpu{rank}={state}:{completed}"
                for rank, state, completed in report
            )
            LOGGER.info(
                "Dense multi-GPU progress %d/%d chunks (%s)",
                encoded,
                total_texts,
                worker_summary,
            )
            last_report = report_tuple

        if all_finished:
            return
        time.sleep(POLL_INTERVAL_SECONDS)


def encode_documents_multi_gpu(
    texts: Sequence[str],
    *,
    devices: Sequence[str],
    config: DenseConfig,
    show_progress: bool,
) -> Any:
    """Encode deterministic shards in independent one-GPU subprocesses.

    Workers exchange paths and JSON status only. No model or CUDA tensor crosses
    a process boundary, which avoids SentenceTransformers' shared-parent pool.
    """

    if not texts:
        raise ValueError("cannot build a dense index from an empty corpus")
    if len(devices) < 2:
        raise ValueError("multi-GPU encoding requires at least two CUDA devices")

    ranges = partition_ranges(len(texts), len(devices))
    selected_devices = list(devices[: len(ranges)])
    work_chunk_size = config.multi_process_chunk_size or DEFAULT_WORK_CHUNK_SIZE
    model_source = _resolve_model_source(config)
    temporary_root = os.environ.get("LEGAL_IR_MULTI_GPU_TMPDIR")

    LOGGER.info(
        "Encoding %d dense documents with %d isolated GPU workers %s",
        len(texts),
        len(selected_devices),
        selected_devices,
    )

    with tempfile.TemporaryDirectory(
        prefix="legal-ir-dense-",
        dir=temporary_root,
    ) as directory:
        root = Path(directory)
        workers: list[dict[str, Any]] = []

        try:
            for rank, ((start, end), device) in enumerate(
                zip(ranges, selected_devices, strict=True)
            ):
                input_path = root / f"worker-{rank}.jsonl"
                output_path = root / f"worker-{rank}.npy"
                status_path = root / f"worker-{rank}.status.json"
                spec_path = root / f"worker-{rank}.spec.json"
                log_path = root / f"worker-{rank}.log"
                _write_text_shard(input_path, texts[start:end])
                _write_json(
                    spec_path,
                    {
                        "format_version": 1,
                        "rank": rank,
                        "model_source": model_source,
                        "model_name": config.model_name,
                        "revision": config.revision,
                        "batch_size": config.batch_size,
                        "max_length": config.max_length,
                        "dtype": config.dtype,
                        "normalize_embeddings": config.normalize_embeddings,
                        "work_chunk_size": work_chunk_size,
                        "expected_count": end - start,
                        "input_path": str(input_path),
                        "output_path": str(output_path),
                        "status_path": str(status_path),
                    },
                )

                log_handle = log_path.open("wb")
                try:
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "legal_ir.dense_worker",
                            "--spec",
                            str(spec_path),
                        ],
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        env=worker_environment(device, len(selected_devices)),
                    )
                except BaseException:
                    log_handle.close()
                    raise
                workers.append(
                    {
                        "rank": rank,
                        "process": process,
                        "log_handle": log_handle,
                        "log_path": log_path,
                        "output_path": output_path,
                        "status_path": status_path,
                        "expected_count": end - start,
                    }
                )

            try:
                _wait_for_workers(
                    workers,
                    total_texts=len(texts),
                    stall_timeout_seconds=config.multi_gpu_stall_timeout_seconds,
                    show_progress=show_progress,
                )
            except BaseException:
                _terminate_processes(workers)
                raise
        finally:
            _terminate_processes(workers)
            for worker in workers:
                worker["log_handle"].close()

        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - dependency error path
            raise RuntimeError("numpy is required for dense retrieval") from exc

        arrays: list[Any] = []
        expected_dimension: int | None = None
        for worker in workers:
            output_path = worker["output_path"]
            if not output_path.is_file():
                raise RuntimeError(
                    _failure_message(worker, "did not produce an embedding shard")
                )
            array = np.load(output_path, mmap_mode="r", allow_pickle=False)
            if array.ndim != 2 or array.shape[0] != worker["expected_count"]:
                raise RuntimeError(
                    f"dense GPU worker {worker['rank']} produced invalid shape "
                    f"{array.shape}; expected {worker['expected_count']} rows"
                )
            dimension = int(array.shape[1])
            if expected_dimension is None:
                expected_dimension = dimension
            elif dimension != expected_dimension:
                raise RuntimeError(
                    "dense GPU workers produced inconsistent embedding dimensions"
                )
            arrays.append(array)

        vectors = np.concatenate(arrays, axis=0, dtype=np.float32)
        if vectors.shape[0] != len(texts):
            raise RuntimeError(
                f"multi-GPU encoding returned {vectors.shape[0]} rows for "
                f"{len(texts)} texts"
            )
        return vectors
