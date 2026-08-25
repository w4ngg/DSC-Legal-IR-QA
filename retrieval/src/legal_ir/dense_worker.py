from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import traceback
from typing import Any, Iterator, Sequence


WORKER_SPEC_VERSION = 1


def _atomic_write_status(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _text_blocks(path: Path, block_size: int) -> Iterator[list[str]]:
    block: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank line in worker input at line {line_number}")
            text = json.loads(line)
            if not isinstance(text, str):
                raise ValueError(
                    f"worker input line {line_number} is not a JSON string"
                )
            block.append(text)
            if len(block) >= block_size:
                yield block
                block = []
    if block:
        yield block


def _torch_dtype(torch: Any, dtype: str) -> Any:
    if dtype != "auto":
        try:
            return getattr(torch, dtype)
        except AttributeError as exc:
            raise ValueError(f"unsupported torch dtype: {dtype}") from exc
    supports_bf16 = getattr(torch.cuda, "is_bf16_supported", None)
    if callable(supports_bf16) and supports_bf16():
        return torch.bfloat16
    return torch.float16


def _load_spec(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        spec = json.load(handle)
    if not isinstance(spec, dict):
        raise ValueError("dense worker spec must be a JSON object")
    if spec.get("format_version") != WORKER_SPEC_VERSION:
        raise ValueError(
            f"unsupported dense worker spec version: {spec.get('format_version')}"
        )
    return spec


def run_worker(spec_path: str | Path) -> None:
    spec = _load_spec(Path(spec_path))
    status_path = Path(spec["status_path"])
    output_path = Path(spec["output_path"])
    partial_output_path = output_path.with_suffix(output_path.suffix + ".partial")
    input_path = Path(spec["input_path"])
    rank = int(spec["rank"])
    expected_count = int(spec["expected_count"])
    batch_size = int(spec["batch_size"])
    work_chunk_size = int(spec["work_chunk_size"])
    if expected_count <= 0:
        raise ValueError("dense worker expected_count must be positive")
    if batch_size <= 0 or work_chunk_size <= 0:
        raise ValueError("dense worker batch and work chunk sizes must be positive")

    started_at = time.monotonic()
    base_status: dict[str, Any] = {
        "format_version": WORKER_SPEC_VERSION,
        "rank": rank,
        "pid": os.getpid(),
        "completed": 0,
        "total": expected_count,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    _atomic_write_status(
        status_path,
        {**base_status, "state": "starting"},
    )

    try:
        import numpy as np
        import torch
        from sentence_transformers import SentenceTransformer

        if not torch.cuda.is_available() or int(torch.cuda.device_count()) != 1:
            raise RuntimeError(
                "dense worker must see exactly one CUDA GPU; "
                f"found {torch.cuda.device_count()}"
            )
        torch.cuda.set_device(0)
        configured_threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
        torch.set_num_threads(max(1, configured_threads))
        gpu_name = str(torch.cuda.get_device_name(0))
        _atomic_write_status(
            status_path,
            {
                **base_status,
                "state": "loading_model",
                "gpu_name": gpu_name,
            },
        )

        load_started = time.monotonic()
        model = SentenceTransformer(
            str(spec["model_source"]),
            device="cuda:0",
            model_kwargs={"torch_dtype": _torch_dtype(torch, str(spec["dtype"]))},
        )
        model.max_seq_length = int(spec["max_length"])
        model.eval()
        model_load_seconds = time.monotonic() - load_started
        _atomic_write_status(
            status_path,
            {
                **base_status,
                "state": "encoding",
                "gpu_name": gpu_name,
                "model_load_seconds": round(model_load_seconds, 3),
            },
        )

        output: Any | None = None
        dimension: int | None = None
        completed = 0
        for block in _text_blocks(input_path, work_chunk_size):
            vectors = model.encode_document(
                block,
                device="cuda:0",
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=bool(spec["normalize_embeddings"]),
            )
            vectors = np.asarray(vectors, dtype=np.float32)
            if vectors.ndim != 2 or vectors.shape[0] != len(block):
                raise RuntimeError(
                    f"worker {rank} received invalid embedding shape {vectors.shape}"
                )
            if not np.isfinite(vectors).all():
                raise RuntimeError(f"worker {rank} produced NaN or infinite embeddings")

            if output is None:
                dimension = int(vectors.shape[1])
                output = np.lib.format.open_memmap(
                    partial_output_path,
                    mode="w+",
                    dtype=np.float32,
                    shape=(expected_count, dimension),
                )
            elif int(vectors.shape[1]) != dimension:
                raise RuntimeError(
                    f"worker {rank} embedding dimension changed from "
                    f"{dimension} to {vectors.shape[1]}"
                )

            next_completed = completed + len(block)
            if next_completed > expected_count:
                raise RuntimeError(
                    f"worker {rank} input has more than {expected_count} records"
                )
            output[completed:next_completed] = vectors
            output.flush()
            completed = next_completed
            _atomic_write_status(
                status_path,
                {
                    **base_status,
                    "state": "encoding",
                    "gpu_name": gpu_name,
                    "model_load_seconds": round(model_load_seconds, 3),
                    "completed": completed,
                    "dimension": dimension,
                },
            )

        if completed != expected_count or output is None or dimension is None:
            raise RuntimeError(
                f"worker {rank} encoded {completed} records; expected {expected_count}"
            )
        output.flush()
        del output
        partial_output_path.replace(output_path)
        _atomic_write_status(
            status_path,
            {
                **base_status,
                "state": "complete",
                "gpu_name": gpu_name,
                "model_load_seconds": round(model_load_seconds, 3),
                "elapsed_seconds": round(time.monotonic() - started_at, 3),
                "completed": completed,
                "dimension": dimension,
            },
        )
    except BaseException as exc:
        _atomic_write_status(
            status_path,
            {
                **base_status,
                "state": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "elapsed_seconds": round(time.monotonic() - started_at, 3),
            },
        )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m legal_ir.dense_worker",
        description="Internal isolated dense embedding GPU worker",
    )
    parser.add_argument("--spec", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_worker(args.spec)
    return 0


if __name__ == "__main__":  # pragma: no cover - subprocess entry point
    raise SystemExit(main())
