from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Sequence


WORKER_SPEC_VERSION = 1
TASK_VERSION = 1


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


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


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _load_spec(path: Path) -> dict[str, Any]:
    spec = _load_json_object(path, label="reranker worker spec")
    if spec.get("format_version") != WORKER_SPEC_VERSION:
        raise ValueError(
            "unsupported reranker worker spec version: "
            f"{spec.get('format_version')}"
        )
    return spec


def _load_task(path: Path) -> tuple[str, list[str]]:
    task = _load_json_object(path, label="reranker task")
    if task.get("format_version") != TASK_VERSION:
        raise ValueError(
            f"unsupported reranker task version: {task.get('format_version')}"
        )
    query = task.get("query")
    passages = task.get("passages")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("reranker task query must be a non-empty string")
    if not isinstance(passages, list) or not passages:
        raise ValueError("reranker task passages must be a non-empty list")
    if any(not isinstance(passage, str) for passage in passages):
        raise ValueError("every reranker task passage must be a string")
    return query, passages


def run_worker(spec_path: str | Path) -> None:
    spec = _load_spec(Path(spec_path))
    status_path = Path(spec["status_path"])
    rank = int(spec["rank"])
    base_status: dict[str, Any] = {
        "format_version": WORKER_SPEC_VERSION,
        "rank": rank,
        "pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    started_at = time.monotonic()
    _atomic_write_json(status_path, {**base_status, "state": "starting"})

    try:
        import numpy as np
        import torch
        from sentence_transformers import CrossEncoder

        if not torch.cuda.is_available() or int(torch.cuda.device_count()) != 1:
            raise RuntimeError(
                "reranker worker must see exactly one CUDA GPU; "
                f"found {torch.cuda.device_count()}"
            )
        torch.cuda.set_device(0)
        configured_threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
        torch.set_num_threads(max(1, configured_threads))
        gpu_name = str(torch.cuda.get_device_name(0))
        _atomic_write_json(
            status_path,
            {**base_status, "state": "loading_model", "gpu_name": gpu_name},
        )

        load_started = time.monotonic()
        model = CrossEncoder(
            str(spec["model_source"]),
            device="cuda:0",
            model_kwargs={
                "torch_dtype": _torch_dtype(torch, str(spec["dtype"]))
            },
            max_length=int(spec["max_length"]),
            activation_fn=torch.nn.Identity(),
        )
        model_load_seconds = time.monotonic() - load_started
        ready_status = {
            **base_status,
            "state": "ready",
            "gpu_name": gpu_name,
            "model_load_seconds": round(model_load_seconds, 3),
        }
        _atomic_write_json(status_path, ready_status)

        for raw_command in sys.stdin:
            raw_command = raw_command.strip()
            if not raw_command:
                continue
            command = json.loads(raw_command)
            if not isinstance(command, dict):
                raise ValueError("reranker worker command must be a JSON object")
            action = command.get("command")
            if action == "shutdown":
                _atomic_write_json(
                    status_path,
                    {
                        **ready_status,
                        "state": "stopped",
                        "elapsed_seconds": round(
                            time.monotonic() - started_at, 3
                        ),
                    },
                )
                return
            if action != "score":
                raise ValueError(f"unsupported reranker worker command: {action!r}")

            task_id = int(command["task_id"])
            input_path = Path(command["input_path"])
            output_path = Path(command["output_path"])
            query, passages = _load_task(input_path)
            _atomic_write_json(
                status_path,
                {
                    **ready_status,
                    "state": "scoring",
                    "task_id": task_id,
                    "total": len(passages),
                    "completed": 0,
                },
            )

            pairs = [(query, passage) for passage in passages]
            raw_scores = model.predict(
                pairs,
                batch_size=int(spec["batch_size"]),
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            scores = [
                float(value) for value in np.asarray(raw_scores).reshape(-1)
            ]
            if len(scores) != len(passages):
                raise RuntimeError(
                    f"reranker worker {rank} returned {len(scores)} scores "
                    f"for {len(passages)} passages"
                )
            _atomic_write_json(
                output_path,
                {
                    "format_version": TASK_VERSION,
                    "task_id": task_id,
                    "scores": scores,
                },
            )
            _atomic_write_json(
                status_path,
                {
                    **ready_status,
                    "state": "ready",
                    "task_id": task_id,
                    "total": len(passages),
                    "completed": len(passages),
                },
            )
    except BaseException as exc:
        _atomic_write_json(
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
        prog="python -m legal_ir.reranker_worker",
        description="Internal persistent one-GPU reranker worker",
    )
    parser.add_argument("--spec", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_worker(args.spec)
    return 0


if __name__ == "__main__":  # pragma: no cover - subprocess entry point
    raise SystemExit(main())
