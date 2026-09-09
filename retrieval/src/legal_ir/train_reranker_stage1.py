from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import math
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .config import PipelineConfig


LOGGER = logging.getLogger(__name__)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True, slots=True)
class TrainingGroup:
    query_id: str
    query: str
    passages: tuple[str, ...]
    chunk_ids: tuple[str, ...]
    document_ids: tuple[str, ...]


def _load_training_groups(
    path: str | Path,
    *,
    expected_negative_count: int,
) -> list[TrainingGroup]:
    source = Path(path)
    groups: list[TrainingGroup] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("row must be an object")
                if int(row.get("format_version", -1)) != 1:
                    raise ValueError("unsupported format_version")
                query_id = _required_string(row.get("query_id"), "query_id")
                query = _required_string(row.get("query"), "query")
                positive = row.get("positive")
                negatives = row.get("negatives")
                gold_ids = row.get("gold_document_ids")
                if not isinstance(positive, dict):
                    raise ValueError("positive must be an object")
                if not isinstance(negatives, list):
                    raise ValueError("negatives must be an array")
                if len(negatives) != expected_negative_count:
                    raise ValueError(
                        f"expected {expected_negative_count} negatives, got "
                        f"{len(negatives)}"
                    )
                if not isinstance(gold_ids, list) or not gold_ids:
                    raise ValueError("gold_document_ids must be a non-empty array")
                normalized_gold_ids = {
                    _required_string(item, "gold_document_ids[]") for item in gold_ids
                }
                records = [positive, *negatives]
                passages = tuple(
                    _required_string(item.get("text"), "candidate.text")
                    for item in records
                )
                chunk_ids = tuple(
                    _required_string(item.get("chunk_id"), "candidate.chunk_id")
                    for item in records
                )
                document_ids = tuple(
                    _required_string(
                        item.get("document_id"),
                        "candidate.document_id",
                    )
                    for item in records
                )
                if document_ids[0] not in normalized_gold_ids:
                    raise ValueError("positive document is not in gold_document_ids")
                if any(item in normalized_gold_ids for item in document_ids[1:]):
                    raise ValueError("a negative belongs to a gold document")
                if len(set(document_ids[1:])) != len(document_ids[1:]):
                    raise ValueError("negative documents must be distinct within a group")
                if len(set(chunk_ids)) != len(chunk_ids):
                    raise ValueError("candidate chunk IDs must be distinct within a group")
                groups.append(
                    TrainingGroup(
                        query_id=query_id,
                        query=query,
                        passages=passages,
                        chunk_ids=chunk_ids,
                        document_ids=document_ids,
                    )
                )
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid Stage 1 training row at {source}:{line_number}: {exc}"
                ) from exc
    if not groups:
        raise ValueError("Stage 1 training dataset is empty")
    return groups


def _load_and_validate_data_manifest(
    dataset_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    try:
        if manifest["artifact_type"] != "stage1_grouped_reranker_training_data":
            raise ValueError("unexpected artifact_type")
        dataset = manifest["dataset"]
        expected_filename = str(dataset["filename"])
        expected_sha256 = str(dataset["sha256"])
        expected_groups = int(dataset["group_count"])
        group_size = int(dataset["group_size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid Stage 1 manifest: {manifest_path}") from exc
    if dataset_path.name != expected_filename:
        raise ValueError(
            f"dataset filename is {dataset_path.name!r}, expected "
            f"{expected_filename!r} from the manifest"
        )
    if _sha256_file(dataset_path) != expected_sha256:
        raise ValueError("Stage 1 dataset SHA-256 differs from its manifest")
    if expected_groups <= 0 or group_size < 2:
        raise ValueError("invalid group_count or group_size in Stage 1 manifest")
    return {
        "raw": manifest,
        "group_count": expected_groups,
        "group_size": group_size,
    }


class GroupCollator:
    def __init__(self, tokenizer: Any, *, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, groups: Sequence[TrainingGroup]) -> dict[str, Any]:
        import torch

        queries: list[str] = []
        passages: list[str] = []
        group_size = len(groups[0].passages)
        for group in groups:
            if len(group.passages) != group_size:
                raise ValueError("all training groups in a batch must have equal size")
            queries.extend(group.query for _ in group.passages)
            passages.extend(group.passages)
        encoded = self.tokenizer(
            queries,
            passages,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded["group_count"] = len(groups)
        encoded["group_size"] = group_size
        encoded["labels"] = torch.zeros(len(groups), dtype=torch.long)
        return encoded


@dataclass(frozen=True, slots=True)
class TrainRuntime:
    distributed: bool
    rank: int
    local_rank: int
    world_size: int
    device: Any
    is_main: bool


def _initialize_runtime() -> TrainRuntime:
    import torch
    import torch.distributed as dist

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}" if distributed else "cuda:0")
        torch.cuda.set_device(device)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    if distributed and not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
    return TrainRuntime(
        distributed=distributed,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        is_main=rank == 0,
    )


def _set_seed(seed: int, *, rank: int) -> None:
    import numpy as np
    import torch

    process_seed = seed + rank
    random.seed(process_seed)
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(process_seed)


def _optimizer_groups(model: Any, *, weight_decay: float) -> list[dict[str, Any]]:
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    decay_parameters = []
    no_decay_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if any(token in name for token in no_decay):
            no_decay_parameters.append(parameter)
        else:
            decay_parameters.append(parameter)
    return [
        {"params": decay_parameters, "weight_decay": weight_decay},
        {"params": no_decay_parameters, "weight_decay": 0.0},
    ]


def _checkpoint_directories(output_dir: Path) -> list[Path]:
    values: list[tuple[int, Path]] = []
    for path in output_dir.glob("checkpoint-step-*"):
        try:
            step = int(path.name.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            continue
        if path.is_dir():
            values.append((step, path))
    return [path for _, path in sorted(values)]


def _save_checkpoint(
    *,
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    output_dir: Path,
    global_step: int,
    epoch: int,
    args: argparse.Namespace,
    runtime: TrainRuntime,
    dataset_sha256: str,
) -> Path:
    import torch
    import torch.distributed as dist

    if runtime.distributed:
        dist.barrier()
    destination = output_dir / f"checkpoint-step-{global_step}"
    if runtime.is_main:
        temporary = output_dir / f".{destination.name}.tmp"
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True, exist_ok=False)
        unwrapped = model.module if hasattr(model, "module") else model
        unwrapped.save_pretrained(temporary, safe_serialization=True)
        tokenizer.save_pretrained(temporary)
        metadata = {
            "format_version": 1,
            "artifact_type": "stage1_reranker_checkpoint",
            "global_step": global_step,
            "completed_epoch": epoch,
            "base_model": args.model_name,
            "base_revision": args.revision,
            "training_dataset": str(Path(args.train_data)),
            "training_dataset_sha256": dataset_sha256,
            "objective": "grouped_listwise_cross_entropy_positive_at_index_zero",
            "training_arguments": {
                key: value
                for key, value in vars(args).items()
                if isinstance(value, (str, int, float, bool, type(None)))
            },
        }
        with (temporary / "stage1_training_manifest.json").open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        if args.save_optimizer_state:
            torch.save(
                {
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict() if scaler is not None else None,
                    "global_step": global_step,
                    "completed_epoch": epoch,
                },
                temporary / "trainer_state.pt",
            )
        if destination.exists():
            shutil.rmtree(destination)
        temporary.replace(destination)

        checkpoints = _checkpoint_directories(output_dir)
        for expired in checkpoints[: -args.save_total_limit]:
            if expired != destination:
                shutil.rmtree(expired)
        (output_dir / "LAST_CHECKPOINT.txt").write_text(
            destination.name + "\n",
            encoding="utf-8",
        )
    if runtime.distributed:
        dist.barrier()
    return destination


def train(args: argparse.Namespace) -> Path:
    import torch
    import torch.distributed as dist
    import torch.nn.functional as functional
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader, DistributedSampler
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    runtime = _initialize_runtime()
    if runtime.is_main:
        logging.basicConfig(
            level=logging.DEBUG if args.verbose else logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
        )
    _set_seed(args.seed, rank=runtime.rank)

    train_data = Path(args.train_data)
    data_manifest_path = Path(args.data_manifest)
    output_dir = Path(args.output_dir)
    data_manifest = _load_and_validate_data_manifest(
        train_data,
        data_manifest_path,
    )
    expected_negative_count = data_manifest["group_size"] - 1
    groups = _load_training_groups(
        train_data,
        expected_negative_count=expected_negative_count,
    )
    if len(groups) != data_manifest["group_count"]:
        raise ValueError(
            f"loaded {len(groups)} groups, manifest declares "
            f"{data_manifest['group_count']}"
        )
    if args.max_groups is not None:
        groups = groups[: args.max_groups]
    if runtime.is_main:
        LOGGER.info(
            "Training groups=%d group_size=%d world_size=%d",
            len(groups),
            data_manifest["group_size"],
            runtime.world_size,
        )

    existing_checkpoints = _checkpoint_directories(output_dir)
    if existing_checkpoints and not args.overwrite_output_dir:
        raise FileExistsError(
            "checkpoint directories already exist; choose another --output-dir "
            "or pass --overwrite-output-dir"
        )
    if runtime.is_main and args.overwrite_output_dir:
        for checkpoint in existing_checkpoints:
            shutil.rmtree(checkpoint)
        (output_dir / "LAST_CHECKPOINT.txt").unlink(missing_ok=True)
    if runtime.distributed:
        dist.barrier()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        revision=args.revision,
        use_fast=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        revision=args.revision,
    )
    if int(getattr(model.config, "num_labels", -1)) != 1:
        raise ValueError(
            "reranker checkpoint must return exactly one raw logit per pair"
        )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    model.to(runtime.device)
    if runtime.distributed:
        model = DistributedDataParallel(
            model,
            device_ids=(
                [runtime.local_rank] if runtime.device.type == "cuda" else None
            ),
            output_device=(
                runtime.local_rank if runtime.device.type == "cuda" else None
            ),
            find_unused_parameters=False,
        )

    sampler = (
        DistributedSampler(
            groups,
            num_replicas=runtime.world_size,
            rank=runtime.rank,
            shuffle=True,
            seed=args.seed,
            drop_last=False,
        )
        if runtime.distributed
        else None
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        groups,
        batch_size=args.per_device_group_batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.dataloader_num_workers,
        pin_memory=runtime.device.type == "cuda",
        collate_fn=GroupCollator(tokenizer, max_length=args.max_length),
        generator=generator,
    )
    updates_per_epoch = math.ceil(len(loader) / args.gradient_accumulation_steps)
    total_updates = updates_per_epoch * args.epochs
    warmup_steps = int(total_updates * args.warmup_ratio)
    optimizer = torch.optim.AdamW(
        _optimizer_groups(model, weight_decay=args.weight_decay),
        lr=args.learning_rate,
        eps=args.adam_epsilon,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )
    use_fp16 = runtime.device.type == "cuda" and args.mixed_precision == "fp16"
    use_bf16 = runtime.device.type == "cuda" and args.mixed_precision == "bf16"
    if use_bf16 and not torch.cuda.is_bf16_supported():
        raise ValueError("bf16 was requested but the CUDA device does not support it")
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    autocast_dtype = torch.float16 if use_fp16 else torch.bfloat16

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_sha256 = _sha256_file(train_data)
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    final_checkpoint: Path | None = None
    for epoch_index in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch_index)
        model.train()
        accumulated_loss = 0.0
        for micro_step, batch in enumerate(loader, start=1):
            group_count = int(batch.pop("group_count"))
            group_size = int(batch.pop("group_size"))
            labels = batch.pop("labels").to(runtime.device, non_blocking=True)
            batch = {
                key: value.to(runtime.device, non_blocking=True)
                for key, value in batch.items()
            }
            should_update = (
                micro_step % args.gradient_accumulation_steps == 0
                or micro_step == len(loader)
            )
            sync_context = (
                contextlib.nullcontext()
                if should_update or not runtime.distributed
                else model.no_sync()
            )
            with sync_context:
                accumulation_window_start = (
                    ((micro_step - 1) // args.gradient_accumulation_steps)
                    * args.gradient_accumulation_steps
                    + 1
                )
                accumulation_divisor = min(
                    args.gradient_accumulation_steps,
                    len(loader) - accumulation_window_start + 1,
                )
                with torch.autocast(
                    device_type=runtime.device.type,
                    dtype=autocast_dtype,
                    enabled=use_fp16 or use_bf16,
                ):
                    outputs = model(**batch)
                    logits = outputs.logits
                    if logits.ndim != 2 or logits.shape[1] != 1:
                        raise ValueError(
                            f"reranker produced logits with shape {tuple(logits.shape)}"
                        )
                    grouped_logits = logits.reshape(group_count, group_size)
                    loss = functional.cross_entropy(grouped_logits, labels)
                    scaled_loss = loss / accumulation_divisor
                scaler.scale(scaled_loss).backward()
            accumulated_loss += float(loss.detach())

            if not should_update:
                continue
            if args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if runtime.is_main and (
                global_step % args.log_every_steps == 0 or global_step == 1
            ):
                LOGGER.info(
                    "epoch=%d step=%d/%d loss=%.6f lr=%.3e",
                    epoch_index + 1,
                    global_step,
                    total_updates,
                    accumulated_loss / max(1, micro_step),
                    scheduler.get_last_lr()[0],
                )

            if args.save_steps > 0 and global_step % args.save_steps == 0:
                final_checkpoint = _save_checkpoint(
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    output_dir=output_dir,
                    global_step=global_step,
                    epoch=epoch_index + 1,
                    args=args,
                    runtime=runtime,
                    dataset_sha256=dataset_sha256,
                )

        final_checkpoint = _save_checkpoint(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            output_dir=output_dir,
            global_step=global_step,
            epoch=epoch_index + 1,
            args=args,
            runtime=runtime,
            dataset_sha256=dataset_sha256,
        )

    if runtime.distributed:
        dist.barrier()
        dist.destroy_process_group()
    assert final_checkpoint is not None
    if runtime.is_main:
        print(f"CHECKPOINT={final_checkpoint}")
    return final_checkpoint


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune Stage 1 reranker with grouped listwise cross-entropy"
    )
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--data-manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name")
    parser.add_argument("--revision")
    parser.add_argument("--epochs", type=_positive_int, default=1)
    parser.add_argument("--per-device-group-batch-size", type=_positive_int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=_positive_int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=_positive_int)
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="fp16")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dataloader-num-workers", type=_non_negative_int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--save-steps", type=_non_negative_int, default=0)
    parser.add_argument("--save-total-limit", type=_positive_int, default=1)
    parser.add_argument("--save-optimizer-state", action="store_true")
    parser.add_argument("--log-every-steps", type=_positive_int, default=25)
    parser.add_argument("--max-groups", type=_positive_int)
    parser.add_argument("--overwrite-output-dir", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = PipelineConfig.from_yaml(args.config)
    if not config.reranker.enabled:
        raise SystemExit("the supplied config has reranker.enabled=false")
    if args.model_name is None:
        args.model_name = config.reranker.model_name
        args.revision = config.reranker.revision
    elif args.revision is None:
        LOGGER.warning("custom --model-name supplied without a pinned --revision")
    if args.max_length is None:
        args.max_length = config.reranker.max_length
    for name in (
        "learning_rate",
        "adam_epsilon",
        "max_grad_norm",
    ):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.weight_decay < 0:
        raise SystemExit("--weight-decay must be non-negative")
    if not 0 <= args.warmup_ratio < 1:
        raise SystemExit("--warmup-ratio must be in [0, 1)")
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
