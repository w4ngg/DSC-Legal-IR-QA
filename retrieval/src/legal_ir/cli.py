from __future__ import annotations

import argparse
import hashlib
import json
import logging
from contextlib import nullcontext
from dataclasses import asdict, replace
from pathlib import Path
from typing import Sequence

from .config import PipelineConfig
from .dual_rerank import DualChunkAssets
from .hyde import (
    JsonlCachedHyDEGenerator,
    QwenHyDEGenerator,
    hyde_cache_namespace,
)
from .indexing import build_indexes, load_indexes
from .io import (
    ChunkStore,
    DeepDiagnosticsWriter,
    load_questions,
    write_diagnostics,
    write_submission,
)
from .pipeline import RetrievalPipeline
from .reranker import VietnameseCrossEncoderReranker


LOGGER = logging.getLogger("legal_ir")


def _config(path: str | None) -> PipelineConfig:
    return PipelineConfig.from_yaml(path) if path else PipelineConfig()


def _runtime_config(args: argparse.Namespace) -> PipelineConfig:
    config = _config(args.config)
    if getattr(args, "disable_hyde", False):
        config = replace(config, hyde=replace(config.hyde, enabled=False))
    if getattr(args, "disable_reranker", False):
        config = replace(config, reranker=replace(config.reranker, enabled=False))
    return config


def _dual_output_spec(
    manifest: dict,
    output_name: str,
) -> tuple[int, str]:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("dual chunk manifest is missing outputs")
    output = outputs.get(output_name)
    if not isinstance(output, dict):
        raise ValueError(
            f"dual chunk manifest is missing outputs.{output_name}"
        )
    records = output.get("records")
    if isinstance(records, bool) or not isinstance(records, int) or records < 0:
        raise ValueError(
            f"dual chunk manifest outputs.{output_name}.records is invalid"
        )
    sha256 = output.get("sha256")
    if (
        not isinstance(sha256, str)
        or len(sha256) != 64
        or any(character not in "0123456789abcdef" for character in sha256)
    ):
        raise ValueError(
            f"dual chunk manifest outputs.{output_name}.sha256 is invalid"
        )
    return records, sha256


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_embedded_short_mappings(
    mapping_path: Path,
    short_chunks: ChunkStore,
) -> tuple[int, str]:
    """Stream-compare standalone mappings with index metadata.

    The validation intentionally retains no mapping rows. This keeps startup
    memory bounded even for the roughly two-million-short-chunk dual index.
    """

    digest = hashlib.sha256()
    record_count = 0
    with mapping_path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            digest.update(raw_line)
            if not raw_line.strip():
                continue
            if record_count >= len(short_chunks):
                raise ValueError(
                    "short_to_long.jsonl contains more rows than indexed "
                    "short chunks"
                )
            try:
                record = json.loads(raw_line)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise ValueError(
                    f"invalid mapping at {mapping_path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"mapping at {mapping_path}:{line_number} must be an object"
                )

            short_chunk = short_chunks[record_count]
            raw_long_ids = record.get("long_chunk_ids")
            if (
                not isinstance(raw_long_ids, list)
                or not raw_long_ids
                or any(not isinstance(item, str) for item in raw_long_ids)
                or len(set(raw_long_ids)) != len(raw_long_ids)
            ):
                raise ValueError(
                    f"mapping at {mapping_path}:{line_number} has invalid "
                    "long_chunk_ids"
                )
            primary_long_id = record.get("primary_long_chunk_id")
            if (
                not isinstance(primary_long_id, str)
                or primary_long_id not in raw_long_ids
            ):
                raise ValueError(
                    f"mapping at {mapping_path}:{line_number} has invalid "
                    "primary_long_chunk_id"
                )

            metadata_long_ids = short_chunk.metadata.get("long_chunk_ids")
            if not isinstance(metadata_long_ids, (list, tuple)):
                raise ValueError(
                    f"indexed short chunk {short_chunk.chunk_id} has no valid "
                    "long_chunk_ids metadata"
                )
            expected = (
                short_chunk.chunk_id,
                short_chunk.document_id,
                short_chunk.metadata.get("primary_long_chunk_id"),
                tuple(metadata_long_ids),
            )
            observed = (
                record.get("short_chunk_id"),
                record.get("document_id"),
                primary_long_id,
                tuple(raw_long_ids),
            )
            if short_chunk.metadata.get("granularity") != "short":
                raise ValueError(
                    f"indexed chunk {short_chunk.chunk_id} is not a short chunk"
                )
            if observed != expected:
                raise ValueError(
                    "standalone mapping does not match indexed short chunk at "
                    f"row {record_count + 1}: {short_chunk.chunk_id}"
                )

            record_count += 1
            if record_count % 250_000 == 0:
                LOGGER.info(
                    "Validated %d/%d short-to-long mappings",
                    record_count,
                    len(short_chunks),
                )

    if record_count != len(short_chunks):
        raise ValueError(
            "short_to_long.jsonl and indexed short chunks have different row "
            f"counts ({record_count} != {len(short_chunks)})"
        )
    return record_count, digest.hexdigest()


def _load_dual_chunk_assets(
    args: argparse.Namespace,
    config: PipelineConfig,
    *,
    short_chunks: ChunkStore,
) -> DualChunkAssets | None:
    dual_chunks_dir = getattr(args, "dual_chunks_dir", None)
    if not config.long_context.enabled:
        if dual_chunks_dir is not None:
            LOGGER.warning(
                "Ignoring --dual-chunks-dir because long_context.enabled is false"
            )
        return None
    if dual_chunks_dir is None:
        raise ValueError(
            "--dual-chunks-dir is required when long_context.enabled is true"
        )

    source = Path(dual_chunks_dir)
    manifest_path = source / "manifest.json"
    long_chunks_path = source / "long_chunks.jsonl"
    mapping_path = source / "short_to_long.jsonl"
    missing = [
        path
        for path in (manifest_path, long_chunks_path, mapping_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "dual chunk directory is incomplete; missing: "
            + ", ".join(str(path) for path in missing)
        )

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError("dual chunk manifest must be a JSON object")
    if manifest.get("artifact_type") != "legal_ir_dual_granularity_chunks":
        raise ValueError(
            "unsupported dual chunk manifest artifact_type: "
            f"{manifest.get('artifact_type')!r}"
        )
    if manifest.get("format_version") != 1:
        raise ValueError(
            "unsupported dual chunk manifest format_version: "
            f"{manifest.get('format_version')!r}"
        )
    statistics = manifest.get("statistics")
    if not isinstance(statistics, dict):
        raise ValueError("dual chunk manifest is missing statistics")
    expected_short_count = int(statistics.get("short_chunks_written", -1))
    expected_mapping_count = int(statistics.get("mappings_written", -1))
    output_mapping_count, expected_mapping_sha256 = _dual_output_spec(
        manifest,
        "short_to_long",
    )
    output_long_count, expected_long_sha256 = _dual_output_spec(
        manifest,
        "long_chunks",
    )
    if expected_short_count != len(short_chunks):
        raise ValueError(
            "dual chunk manifest and indexed short chunks have different row "
            f"counts ({expected_short_count} != {len(short_chunks)})"
        )
    if expected_mapping_count != expected_short_count:
        raise ValueError(
            "dual chunk manifest does not contain exactly one mapping per "
            "short chunk"
        )
    if output_mapping_count != expected_mapping_count:
        raise ValueError(
            "dual chunk manifest mapping counts are inconsistent"
        )

    LOGGER.info(
        "Validating %d standalone short-to-long mappings against the index",
        expected_mapping_count,
    )
    actual_mapping_count, actual_mapping_sha256 = (
        _validate_embedded_short_mappings(mapping_path, short_chunks)
    )
    if actual_mapping_count != expected_mapping_count:
        raise ValueError(
            "dual chunk manifest and short_to_long.jsonl have different row "
            f"counts ({expected_mapping_count} != {actual_mapping_count})"
        )
    if actual_mapping_sha256 != expected_mapping_sha256:
        raise ValueError(
            "short_to_long.jsonl SHA-256 does not match the dual manifest"
        )

    expected_long_count = int(statistics.get("long_chunks_written", -1))
    if output_long_count != expected_long_count:
        raise ValueError("dual chunk manifest long chunk counts are inconsistent")
    LOGGER.info("Validating long_chunks.jsonl SHA-256")
    if _file_sha256(long_chunks_path) != expected_long_sha256:
        raise ValueError(
            "long_chunks.jsonl SHA-256 does not match the dual manifest"
        )

    # The index copy of short_chunks.jsonl already retains long_chunk_ids in
    # metadata. Reusing it avoids materializing a second ~2M-row mapping dict;
    # the standalone mapping file is still required as part of the portable
    # dual artifact and for offline audits.
    LOGGER.info("Loading long chunk store from %s", long_chunks_path)
    assets = DualChunkAssets.from_indexed_short_chunks(
        short_chunks=short_chunks,
        long_chunks_path=long_chunks_path,
    )
    if expected_long_count != len(assets.long_chunks):
        raise ValueError(
            "dual chunk manifest and long_chunks.jsonl have different row "
            f"counts ({expected_long_count} != {len(assets.long_chunks)})"
        )
    # Fail at startup for the most common wrong-dataset case. Every retrieved
    # candidate is still validated lazily during search.
    sample_positions = {0, len(short_chunks) - 1}
    for position in sorted(sample_positions):
        short_chunk = short_chunks[position]
        mapping = assets.mappings.get(short_chunk.chunk_id)
        assets.validate_mapping(
            mapping,
            expected_document_id=short_chunk.document_id,
        )
    LOGGER.info(
        "Loaded dual chunks: %d indexed short chunks, %d long chunks",
        len(short_chunks),
        len(assets.long_chunks),
    )
    LOGGER.info(
        "Long-context mode: BM25@%d + dense@%d%s; candidate pool=%s; "
        "reranker top-%d long -> MaxP -> top-%d documents",
        config.bm25.top_k_chunks,
        config.dense.top_k_chunks,
        (
            f" + HyDE@{config.hyde.top_k_chunks}"
            if config.hyde.enabled
            else ""
        ),
        (
            "full"
            if config.long_context.candidate_mode == "full"
            else f"top-{config.long_context.candidate_top_k}"
        ),
        config.long_context.rerank_top_k_chunks,
        config.reranker.final_top_k_documents,
    )
    return assets


def _pipeline(
    args: argparse.Namespace, config: PipelineConfig
) -> RetrievalPipeline:
    bundle = load_indexes(args.index_dir, config)
    dual_chunk_assets = _load_dual_chunk_assets(
        args,
        config,
        short_chunks=bundle.chunks,
    )
    hyde_generator = None
    if config.hyde.enabled:
        hyde_generator = QwenHyDEGenerator(config.hyde)
        if getattr(args, "hyde_cache", None):
            hyde_generator = JsonlCachedHyDEGenerator(
                hyde_generator,
                args.hyde_cache,
                namespace=hyde_cache_namespace(config.hyde),
            )
    reranker = (
        VietnameseCrossEncoderReranker(config.reranker)
        if config.reranker.enabled
        else None
    )
    return RetrievalPipeline(
        chunks=bundle.chunks,
        bm25=bundle.bm25,
        dense=bundle.dense,
        hyde_generator=hyde_generator,
        reranker=reranker,
        dual_chunk_assets=dual_chunk_assets,
        config=config,
    )


def _build_command(args: argparse.Namespace) -> int:
    config = _config(args.config)
    bundle = build_indexes(
        args.chunks,
        args.index_dir,
        config,
        overwrite=args.overwrite,
    )
    LOGGER.info("Built BM25 and dense indexes for %d chunks", len(bundle.chunks))
    return 0


def _validate_search_output_paths(args: argparse.Namespace) -> None:
    paths = [("output", Path(args.output))]
    if args.diagnostics:
        paths.append(("diagnostics", Path(args.diagnostics)))
    if getattr(args, "deep_diagnostics", None):
        paths.append(("deep-diagnostics", Path(args.deep_diagnostics)))

    seen: dict[Path, str] = {}
    for label, path in paths:
        resolved = path.expanduser().resolve(strict=False)
        previous = seen.get(resolved)
        if previous is not None:
            raise ValueError(
                f"--{label} and --{previous} must use different output paths"
            )
        seen[resolved] = label


def _search_command(args: argparse.Namespace) -> int:
    _validate_search_output_paths(args)
    config = _runtime_config(args)
    pipeline = _pipeline(args, config)
    try:
        questions = load_questions(args.queries)
        responses = {}
        deep_path = getattr(args, "deep_diagnostics", None)
        writer_context = (
            DeepDiagnosticsWriter(deep_path, pipeline_config=asdict(config))
            if deep_path
            else nullcontext(None)
        )
        with writer_context as deep_writer:
            for position, (query_id, question) in enumerate(
                questions.items(), start=1
            ):
                if deep_writer is None:
                    response = pipeline.search(question)
                else:
                    response, deep_diagnostics = (
                        pipeline.search_with_deep_diagnostics(question)
                    )
                    deep_writer.write(query_id, deep_diagnostics)
                responses[query_id] = response
                if (
                    position == 1
                    or position % 10 == 0
                    or position == len(questions)
                ):
                    LOGGER.info(
                        "Retrieved %d/%d queries", position, len(questions)
                    )
        write_submission(responses, args.output)
        if args.diagnostics:
            write_diagnostics(responses, args.diagnostics)
        return 0
    finally:
        pipeline.close()


def _search_one_command(args: argparse.Namespace) -> int:
    config = _runtime_config(args)
    pipeline = _pipeline(args, config)
    try:
        response = pipeline.search(args.query)
        print(json.dumps(response.to_dict(), ensure_ascii=False, indent=2))
        return 0
    finally:
        pipeline.close()


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--index-dir", required=True, type=Path)
    parser.add_argument("--config", type=str)
    parser.add_argument("--disable-hyde", action="store_true")
    parser.add_argument("--disable-reranker", action="store_true")
    parser.add_argument(
        "--dual-chunks-dir",
        type=Path,
        help=(
            "directory containing manifest.json, long_chunks.jsonl, and "
            "short_to_long.jsonl; required by long-context mode"
        ),
    )
    parser.add_argument(
        "--hyde-cache",
        type=Path,
        help="optional append-only JSONL cache for SLM generations",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="legal-ir",
        description="DSC LegalIR hybrid retrieval pipeline",
    )
    parser.add_argument("--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-index", help="build BM25 and dense indexes")
    build.add_argument("--chunks", required=True, type=Path)
    build.add_argument("--index-dir", required=True, type=Path)
    build.add_argument("--config", type=str)
    build.add_argument("--overwrite", action="store_true")
    build.set_defaults(handler=_build_command)

    search = subparsers.add_parser("search", help="retrieve an official query JSON file")
    _add_runtime_arguments(search)
    search.add_argument("--queries", required=True, type=Path)
    search.add_argument("--output", required=True, type=Path)
    search.add_argument("--diagnostics", type=Path)
    search.add_argument(
        "--deep-diagnostics",
        type=Path,
        help="optional full pre-fusion BM25/dense/HyDE trace JSON",
    )
    search.set_defaults(handler=_search_command)

    one = subparsers.add_parser("search-one", help="retrieve one query")
    _add_runtime_arguments(one)
    one.add_argument("--query", required=True)
    one.set_defaults(handler=_search_one_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    return int(args.handler(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
