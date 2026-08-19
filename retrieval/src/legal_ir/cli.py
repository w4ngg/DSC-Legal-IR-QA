from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from .config import PipelineConfig
from .hyde import (
    JsonlCachedHyDEGenerator,
    QwenHyDEGenerator,
    hyde_cache_namespace,
)
from .indexing import build_indexes, load_indexes
from .io import load_questions, write_diagnostics, write_submission
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


def _pipeline(
    args: argparse.Namespace, config: PipelineConfig
) -> RetrievalPipeline:
    bundle = load_indexes(args.index_dir, config)
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


def _search_command(args: argparse.Namespace) -> int:
    config = _runtime_config(args)
    pipeline = _pipeline(args, config)
    questions = load_questions(args.queries)
    responses = {}
    for position, (query_id, question) in enumerate(questions.items(), start=1):
        responses[query_id] = pipeline.search(question)
        if position == 1 or position % 10 == 0 or position == len(questions):
            LOGGER.info("Retrieved %d/%d queries", position, len(questions))
    write_submission(responses, args.output)
    if args.diagnostics:
        write_diagnostics(responses, args.diagnostics)
    return 0


def _search_one_command(args: argparse.Namespace) -> int:
    config = _runtime_config(args)
    response = _pipeline(args, config).search(args.query)
    print(json.dumps(response.to_dict(), ensure_ascii=False, indent=2))
    return 0


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--index-dir", required=True, type=Path)
    parser.add_argument("--config", type=str)
    parser.add_argument("--disable-hyde", action="store_true")
    parser.add_argument("--disable-reranker", action="store_true")
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
