from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import math
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TextIO

from .config import PipelineConfig
from .dense import FaissDenseIndex
from .dual_rerank import (
    DualChunkAssets,
    LongCandidate,
    ShortChunkMetadataLookup,
    build_long_candidate_pool,
)
from .fusion import rank_channels
from .indexing import INDEX_FORMAT_VERSION, _chunk_records_hash, load_indexes
from .io import ChunkStore
from .reranker import VietnameseCrossEncoderReranker


LOGGER = logging.getLogger(__name__)
DATASET_FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class GoldQuery:
    query_id: str
    question: str
    document_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DenseLongHit:
    chunk_id: str
    document_id: str
    score: float
    rank: int


@dataclass(frozen=True, slots=True)
class LongDenseMiningResult:
    positive_shortlists: Mapping[str, tuple[DenseLongHit, ...]]
    global_hits: tuple[DenseLongHit, ...]


@dataclass(slots=True)
class _ScoredCandidate:
    chunk_id: str
    document_id: str
    teacher_score: float
    sources: set[str]
    mapped_retrieval_rank: int | None = None
    retrieval_support_score: float | None = None
    long_dense_rank: int | None = None
    long_dense_score: float | None = None


@dataclass(frozen=True, slots=True)
class NegativeDocument:
    chunk_id: str
    document_id: str
    teacher_score: float
    sources: tuple[str, ...]
    mapped_retrieval_rank: int | None
    retrieval_support_score: float | None
    long_dense_rank: int | None
    long_dense_score: float | None


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_gold(path: str | Path) -> list[GoldQuery]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not payload:
        raise ValueError("gold must be a non-empty JSON object keyed by query ID")

    records: list[GoldQuery] = []
    for raw_query_id, raw_record in payload.items():
        query_id = str(raw_query_id).strip()
        if not query_id or not isinstance(raw_record, dict):
            raise ValueError("every gold query must have a non-empty ID and object value")
        question = raw_record.get("question")
        answers = raw_record.get("answer")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"gold query {query_id}.question must be non-empty")
        if not isinstance(answers, list) or not answers:
            raise ValueError(f"gold query {query_id}.answer must be a non-empty array")
        document_ids = tuple(str(item).strip() for item in answers)
        if any(not item for item in document_ids):
            raise ValueError(f"gold query {query_id}.answer contains an empty ID")
        if len(set(document_ids)) != len(document_ids):
            raise ValueError(f"gold query {query_id}.answer contains duplicate IDs")
        records.append(
            GoldQuery(
                query_id=query_id,
                question=question.strip(),
                document_ids=document_ids,
            )
        )
    return records


def _verify_training_split(
    gold_path: Path,
    records: Sequence[GoldQuery],
    *,
    split_manifest_path: Path | None,
    allow_unverified: bool,
) -> dict[str, Any]:
    manifest_path = split_manifest_path or gold_path.parent / "split_manifest.json"
    if not manifest_path.is_file():
        if allow_unverified:
            LOGGER.warning(
                "No split manifest found; proceeding because "
                "--allow-unverified-training-data was supplied"
            )
            return {"verified": False, "manifest": None}
        raise FileNotFoundError(
            f"split manifest not found: {manifest_path}. Stage 1 must use the "
            "train split, not val/test or the unsplit source. Supply "
            "--split-manifest or explicitly pass --allow-unverified-training-data."
        )

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    try:
        train = manifest["outputs"]["train"]
        expected_name = str(train["filename"])
        expected_count = int(train["query_count"])
        expected_hash = str(train["sha256"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid split manifest schema: {manifest_path}") from exc
    if gold_path.name != expected_name:
        raise ValueError(
            f"gold file is {gold_path.name!r}, but split manifest identifies "
            f"{expected_name!r} as the train split"
        )
    if len(records) != expected_count:
        raise ValueError(
            f"train query count is {len(records)}, expected {expected_count}"
        )
    actual_hash = _sha256_file(gold_path)
    if actual_hash != expected_hash:
        raise ValueError("gold train file SHA-256 differs from split_manifest.json")
    return {
        "verified": True,
        "manifest": str(manifest_path),
        "sha256": actual_hash,
        "query_count": len(records),
    }


def _expected_dense_manifest(config: PipelineConfig) -> dict[str, Any]:
    return {
        "model_name": config.dense.model_name,
        "revision": config.dense.revision,
        "max_length": config.dense.max_length,
        "dtype": config.dense.dtype,
        "normalize_embeddings": config.dense.normalize_embeddings,
        "index_type": config.dense.index_type,
        "hnsw_m": config.dense.hnsw_m,
        "hnsw_ef_construction": config.dense.hnsw_ef_construction,
    }


def _load_long_dense_index(
    index_dir: str | Path,
    config: PipelineConfig,
) -> tuple[ChunkStore, FaissDenseIndex, dict[str, Any]]:
    source = Path(index_dir)
    for artifact in (
        source / "manifest.json",
        source / "chunks.jsonl",
        source / "dense.faiss",
    ):
        if not artifact.is_file():
            raise FileNotFoundError(f"long index is missing: {artifact}")
    with (source / "manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("format_version") != INDEX_FORMAT_VERSION:
        raise ValueError("unsupported long index manifest format")

    chunks = ChunkStore.load_jsonl(
        source / "chunks.jsonl",
        compact_for_search=True,
    )
    if int(manifest.get("chunk_count", -1)) != len(chunks):
        raise ValueError("long index manifest and chunks.jsonl row counts differ")
    if manifest.get("chunk_records_sha256") != _chunk_records_hash(chunks):
        raise ValueError("long chunks or ordering do not match the index manifest")
    if manifest.get("dense") != _expected_dense_manifest(config):
        raise ValueError(
            "long dense index was built with a different dense config; use its "
            "original model/revision/max_length/dtype/index settings"
        )
    invalid = [
        chunk.chunk_id
        for chunk in chunks.chunks
        if chunk.metadata.get("granularity") not in (None, "long")
    ]
    if invalid:
        raise ValueError(f"long index contains non-long chunks, e.g. {invalid[0]}")
    return (
        chunks,
        FaissDenseIndex.load(chunks, source / "dense.faiss", config.dense),
        manifest,
    )


def _rank_restricted_dense_hits(
    *,
    query_vector: Any,
    document_id: str,
    positions: Sequence[int],
    corpus_vectors: Any,
    chunks: ChunkStore,
    top_k: int,
) -> tuple[DenseLongHit, ...]:
    import numpy as np

    if not positions:
        return ()
    row_ids = np.asarray(positions, dtype=np.int64)
    scores = corpus_vectors[row_ids] @ query_vector
    ordered = sorted(
        zip(positions, scores, strict=True),
        key=lambda item: (-float(item[1]), chunks[item[0]].chunk_id),
    )[:top_k]
    return tuple(
        DenseLongHit(
            chunk_id=chunks[position].chunk_id,
            document_id=document_id,
            score=float(score),
            rank=rank,
        )
        for rank, (position, score) in enumerate(ordered, start=1)
    )


def _precompute_long_dense_candidates(
    records: Sequence[GoldQuery],
    *,
    chunks: ChunkStore,
    dense: FaissDenseIndex,
    positive_top_k: int,
    global_top_k: int,
    query_batch_size: int,
    log_every: int,
) -> dict[str, LongDenseMiningResult]:
    document_positions: dict[str, list[int]] = defaultdict(list)
    for position, chunk in enumerate(chunks.chunks):
        document_positions[chunk.document_id].append(position)

    missing_documents = sorted(
        {
            document_id
            for record in records
            for document_id in record.document_ids
            if document_id not in document_positions
        }
    )
    if missing_documents:
        LOGGER.warning(
            "%d gold documents have no long chunks and their groups will be skipped",
            len(missing_documents),
        )

    LOGGER.info("Reconstructing long-index vectors for restricted positive mining")
    corpus_vectors = dense.reconstruct_all_vectors()
    results: dict[str, LongDenseMiningResult] = {}
    for batch_start in range(0, len(records), query_batch_size):
        batch = records[batch_start : batch_start + query_batch_size]
        vectors = dense.encode_queries(
            [record.question for record in batch],
            show_progress=False,
        )
        global_batches = dense.search_encoded(vectors, global_top_k)
        for record, vector, global_hits in zip(
            batch,
            vectors,
            global_batches,
            strict=True,
        ):
            positive_shortlists = {
                document_id: _rank_restricted_dense_hits(
                    query_vector=vector,
                    document_id=document_id,
                    positions=document_positions.get(document_id, ()),
                    corpus_vectors=corpus_vectors,
                    chunks=chunks,
                    top_k=positive_top_k,
                )
                for document_id in record.document_ids
            }
            results[record.query_id] = LongDenseMiningResult(
                positive_shortlists=positive_shortlists,
                global_hits=tuple(
                    DenseLongHit(
                        chunk_id=hit.chunk_id,
                        document_id=hit.document_id,
                        score=hit.score,
                        rank=rank,
                    )
                    for rank, hit in enumerate(global_hits, start=1)
                ),
            )
        processed = min(batch_start + len(batch), len(records))
        if processed % log_every == 0 or processed == len(records):
            LOGGER.info("Long dense mining progress %d/%d queries", processed, len(records))

    del corpus_vectors
    gc.collect()
    return results


def _add_candidate(
    candidates: dict[str, _ScoredCandidate],
    *,
    chunk_id: str,
    document_id: str,
    source: str,
    mapped: LongCandidate | None = None,
    dense_hit: DenseLongHit | None = None,
) -> None:
    candidate = candidates.get(chunk_id)
    if candidate is None:
        candidate = _ScoredCandidate(
            chunk_id=chunk_id,
            document_id=document_id,
            teacher_score=math.nan,
            sources=set(),
        )
        candidates[chunk_id] = candidate
    elif candidate.document_id != document_id:
        raise ValueError(f"long chunk {chunk_id} has inconsistent document IDs")
    candidate.sources.add(source)
    if mapped is not None:
        candidate.mapped_retrieval_rank = mapped.retrieval_rank
        candidate.retrieval_support_score = mapped.retrieval_support_score
    if dense_hit is not None:
        if candidate.long_dense_rank is None or dense_hit.rank < candidate.long_dense_rank:
            candidate.long_dense_rank = dense_hit.rank
            candidate.long_dense_score = dense_hit.score


def _maxp_wrong_documents(
    candidates: Iterable[_ScoredCandidate],
    *,
    gold_document_ids: frozenset[str],
) -> list[NegativeDocument]:
    best: dict[str, _ScoredCandidate] = {}
    for candidate in candidates:
        if candidate.document_id in gold_document_ids:
            continue
        current = best.get(candidate.document_id)
        candidate_key = (
            -candidate.teacher_score,
            candidate.mapped_retrieval_rank or 10**12,
            candidate.long_dense_rank or 10**12,
            candidate.chunk_id,
        )
        if current is None:
            best[candidate.document_id] = candidate
            continue
        current_key = (
            -current.teacher_score,
            current.mapped_retrieval_rank or 10**12,
            current.long_dense_rank or 10**12,
            current.chunk_id,
        )
        if candidate_key < current_key:
            best[candidate.document_id] = candidate
    return [
        NegativeDocument(
            chunk_id=item.chunk_id,
            document_id=item.document_id,
            teacher_score=item.teacher_score,
            sources=tuple(sorted(item.sources)),
            mapped_retrieval_rank=item.mapped_retrieval_rank,
            retrieval_support_score=item.retrieval_support_score,
            long_dense_rank=item.long_dense_rank,
            long_dense_score=item.long_dense_score,
        )
        for item in sorted(
            best.values(),
            key=lambda value: (-value.teacher_score, value.document_id),
        )
    ]


def select_stage1_negatives(
    candidates: Sequence[NegativeDocument],
    *,
    positive_score: float,
    count: int,
    near_margin: float,
    max_violating: int = 3,
    max_near: int = 2,
    min_direct_dense: int = 1,
) -> list[tuple[NegativeDocument, str]]:
    """Select distinct-document negatives using deterministic hardness buckets."""

    if count <= 0:
        raise ValueError("negative count must be positive")
    if near_margin < 0:
        raise ValueError("near margin must be non-negative")
    ordered = sorted(
        candidates,
        key=lambda value: (-value.teacher_score, value.document_id),
    )
    chosen: list[tuple[NegativeDocument, str]] = []
    chosen_documents: set[str] = set()

    def take(pool: Iterable[NegativeDocument], limit: int, category: str) -> None:
        if limit <= 0:
            return
        added = 0
        for candidate in pool:
            if len(chosen) >= count or added >= limit:
                break
            if candidate.document_id in chosen_documents:
                continue
            chosen.append((candidate, category))
            chosen_documents.add(candidate.document_id)
            added += 1

    violating = [item for item in ordered if item.teacher_score >= positive_score]
    near = [
        item
        for item in ordered
        if positive_score - near_margin <= item.teacher_score < positive_score
    ]
    direct = [item for item in ordered if "long_dense" in item.sources]
    mapped = [item for item in ordered if "mapped_short_pool" in item.sources]
    take(violating, max_violating, "violating")
    take(near, max_near, "near_margin")
    take(direct, min_direct_dense, "long_dense")
    take(mapped, count, "mapped_hard")
    take(ordered, count, "score_fallback")
    return chosen


def _candidate_to_json(
    candidate: NegativeDocument,
    *,
    text: str,
    category: str,
) -> dict[str, Any]:
    return {
        "chunk_id": candidate.chunk_id,
        "document_id": candidate.document_id,
        "text": text,
        "teacher_score": candidate.teacher_score,
        "category": category,
        "sources": list(candidate.sources),
        "mapped_retrieval_rank": candidate.mapped_retrieval_rank,
        "retrieval_support_score": candidate.retrieval_support_score,
        "long_dense_rank": candidate.long_dense_rank,
        "long_dense_score": candidate.long_dense_score,
    }


def _open_temporary(destination: Path) -> tuple[TextIO, Path]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    return os.fdopen(descriptor, "w", encoding="utf-8", newline="\n"), Path(name)


def _release_cuda_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def mine_stage1_dataset(args: argparse.Namespace) -> dict[str, Any]:
    gold_path = Path(args.gold)
    short_index_dir = Path(args.short_index_dir)
    long_index_dir = Path(args.long_index_dir)
    config_path = Path(args.config)
    output_dir = Path(args.output_dir)
    records = _load_gold(gold_path)
    split_verification = _verify_training_split(
        gold_path,
        records,
        split_manifest_path=(
            Path(args.split_manifest) if args.split_manifest else None
        ),
        allow_unverified=args.allow_unverified_training_data,
    )
    if args.max_queries is not None:
        records = records[: args.max_queries]

    config = PipelineConfig.from_yaml(config_path)
    if config.hyde.enabled:
        raise ValueError("Stage 1 currently requires hyde.enabled=false")
    if not config.reranker.enabled or not config.long_context.enabled:
        raise ValueError(
            "Stage 1 requires reranker.enabled=true and long_context.enabled=true"
        )
    if config.bm25.top_k_chunks != 50 or config.dense.top_k_chunks != 100:
        LOGGER.warning(
            "Mining with config top-k bm25=%d dense=%d (expected experiment: 50/100)",
            config.bm25.top_k_chunks,
            config.dense.top_k_chunks,
        )

    outputs = {
        "dataset": output_dir / "stage1_train.jsonl",
        "skipped": output_dir / "stage1_skipped.jsonl",
        "manifest": output_dir / "stage1_manifest.json",
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Stage 1 outputs already exist; pass --overwrite: "
            + ", ".join(str(path) for path in existing)
        )

    LOGGER.info("Loading long index and mining dense positive/negative shortlists")
    long_chunks, long_dense, long_manifest = _load_long_dense_index(
        long_index_dir,
        config,
    )
    long_dense_results = _precompute_long_dense_candidates(
        records,
        chunks=long_chunks,
        dense=long_dense,
        positive_top_k=args.positive_dense_top_k,
        global_top_k=args.direct_long_top_k,
        query_batch_size=args.dense_query_batch_size,
        log_every=args.log_every,
    )
    del long_dense
    _release_cuda_cache()

    LOGGER.info("Loading short BM25/dense index with embedded short-to-long mapping")
    short_bundle = load_indexes(short_index_dir, config)
    assets = DualChunkAssets(
        mappings=ShortChunkMetadataLookup(short_bundle.chunks),
        long_chunks=long_chunks,
    )
    by_long_id = long_chunks
    candidate_limit = (
        None
        if config.long_context.candidate_mode == "full"
        else config.long_context.candidate_top_k
    )

    dataset_handle, dataset_tmp = _open_temporary(outputs["dataset"])
    skipped_handle, skipped_tmp = _open_temporary(outputs["skipped"])
    group_count = 0
    emitted_queries: set[str] = set()
    skipped_reasons: Counter[str] = Counter()
    negative_categories: Counter[str] = Counter()
    negative_sources: Counter[str] = Counter()
    pool_long_counts: list[int] = []
    try:
        with VietnameseCrossEncoderReranker(config.reranker) as reranker:
            for record_number, record in enumerate(records, start=1):
                bm25_hits = short_bundle.bm25.search(
                    record.question,
                    config.bm25.top_k_chunks,
                )
                dense_hits = short_bundle.dense.search(
                    record.question,
                    config.dense.top_k_chunks,
                )
                ranked_channels = rank_channels(
                    {"bm25": bm25_hits, "dense": dense_hits}
                )
                pool = build_long_candidate_pool(
                    ranked_channels,
                    assets=assets,
                    channel_weights=config.fusion.channel_weights,
                    rrf_k=config.fusion.rrf_k,
                    long_candidate_limit=candidate_limit,
                )
                pool_long_counts.append(pool.selected_long_chunk_count)
                dense_result = long_dense_results[record.query_id]
                candidates: dict[str, _ScoredCandidate] = {}
                for mapped in pool.candidates:
                    _add_candidate(
                        candidates,
                        chunk_id=mapped.long_chunk_id,
                        document_id=mapped.document_id,
                        source="mapped_short_pool",
                        mapped=mapped,
                    )
                for hit in dense_result.global_hits:
                    _add_candidate(
                        candidates,
                        chunk_id=hit.chunk_id,
                        document_id=hit.document_id,
                        source="long_dense",
                        dense_hit=hit,
                    )
                for document_hits in dense_result.positive_shortlists.values():
                    for hit in document_hits:
                        _add_candidate(
                            candidates,
                            chunk_id=hit.chunk_id,
                            document_id=hit.document_id,
                            source="positive_dense_restricted",
                            dense_hit=hit,
                        )

                ordered_candidates = sorted(candidates.values(), key=lambda item: item.chunk_id)
                passages = [
                    by_long_id.get(candidate.chunk_id).index_text
                    for candidate in ordered_candidates
                ]
                scores = reranker.score(record.question, passages)
                if len(scores) != len(ordered_candidates):
                    raise ValueError("teacher reranker returned the wrong score count")
                for candidate, raw_score in zip(
                    ordered_candidates,
                    scores,
                    strict=True,
                ):
                    score = float(raw_score)
                    if not math.isfinite(score):
                        raise ValueError("teacher reranker returned a non-finite score")
                    candidate.teacher_score = score

                gold_set = frozenset(record.document_ids)
                wrong_documents = _maxp_wrong_documents(
                    ordered_candidates,
                    gold_document_ids=gold_set,
                )
                for gold_document_id in record.document_ids:
                    positive_dense_hits = {
                        hit.chunk_id: hit
                        for hit in dense_result.positive_shortlists.get(
                            gold_document_id,
                            (),
                        )
                    }
                    positives = [
                        (candidate, positive_dense_hits[candidate.chunk_id])
                        for candidate in ordered_candidates
                        if candidate.chunk_id in positive_dense_hits
                    ]
                    if not positives:
                        reason = "gold_document_has_no_long_chunks"
                        skipped_reasons[reason] += 1
                        skipped_handle.write(
                            json.dumps(
                                {
                                    "query_id": record.query_id,
                                    "document_id": gold_document_id,
                                    "reason": reason,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        continue
                    positive, positive_dense_hit = min(
                        positives,
                        key=lambda item: (
                            -item[0].teacher_score,
                            item[1].rank,
                            item[0].chunk_id,
                        ),
                    )
                    selected_negatives = select_stage1_negatives(
                        wrong_documents,
                        positive_score=positive.teacher_score,
                        count=args.negatives_per_positive,
                        near_margin=args.near_margin,
                        max_violating=args.max_violating_negatives,
                        max_near=args.max_near_negatives,
                        min_direct_dense=args.min_direct_dense_negatives,
                    )
                    if len(selected_negatives) < args.negatives_per_positive:
                        reason = "insufficient_distinct_negative_documents"
                        skipped_reasons[reason] += 1
                        skipped_handle.write(
                            json.dumps(
                                {
                                    "query_id": record.query_id,
                                    "document_id": gold_document_id,
                                    "reason": reason,
                                    "available": len(selected_negatives),
                                    "required": args.negatives_per_positive,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        continue

                    positive_chunk = by_long_id.get(positive.chunk_id)
                    row = {
                        "format_version": DATASET_FORMAT_VERSION,
                        "query_id": record.query_id,
                        "query": record.question,
                        "gold_document_ids": list(record.document_ids),
                        "positive": {
                            "chunk_id": positive.chunk_id,
                            "document_id": positive.document_id,
                            "text": positive_chunk.index_text,
                            "teacher_score": positive.teacher_score,
                            "long_dense_rank_within_gold_document": (
                                positive_dense_hit.rank
                            ),
                            "long_dense_score": positive_dense_hit.score,
                            "selection": "dense_top_k_within_gold_then_teacher_max",
                        },
                        "negatives": [
                            _candidate_to_json(
                                negative,
                                text=by_long_id.get(negative.chunk_id).index_text,
                                category=category,
                            )
                            for negative, category in selected_negatives
                        ],
                        "mining": {
                            "bm25_short_top_k": config.bm25.top_k_chunks,
                            "dense_short_top_k": config.dense.top_k_chunks,
                            "mapped_long_candidate_count": pool.selected_long_chunk_count,
                            "direct_long_dense_top_k": args.direct_long_top_k,
                            "all_gold_documents_excluded_from_negatives": True,
                            "negative_document_aggregation": "maxp_teacher_score",
                        },
                    }
                    dataset_handle.write(
                        json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                    )
                    group_count += 1
                    emitted_queries.add(record.query_id)
                    for negative, category in selected_negatives:
                        negative_categories[category] += 1
                        for source in negative.sources:
                            negative_sources[source] += 1

                if record_number % args.log_every == 0 or record_number == len(records):
                    LOGGER.info(
                        "Stage 1 scoring progress %d/%d queries; %d groups",
                        record_number,
                        len(records),
                        group_count,
                    )

        dataset_handle.flush()
        os.fsync(dataset_handle.fileno())
        skipped_handle.flush()
        os.fsync(skipped_handle.fileno())
        dataset_handle.close()
        skipped_handle.close()
        dataset_tmp.replace(outputs["dataset"])
        skipped_tmp.replace(outputs["skipped"])
    except BaseException:
        dataset_handle.close()
        skipped_handle.close()
        dataset_tmp.unlink(missing_ok=True)
        skipped_tmp.unlink(missing_ok=True)
        raise

    short_manifest_path = short_index_dir / "manifest.json"
    long_manifest_path = long_index_dir / "manifest.json"
    manifest = {
        "format_version": DATASET_FORMAT_VERSION,
        "artifact_type": "stage1_grouped_reranker_training_data",
        "dataset": {
            "filename": outputs["dataset"].name,
            "sha256": _sha256_file(outputs["dataset"]),
            "group_count": group_count,
            "query_count": len(emitted_queries),
            "group_size": 1 + args.negatives_per_positive,
            "skipped_filename": outputs["skipped"].name,
            "skipped_group_count": sum(skipped_reasons.values()),
            "skipped_reasons": dict(sorted(skipped_reasons.items())),
        },
        "training_split": split_verification,
        "source": {
            "gold": str(gold_path),
            "config": str(config_path),
            "config_sha256": _sha256_file(config_path),
            "short_index_dir": str(short_index_dir),
            "short_index_manifest_sha256": _sha256_file(short_manifest_path),
            "long_index_dir": str(long_index_dir),
            "long_index_manifest_sha256": _sha256_file(long_manifest_path),
            "long_chunk_count": int(long_manifest["chunk_count"]),
        },
        "mining": {
            "algorithm": "dense_restricted_positive_teacher_max_and_document_maxp_hard_negatives_v1",
            "positive_dense_top_k": args.positive_dense_top_k,
            "direct_long_top_k": args.direct_long_top_k,
            "negative_count": args.negatives_per_positive,
            "near_margin": args.near_margin,
            "max_violating_negatives": args.max_violating_negatives,
            "max_near_negatives": args.max_near_negatives,
            "min_direct_dense_negatives": args.min_direct_dense_negatives,
            "teacher_model": config.reranker.model_name,
            "teacher_revision": config.reranker.revision,
            "all_gold_documents_excluded_from_negatives": True,
            "negative_document_aggregation": "maxp",
            "negative_category_counts": dict(sorted(negative_categories.items())),
            "negative_source_counts": dict(sorted(negative_sources.items())),
        },
        "retrieval": {
            "bm25_short_top_k": config.bm25.top_k_chunks,
            "dense_short_top_k": config.dense.top_k_chunks,
            "long_candidate_mode": config.long_context.candidate_mode,
            "mean_mapped_long_candidates": (
                sum(pool_long_counts) / len(pool_long_counts)
                if pool_long_counts
                else 0.0
            ),
        },
        "smoke_test": args.max_queries is not None,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_tmp = outputs["manifest"].with_name(
        f".{outputs['manifest'].name}.{os.getpid()}.tmp"
    )
    with manifest_tmp.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    manifest_tmp.replace(outputs["manifest"])
    return manifest


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Mine Stage 1 grouped reranker data from existing short and long indexes"
        )
    )
    parser.add_argument("--gold", required=True, help="Path to seed_2026/train.json")
    parser.add_argument("--split-manifest")
    parser.add_argument("--short-index-dir", required=True)
    parser.add_argument("--long-index-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--positive-dense-top-k", type=_positive_int, default=8)
    parser.add_argument("--direct-long-top-k", type=_positive_int, default=60)
    parser.add_argument("--negatives-per-positive", type=_positive_int, default=7)
    parser.add_argument("--near-margin", type=float, default=2.0)
    parser.add_argument("--max-violating-negatives", type=int, default=3)
    parser.add_argument("--max-near-negatives", type=int, default=2)
    parser.add_argument("--min-direct-dense-negatives", type=int, default=1)
    parser.add_argument("--dense-query-batch-size", type=_positive_int, default=16)
    parser.add_argument("--log-every", type=_positive_int, default=25)
    parser.add_argument("--max-queries", type=_positive_int)
    parser.add_argument("--allow-unverified-training-data", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.near_margin < 0:
        raise SystemExit("--near-margin must be non-negative")
    for name in (
        "max_violating_negatives",
        "max_near_negatives",
        "min_direct_dense_negatives",
    ):
        if getattr(args, name) < 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be non-negative")
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    manifest = mine_stage1_dataset(args)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
