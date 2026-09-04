from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import logging
import os
import re
import tempfile
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .chunk_fixed_size import (
    _document_titles,
    _read_source_document,
    discover_context_files,
)


LOGGER = logging.getLogger("legal_ir.chunk_dual_granularity")

FORMAT_VERSION = 1
CHUNKING_VERSION = "dual_char_v1"
NORMALIZATION_VERSION = "legal_nfc_lines_v1"
SHORT_CHUNK_STRATEGY = "clause_character"
LONG_CHUNK_STRATEGY = "overlapping_short_window_character"

_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0e-\x1f\x7f-\x9f]")
_HORIZONTAL_WHITESPACE = re.compile(r"[^\S\n]+")
_HTML_SPACE_ENTITY = re.compile(r"&(?:nbsp|#0*160|#x0*a0);", re.IGNORECASE)
_INVISIBLE_TRANSLATION = str.maketrans(
    {
        "\u00ad": None,
        "\u200b": " ",
        "\u200c": None,
        "\u200d": None,
        "\u200e": None,
        "\u200f": None,
        "\u2060": None,
        "\ufeff": None,
        "\u2028": "\n",
        "\u2029": "\n",
    }
)
_STRUCTURAL_LINE_START = re.compile(
    r"^(?:"
    r"(?:phần|chương|mục|tiểu\s+mục|điều|khoản)\s+"
    r"(?:số\s+)?(?:\d+[a-zđ]?|[ivxlcdm]+)(?:[.):-]|\s|$)"
    r"|phụ\s+lục\s+\S+"
    r"|(?:\d+(?:\.\d+){0,4}|[a-zđ]|[ivxlcdm]+)[.)](?:\s|$)"
    r"|[-–—•](?:\s|$)"
    r")",
    re.IGNORECASE,
)
_SENTENCE_END = re.compile(r"[.!?;:][\"'”’)]*(?=\s|$)")


@dataclass(frozen=True, slots=True)
class TextSpan:
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError(f"invalid text span: {self.start}:{self.end}")

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class DualGranularityChunkingConfig:
    """One-pass character chunking for short retrieval and long reranking text."""

    input_dir: Path
    output_dir: Path
    short_max_characters: int = 450
    long_max_characters: int = 2000
    long_overlap_short_chunks: int = 1
    input_pattern: str = "context_*.json"
    include_document_title: bool = True
    title_max_characters: int = 300
    log_every_documents: int = 100
    overwrite: bool = False

    def __post_init__(self) -> None:
        if self.short_max_characters <= 0:
            raise ValueError("short_max_characters must be positive")
        if self.long_max_characters < self.short_max_characters:
            raise ValueError(
                "long_max_characters must be >= short_max_characters"
            )
        if self.long_overlap_short_chunks < 0:
            raise ValueError("long_overlap_short_chunks must be non-negative")
        if self.title_max_characters <= 0:
            raise ValueError("title_max_characters must be positive")
        if self.log_every_documents <= 0:
            raise ValueError("log_every_documents must be positive")
        if not self.input_pattern.strip():
            raise ValueError("input_pattern must not be empty")

    @property
    def short_chunks_path(self) -> Path:
        return self.output_dir / "short_chunks.jsonl"

    @property
    def long_chunks_path(self) -> Path:
        return self.output_dir / "long_chunks.jsonl"

    @property
    def mapping_path(self) -> Path:
        return self.output_dir / "short_to_long.jsonl"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"

    @property
    def artifact_paths(self) -> tuple[Path, ...]:
        return (
            self.short_chunks_path,
            self.long_chunks_path,
            self.mapping_path,
            self.manifest_path,
        )


@dataclass(slots=True)
class DualChunkingStatistics:
    documents_discovered: int = 0
    documents_chunked: int = 0
    documents_skipped_empty: int = 0
    normalized_characters: int = 0
    short_chunks_written: int = 0
    long_chunks_written: int = 0
    mappings_written: int = 0
    maximum_short_chunk_characters: int = 0
    maximum_long_chunk_characters: int = 0


def normalize_legal_text_with_lines(value: Any) -> str:
    """Normalize legal text while retaining non-empty source line boundaries.

    The fixed-token baseline collapses all whitespace. Dual chunking needs stable
    structural boundaries, so this policy collapses horizontal whitespace inside
    each source line but keeps a single ``\n`` between consecutive non-empty lines.
    """

    if value is None:
        return ""
    text = _HTML_SPACE_ENTITY.sub(" ", str(value))
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_INVISIBLE_TRANSLATION)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\f", "\n").replace("\v", "\n")
    text = _CONTROL_CHARACTERS.sub(" ", text)
    lines = [
        _HORIZONTAL_WHITESPACE.sub(" ", line).strip()
        for line in text.split("\n")
    ]
    return "\n".join(line for line in lines if line)


def _trim_span(text: str, start: int, end: int) -> TextSpan | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return TextSpan(start, end) if start < end else None


def _line_spans(text: str) -> list[TextSpan]:
    return [TextSpan(match.start(), match.end()) for match in re.finditer(r"[^\n]+", text)]


def _logical_clause_spans(text: str) -> list[TextSpan]:
    """Use legal/list markers at source-line starts as hard short-chunk borders."""

    lines = _line_spans(text)
    if not lines:
        return []

    units: list[TextSpan] = []
    unit_start = lines[0].start
    previous_end = lines[0].end
    for line in lines[1:]:
        line_text = text[line.start : line.end]
        if _STRUCTURAL_LINE_START.match(line_text):
            unit = _trim_span(text, unit_start, previous_end)
            if unit is not None:
                units.append(unit)
            unit_start = line.start
        previous_end = line.end

    final_unit = _trim_span(text, unit_start, previous_end)
    if final_unit is not None:
        units.append(final_unit)
    return units


def _preferred_split(text: str, start: int, limit: int) -> int:
    """Find a coherent boundary not exceeding ``limit``."""

    minimum = start + max(1, (limit - start) // 2)
    region = text[start:limit]

    sentence_ends = [
        start + match.end()
        for match in _SENTENCE_END.finditer(region)
        if start + match.end() >= minimum
    ]
    if sentence_ends:
        return sentence_ends[-1]

    newline = text.rfind("\n", minimum, limit + 1)
    if newline >= minimum:
        return newline

    for position in range(limit, minimum - 1, -1):
        if position < len(text) and text[position].isspace():
            return position
    return limit


def _split_span(text: str, span: TextSpan, maximum: int) -> list[TextSpan]:
    pieces: list[TextSpan] = []
    cursor = span.start
    while cursor < span.end:
        while cursor < span.end and text[cursor].isspace():
            cursor += 1
        if cursor >= span.end:
            break

        limit = min(cursor + maximum, span.end)
        split = span.end if limit == span.end else _preferred_split(text, cursor, limit)
        if split <= cursor:
            split = limit
        piece = _trim_span(text, cursor, split)
        if piece is not None:
            if piece.length > maximum:
                raise AssertionError("short chunk exceeds configured maximum")
            pieces.append(piece)
        cursor = split
    return pieces


def short_chunk_spans(text: str, maximum: int) -> list[TextSpan]:
    if maximum <= 0:
        raise ValueError("maximum must be positive")
    chunks: list[TextSpan] = []
    for unit in _logical_clause_spans(text):
        chunks.extend(_split_span(text, unit, maximum))
    return chunks


def long_chunk_spans(
    short_spans: Sequence[TextSpan],
    *,
    maximum: int,
    overlap_short_chunks: int,
) -> list[TextSpan]:
    """Pack short chunks into long windows and overlap by complete short chunks."""

    if maximum <= 0:
        raise ValueError("maximum must be positive")
    if overlap_short_chunks < 0:
        raise ValueError("overlap_short_chunks must be non-negative")
    if not short_spans:
        return []
    if any(span.length > maximum for span in short_spans):
        raise ValueError("a short chunk is larger than the long chunk maximum")

    result: list[TextSpan] = []
    start_index = 0
    while start_index < len(short_spans):
        end_index = start_index + 1
        while end_index < len(short_spans):
            proposed_length = (
                short_spans[end_index].end - short_spans[start_index].start
            )
            if proposed_length > maximum:
                break
            end_index += 1

        window = TextSpan(
            short_spans[start_index].start,
            short_spans[end_index - 1].end,
        )
        if window.length > maximum:
            raise AssertionError("long chunk exceeds configured maximum")
        result.append(window)
        if end_index == len(short_spans):
            break
        start_index = max(
            start_index + 1,
            end_index - overlap_short_chunks,
        )
    return result


def _artifact_namespace(config: DualGranularityChunkingConfig) -> str:
    return (
        f"{CHUNKING_VERSION}_{NORMALIZATION_VERSION}_"
        f"s{config.short_max_characters}_"
        f"l{config.long_max_characters}_"
        f"o{config.long_overlap_short_chunks}"
    )


def _chunk_id(
    document_id: str,
    namespace: str,
    granularity: str,
    index: int,
) -> str:
    return f"{document_id}:{namespace}:{granularity}:{index:06d}"


def _primary_long_index(short: TextSpan, long_spans: Sequence[TextSpan]) -> int:
    candidates: list[tuple[int, int, int]] = []
    for index, long_span in enumerate(long_spans):
        if long_span.start <= short.start and short.end <= long_span.end:
            balanced_context = min(
                short.start - long_span.start,
                long_span.end - short.end,
            )
            candidates.append((balanced_context, long_span.length, -index))
    if not candidates:
        raise ValueError(
            f"short span {short.start}:{short.end} has no containing long chunk"
        )
    return -max(candidates)[2]


def _retrieval_text(passage: str, retrieval_title: str, include_title: bool) -> str:
    if include_title and retrieval_title:
        return f"Tên văn bản: {retrieval_title}\n{passage}"
    return passage


def build_document_dual_chunks(
    document: Mapping[str, Any],
    source_path: Path,
    config: DualGranularityChunkingConfig,
    *,
    normalized_text: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Create both granularities and their mapping from one normalized document."""

    document_id = str(document["id"]).strip()
    text = (
        normalize_legal_text_with_lines(document.get("passage"))
        if normalized_text is None
        else normalized_text
    )
    if not text:
        return [], [], []

    short_spans = short_chunk_spans(text, config.short_max_characters)
    long_spans = long_chunk_spans(
        short_spans,
        maximum=config.long_max_characters,
        overlap_short_chunks=config.long_overlap_short_chunks,
    )
    if not short_spans or not long_spans:
        raise ValueError(f"chunking produced no chunks for document {document_id}")

    namespace = _artifact_namespace(config)
    short_ids = [
        _chunk_id(document_id, namespace, "short", index)
        for index in range(len(short_spans))
    ]
    long_ids = [
        _chunk_id(document_id, namespace, "long", index)
        for index in range(len(long_spans))
    ]
    original_title, retrieval_title = _document_titles(
        document, config.title_max_characters
    )
    source_link = str(document.get("link") or "").strip()

    long_members: list[list[int]] = []
    for long_span in long_spans:
        members = [
            index
            for index, short_span in enumerate(short_spans)
            if long_span.start <= short_span.start
            and short_span.end <= long_span.end
        ]
        if not members:
            raise AssertionError("long chunk contains no complete short chunk")
        long_members.append(members)

    containing_long_indices: list[list[int]] = []
    primary_long_indices: list[int] = []
    for short_span in short_spans:
        containing = [
            index
            for index, long_span in enumerate(long_spans)
            if long_span.start <= short_span.start
            and short_span.end <= long_span.end
        ]
        containing_long_indices.append(containing)
        primary_long_indices.append(_primary_long_index(short_span, long_spans))

    common_metadata: dict[str, Any] = {
        "normalization_version": NORMALIZATION_VERSION,
        "normalized_document_characters": len(text),
        "source_file": source_path.name,
    }
    if original_title:
        common_metadata["document_name"] = original_title
    if retrieval_title:
        common_metadata["retrieval_title"] = retrieval_title
    if source_link:
        common_metadata["source_link"] = source_link

    short_records: list[dict[str, Any]] = []
    mappings: list[dict[str, Any]] = []
    for index, span in enumerate(short_spans):
        passage = text[span.start : span.end]
        mapped_long_ids = [long_ids[item] for item in containing_long_indices[index]]
        primary_long_id = long_ids[primary_long_indices[index]]
        metadata = {
            **common_metadata,
            "granularity": "short",
            "chunk_strategy": SHORT_CHUNK_STRATEGY,
            "chunk_index": index,
            "chunk_count_in_document": len(short_spans),
            "max_characters": config.short_max_characters,
            "character_count": len(passage),
            "normalized_character_start": span.start,
            "normalized_character_end": span.end,
            "primary_long_chunk_id": primary_long_id,
            "long_chunk_ids": mapped_long_ids,
        }
        short_records.append(
            {
                "chunk_id": short_ids[index],
                "document_id": document_id,
                "passage": passage,
                "retrieval_text": _retrieval_text(
                    passage,
                    retrieval_title,
                    config.include_document_title,
                ),
                "metadata": metadata,
            }
        )
        mappings.append(
            {
                "short_chunk_id": short_ids[index],
                "document_id": document_id,
                "primary_long_chunk_id": primary_long_id,
                "long_chunk_ids": mapped_long_ids,
            }
        )

    long_records: list[dict[str, Any]] = []
    for index, span in enumerate(long_spans):
        passage = text[span.start : span.end]
        member_short_ids = [short_ids[item] for item in long_members[index]]
        metadata = {
            **common_metadata,
            "granularity": "long",
            "chunk_strategy": LONG_CHUNK_STRATEGY,
            "chunk_index": index,
            "chunk_count_in_document": len(long_spans),
            "max_characters": config.long_max_characters,
            "character_count": len(passage),
            "normalized_character_start": span.start,
            "normalized_character_end": span.end,
            "overlap_short_chunks": config.long_overlap_short_chunks,
            "short_chunk_ids": member_short_ids,
        }
        long_records.append(
            {
                "chunk_id": long_ids[index],
                "document_id": document_id,
                "passage": passage,
                "retrieval_text": _retrieval_text(
                    passage,
                    retrieval_title,
                    config.include_document_title,
                ),
                "metadata": metadata,
            }
        )

    return short_records, long_records, mappings


def _temporary_path(path: Path) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    os.close(descriptor)
    return Path(temporary_name)


def _serialized_line(record: Mapping[str, Any]) -> str:
    return json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"


def _manifest_payload(
    config: DualGranularityChunkingConfig,
    statistics: DualChunkingStatistics,
    *,
    input_sha256: str,
    output_sha256: Mapping[str, str],
    skipped_empty_document_ids: Sequence[str],
) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
        "artifact_type": "legal_ir_dual_granularity_chunks",
        "strategy": "dual_character",
        "strategy_version": CHUNKING_VERSION,
        "source_passes": 1,
        "normalization": {
            "version": NORMALIZATION_VERSION,
            "unicode_form": "NFC",
            "html_entity_policy": "decode_non_breaking_space_only",
            "horizontal_whitespace_policy": "collapse_to_ascii_space",
            "line_policy": "retain_non_empty_source_lines",
            "lowercase": False,
            "strip_accents": False,
            "preserve_punctuation_and_numbers": True,
        },
        "input_dir": str(config.input_dir),
        "input_pattern": config.input_pattern,
        "input_sha256": input_sha256,
        "outputs": {
            "short_chunks": {
                "path": str(config.short_chunks_path),
                "sha256": output_sha256["short_chunks"],
                "records": statistics.short_chunks_written,
                "purpose": "bm25_and_dense_retrieval_index",
            },
            "long_chunks": {
                "path": str(config.long_chunks_path),
                "sha256": output_sha256["long_chunks"],
                "records": statistics.long_chunks_written,
                "purpose": "reranker_context_only_not_retrieval_index",
            },
            "short_to_long": {
                "path": str(config.mapping_path),
                "sha256": output_sha256["short_to_long"],
                "records": statistics.mappings_written,
            },
        },
        "chunking": {
            "short": {
                "strategy": SHORT_CHUNK_STRATEGY,
                "max_characters": config.short_max_characters,
                "overlap": False,
                "boundary_policy": "legal_or_list_marker_then_sentence_line_word",
            },
            "long": {
                "strategy": LONG_CHUNK_STRATEGY,
                "max_characters": config.long_max_characters,
                "overlap_short_chunks": config.long_overlap_short_chunks,
                "boundary_policy": "whole_short_chunks_only",
            },
            "include_document_title": config.include_document_title,
            "title_max_characters": config.title_max_characters,
        },
        "statistics": asdict(statistics),
        "skipped_empty_document_ids": list(skipped_empty_document_ids),
    }


def build_dual_granularity_chunks(
    config: DualGranularityChunkingConfig,
) -> dict[str, Any]:
    """Stream a corpus once and atomically publish short, long, and mapping files."""

    source_paths = discover_context_files(config.input_dir, config.input_pattern)
    existing = [path for path in config.artifact_paths if path.exists()]
    if existing and not config.overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"dual chunk artifacts already exist: {joined}; pass --overwrite explicitly"
        )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    temporary_paths = {
        "short_chunks": _temporary_path(config.short_chunks_path),
        "long_chunks": _temporary_path(config.long_chunks_path),
        "short_to_long": _temporary_path(config.mapping_path),
        "manifest": _temporary_path(config.manifest_path),
    }
    digests = {
        "short_chunks": hashlib.sha256(),
        "long_chunks": hashlib.sha256(),
        "short_to_long": hashlib.sha256(),
    }
    statistics = DualChunkingStatistics(documents_discovered=len(source_paths))
    skipped_empty_document_ids: list[str] = []
    seen_document_ids: set[str] = set()
    input_digest = hashlib.sha256()

    try:
        with ExitStack() as stack:
            handles = {
                name: stack.enter_context(
                    temporary_paths[name].open("w", encoding="utf-8", newline="\n")
                )
                for name in ("short_chunks", "long_chunks", "short_to_long")
            }
            for position, source_path in enumerate(source_paths, start=1):
                document, raw_source = _read_source_document(source_path)
                relative_name = source_path.relative_to(config.input_dir).as_posix()
                input_digest.update(relative_name.encode("utf-8"))
                input_digest.update(b"\0")
                input_digest.update(raw_source)
                input_digest.update(b"\0")

                document_id = str(document["id"]).strip()
                if document_id in seen_document_ids:
                    raise ValueError(f"duplicate document id: {document_id}")
                seen_document_ids.add(document_id)

                normalized_text = normalize_legal_text_with_lines(
                    document.get("passage")
                )
                if not normalized_text:
                    statistics.documents_skipped_empty += 1
                    skipped_empty_document_ids.append(document_id)
                else:
                    short_records, long_records, mappings = (
                        build_document_dual_chunks(
                            document,
                            source_path,
                            config,
                            normalized_text=normalized_text,
                        )
                    )
                    statistics.documents_chunked += 1
                    statistics.normalized_characters += len(normalized_text)
                    for output_name, records in (
                        ("short_chunks", short_records),
                        ("long_chunks", long_records),
                        ("short_to_long", mappings),
                    ):
                        for record in records:
                            line = _serialized_line(record)
                            handles[output_name].write(line)
                            digests[output_name].update(line.encode("utf-8"))

                    statistics.short_chunks_written += len(short_records)
                    statistics.long_chunks_written += len(long_records)
                    statistics.mappings_written += len(mappings)
                    statistics.maximum_short_chunk_characters = max(
                        statistics.maximum_short_chunk_characters,
                        *(len(record["passage"]) for record in short_records),
                    )
                    statistics.maximum_long_chunk_characters = max(
                        statistics.maximum_long_chunk_characters,
                        *(len(record["passage"]) for record in long_records),
                    )

                if (
                    position == 1
                    or position % config.log_every_documents == 0
                    or position == len(source_paths)
                ):
                    LOGGER.info(
                        "Processed %d/%d documents; wrote %d short and %d long chunks",
                        position,
                        len(source_paths),
                        statistics.short_chunks_written,
                        statistics.long_chunks_written,
                    )

        if statistics.short_chunks_written == 0:
            raise ValueError("dual chunking produced no output chunks")
        if statistics.mappings_written != statistics.short_chunks_written:
            raise AssertionError("every short chunk must have exactly one mapping row")

        manifest = _manifest_payload(
            config,
            statistics,
            input_sha256=input_digest.hexdigest(),
            output_sha256={name: digest.hexdigest() for name, digest in digests.items()},
            skipped_empty_document_ids=skipped_empty_document_ids,
        )
        with temporary_paths["manifest"].open(
            "w", encoding="utf-8", newline="\n"
        ) as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

        # The manifest is the commit marker and is therefore published last.
        temporary_paths["short_chunks"].replace(config.short_chunks_path)
        temporary_paths["long_chunks"].replace(config.long_chunks_path)
        temporary_paths["short_to_long"].replace(config.mapping_path)
        temporary_paths["manifest"].replace(config.manifest_path)
        return manifest
    except BaseException:
        for temporary_path in temporary_paths.values():
            temporary_path.unlink(missing_ok=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="legal-ir-chunk-dual",
        description=(
            "Create short retrieval chunks and overlapping long reranker chunks "
            "from selected-contexts in one corpus pass."
        ),
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--input-pattern", default="context_*.json")
    parser.add_argument("--short-max-characters", type=int, default=450)
    parser.add_argument("--long-max-characters", type=int, default=2000)
    parser.add_argument("--long-overlap-short-chunks", type=int, default=1)
    parser.add_argument("--title-max-characters", type=int, default=300)
    parser.add_argument("--without-document-title", action="store_true")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    config = DualGranularityChunkingConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        short_max_characters=args.short_max_characters,
        long_max_characters=args.long_max_characters,
        long_overlap_short_chunks=args.long_overlap_short_chunks,
        input_pattern=args.input_pattern,
        include_document_title=not args.without_document_title,
        title_max_characters=args.title_max_characters,
        log_every_documents=args.log_every,
        overwrite=args.overwrite,
    )
    manifest = build_dual_granularity_chunks(config)
    LOGGER.info(
        "Finished: %d short chunks and %d long chunks -> %s",
        manifest["statistics"]["short_chunks_written"],
        manifest["statistics"]["long_chunks_written"],
        config.output_dir,
    )
    LOGGER.info("Use only %s for build-index", config.short_chunks_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
