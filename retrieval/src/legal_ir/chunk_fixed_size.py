from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import tempfile
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence
from urllib.parse import unquote, urlparse


LOGGER = logging.getLogger("legal_ir.chunk_fixed_size")

CHUNK_STRATEGY = "fixed_token"
NORMALIZATION_VERSION = "legal_nfc_ws_v1"
DEFAULT_TOKENIZER = "AITeamVN/Vietnamese_Embedding_v2"
DEFAULT_TOKENIZER_REVISION = "18b44161e041bf1d3a333ab5144b5b7b93f914d2"

_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0e-\x1f\x7f-\x9f]")
_WHITESPACE = re.compile(r"\s+")
_HTML_SPACE_ENTITY = re.compile(r"&(?:nbsp|#0*160|#x0*a0);", re.IGNORECASE)
_CONTEXT_FILENAME = re.compile(r"^context_(.+)$")
_INVISIBLE_TRANSLATION = str.maketrans(
    {
        "\u00ad": None,  # soft hyphen
        "\u200b": " ",  # zero-width space must not join adjacent words
        "\u200c": None,  # zero-width non-joiner
        "\u200d": None,  # zero-width joiner
        "\u200e": None,  # left-to-right mark
        "\u200f": None,  # right-to-left mark
        "\u2060": None,  # word joiner
        "\ufeff": None,  # byte-order mark / zero-width no-break space
        "\u2028": "\n",  # Unicode line separator
        "\u2029": "\n",  # Unicode paragraph separator
    }
)


@dataclass(frozen=True, slots=True)
class FixedSizeChunkingConfig:
    """Configuration for the reproducible fixed-token baseline."""

    input_dir: Path
    output_path: Path
    manifest_path: Path
    tokenizer_name: str = DEFAULT_TOKENIZER
    tokenizer_revision: str | None = DEFAULT_TOKENIZER_REVISION
    tokenizer_cache_dir: Path | None = None
    local_files_only: bool = False
    chunk_size_tokens: int = 384
    overlap_tokens: int = 64
    input_pattern: str = "context_*.json"
    include_document_title: bool = True
    title_max_characters: int = 300
    log_every_documents: int = 100
    overwrite: bool = False

    def __post_init__(self) -> None:
        if self.chunk_size_tokens <= 0:
            raise ValueError("chunk_size_tokens must be positive")
        if self.overlap_tokens < 0:
            raise ValueError("overlap_tokens must be non-negative")
        if self.overlap_tokens >= self.chunk_size_tokens:
            raise ValueError("overlap_tokens must be smaller than chunk_size_tokens")
        if self.title_max_characters <= 0:
            raise ValueError("title_max_characters must be positive")
        if self.log_every_documents <= 0:
            raise ValueError("log_every_documents must be positive")
        if not self.input_pattern.strip():
            raise ValueError("input_pattern must not be empty")
        if not self.tokenizer_name.strip():
            raise ValueError("tokenizer_name must not be empty")
        if self.tokenizer_revision is not None and not self.tokenizer_revision.strip():
            raise ValueError("tokenizer_revision must be null or non-empty")
        if self.output_path == self.manifest_path:
            raise ValueError("output_path and manifest_path must be different files")


@dataclass(frozen=True, slots=True)
class TokenWindow:
    chunk_index: int
    token_start: int
    token_end: int
    character_start: int
    character_end: int
    passage: str

    @property
    def token_count(self) -> int:
        return self.token_end - self.token_start


@dataclass(slots=True)
class ChunkingStatistics:
    documents_discovered: int = 0
    documents_chunked: int = 0
    documents_skipped_empty: int = 0
    chunks_written: int = 0
    normalized_characters: int = 0
    content_tokens: int = 0


def normalize_legal_text(value: Any) -> str:
    """Normalize OCR/web whitespace without changing Vietnamese legal wording.

    NFC is deliberately used instead of NFKC: compatibility normalization can
    rewrite legally meaningful symbols, numbering, or full-width characters.
    Case, punctuation, hyphens, article numbers, and text order are kept.
    """

    if value is None:
        return ""
    # Only decode HTML entities that unambiguously represent whitespace. A
    # general html.unescape() could rewrite literal legal text such as &sect;.
    text = _HTML_SPACE_ENTITY.sub(" ", str(value))
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_INVISIBLE_TRANSLATION)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\f", "\n").replace("\v", "\n")
    text = _CONTROL_CHARACTERS.sub(" ", text)
    text = _WHITESPACE.sub(" ", text)
    return text.strip()


def _source_sort_key(path: Path) -> tuple[int, int | str, str]:
    match = _CONTEXT_FILENAME.match(path.stem)
    if match and match.group(1).isdigit():
        return (0, int(match.group(1)), path.name)
    return (1, path.name, path.name)


def discover_context_files(input_dir: Path, pattern: str) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"input path is not a directory: {input_dir}")
    paths = sorted(input_dir.glob(pattern), key=_source_sort_key)
    if not paths:
        raise FileNotFoundError(
            f"no source documents matched {pattern!r} under {input_dir}"
        )
    return paths


def _read_source_document(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw_bytes = path.read_bytes()
        value = json.loads(raw_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read source document {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"source document must be a JSON object: {path}")
    if value.get("id") is None or not str(value["id"]).strip():
        raise ValueError(f"source document has no valid id: {path}")
    if "passage" not in value:
        raise ValueError(f"source document has no passage field: {path}")
    if not isinstance(value["passage"], str):
        raise ValueError(f"source document passage must be a string: {path}")

    document_id = str(value["id"]).strip()
    match = _CONTEXT_FILENAME.match(path.stem)
    if match and match.group(1) != document_id:
        raise ValueError(
            f"filename/document id mismatch: {path.name} contains {document_id!r}"
        )
    return value, raw_bytes


def load_source_document(path: Path) -> dict[str, Any]:
    document, _ = _read_source_document(path)
    return document


def load_fast_tokenizer(
    model_name: str,
    revision: str | None,
    *,
    cache_dir: Path | None = None,
    local_files_only: bool = False,
) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("transformers is required for fixed-size chunking") from exc

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        use_fast=True,
        trust_remote_code=False,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        local_files_only=local_files_only,
    )
    if not tokenizer.is_fast:
        raise RuntimeError(
            "fixed-size chunking requires a fast tokenizer with offset mappings"
        )
    return tokenizer


def _character_span(
    offsets: Sequence[Sequence[int]], token_start: int, token_end: int
) -> tuple[int, int] | None:
    character_start: int | None = None
    character_end: int | None = None
    for raw_start, raw_end in offsets[token_start:token_end]:
        start = int(raw_start)
        end = int(raw_end)
        if end <= start:
            continue
        if character_start is None:
            character_start = start
        character_end = end
    if character_start is None or character_end is None:
        return None
    return character_start, character_end


def iter_token_windows(
    normalized_text: str,
    tokenizer: Any,
    *,
    chunk_size_tokens: int,
    overlap_tokens: int,
) -> tuple[int, Iterator[TokenWindow]]:
    """Tokenize once and expose fixed windows as slices of normalized text."""

    if chunk_size_tokens <= 0:
        raise ValueError("chunk_size_tokens must be positive")
    if overlap_tokens < 0 or overlap_tokens >= chunk_size_tokens:
        raise ValueError(
            "overlap_tokens must be non-negative and smaller than chunk_size_tokens"
        )

    encoded = tokenizer(
        normalized_text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
        return_offsets_mapping=True,
        truncation=False,
        verbose=False,
    )
    input_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    if len(input_ids) != len(offsets):
        raise ValueError("tokenizer returned different token and offset counts")

    token_count = len(input_ids)

    def generate() -> Iterator[TokenWindow]:
        if token_count == 0:
            return
        step = chunk_size_tokens - overlap_tokens
        token_start = 0
        chunk_index = 0
        while token_start < token_count:
            token_end = min(token_start + chunk_size_tokens, token_count)
            character_span = _character_span(offsets, token_start, token_end)
            if character_span is None:
                raise ValueError(
                    f"token window {token_start}:{token_end} has no valid offsets"
                )
            character_start, character_end = character_span
            if not 0 <= character_start < character_end <= len(normalized_text):
                raise ValueError(
                    f"token window {token_start}:{token_end} returned invalid "
                    f"character offsets {character_start}:{character_end}"
                )
            raw_passage = normalized_text[character_start:character_end]
            left_trimmed = len(raw_passage) - len(raw_passage.lstrip())
            right_trimmed = len(raw_passage) - len(raw_passage.rstrip())
            character_start += left_trimmed
            character_end -= right_trimmed
            passage = normalized_text[character_start:character_end]
            if not passage:
                raise ValueError(
                    f"token window {token_start}:{token_end} produced empty text"
                )
            yield TokenWindow(
                chunk_index=chunk_index,
                token_start=token_start,
                token_end=token_end,
                character_start=character_start,
                character_end=character_end,
                passage=passage,
            )
            chunk_index += 1
            if token_end == token_count:
                break
            token_start += step

    return token_count, generate()


def _document_titles(
    document: Mapping[str, Any], maximum_characters: int
) -> tuple[str, str]:
    original_title = normalize_legal_text(document.get("name"))
    if not original_title:
        link = str(document.get("link") or "").strip()
        if link:
            slug = Path(unquote(urlparse(link).path)).name
            original_title = normalize_legal_text(Path(slug).stem)
    original_title = " ".join(original_title.splitlines()).strip()
    retrieval_title = re.sub(r"[-_]+", " ", original_title)
    retrieval_title = _WHITESPACE.sub(" ", retrieval_title).strip()
    if len(retrieval_title) > maximum_characters:
        retrieval_title = retrieval_title[:maximum_characters].rstrip()
    return original_title, retrieval_title


def _chunk_id(
    document_id: str,
    chunk_size_tokens: int,
    overlap_tokens: int,
    chunk_index: int,
) -> str:
    return (
        f"{document_id}:fixed_{NORMALIZATION_VERSION}_"
        f"{chunk_size_tokens}_{overlap_tokens}:"
        f"{chunk_index:06d}"
    )


def iter_document_chunks(
    document: Mapping[str, Any],
    source_path: Path,
    tokenizer: Any,
    config: FixedSizeChunkingConfig,
    *,
    normalized_text: str | None = None,
) -> tuple[int, Iterator[dict[str, Any]]]:
    document_id = str(document["id"]).strip()
    if normalized_text is None:
        normalized_text = normalize_legal_text(document.get("passage"))
    if not normalized_text:
        return 0, iter(())

    original_title, retrieval_title = _document_titles(
        document, config.title_max_characters
    )
    source_link = str(document.get("link") or "").strip()
    token_count, windows = iter_token_windows(
        normalized_text,
        tokenizer,
        chunk_size_tokens=config.chunk_size_tokens,
        overlap_tokens=config.overlap_tokens,
    )
    if token_count == 0:
        raise ValueError(
            f"tokenizer produced no content tokens for document {document_id}"
        )
    step = config.chunk_size_tokens - config.overlap_tokens
    chunk_count = (
        1
        if token_count <= config.chunk_size_tokens
        else 1
        + (token_count - config.chunk_size_tokens + step - 1) // step
    )

    def generate() -> Iterator[dict[str, Any]]:
        for window in windows:
            retrieval_text = window.passage
            if config.include_document_title and retrieval_title:
                retrieval_text = (
                    f"Tên văn bản: {retrieval_title}\n{window.passage}"
                )
            metadata: dict[str, Any] = {
                "chunk_strategy": CHUNK_STRATEGY,
                "normalization_version": NORMALIZATION_VERSION,
                "chunk_index": window.chunk_index,
                "chunk_count_in_document": chunk_count,
                "token_start": window.token_start,
                "token_end": window.token_end,
                "token_count": window.token_count,
                "normalized_character_start": window.character_start,
                "normalized_character_end": window.character_end,
                "normalized_document_characters": len(normalized_text),
                "document_token_count": token_count,
                "source_file": source_path.name,
            }
            if original_title:
                metadata["document_name"] = original_title
            if retrieval_title:
                metadata["retrieval_title"] = retrieval_title
            if source_link:
                metadata["source_link"] = source_link
            yield {
                "chunk_id": _chunk_id(
                    document_id,
                    config.chunk_size_tokens,
                    config.overlap_tokens,
                    window.chunk_index,
                ),
                "document_id": document_id,
                "passage": window.passage,
                "retrieval_text": retrieval_text,
                "metadata": metadata,
            }

    return token_count, generate()


def _temporary_path(path: Path) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    os.close(descriptor)
    return Path(temporary_name)


def _tokenizer_fingerprint(tokenizer: Any) -> str:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None or not hasattr(backend, "to_str"):
        raise RuntimeError("fast tokenizer backend cannot be fingerprinted")
    serialized = backend.to_str()
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _manifest_payload(
    config: FixedSizeChunkingConfig,
    statistics: ChunkingStatistics,
    *,
    tokenizer: Any,
    tokenizer_source: str,
    effective_tokenizer_revision: str | None,
    input_sha256: str,
    output_sha256: str,
    skipped_empty_document_ids: Sequence[str],
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "artifact_type": "legal_ir_chunks",
        "strategy": CHUNK_STRATEGY,
        "normalization": {
            "version": NORMALIZATION_VERSION,
            "unicode_form": "NFC",
            "html_entity_policy": "decode_non_breaking_space_only",
            "whitespace_policy": "collapse_to_ascii_space",
            "lowercase": False,
            "strip_accents": False,
            "preserve_punctuation_and_numbers": True,
        },
        "input_dir": str(config.input_dir),
        "input_pattern": config.input_pattern,
        "input_sha256": input_sha256,
        "output_path": str(config.output_path),
        "output_sha256": output_sha256,
        "tokenizer": {
            "model_name": config.tokenizer_name,
            "requested_revision": config.tokenizer_revision,
            "effective_revision": effective_tokenizer_revision,
            "source": tokenizer_source,
            "use_fast": True,
            "class": type(tokenizer).__name__,
            "vocab_size": getattr(tokenizer, "vocab_size", None),
            "model_max_length": getattr(tokenizer, "model_max_length", None),
            "local_files_only": config.local_files_only,
            "backend_sha256": _tokenizer_fingerprint(tokenizer),
        },
        "chunking": {
            "chunk_size_tokens": config.chunk_size_tokens,
            "overlap_tokens": config.overlap_tokens,
            "stride_tokens": config.chunk_size_tokens - config.overlap_tokens,
            "tail_policy": "emit_remainder",
            "add_special_tokens": False,
            "include_document_title": config.include_document_title,
            "title_max_characters": config.title_max_characters,
        },
        "statistics": asdict(statistics),
        "skipped_empty_document_ids": list(skipped_empty_document_ids),
    }


def build_fixed_size_chunks(config: FixedSizeChunkingConfig) -> dict[str, Any]:
    """Build the JSONL atomically; intended to run in a Kaggle notebook."""

    source_paths = discover_context_files(config.input_dir, config.input_pattern)
    if config.output_path.exists() and not config.overwrite:
        raise FileExistsError(
            f"output already exists: {config.output_path}; pass --overwrite explicitly"
        )
    if config.manifest_path.exists() and not config.overwrite:
        raise FileExistsError(
            f"manifest already exists: {config.manifest_path}; pass --overwrite explicitly"
        )

    tokenizer_path = Path(config.tokenizer_name)
    tokenizer_is_local = tokenizer_path.exists() and tokenizer_path.is_dir()
    effective_tokenizer_revision = (
        None if tokenizer_is_local else config.tokenizer_revision
    )
    tokenizer = load_fast_tokenizer(
        config.tokenizer_name,
        effective_tokenizer_revision,
        cache_dir=config.tokenizer_cache_dir,
        local_files_only=config.local_files_only,
    )
    statistics = ChunkingStatistics(documents_discovered=len(source_paths))
    skipped_empty_document_ids: list[str] = []
    seen_document_ids: set[str] = set()
    input_digest = hashlib.sha256()
    digest = hashlib.sha256()

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = _temporary_path(config.output_path)
    temporary_manifest: Path | None = None

    try:
        temporary_manifest = _temporary_path(config.manifest_path)
        with temporary_output.open(
            "w", encoding="utf-8", newline="\n"
        ) as output_handle:
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

                normalized_text = normalize_legal_text(document.get("passage"))
                if not normalized_text:
                    statistics.documents_skipped_empty += 1
                    skipped_empty_document_ids.append(document_id)
                else:
                    token_count, chunks = iter_document_chunks(
                        document,
                        source_path,
                        tokenizer,
                        config,
                        normalized_text=normalized_text,
                    )
                    statistics.documents_chunked += 1
                    statistics.normalized_characters += len(normalized_text)
                    statistics.content_tokens += token_count
                    for chunk in chunks:
                        serialized = json.dumps(
                            chunk,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        serialized_line = serialized + "\n"
                        output_handle.write(serialized_line)
                        digest.update(serialized_line.encode("utf-8"))
                        statistics.chunks_written += 1

                if (
                    position == 1
                    or position % config.log_every_documents == 0
                    or position == len(source_paths)
                ):
                    LOGGER.info(
                        "Processed %d/%d documents; wrote %d chunks",
                        position,
                        len(source_paths),
                        statistics.chunks_written,
                    )

        if statistics.chunks_written == 0:
            raise ValueError("chunking produced no output chunks")

        manifest = _manifest_payload(
            config,
            statistics,
            tokenizer=tokenizer,
            tokenizer_source="local" if tokenizer_is_local else "huggingface",
            effective_tokenizer_revision=effective_tokenizer_revision,
            input_sha256=input_digest.hexdigest(),
            output_sha256=digest.hexdigest(),
            skipped_empty_document_ids=skipped_empty_document_ids,
        )
        with temporary_manifest.open(
            "w", encoding="utf-8", newline="\n"
        ) as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

        temporary_output.replace(config.output_path)
        temporary_manifest.replace(config.manifest_path)
        return manifest
    except BaseException:
        temporary_output.unlink(missing_ok=True)
        if temporary_manifest is not None:
            temporary_manifest.unlink(missing_ok=True)
        raise


def _default_manifest_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}.manifest.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="legal-ir-chunk-fixed-size",
        description=(
            "Normalize selected-contexts and create fixed-token JSONL chunks "
            "for the DSC LegalIR retrieval pipeline."
        ),
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--input-pattern", default="context_*.json")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument(
        "--tokenizer-revision",
        help=(
            "Hugging Face revision; defaults to the pinned revision only for "
            "the default remote tokenizer. Use 'none' for an unpinned model."
        ),
    )
    parser.add_argument("--tokenizer-cache-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=384)
    parser.add_argument("--overlap", type=int, default=64)
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
    manifest_path = args.manifest or _default_manifest_path(args.output)
    tokenizer_revision = args.tokenizer_revision
    if tokenizer_revision is not None and tokenizer_revision.lower() in {
        "none",
        "null",
    }:
        tokenizer_revision = None
    elif tokenizer_revision is None and args.tokenizer == DEFAULT_TOKENIZER:
        tokenizer_revision = DEFAULT_TOKENIZER_REVISION
    config = FixedSizeChunkingConfig(
        input_dir=args.input_dir,
        output_path=args.output,
        manifest_path=manifest_path,
        tokenizer_name=args.tokenizer,
        tokenizer_revision=tokenizer_revision,
        tokenizer_cache_dir=args.tokenizer_cache_dir,
        local_files_only=args.local_files_only,
        chunk_size_tokens=args.chunk_size,
        overlap_tokens=args.overlap,
        input_pattern=args.input_pattern,
        include_document_title=not args.without_document_title,
        title_max_characters=args.title_max_characters,
        log_every_documents=args.log_every,
        overwrite=args.overwrite,
    )
    manifest = build_fixed_size_chunks(config)
    LOGGER.info(
        "Finished: %s chunks -> %s",
        manifest["statistics"]["chunks_written"],
        config.output_path,
    )
    LOGGER.info("Manifest: %s", config.manifest_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
