from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from multilingual_rag.chunking import (
    DEFAULT_OVERLAP_TOKENS,
    DEFAULT_TARGET_TOKENS,
    chunk_pages,
)
from multilingual_rag.config import Settings
from multilingual_rag.db import (
    get_document_name,
    get_document_text,
    get_source_pages,
    store_chunks,
    store_embeddings,
    store_parsed_document,
    update_document_status,
    update_document_and_chunk_language,
)
from multilingual_rag.embeddings import embed_texts
from multilingual_rag.language import detect_language
from multilingual_rag.parsers import parse_document


@dataclass(frozen=True)
class ProcessedDocument:
    document_id: str
    document_name: str
    language: str
    pages: int
    chunks: int
    embeddings: int


def ingest_and_process_file(
    settings: Settings,
    path: Path,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> ProcessedDocument:
    parsed = parse_document(path)
    stored = store_parsed_document(settings.database_url, parsed)
    return process_document(
        settings=settings,
        document_id=stored.document_id,
        target_tokens=target_tokens,
        overlap_tokens=overlap_tokens,
    )


def process_document(
    settings: Settings,
    document_id: str,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> ProcessedDocument:
    try:
        update_document_status(settings.database_url, document_id, "processing")
        pages = get_source_pages(settings.database_url, document_id)
        chunks = chunk_pages(
            document_id=document_id,
            pages=pages,
            target_tokens=target_tokens,
            overlap_tokens=overlap_tokens,
        )
        chunk_result = store_chunks(settings.database_url, document_id, chunks)

        text = get_document_text(settings.database_url, document_id)
        language = detect_language(text).language
        update_document_and_chunk_language(settings.database_url, document_id, language)

        vectors = embed_texts(settings.embedding_model_name, [chunk.text for chunk in chunks])
        embedding_result = store_embeddings(
            settings.database_url,
            settings.embedding_model_name,
            [(chunk.chunk_id, vector) for chunk, vector in zip(chunks, vectors)],
        )

        update_document_status(settings.database_url, document_id, "ready")
        return ProcessedDocument(
            document_id=document_id,
            document_name=get_document_name(settings.database_url, document_id),
            language=language,
            pages=len(pages),
            chunks=chunk_result.chunk_count,
            embeddings=embedding_result.stored_count,
        )
    except Exception as exc:
        update_document_status(settings.database_url, document_id, "failed", str(exc))
        raise
