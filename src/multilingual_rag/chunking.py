from __future__ import annotations

import re
from dataclasses import dataclass
from uuid import UUID


DEFAULT_TARGET_TOKENS = 450
DEFAULT_OVERLAP_TOKENS = 75


@dataclass(frozen=True)
class SourcePage:
    page_number: int | None
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    page_number: int | None
    chunk_index: int
    text: str
    token_count: int


def estimate_tokens(text: str) -> int:
    return len(re.findall(r"\S+", text))


def split_paragraphs(text: str) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if paragraphs:
        return paragraphs
    return [text.strip()] if text.strip() else []


def _split_long_text(text: str, target_tokens: int) -> list[str]:
    words = re.findall(r"\S+", text)
    if not words:
        return []

    return [
        " ".join(words[start : start + target_tokens])
        for start in range(0, len(words), target_tokens)
    ]


def _overlap_tail(text: str, overlap_tokens: int) -> str:
    if overlap_tokens <= 0:
        return ""

    words = re.findall(r"\S+", text)
    return " ".join(words[-overlap_tokens:])


def chunk_pages(
    document_id: str,
    pages: list[SourcePage],
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[Chunk]:
    if target_tokens <= 0:
        raise ValueError("target_tokens must be greater than 0")

    if overlap_tokens < 0:
        raise ValueError("overlap_tokens must be 0 or greater")

    if overlap_tokens >= target_tokens:
        raise ValueError("overlap_tokens must be smaller than target_tokens")

    doc_prefix = UUID(document_id).hex[:12]
    chunks: list[Chunk] = []
    chunk_index = 0

    for page in pages:
        current_parts: list[str] = []
        current_tokens = 0

        for paragraph in split_paragraphs(page.text):
            paragraph_tokens = estimate_tokens(paragraph)
            paragraph_parts = (
                _split_long_text(paragraph, target_tokens)
                if paragraph_tokens > target_tokens
                else [paragraph]
            )

            for part in paragraph_parts:
                part_tokens = estimate_tokens(part)
                if current_parts and current_tokens + part_tokens > target_tokens:
                    chunk_text = "\n\n".join(current_parts).strip()
                    chunks.append(
                        _build_chunk(
                            doc_prefix=doc_prefix,
                            page_number=page.page_number,
                            chunk_index=chunk_index,
                            text=chunk_text,
                        )
                    )
                    chunk_index += 1

                    overlap = _overlap_tail(chunk_text, overlap_tokens)
                    current_parts = [overlap] if overlap else []
                    current_tokens = estimate_tokens(overlap)

                current_parts.append(part)
                current_tokens += part_tokens

        if current_parts:
            chunk_text = "\n\n".join(current_parts).strip()
            chunks.append(
                _build_chunk(
                    doc_prefix=doc_prefix,
                    page_number=page.page_number,
                    chunk_index=chunk_index,
                    text=chunk_text,
                )
            )
            chunk_index += 1

    return chunks


def _build_chunk(
    doc_prefix: str,
    page_number: int | None,
    chunk_index: int,
    text: str,
) -> Chunk:
    page_part = f"p{page_number:04d}" if page_number is not None else "pnone"
    chunk_id = f"doc_{doc_prefix}_{page_part}_c{chunk_index:04d}"
    return Chunk(
        chunk_id=chunk_id,
        page_number=page_number,
        chunk_index=chunk_index,
        text=text,
        token_count=estimate_tokens(text),
    )

