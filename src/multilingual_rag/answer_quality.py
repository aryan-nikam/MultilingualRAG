from __future__ import annotations

import re
from typing import Any

from multilingual_rag.db import RetrievedChunk
from multilingual_rag.language import detect_language


SOURCE_MARKER_PATTERN = re.compile(r"\[S(\d+)\]")
UNKNOWN_ANSWER_MARKERS = {
    "i do not know",
    "i don't know",
    "do not know",
    "don't know",
    "मुझे नहीं पता",
    "माहित नाही",
}


def build_citations(chunks: list[RetrievedChunk]) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks, start=1):
        citations.append(
            {
                "source_id": f"S{index}",
                "document_name": chunk.document_name,
                "page_number": chunk.page_number,
                "chunk_id": chunk.chunk_id,
            }
        )
    return citations


def validate_rag_response(
    chunks: list[RetrievedChunk],
    answer: str | None,
    no_generate: bool,
    expected_language: str | None = None,
) -> dict[str, Any]:
    has_context = bool(chunks)
    has_answer = bool(answer and answer.strip())
    citations = build_citations(chunks)
    valid_source_ids = {citation["source_id"] for citation in citations}
    cited_source_ids = _extract_source_ids(answer or "")
    citation_required = has_context and has_answer and not _is_unknown_answer(answer or "")
    detected_answer_language = detect_language(answer).language if has_answer else None

    return {
        "has_context": has_context,
        "has_answer": has_answer,
        "generation_skipped": no_generate,
        "citation_count": len(citations),
        "source_presence_ok": has_context and bool(citations),
        "empty_context_ok": has_context or not has_answer,
        "cited_source_ids": sorted(cited_source_ids),
        "citation_markers_valid": cited_source_ids <= valid_source_ids,
        "answer_has_required_citation": (not citation_required) or bool(cited_source_ids),
        "answer_language": detected_answer_language,
        "answer_language_ok": (not has_answer)
        or expected_language is None
        or expected_language == "unknown"
        or detected_answer_language == expected_language,
        "unsupported_claim_check": "skipped_generation"
        if no_generate
        else _basic_unsupported_claim_check(answer or "", cited_source_ids, citation_required),
    }


def enforce_answer_citations(answer: str, citations: list[dict[str, Any]]) -> str:
    clean_answer = answer.strip()
    if not clean_answer or _is_unknown_answer(clean_answer):
        return clean_answer

    if _extract_source_ids(clean_answer):
        return clean_answer

    source_ids = " ".join(f"[{citation['source_id']}]" for citation in citations[:2])
    return f"{clean_answer} {source_ids}".strip()


def chunks_for_query_log(chunks: list[RetrievedChunk]) -> list[dict[str, Any]]:
    return [
        {
            "rank": index,
            "score": chunk.score,
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "document_name": chunk.document_name,
            "page_number": chunk.page_number,
            "language": chunk.language,
            "retrieval_source": chunk.retrieval_source,
            "vector_score": chunk.vector_score,
            "keyword_score": chunk.keyword_score,
            "rerank_score": chunk.rerank_score,
        }
        for index, chunk in enumerate(chunks, start=1)
    ]


def _extract_source_ids(answer: str) -> set[str]:
    return {f"S{match}" for match in SOURCE_MARKER_PATTERN.findall(answer)}


def _is_unknown_answer(answer: str) -> bool:
    normalized = answer.casefold()
    return any(marker in normalized for marker in UNKNOWN_ANSWER_MARKERS)


def _basic_unsupported_claim_check(
    answer: str,
    cited_source_ids: set[str],
    citation_required: bool,
) -> str:
    if not citation_required:
        return "not_required"

    if cited_source_ids:
        return "citation_present_basic_check_passed"

    return "missing_citation_basic_check_failed"
