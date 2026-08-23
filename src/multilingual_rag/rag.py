from __future__ import annotations

from dataclasses import dataclass, replace

from multilingual_rag.config import Settings
from multilingual_rag.db import (
    RetrievedChunk,
    retrieve_keyword_chunks,
    retrieve_similar_chunks,
)
from multilingual_rag.embeddings import embed_text
from multilingual_rag.language import LanguageDetection, detect_language
from multilingual_rag.reranking import rerank_chunks


RETRIEVAL_MODES = {"vector", "keyword", "hybrid"}


@dataclass(frozen=True)
class RetrievalResult:
    question: str
    query_language: LanguageDetection
    retrieval_mode: str
    rerank: bool
    chunks: list[RetrievedChunk]
    prompt: str


def build_rag_prompt(
    question: str,
    query_language: str,
    chunks: list[RetrievedChunk],
) -> str:
    context_parts = []
    for index, chunk in enumerate(chunks, start=1):
        page = chunk.page_number if chunk.page_number is not None else "unknown"
        context_parts.append(
            f"[Source {index}: {chunk.document_name}, page {page}]\n{chunk.text}"
        )

    context = "\n\n".join(context_parts)
    return (
        "Answer using only the provided context. "
        "If the answer is not present, say you do not know. "
        f"Respond in the user's language. User language code: {query_language}.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{question}\n\n"
        "ANSWER:"
    )


def retrieve_for_question(
    settings: Settings,
    question: str,
    top_k: int | None = None,
    retrieval_mode: str = "hybrid",
    rerank: bool | None = None,
) -> RetrievalResult:
    if retrieval_mode not in RETRIEVAL_MODES:
        raise ValueError(f"unsupported retrieval_mode: {retrieval_mode}")

    effective_top_k = top_k or settings.retrieval_top_k
    use_reranker = settings.reranker_enabled if rerank is None else rerank
    query_language = detect_language(question)
    candidate_limit = _candidate_limit(effective_top_k)

    if retrieval_mode == "keyword":
        limit = candidate_limit if use_reranker else effective_top_k
        chunks = retrieve_keyword_chunks(settings.database_url, question, limit)
    else:
        limit = candidate_limit if retrieval_mode == "hybrid" or use_reranker else effective_top_k
        query_vector = embed_text(settings.embedding_model_name, question)
        vector_candidates = retrieve_similar_chunks(
            settings.database_url,
            settings.embedding_model_name,
            query_vector,
            limit,
        )

        if retrieval_mode == "vector":
            chunks = vector_candidates
        else:
            keyword_candidates = retrieve_keyword_chunks(
                settings.database_url,
                question,
                candidate_limit,
            )
            chunks = _merge_hybrid_candidates(
                vector_candidates=vector_candidates,
                keyword_candidates=keyword_candidates,
                top_k=candidate_limit,
            )

    if use_reranker:
        chunks = rerank_chunks(
            model_name=settings.reranker_model_name,
            question=question,
            chunks=chunks,
            top_k=effective_top_k,
        )
    else:
        chunks = chunks[:effective_top_k]

    prompt = build_rag_prompt(question, query_language.language, chunks)

    return RetrievalResult(
        question=question,
        query_language=query_language,
        retrieval_mode=retrieval_mode,
        rerank=use_reranker,
        chunks=chunks,
        prompt=prompt,
    )


def _candidate_limit(top_k: int) -> int:
    return max(top_k * 4, 20)


def _merge_hybrid_candidates(
    vector_candidates: list[RetrievedChunk],
    keyword_candidates: list[RetrievedChunk],
    top_k: int,
) -> list[RetrievedChunk]:
    max_vector_score = max((chunk.score for chunk in vector_candidates), default=1.0)
    max_keyword_score = max((chunk.score for chunk in keyword_candidates), default=1.0)
    merged: dict[str, RetrievedChunk] = {}
    vector_scores: dict[str, float] = {}
    keyword_scores: dict[str, float] = {}

    for chunk in vector_candidates:
        vector_scores[chunk.chunk_id] = chunk.score / max_vector_score if max_vector_score else 0.0
        merged[chunk.chunk_id] = chunk

    for chunk in keyword_candidates:
        keyword_scores[chunk.chunk_id] = chunk.score / max_keyword_score if max_keyword_score else 0.0
        merged.setdefault(chunk.chunk_id, chunk)

    scored_chunks: list[RetrievedChunk] = []
    for chunk_id, chunk in merged.items():
        vector_score = vector_scores.get(chunk_id, 0.0)
        keyword_score = keyword_scores.get(chunk_id, 0.0)
        combined_score = (0.7 * vector_score) + (0.3 * keyword_score)
        source = "hybrid"
        if vector_score and not keyword_score:
            source = "vector"
        elif keyword_score and not vector_score:
            source = "keyword"

        scored_chunks.append(
            replace(
                chunk,
                score=combined_score,
                vector_score=vector_score or None,
                keyword_score=keyword_score or None,
                retrieval_source=source,
            )
        )

    return sorted(scored_chunks, key=lambda item: item.score, reverse=True)[:top_k]
