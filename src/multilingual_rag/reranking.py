from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

from sentence_transformers import CrossEncoder

from multilingual_rag.db import RetrievedChunk


@lru_cache(maxsize=2)
def _load_reranker(model_name: str) -> CrossEncoder:
    return CrossEncoder(model_name)


def warmup_reranker(model_name: str) -> None:
    reranker = _load_reranker(model_name)
    reranker.predict([("warmup question", "warmup passage")])


def rerank_chunks(
    model_name: str,
    question: str,
    chunks: list[RetrievedChunk],
    top_k: int,
) -> list[RetrievedChunk]:
    if not chunks:
        return []

    reranker = _load_reranker(model_name)
    pairs = [(question, _reranker_passage(chunk)) for chunk in chunks]
    scores = reranker.predict(pairs)
    rerank_scores = [float(score) for score in scores]
    normalized_rerank_scores = _normalize(rerank_scores)
    normalized_base_scores = _normalize([chunk.score for chunk in chunks])

    reranked = [
        replace(
            chunk,
            score=(0.75 * rerank_score) + (0.25 * base_score),
            rerank_score=raw_score,
        )
        for chunk, raw_score, rerank_score, base_score in zip(
            chunks,
            rerank_scores,
            normalized_rerank_scores,
            normalized_base_scores,
        )
    ]
    return sorted(reranked, key=lambda item: item.score, reverse=True)[:top_k]


def _reranker_passage(chunk: RetrievedChunk) -> str:
    page = chunk.page_number if chunk.page_number is not None else "unknown"
    document_name = chunk.document_name.replace("_", " ")
    language = chunk.language or "unknown"
    return (
        f"Document: {document_name}\n"
        f"Language: {language}\n"
        f"Page: {page}\n\n"
        f"{chunk.text}"
    )


def _normalize(scores: list[float]) -> list[float]:
    if not scores:
        return []

    lowest = min(scores)
    highest = max(scores)
    if highest == lowest:
        return [1.0 for _ in scores]

    return [(score - lowest) / (highest - lowest) for score in scores]
