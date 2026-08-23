from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from multilingual_rag.config import load_settings
from multilingual_rag.db import check_database, list_documents
from multilingual_rag.embeddings import embed_text
from multilingual_rag.rag import RetrievalResult, retrieve_for_question


settings = load_settings()
retrieval_cache: dict[tuple[str, int, str], RetrievalResult] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    start = time.perf_counter()
    embed_text(settings.embedding_model_name, "warmup")
    app.state.embedding_warmup_seconds = round(time.perf_counter() - start, 4)
    yield


app = FastAPI(title="Multilingual RAG API", version="0.1.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    no_generate: bool = True
    retrieval_mode: str = Field(default="hybrid", pattern="^(vector|keyword|hybrid)$")


class SourceResponse(BaseModel):
    rank: int
    score: float
    chunk_id: str
    document_id: str
    document_name: str
    page_number: int | None
    language: str | None
    retrieval_source: str
    vector_score: float | None
    keyword_score: float | None
    preview: str


class AskResponse(BaseModel):
    question: str
    query_language: str
    retrieval_mode: str
    top_k: int
    cache_hit: bool
    latency_ms: float
    sources: list[SourceResponse]
    prompt_preview: str
    answer: str | None = None


def _source_response(rank: int, chunk: Any) -> SourceResponse:
    return SourceResponse(
        rank=rank,
        score=round(chunk.score, 4),
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        document_name=chunk.document_name,
        page_number=chunk.page_number,
        language=chunk.language,
        retrieval_source=chunk.retrieval_source,
        vector_score=None if chunk.vector_score is None else round(chunk.vector_score, 4),
        keyword_score=None if chunk.keyword_score is None else round(chunk.keyword_score, 4),
        preview=chunk.text[:220].replace("\n", " "),
    )


@app.get("/health")
def health() -> dict[str, Any]:
    db = check_database(settings.database_url)
    return {
        "status": "ok",
        "embedding_model": settings.embedding_model_name,
        "embedding_warmup_seconds": getattr(app.state, "embedding_warmup_seconds", None),
        "database_connected": db.connected,
        "pgvector_installed": db.vector_extension_installed,
        "missing_tables": sorted(db.missing_tables),
    }


@app.get("/documents")
def documents() -> list[dict[str, Any]]:
    return list_documents(settings.database_url)


@app.post("/ask")
def ask(request: AskRequest) -> AskResponse:
    start = time.perf_counter()
    normalized_question = " ".join(request.question.split())
    cache_key = (
        normalized_question.casefold(),
        request.top_k,
        request.retrieval_mode,
        settings.embedding_model_name,
    )
    cache_hit = cache_key in retrieval_cache

    if cache_hit:
        result = retrieval_cache[cache_key]
    else:
        result = retrieve_for_question(
            settings=settings,
            question=normalized_question,
            top_k=request.top_k,
            retrieval_mode=request.retrieval_mode,
        )
        retrieval_cache[cache_key] = result

    return AskResponse(
        question=result.question,
        query_language=result.query_language.language,
        retrieval_mode=result.retrieval_mode,
        top_k=len(result.chunks),
        cache_hit=cache_hit,
        latency_ms=round((time.perf_counter() - start) * 1000, 2),
        sources=[
            _source_response(rank, chunk)
            for rank, chunk in enumerate(result.chunks, start=1)
        ],
        prompt_preview=result.prompt[:1600],
        answer=None if request.no_generate else "Generation via llama-server is next.",
    )


@app.get("/cache")
def cache_status() -> dict[str, Any]:
    return {"retrieval_cache_entries": len(retrieval_cache)}


@app.delete("/cache")
def clear_cache() -> dict[str, Any]:
    retrieval_cache.clear()
    return {"retrieval_cache_entries": 0}
