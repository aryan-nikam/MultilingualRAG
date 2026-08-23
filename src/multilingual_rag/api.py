from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from multilingual_rag.answer_quality import (
    build_citations,
    chunks_for_query_log,
    enforce_answer_citations,
    validate_rag_response,
)
from multilingual_rag.config import PROJECT_ROOT, load_settings
from multilingual_rag.db import (
    check_database,
    delete_document,
    list_documents,
    log_query_result,
    store_parsed_document,
)
from multilingual_rag.embeddings import embed_text
from multilingual_rag.llama_server import generate_with_llama_server, llama_server_health
from multilingual_rag.parsers import SUPPORTED_EXTENSIONS, parse_document
from multilingual_rag.processing import ProcessedDocument, process_document
from multilingual_rag.reranking import warmup_reranker
from multilingual_rag.rag import RetrievalResult, retrieve_for_question


settings = load_settings()
retrieval_cache: dict[tuple[str, int, str, bool, str, str], RetrievalResult] = {}
UPLOAD_DIR = Path("data/input/uploads")


@asynccontextmanager
async def lifespan(app: FastAPI):
    start = time.perf_counter()
    embed_text(settings.embedding_model_name, "warmup")
    if settings.reranker_enabled:
        warmup_reranker(settings.reranker_model_name)
    app.state.embedding_warmup_seconds = round(time.perf_counter() - start, 4)
    yield


app = FastAPI(title="Multilingual RAG API", version="0.1.0", lifespan=lifespan)
FRONTEND_DIR = PROJECT_ROOT / "frontend"
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    no_generate: bool = True
    retrieval_mode: str = Field(default="hybrid", pattern="^(vector|keyword|hybrid)$")
    rerank: bool | None = None


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
    rerank_score: float | None
    preview: str


class CitationResponse(BaseModel):
    source_id: str
    document_name: str
    page_number: int | None
    chunk_id: str


class AskResponse(BaseModel):
    query_id: str | None
    question: str
    query_language: str
    retrieval_mode: str
    rerank: bool
    reranker_model: str | None
    top_k: int
    cache_hit: bool
    latency_ms: float
    sources: list[SourceResponse]
    citations: list[CitationResponse]
    validation: dict[str, Any]
    prompt_preview: str
    answer: str | None = None


class DocumentUploadResponse(BaseModel):
    document_id: str
    document_name: str
    file_type: str
    pages: int
    status: str


class ProcessDocumentResponse(BaseModel):
    document_id: str
    document_name: str
    language: str
    pages: int
    chunks: int
    embeddings: int
    status: str


class DeleteDocumentResponse(BaseModel):
    document_id: str
    deleted: bool


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
        rerank_score=None if chunk.rerank_score is None else round(chunk.rerank_score, 4),
        preview=chunk.text[:220].replace("\n", " "),
    )


@app.get("/health")
def health() -> dict[str, Any]:
    db = check_database(settings.database_url)
    return {
        "status": "ok",
        "embedding_model": settings.embedding_model_name,
        "reranker_model": settings.reranker_model_name
        if settings.reranker_enabled
        else None,
        "llama_server_url": settings.llama_server_url,
        "llama_server_reachable": llama_server_health(settings),
        "embedding_warmup_seconds": getattr(app.state, "embedding_warmup_seconds", None),
        "database_connected": db.connected,
        "pgvector_installed": db.vector_extension_installed,
        "missing_tables": sorted(db.missing_tables),
    }


@app.get("/", response_class=HTMLResponse)
def frontend() -> str:
    return (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/documents")
def documents() -> list[dict[str, Any]]:
    return list_documents(settings.database_url)


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)) -> DocumentUploadResponse:
    filename = Path(file.filename or "").name
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"unsupported file type: {extension}")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    target_path = UPLOAD_DIR / filename
    content = await file.read()
    target_path.write_bytes(content)

    try:
        parsed = parse_document(target_path)
        stored = store_parsed_document(settings.database_url, parsed)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"upload parse failed: {exc}") from exc

    retrieval_cache.clear()
    return DocumentUploadResponse(
        document_id=stored.document_id,
        document_name=stored.document_name,
        file_type=stored.file_type,
        pages=stored.page_count,
        status="parsed",
    )


@app.post("/documents/{document_id}/process")
def process_uploaded_document(document_id: str) -> ProcessDocumentResponse:
    try:
        result = process_document(settings=settings, document_id=document_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"document processing failed: {exc}") from exc

    retrieval_cache.clear()
    return _process_document_response(result)


@app.delete("/documents/{document_id}")
def remove_document(document_id: str) -> DeleteDocumentResponse:
    deleted = delete_document(settings.database_url, document_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="document not found")

    retrieval_cache.clear()
    return DeleteDocumentResponse(document_id=document_id, deleted=True)


@app.post("/ask")
def ask(request: AskRequest) -> AskResponse:
    return _ask(request)


@app.post("/chat")
def chat(request: AskRequest) -> AskResponse:
    return _ask(request)


def _ask(request: AskRequest) -> AskResponse:
    start = time.perf_counter()
    normalized_question = " ".join(request.question.split())
    cache_key = (
        normalized_question.casefold(),
        request.top_k,
        request.retrieval_mode,
        settings.reranker_enabled if request.rerank is None else request.rerank,
        settings.embedding_model_name,
        settings.reranker_model_name,
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
            rerank=request.rerank,
        )
        retrieval_cache[cache_key] = result

    citations = build_citations(result.chunks)
    answer = None
    if not request.no_generate:
        try:
            answer = generate_with_llama_server(settings, result.prompt).text
        except (ConnectionError, TimeoutError, ValueError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        answer = enforce_answer_citations(answer, citations)

    validation = validate_rag_response(
        result.chunks,
        answer,
        request.no_generate,
        expected_language=result.query_language.language,
    )
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    query_id = log_query_result(
        database_url=settings.database_url,
        question=result.question,
        query_language=result.query_language.language,
        retrieval_mode=result.retrieval_mode,
        rerank=result.rerank,
        retrieved_chunks=chunks_for_query_log(result.chunks),
        answer=answer,
        citations=citations,
        validation=validation,
        latency_ms=latency_ms,
    ).query_id

    return AskResponse(
        query_id=query_id,
        question=result.question,
        query_language=result.query_language.language,
        retrieval_mode=result.retrieval_mode,
        rerank=result.rerank,
        reranker_model=settings.reranker_model_name if result.rerank else None,
        top_k=len(result.chunks),
        cache_hit=cache_hit,
        latency_ms=latency_ms,
        sources=[
            _source_response(rank, chunk)
            for rank, chunk in enumerate(result.chunks, start=1)
        ],
        citations=[CitationResponse(**citation) for citation in citations],
        validation=validation,
        prompt_preview=result.prompt[:1600],
        answer=answer,
    )


@app.get("/cache")
def cache_status() -> dict[str, Any]:
    return {"retrieval_cache_entries": len(retrieval_cache)}


@app.delete("/cache")
def clear_cache() -> dict[str, Any]:
    retrieval_cache.clear()
    return {"retrieval_cache_entries": 0}


def _process_document_response(result: ProcessedDocument) -> ProcessDocumentResponse:
    return ProcessDocumentResponse(
        document_id=result.document_id,
        document_name=result.document_name,
        language=result.language,
        pages=result.pages,
        chunks=result.chunks,
        embeddings=result.embeddings,
        status="ready",
    )
