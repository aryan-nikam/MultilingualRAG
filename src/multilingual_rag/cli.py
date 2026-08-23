from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from multilingual_rag.chunking import (
    DEFAULT_OVERLAP_TOKENS,
    DEFAULT_TARGET_TOKENS,
    chunk_pages,
)
from multilingual_rag.config import load_settings
from multilingual_rag.db import (
    check_database,
    get_chunks,
    get_document_pages,
    get_document_text,
    get_source_pages,
    get_chunks_for_embedding,
    list_documents,
    run_smoke_test,
    store_chunks,
    store_embeddings,
    store_parsed_document,
    update_document_and_chunk_language,
)
from multilingual_rag.embeddings import embed_text, embed_texts
from multilingual_rag.language import detect_language
from multilingual_rag.llama_cpp import run_prompt
from multilingual_rag.parsers import iter_supported_files, parse_document
from multilingual_rag.rag import retrieve_for_question


def _status_line(label: str, ok: bool, detail: str) -> str:
    marker = "OK" if ok else "MISSING"
    return f"[{marker}] {label}: {detail}"


def env_check() -> int:
    settings = load_settings()

    checks = [
        (
            "llama.cpp directory",
            settings.llama_cpp_dir.exists(),
            str(settings.llama_cpp_dir),
        ),
        (
            "llama-cli executable",
            settings.llama_cli_path.exists(),
            str(settings.llama_cli_path),
        ),
        (
            "LLM model",
            settings.llama_model_path.exists(),
            str(settings.llama_model_path),
        ),
        (
            "psql on PATH",
            shutil.which("psql") is not None,
            shutil.which("psql") or "not found",
        ),
    ]

    for label, ok, detail in checks:
        print(_status_line(label, ok, detail))

    print(f"[INFO] database_url: {settings.database_url}")
    return 0 if all(ok for _, ok, _ in checks[:3]) else 1


def llm_prompt(prompt: str, max_tokens: int | None, temperature: float | None) -> int:
    settings = load_settings()

    if not settings.llama_cli_path.exists():
        print(f"llama-cli not found: {settings.llama_cli_path}", file=sys.stderr)
        return 1

    if not settings.llama_model_path.exists():
        print(f"model not found: {settings.llama_model_path}", file=sys.stderr)
        return 1

    result = run_prompt(
        settings=settings,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    if result.stderr:
        print(result.stderr, file=sys.stderr)

    if result.stdout:
        print(result.stdout)

    return result.returncode


def db_check() -> int:
    settings = load_settings()

    try:
        result = check_database(settings.database_url)
    except Exception as exc:
        print(f"[MISSING] database connection: {exc}")
        return 1

    print("[OK] database connection")
    print(f"[INFO] postgres_version: {result.postgres_version}")

    if result.vector_extension_installed:
        print("[OK] pgvector extension: installed")
    else:
        print("[MISSING] pgvector extension: not installed")

    for table in sorted(result.existing_tables):
        print(f"[OK] table: {table}")

    for table in sorted(result.missing_tables):
        print(f"[MISSING] table: {table}")

    return 0 if result.vector_extension_installed and not result.missing_tables else 1


def db_smoke_test() -> int:
    settings = load_settings()

    try:
        result = run_smoke_test(settings.database_url)
    except Exception as exc:
        print(f"[MISSING] database smoke test: {exc}")
        return 1

    print("[OK] database smoke test")
    print(f"[INFO] inserted_document_id: {result.document_id}")
    print(f"[INFO] inserted_chunk_id: {result.chunk_id}")
    print(f"[INFO] retrieved_text: {result.retrieved_text}")
    print("[OK] cleanup: smoke-test document deleted")
    return 0


def ingest(path: str) -> int:
    settings = load_settings()

    try:
        files = iter_supported_files(Path(path))
    except Exception as exc:
        print(f"[MISSING] ingest path: {exc}")
        return 1

    if not files:
        print("[MISSING] ingest files: no supported PDF/TXT files found")
        return 1

    exit_code = 0
    print(f"[INFO] ingest_files: {len(files)}")

    for file_path in files:
        try:
            parsed = parse_document(file_path)
            stored = store_parsed_document(settings.database_url, parsed)
        except Exception as exc:
            print(f"[MISSING] ingest failed: {file_path} :: {exc}")
            exit_code = 1
            continue

        print(
            "[OK] ingested "
            f"document_id={stored.document_id} "
            f"name={stored.document_name} "
            f"type={stored.file_type} "
            f"pages={stored.page_count} "
            f"chars={stored.character_count}"
        )

    return exit_code


def show_pages(document_id: str) -> int:
    settings = load_settings()

    try:
        pages = get_document_pages(settings.database_url, document_id)
    except Exception as exc:
        print(f"[MISSING] show pages: {exc}")
        return 1

    if not pages:
        print(f"[MISSING] pages: no pages found for document_id={document_id}")
        return 1

    for page in pages:
        print(
            "[OK] page "
            f"number={page.page_number} "
            f"parser={page.parser_name} "
            f"chars={page.character_count} "
            f"preview={page.preview}"
        )

    return 0


def chunk_document(document_id: str, target_tokens: int, overlap_tokens: int) -> int:
    settings = load_settings()

    try:
        pages = get_source_pages(settings.database_url, document_id)
    except Exception as exc:
        print(f"[MISSING] chunk pages: {exc}")
        return 1

    if not pages:
        print(f"[MISSING] chunk pages: no parsed pages found for document_id={document_id}")
        return 1

    try:
        chunks = chunk_pages(
            document_id=document_id,
            pages=pages,
            target_tokens=target_tokens,
            overlap_tokens=overlap_tokens,
        )
        result = store_chunks(settings.database_url, document_id, chunks)
    except Exception as exc:
        print(f"[MISSING] chunk failed: {exc}")
        return 1

    print(
        "[OK] chunked "
        f"document_id={result.document_id} "
        f"chunks={result.chunk_count} "
        f"target_tokens={target_tokens} "
        f"overlap_tokens={overlap_tokens}"
    )
    return 0


def show_chunks(document_id: str) -> int:
    settings = load_settings()

    try:
        chunks = get_chunks(settings.database_url, document_id)
    except Exception as exc:
        print(f"[MISSING] show chunks: {exc}")
        return 1

    if not chunks:
        print(f"[MISSING] chunks: no chunks found for document_id={document_id}")
        return 1

    for chunk in chunks:
        print(
            "[OK] chunk "
            f"id={chunk.chunk_id} "
            f"index={chunk.chunk_index} "
            f"page={chunk.page_number} "
            f"tokens={chunk.token_count} "
            f"chars={chunk.character_count} "
            f"preview={chunk.preview}"
        )

    return 0


def list_docs() -> int:
    settings = load_settings()
    try:
        documents = list_documents(settings.database_url)
    except Exception as exc:
        print(f"[MISSING] list documents: {exc}")
        return 1

    if not documents:
        print("[MISSING] documents: no documents found")
        return 1

    for document in documents:
        print(
            "[OK] document "
            f"id={document['document_id']} "
            f"name={document['document_name']} "
            f"type={document['file_type']} "
            f"language={document['language']} "
            f"status={document['status']} "
            f"pages={document['pages']} "
            f"chunks={document['chunks']}"
        )

    return 0


def detect_text_language(text: str) -> int:
    result = detect_language(text)
    print(f"[OK] language={result.language} confidence={result.confidence}")
    return 0


def detect_document_language(document_id: str) -> int:
    settings = load_settings()

    try:
        text = get_document_text(settings.database_url, document_id)
        result = detect_language(text)
        update = update_document_and_chunk_language(
            settings.database_url,
            document_id,
            result.language,
        )
    except Exception as exc:
        print(f"[MISSING] detect document language: {exc}")
        return 1

    print(
        "[OK] document language "
        f"document_id={update.document_id} "
        f"language={update.language} "
        f"confidence={result.confidence} "
        f"chunks_updated={update.chunk_count}"
    )
    return 0


def embed_chunks(document_id: str | None = None) -> int:
    settings = load_settings()

    try:
        chunks = get_chunks_for_embedding(settings.database_url, document_id)
    except Exception as exc:
        print(f"[MISSING] load chunks for embedding: {exc}")
        return 1

    if not chunks:
        print("[MISSING] chunks: no chunks found to embed")
        return 1

    try:
        vectors = embed_texts(
            settings.embedding_model_name,
            [chunk.text for chunk in chunks],
        )
    except Exception as exc:
        print(f"[MISSING] embedding model: {exc}")
        return 1

    wrong_dimensions = [
        len(vector)
        for vector in vectors
        if len(vector) != settings.embedding_dimension
    ]
    if wrong_dimensions:
        print(
            "[MISSING] embedding dimension mismatch: "
            f"expected={settings.embedding_dimension} got={wrong_dimensions[0]}"
        )
        return 1

    result = store_embeddings(
        settings.database_url,
        settings.embedding_model_name,
        [(chunk.chunk_id, vector) for chunk, vector in zip(chunks, vectors)],
    )

    print(
        "[OK] embedded chunks "
        f"model={result.embedding_model} "
        f"stored={result.stored_count}"
    )
    return 0


def embed_sample(text: str) -> int:
    settings = load_settings()

    try:
        vector = embed_text(settings.embedding_model_name, text)
    except Exception as exc:
        print(f"[MISSING] embedding sample: {exc}")
        return 1

    print(
        "[OK] embedding "
        f"model={settings.embedding_model_name} "
        f"dimension={len(vector)} "
        f"preview={vector[:5]}"
    )
    return 0


def ask(
    question: str,
    top_k: int | None,
    no_generate: bool,
    retrieval_mode: str,
    rerank: bool,
) -> int:
    settings = load_settings()

    try:
        result = retrieve_for_question(settings, question, top_k, retrieval_mode, rerank)
    except Exception as exc:
        print(f"[MISSING] retrieval: {exc}")
        return 1

    if not result.chunks:
        print("[MISSING] retrieval: no embedded chunks found")
        return 1

    print(
        "[OK] retrieval "
        f"query_language={result.query_language.language} "
        f"mode={result.retrieval_mode} "
        f"rerank={result.rerank} "
        f"top_k={len(result.chunks)}"
    )
    for index, chunk in enumerate(result.chunks, start=1):
        preview = chunk.text[:180].replace("\n", " ")
        print(
            f"[SOURCE {index}] "
            f"score={chunk.score:.4f} "
            f"document={chunk.document_name} "
            f"page={chunk.page_number} "
            f"language={chunk.language} "
            f"source={chunk.retrieval_source} "
            f"vector_score={chunk.vector_score} "
            f"keyword_score={chunk.keyword_score} "
            f"rerank_score={chunk.rerank_score} "
            f"chunk_id={chunk.chunk_id} "
            f"preview={preview}"
        )

    if no_generate:
        print("[INFO] generation skipped")
        print("[INFO] prompt_preview:")
        print(result.prompt[:1200])
        return 0

    return llm_prompt(result.prompt, settings.llama_max_tokens, settings.llama_temperature)


def serve_api(host: str, port: int) -> int:
    import uvicorn

    uvicorn.run("multilingual_rag.api:app", host=host, port=port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mrag")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("env-check", help="Check local Phase 0 prerequisites.")
    subparsers.add_parser("db-check", help="Check PostgreSQL and pgvector setup.")
    subparsers.add_parser("list-documents", help="List ingested documents.")
    subparsers.add_parser(
        "db-smoke-test",
        help="Insert, read, and delete a small document/chunk/embedding record.",
    )

    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Parse PDF/TXT files and store document/page text in PostgreSQL.",
    )
    ingest_parser.add_argument("--path", required=True)

    pages_parser = subparsers.add_parser(
        "show-pages",
        help="Show stored parsed page previews for a document.",
    )
    pages_parser.add_argument("--document-id", required=True)

    chunk_parser = subparsers.add_parser(
        "chunk",
        help="Create chunks from previously parsed document pages.",
    )
    chunk_parser.add_argument("--document-id", required=True)
    chunk_parser.add_argument("--target-tokens", type=int, default=DEFAULT_TARGET_TOKENS)
    chunk_parser.add_argument("--overlap-tokens", type=int, default=DEFAULT_OVERLAP_TOKENS)

    chunks_parser = subparsers.add_parser(
        "show-chunks",
        help="Show stored chunk previews for a document.",
    )
    chunks_parser.add_argument("--document-id", required=True)

    detect_text_parser = subparsers.add_parser(
        "detect-language",
        help="Detect language for a text string.",
    )
    detect_text_parser.add_argument("--text", required=True)

    detect_doc_parser = subparsers.add_parser(
        "detect-document-language",
        help="Detect and store language for a parsed/chunked document.",
    )
    detect_doc_parser.add_argument("--document-id", required=True)

    embed_parser = subparsers.add_parser(
        "embed-chunks",
        help="Generate and store embeddings for chunks.",
    )
    embed_parser.add_argument("--document-id", default=None)

    embed_sample_parser = subparsers.add_parser(
        "embed-sample",
        help="Generate one embedding and print its dimension.",
    )
    embed_sample_parser.add_argument("--text", required=True)

    ask_parser = subparsers.add_parser(
        "ask",
        help="Run basic semantic RAG over embedded chunks.",
    )
    ask_parser.add_argument("--question", required=True)
    ask_parser.add_argument("--top-k", type=int, default=None)
    ask_parser.add_argument(
        "--retrieval-mode",
        choices=["vector", "keyword", "hybrid"],
        default="hybrid",
    )
    ask_parser.add_argument(
        "--no-generate",
        action="store_true",
        help="Show retrieval/context only and skip llama.cpp generation.",
    )
    ask_parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Skip cross-encoder reranking and use raw retrieval scores.",
    )

    serve_parser = subparsers.add_parser(
        "serve-api",
        help="Start the local FastAPI backend.",
    )
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)

    prompt_parser = subparsers.add_parser(
        "llm-prompt",
        help="Send one prompt to the configured llama.cpp model.",
    )
    prompt_parser.add_argument("prompt")
    prompt_parser.add_argument("--max-tokens", type=int, default=None)
    prompt_parser.add_argument("--temperature", type=float, default=None)

    return parser


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = build_parser()
    args = parser.parse_args()

    if args.command == "env-check":
        return env_check()

    if args.command == "llm-prompt":
        return llm_prompt(args.prompt, args.max_tokens, args.temperature)

    if args.command == "db-check":
        return db_check()

    if args.command == "list-documents":
        return list_docs()

    if args.command == "db-smoke-test":
        return db_smoke_test()

    if args.command == "ingest":
        return ingest(args.path)

    if args.command == "show-pages":
        return show_pages(args.document_id)

    if args.command == "chunk":
        return chunk_document(args.document_id, args.target_tokens, args.overlap_tokens)

    if args.command == "show-chunks":
        return show_chunks(args.document_id)

    if args.command == "detect-language":
        return detect_text_language(args.text)

    if args.command == "detect-document-language":
        return detect_document_language(args.document_id)

    if args.command == "embed-chunks":
        return embed_chunks(args.document_id)

    if args.command == "embed-sample":
        return embed_sample(args.text)

    if args.command == "ask":
        return ask(
            args.question,
            args.top_k,
            args.no_generate,
            args.retrieval_mode,
            not args.no_rerank,
        )

    if args.command == "serve-api":
        return serve_api(args.host, args.port)

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
