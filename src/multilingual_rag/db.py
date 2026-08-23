from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import psycopg

from multilingual_rag.chunking import Chunk, SourcePage
from multilingual_rag.embeddings import vector_literal
from multilingual_rag.parsers import ParsedDocument


EXPECTED_TABLES = {
    "documents",
    "ingestion_status",
    "document_pages",
    "chunks",
    "embeddings",
}

KEYWORD_STOP_WORDS = {
    "a",
    "an",
    "and",
    "about",
    "can",
    "does",
    "how",
    "is",
    "of",
    "say",
    "the",
    "to",
    "what",
    "with",
}


def _with_connect_timeout(database_url: str, seconds: int = 3) -> str:
    parts = urlsplit(database_url)
    query = dict(parse_qsl(parts.query))
    query.setdefault("connect_timeout", str(seconds))
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


@dataclass(frozen=True)
class DatabaseCheck:
    connected: bool
    postgres_version: str | None
    vector_extension_installed: bool
    existing_tables: set[str]
    missing_tables: set[str]


@dataclass(frozen=True)
class DatabaseSmokeTest:
    document_id: str
    chunk_id: str
    retrieved_text: str


@dataclass(frozen=True)
class StoredDocument:
    document_id: str
    document_name: str
    file_type: str
    page_count: int
    character_count: int


@dataclass(frozen=True)
class StoredPage:
    page_number: int
    parser_name: str | None
    character_count: int
    preview: str


@dataclass(frozen=True)
class StoredChunk:
    chunk_id: str
    page_number: int | None
    chunk_index: int
    token_count: int | None
    character_count: int
    preview: str


@dataclass(frozen=True)
class ChunkingResult:
    document_id: str
    chunk_count: int


@dataclass(frozen=True)
class DocumentLanguageUpdate:
    document_id: str
    language: str
    chunk_count: int


@dataclass(frozen=True)
class ChunkForEmbedding:
    chunk_id: str
    text: str


@dataclass(frozen=True)
class EmbeddingResult:
    embedding_model: str
    stored_count: int


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    document_id: str
    document_name: str
    page_number: int | None
    language: str | None
    text: str
    score: float
    vector_score: float | None = None
    keyword_score: float | None = None
    rerank_score: float | None = None
    retrieval_source: str = "vector"


def check_database(database_url: str) -> DatabaseCheck:
    with psycopg.connect(_with_connect_timeout(database_url)) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT version();")
            postgres_version = cursor.fetchone()[0]

            cursor.execute(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM pg_extension
                    WHERE extname = 'vector'
                );
                """
            )
            vector_extension_installed = bool(cursor.fetchone()[0])

            cursor.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public';
                """
            )
            existing_tables = {row[0] for row in cursor.fetchall()}

    return DatabaseCheck(
        connected=True,
        postgres_version=postgres_version,
        vector_extension_installed=vector_extension_installed,
        existing_tables=existing_tables,
        missing_tables=EXPECTED_TABLES - existing_tables,
    )


def run_smoke_test(database_url: str) -> DatabaseSmokeTest:
    smoke_id = uuid4().hex
    chunk_id = f"smoke_{smoke_id}"
    embedding = "[" + ",".join(["0"] * 1024) + "]"

    with psycopg.connect(_with_connect_timeout(database_url)) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO documents (document_name, file_type, source_path, language, status)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING document_id;
                """,
                (
                    "smoke-test.txt",
                    "txt",
                    "data/input/smoke-test.txt",
                    "en",
                    "ready",
                ),
            )
            document_id = cursor.fetchone()[0]

            cursor.execute(
                """
                INSERT INTO ingestion_status (document_id, stage, status, message)
                VALUES (%s, %s, %s, %s);
                """,
                (document_id, "smoke_test", "done", "Smoke test inserted a document."),
            )

            cursor.execute(
                """
                INSERT INTO document_pages (document_id, page_number, text, parser_name)
                VALUES (%s, %s, %s, %s);
                """,
                (
                    document_id,
                    1,
                    "This is a smoke-test page for the multilingual RAG database.",
                    "smoke-test",
                ),
            )

            cursor.execute(
                """
                INSERT INTO chunks (
                    chunk_id, document_id, page_number, language,
                    chunk_index, text, token_count
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s);
                """,
                (
                    chunk_id,
                    document_id,
                    1,
                    "en",
                    0,
                    "This is a smoke-test chunk for the multilingual RAG database.",
                    11,
                ),
            )

            cursor.execute(
                """
                INSERT INTO embeddings (chunk_id, embedding_model, embedding)
                VALUES (%s, %s, %s);
                """,
                (chunk_id, "smoke-test-1024", embedding),
            )

            cursor.execute("SELECT text FROM chunks WHERE chunk_id = %s;", (chunk_id,))
            retrieved_text = cursor.fetchone()[0]

            cursor.execute("DELETE FROM documents WHERE document_id = %s;", (document_id,))

    return DatabaseSmokeTest(
        document_id=str(document_id),
        chunk_id=chunk_id,
        retrieved_text=retrieved_text,
    )


def store_parsed_document(database_url: str, parsed: ParsedDocument) -> StoredDocument:
    character_count = sum(len(page.text) for page in parsed.pages)

    with psycopg.connect(_with_connect_timeout(database_url)) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO documents (
                    document_name, file_type, source_path, status
                )
                VALUES (%s, %s, %s, %s)
                RETURNING document_id;
                """,
                (
                    parsed.document_name,
                    parsed.file_type,
                    str(parsed.source_path),
                    "parsed",
                ),
            )
            document_id = cursor.fetchone()[0]

            cursor.execute(
                """
                INSERT INTO ingestion_status (document_id, stage, status, message)
                VALUES (%s, %s, %s, %s);
                """,
                (
                    document_id,
                    "parse",
                    "done",
                    f"Parsed {len(parsed.pages)} page(s) with {parsed.parser_name}.",
                ),
            )

            for page in parsed.pages:
                cursor.execute(
                    """
                    INSERT INTO document_pages (
                        document_id, page_number, text, parser_name
                    )
                    VALUES (%s, %s, %s, %s);
                    """,
                    (
                        document_id,
                        page.page_number,
                        page.text,
                        parsed.parser_name,
                    ),
                )

    return StoredDocument(
        document_id=str(document_id),
        document_name=parsed.document_name,
        file_type=parsed.file_type,
        page_count=len(parsed.pages),
        character_count=character_count,
    )


def get_document_pages(database_url: str, document_id: str) -> list[StoredPage]:
    with psycopg.connect(_with_connect_timeout(database_url)) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT page_number, parser_name, length(text), left(text, 160)
                FROM document_pages
                WHERE document_id = %s
                ORDER BY page_number NULLS LAST;
                """,
                (document_id,),
            )
            rows = cursor.fetchall()

    return [
        StoredPage(
            page_number=row[0],
            parser_name=row[1],
            character_count=row[2],
            preview=row[3].replace("\n", " "),
        )
        for row in rows
    ]


def get_source_pages(database_url: str, document_id: str) -> list[SourcePage]:
    with psycopg.connect(_with_connect_timeout(database_url)) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT page_number, text
                FROM document_pages
                WHERE document_id = %s
                ORDER BY page_number NULLS LAST;
                """,
                (document_id,),
            )
            rows = cursor.fetchall()

    return [SourcePage(page_number=row[0], text=row[1]) for row in rows]


def store_chunks(
    database_url: str,
    document_id: str,
    chunks: list[Chunk],
    language: str | None = None,
) -> ChunkingResult:
    with psycopg.connect(_with_connect_timeout(database_url)) as connection:
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM chunks WHERE document_id = %s;", (document_id,))

            for chunk in chunks:
                cursor.execute(
                    """
                    INSERT INTO chunks (
                        chunk_id, document_id, page_number, language,
                        chunk_index, text, token_count
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s);
                    """,
                    (
                        chunk.chunk_id,
                        document_id,
                        chunk.page_number,
                        language,
                        chunk.chunk_index,
                        chunk.text,
                        chunk.token_count,
                    ),
                )

            cursor.execute(
                """
                INSERT INTO ingestion_status (document_id, stage, status, message)
                VALUES (%s, %s, %s, %s);
                """,
                (document_id, "chunk", "done", f"Stored {len(chunks)} chunk(s)."),
            )

            cursor.execute(
                """
                UPDATE documents
                SET status = %s, updated_at = now()
                WHERE document_id = %s;
                """,
                ("chunked", document_id),
            )

    return ChunkingResult(document_id=document_id, chunk_count=len(chunks))


def get_chunks(database_url: str, document_id: str) -> list[StoredChunk]:
    with psycopg.connect(_with_connect_timeout(database_url)) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT chunk_id, page_number, chunk_index, token_count, length(text), left(text, 200)
                FROM chunks
                WHERE document_id = %s
                ORDER BY chunk_index;
                """,
                (document_id,),
            )
            rows = cursor.fetchall()

    return [
        StoredChunk(
            chunk_id=row[0],
            page_number=row[1],
            chunk_index=row[2],
            token_count=row[3],
            character_count=row[4],
            preview=row[5].replace("\n", " "),
        )
        for row in rows
    ]


def list_documents(database_url: str) -> list[dict[str, Any]]:
    with psycopg.connect(_with_connect_timeout(database_url)) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    d.document_id,
                    d.document_name,
                    d.file_type,
                    d.language,
                    d.status,
                    count(DISTINCT p.page_id) AS pages,
                    count(DISTINCT c.chunk_id) AS chunks
                FROM documents d
                LEFT JOIN document_pages p USING (document_id)
                LEFT JOIN chunks c USING (document_id)
                GROUP BY d.document_id, d.document_name, d.file_type, d.language, d.status
                ORDER BY d.created_at;
                """
            )
            rows = cursor.fetchall()

    return [
        {
            "document_id": str(row[0]),
            "document_name": row[1],
            "file_type": row[2],
            "language": row[3],
            "status": row[4],
            "pages": row[5],
            "chunks": row[6],
        }
        for row in rows
    ]


def update_document_and_chunk_language(
    database_url: str,
    document_id: str,
    language: str,
) -> DocumentLanguageUpdate:
    with psycopg.connect(_with_connect_timeout(database_url)) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE documents
                SET language = %s, updated_at = now()
                WHERE document_id = %s;
                """,
                (language, document_id),
            )

            cursor.execute(
                """
                UPDATE chunks
                SET language = %s
                WHERE document_id = %s;
                """,
                (language, document_id),
            )

            cursor.execute(
                """
                INSERT INTO ingestion_status (document_id, stage, status, message)
                VALUES (%s, %s, %s, %s);
                """,
                (document_id, "language_detect", "done", f"Detected language {language}."),
            )

            cursor.execute(
                "SELECT count(*) FROM chunks WHERE document_id = %s;",
                (document_id,),
            )
            chunk_count = cursor.fetchone()[0]

    return DocumentLanguageUpdate(
        document_id=document_id,
        language=language,
        chunk_count=chunk_count,
    )


def get_document_text(database_url: str, document_id: str, max_chars: int = 10000) -> str:
    with psycopg.connect(_with_connect_timeout(database_url)) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT string_agg(text, E'\n\n' ORDER BY page_number)
                FROM document_pages
                WHERE document_id = %s;
                """,
                (document_id,),
            )
            text = cursor.fetchone()[0] or ""

    return text[:max_chars]


def get_chunks_for_embedding(
    database_url: str,
    document_id: str | None = None,
) -> list[ChunkForEmbedding]:
    params: tuple[Any, ...] = ()
    where_clause = ""
    if document_id is not None:
        where_clause = "WHERE document_id = %s"
        params = (document_id,)

    with psycopg.connect(_with_connect_timeout(database_url)) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT chunk_id, text
                FROM chunks
                {where_clause}
                ORDER BY document_id, chunk_index;
                """,
                params,
            )
            rows = cursor.fetchall()

    return [ChunkForEmbedding(chunk_id=row[0], text=row[1]) for row in rows]


def store_embeddings(
    database_url: str,
    embedding_model: str,
    chunk_vectors: list[tuple[str, list[float]]],
) -> EmbeddingResult:
    with psycopg.connect(_with_connect_timeout(database_url)) as connection:
        with connection.cursor() as cursor:
            for chunk_id, vector in chunk_vectors:
                cursor.execute(
                    """
                    INSERT INTO embeddings (chunk_id, embedding_model, embedding)
                    VALUES (%s, %s, %s::vector)
                    ON CONFLICT (chunk_id, embedding_model)
                    DO UPDATE SET embedding = EXCLUDED.embedding, created_at = now();
                    """,
                    (chunk_id, embedding_model, vector_literal(vector)),
                )

    return EmbeddingResult(
        embedding_model=embedding_model,
        stored_count=len(chunk_vectors),
    )


def retrieve_similar_chunks(
    database_url: str,
    embedding_model: str,
    query_vector: list[float],
    top_k: int,
) -> list[RetrievedChunk]:
    query_literal = vector_literal(query_vector)

    with psycopg.connect(_with_connect_timeout(database_url)) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    c.chunk_id,
                    c.document_id,
                    d.document_name,
                    c.page_number,
                    c.language,
                    c.text,
                    1 - (e.embedding <=> %s::vector) AS score
                FROM embeddings e
                JOIN chunks c USING (chunk_id)
                JOIN documents d USING (document_id)
                WHERE e.embedding_model = %s
                ORDER BY e.embedding <=> %s::vector
                LIMIT %s;
                """,
                (query_literal, embedding_model, query_literal, top_k),
            )
            rows = cursor.fetchall()

    return [
        RetrievedChunk(
            chunk_id=row[0],
            document_id=str(row[1]),
            document_name=row[2],
            page_number=row[3],
            language=row[4],
            text=row[5],
            score=float(row[6]),
            vector_score=float(row[6]),
            retrieval_source="vector",
        )
        for row in rows
    ]


def retrieve_keyword_chunks(
    database_url: str,
    query: str,
    top_k: int,
) -> list[RetrievedChunk]:
    terms = _keyword_terms(query)
    if not terms:
        return []

    with psycopg.connect(_with_connect_timeout(database_url)) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                WITH terms AS (
                    SELECT unnest(%s::text[]) AS term
                ),
                scored AS (
                    SELECT
                        c.chunk_id,
                        c.document_id,
                        d.document_name,
                        c.page_number,
                        c.language,
                        c.text,
                        sum(
                            CASE
                                WHEN lower(c.text) LIKE ('%%' || lower(t.term) || '%%')
                                THEN 1.0
                                ELSE 0.0
                            END
                            +
                            CASE
                                WHEN lower(replace(d.document_name, '_', ' ')) LIKE ('%%' || lower(t.term) || '%%')
                                THEN 5.0
                                ELSE 0.0
                            END
                        ) AS score
                    FROM chunks c
                    JOIN documents d USING (document_id)
                    CROSS JOIN terms t
                    GROUP BY
                        c.chunk_id,
                        c.document_id,
                        d.document_name,
                        c.page_number,
                        c.language,
                        c.text
                )
                SELECT
                    chunk_id,
                    document_id,
                    document_name,
                    page_number,
                    language,
                    text,
                    score
                FROM scored
                WHERE score > 0
                ORDER BY score DESC
                LIMIT %s;
                """,
                (terms, top_k),
            )
            rows = cursor.fetchall()

    return [
        RetrievedChunk(
            chunk_id=row[0],
            document_id=str(row[1]),
            document_name=row[2],
            page_number=row[3],
            language=row[4],
            text=row[5],
            score=float(row[6]),
            keyword_score=float(row[6]),
            retrieval_source="keyword",
        )
        for row in rows
    ]


def _keyword_terms(query: str) -> list[str]:
    seen: set[str] = set()
    terms: list[str] = []
    for raw_term in re.findall(r"[\w]+", query.casefold()):
        term = raw_term.strip("_")
        if len(term) < 3 or term in KEYWORD_STOP_WORDS or term in seen:
            continue
        seen.add(term)
        terms.append(term)

    return terms[:12]
