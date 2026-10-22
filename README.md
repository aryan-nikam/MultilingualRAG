# Multilingual RAG System

A local-first Retrieval-Augmented Generation (RAG) application that lets users upload PDF and TXT documents and ask questions in **English, Hindi, or Marathi**. It combines multilingual embeddings, hybrid retrieval, cross-encoder reranking, and a locally hosted language model to produce context-grounded answers with source citations.



## Features

- **PDF and TXT ingestion** with page-level text extraction and source metadata.
- **Language detection** for English, Hindi, and Marathi using Lingua.
- **Paragraph-aware chunking** with a default target of approximately 450 tokens and 75-token overlap.
- **Multilingual embeddings** using BGE-M3 through `sentence-transformers`.
- **PostgreSQL + pgvector** for storing document chunks and dense vectors.
- **Hybrid retrieval** combining vector similarity and PostgreSQL full-text/keyword search.
- **Cross-encoder reranking** using `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`.
- **Local LLM inference** using Qwen 2.5 0.5B Instruct (GGUF) with `llama.cpp`.
- **Grounded answers and citations** that identify source documents and page numbers.
- **Answer checks** for citation presence/validity, empty context, expected language, and basic unsupported-claim signals.
- **FastAPI backend** and a basic browser UI for document upload, processing, listing, deletion, and chat.
- **Evaluation CLI** reporting retrieval Recall@K and expected-source ranking.

## Architecture

```text
PDF / TXT files
      |
      v
Parsing and text cleaning
      |
      v
Page-aware / paragraph-aware chunking
      |
      v
Language detection (English / Hindi / Marathi)
      |
      v
BGE-M3 embeddings
      |
      v
PostgreSQL + pgvector <---- PostgreSQL full-text / keyword search
      |                                  |
      +----------- Hybrid retrieval -----+
                         |
                         v
               Cross-encoder reranking
                         |
                         v
                Top-ranked context
                         |
                         v
             Local Qwen 2.5 via llama.cpp
                         |
                         v
            Answer validation + citations
                         |
                         v
                    CLI / FastAPI
```

## Technology Stack

| Area | Technology |
|---|---|
| Language | Python 3.11 |
| Dependency / command runner | `uv` |
| API | FastAPI |
| Database | PostgreSQL |
| Vector search | pgvector |
| PDF parsing | PyMuPDF |
| Language detection | Lingua |
| Embeddings | BGE-M3, `sentence-transformers` |
| Reranking | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` |
| Local LLM | Qwen 2.5 0.5B Instruct (GGUF) |
| Inference runtime | `llama.cpp` / `llama-server` |
| Evaluation data | JSONL |

## Current Evaluation Results

The implementation status tracker records the following results from the project's seed evaluation and manual checks:

- **Retrieval evaluation:** 10/10 seed retrieval checks passed.
- **Recall@3:** 1.000 on the current 10-question seed evaluation using hybrid retrieval and reranking.
- **Cross-lingual retrieval:** Six English/Hindi/Marathi cross-document checks ranked the expected document first after hybrid retrieval and reranking.
- **Warm API reranked retrieval latency:** approximately 2.2–2.5 seconds before cache, as recorded in the status tracker.
- **Cache hits:** the tracker records around 5–15 ms for the FastAPI in-memory retrieval cache in one section; another earlier measurement records around 0.01 ms. Treat these as local observations, not production benchmarks.

These are **small, development-stage results**, not a claim of production-scale performance. Generated-answer evaluation should be run with the local `llama-server` available.

## Prerequisites

Install and configure the following:

- Python 3.11
- [`uv`](https://docs.astral.sh/uv/)
- PostgreSQL
- pgvector extension compatible with your PostgreSQL installation
- A working `llama.cpp` build and `llama-server`
- The selected Qwen GGUF model file
- Enough local memory and disk space for the embedding model, reranker, database, and LLM

The implementation was developed on Windows. Adjust paths and commands for your environment where necessary.

## Setup

> **Important:** Use your own database password and keep credentials local. Do not commit `.env`, passwords, API secrets, or private document data to Git.

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_PROJECT_DIRECTORY>
```

Replace the placeholders with your repository URL and directory name.

### 2. Create the environment and install dependencies

The project uses `uv`. From the project root, run the dependency installation command supported by the repository's current `pyproject.toml` and lockfile. For example:

```bash
uv sync
```

If the repository does not yet contain a committed lockfile, use the dependency workflow already defined in `pyproject.toml`.

### 3. Configure environment variables

Review `.env.example` and configure the local model path and database connection settings in your local `.env` file. Do not commit the populated `.env` file.

The implementation status tracker records a local PostgreSQL database named `multilingual_rag` listening on port `5433`; these are development settings, not requirements for every installation.

### 4. Prepare PostgreSQL and pgvector

Create the project database, enable pgvector, and apply the SQL initialization files included in the repository. The initial schema is in `db/init/001_init.sql`; hybrid full-text retrieval additions are in `db/init/002_hybrid_retrieval.sql`.

Example SQL to enable pgvector after connecting to the target database:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

Ensure the database schema and connection settings match the application's configuration before running ingestion commands.

### 5. Start the local LLM server

The project uses a Qwen 2.5 0.5B Instruct GGUF model through `llama.cpp`. Start `llama-server` with your actual model path and the port expected by the application configuration. The implementation tracker records the expected local endpoint as:

```text
http://127.0.0.1:8080
```

Example command (adjust the executable and model path to your installation):

```powershell
llama-server.exe -m "C:\path\to\qwen2.5-0.5b-instruct-q4_k_m.gguf" --host 127.0.0.1 --port 8080
```

Keep the server running in a separate terminal when testing generated answers.

## Usage

The CLI command names below are recorded in the implementation status tracker. Run commands from the project root.

### Check the database

```bash
uv run mrag db-check
uv run mrag db-smoke-test
```

### Ingest PDF or TXT documents

```bash
uv run mrag ingest --path data/input/sample_english.txt
```

You can also provide a PDF file or a folder containing supported documents:

```bash
uv run mrag ingest --path data/input
```

### Inspect parsed pages and chunks

```bash
uv run mrag show-pages
uv run mrag show-chunks --document-id <DOCUMENT_ID>
```

Use the exact arguments shown by the CLI help if a command requires additional options.

### Generate embeddings

```bash
uv run mrag embed-chunks
uv run mrag embed-sample --text "How does solar energy work?"
```

### Ask a question

```bash
uv run mrag ask --question "What does the document say about solar energy?"
```

The RAG pipeline detects the query language, embeds the query, retrieves and reranks relevant chunks, builds a context-grounded prompt, and can send it to the local model for answer generation. Source document and page information are included in the output.

For retrieval-only debugging, the implementation supports a `--no-generate` path:

```bash
uv run mrag ask --question "What does the document say about solar energy?" --no-generate
```

### Run the API and web UI

Start FastAPI:

```bash
uv run mrag serve-api --host 127.0.0.1 --port 8000
```

Then open the local UI at:

```text
http://127.0.0.1:8000
```

The implementation includes endpoints for health checks, document upload and processing, document listing and deletion, asking questions, and chat. Check the actual API route definitions or interactive API documentation in your running application for exact request schemas.

### Run retrieval evaluation

```bash
uv run mrag eval-retrieval
```

To include generated-answer checks, ensure `llama-server` is running and reachable, then run:

```bash
uv run mrag eval-retrieval --generate
```

The seed evaluation dataset is stored at `data/eval/seed_eval.jsonl`. It includes multilingual retrieval checks and expected source information.

## Repository Structure

The following is a conceptual outline based on the implementation tracker; confirm exact filenames against the current repository.

```text
.
├── config/
├── data/
│   ├── eval/
│   │   └── seed_eval.jsonl
│   └── input/
├── db/
│   └── init/
│       ├── 001_init.sql
│       └── 002_hybrid_retrieval.sql
├── scripts/
├── src/
│   └── multilingual_rag/
│       ├── api.py
│       ├── chunking.py
│       ├── embeddings.py
│       ├── parsers.py
│       └── rag.py
├── tests/
├── .env.example
└── IMPLEMENTATION_STATUS.txt
```

Additional modules and files may exist in the actual repository.


## Security Notes

- Never commit database passwords, `.env` files, access tokens, or private documents.
- Use a dedicated database role with a strong password rather than a simple development password.
- Do not expose the local API or `llama-server` publicly without appropriate authentication, network restrictions, and security review.
- Tenant isolation, authentication, rate limiting, and robust upload validation are not yet implemented according to the status tracker.

## License

No license has been specified yet. Add a `LICENSE` file before presenting the repository as open source.
