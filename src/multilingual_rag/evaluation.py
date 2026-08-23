from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from multilingual_rag.answer_quality import (
    build_citations,
    enforce_answer_citations,
    validate_rag_response,
)
from multilingual_rag.config import Settings
from multilingual_rag.llama_server import generate_with_llama_server
from multilingual_rag.rag import retrieve_for_question


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    question: str
    language: str
    expected_sources: list[str]
    expected_answer_contains: list[str]
    notes: str


@dataclass(frozen=True)
class EvalResult:
    case_id: str
    expected_sources: list[str]
    retrieved_sources: list[str]
    passed: bool
    best_rank: int | None
    query_language: str
    answer_language_ok: bool | None
    citation_ok: bool | None
    expected_answer_ok: bool | None


@dataclass(frozen=True)
class EvalSummary:
    total: int
    passed: int
    recall_at_k: float
    results: list[EvalResult]


def load_eval_cases(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue

        data = json.loads(line)
        cases.append(
            EvalCase(
                case_id=data.get("id") or f"case_{line_number}",
                question=data["question"],
                language=data["language"],
                expected_sources=list(data["expected_sources"]),
                expected_answer_contains=list(data.get("expected_answer_contains", [])),
                notes=data.get("notes", ""),
            )
        )

    return cases


def run_retrieval_eval(
    settings: Settings,
    cases: list[EvalCase],
    top_k: int,
    retrieval_mode: str,
    rerank: bool,
    generate_answers: bool = False,
) -> EvalSummary:
    results: list[EvalResult] = []
    for case in cases:
        retrieval = retrieve_for_question(
            settings=settings,
            question=case.question,
            top_k=top_k,
            retrieval_mode=retrieval_mode,
            rerank=rerank,
        )
        retrieved_sources = [chunk.document_name for chunk in retrieval.chunks]
        ranks = [
            retrieved_sources.index(expected) + 1
            for expected in case.expected_sources
            if expected in retrieved_sources
        ]
        best_rank = min(ranks) if ranks else None
        answer_language_ok: bool | None = None
        citation_ok: bool | None = None
        expected_answer_ok: bool | None = None

        if generate_answers:
            citations = build_citations(retrieval.chunks)
            answer = generate_with_llama_server(settings, retrieval.prompt).text
            answer = enforce_answer_citations(answer, citations)
            validation = validate_rag_response(
                retrieval.chunks,
                answer,
                no_generate=False,
                expected_language=case.language,
            )
            answer_language_ok = bool(validation["answer_language_ok"])
            citation_ok = bool(
                validation["citation_markers_valid"]
                and validation["answer_has_required_citation"]
            )
            answer_text = answer.casefold()
            expected_answer_ok = all(
                expected.casefold() in answer_text
                for expected in case.expected_answer_contains
            )

        results.append(
            EvalResult(
                case_id=case.case_id,
                expected_sources=case.expected_sources,
                retrieved_sources=retrieved_sources,
                passed=best_rank is not None,
                best_rank=best_rank,
                query_language=retrieval.query_language.language,
                answer_language_ok=answer_language_ok,
                citation_ok=citation_ok,
                expected_answer_ok=expected_answer_ok,
            )
        )

    passed = sum(1 for result in results if result.passed)
    recall_at_k = passed / len(results) if results else 0.0
    return EvalSummary(
        total=len(results),
        passed=passed,
        recall_at_k=recall_at_k,
        results=results,
    )
