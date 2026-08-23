from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf


SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


@dataclass(frozen=True)
class ParsedPage:
    page_number: int
    text: str


@dataclass(frozen=True)
class ParsedDocument:
    document_name: str
    file_type: str
    source_path: Path
    parser_name: str
    pages: list[ParsedPage]


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_txt(path: Path) -> ParsedDocument:
    raw_text = path.read_text(encoding="utf-8-sig", errors="replace")
    text = clean_text(raw_text)
    return ParsedDocument(
        document_name=path.name,
        file_type="txt",
        source_path=path,
        parser_name="txt",
        pages=[ParsedPage(page_number=1, text=text)],
    )


def parse_pdf(path: Path) -> ParsedDocument:
    pages: list[ParsedPage] = []

    with pymupdf.open(path) as document:
        for page_index, page in enumerate(document, start=1):
            text = clean_text(page.get_text("text"))
            pages.append(ParsedPage(page_number=page_index, text=text))

    return ParsedDocument(
        document_name=path.name,
        file_type="pdf",
        source_path=path,
        parser_name="pymupdf",
        pages=pages,
    )


def parse_document(path: Path) -> ParsedDocument:
    path = path.resolve()
    extension = path.suffix.lower()

    if extension == ".txt":
        return parse_txt(path)

    if extension == ".pdf":
        return parse_pdf(path)

    raise ValueError(f"unsupported file type: {path.suffix}")


def iter_supported_files(path: Path) -> list[Path]:
    path = path.resolve()

    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"unsupported file type: {path.suffix}")
        return [path]

    if path.is_dir():
        return sorted(
            file_path
            for file_path in path.rglob("*")
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
        )

    raise FileNotFoundError(f"path not found: {path}")

