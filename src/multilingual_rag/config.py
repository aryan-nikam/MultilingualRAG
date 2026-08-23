from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


@dataclass(frozen=True)
class Settings:
    llama_cpp_dir: Path
    llama_cli_path: Path
    llama_model_path: Path
    llama_threads: int
    llama_context_size: int
    llama_max_tokens: int
    llama_temperature: float
    database_url: str
    embedding_model_name: str
    embedding_dimension: int
    retrieval_top_k: int


def load_settings() -> Settings:
    _load_env_file(PROJECT_ROOT / ".env")

    llama_cpp_dir = Path(
        os.environ.get("LLAMA_CPP_DIR", r"C:\Users\Aryan\Desktop\llama.cpp")
    )

    return Settings(
        llama_cpp_dir=llama_cpp_dir,
        llama_cli_path=Path(
            os.environ.get(
                "LLAMA_CLI_PATH",
                str(llama_cpp_dir / "build" / "llama-cli.exe"),
            )
        ),
        llama_model_path=Path(
            os.environ.get(
                "LLAMA_MODEL_PATH",
                str(
                    llama_cpp_dir
                    / "models"
                    / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
                ),
            )
        ),
        llama_threads=int(os.environ.get("LLAMA_THREADS", "-1")),
        llama_context_size=int(os.environ.get("LLAMA_CONTEXT_SIZE", "2048")),
        llama_max_tokens=int(os.environ.get("LLAMA_MAX_TOKENS", "128")),
        llama_temperature=float(os.environ.get("LLAMA_TEMPERATURE", "0.2")),
        database_url=os.environ.get(
            "DATABASE_URL",
            "postgresql://postgres:root@localhost:5433/multilingual_rag",
        ),
        embedding_model_name=os.environ.get("EMBEDDING_MODEL_NAME", "BAAI/bge-m3"),
        embedding_dimension=int(os.environ.get("EMBEDDING_DIMENSION", "1024")),
        retrieval_top_k=int(os.environ.get("RETRIEVAL_TOP_K", "5")),
    )
