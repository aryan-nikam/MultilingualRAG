from __future__ import annotations

import subprocess
from dataclasses import dataclass

from multilingual_rag.config import Settings


@dataclass(frozen=True)
class LlamaResult:
    stdout: str
    stderr: str
    returncode: int


def run_prompt(
    settings: Settings,
    prompt: str,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> LlamaResult:
    tokens = settings.llama_max_tokens if max_tokens is None else max_tokens
    temp = settings.llama_temperature if temperature is None else temperature

    command = [
        str(settings.llama_cli_path),
        "--model",
        str(settings.llama_model_path),
        "--prompt",
        prompt,
        "--ctx-size",
        str(settings.llama_context_size),
        "--threads",
        str(settings.llama_threads),
        "--predict",
        str(tokens),
        "--temp",
        str(temp),
        "--simple-io",
        "--no-display-prompt",
        "--no-warmup",
        "--no-show-timings",
    ]

    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    return LlamaResult(
        stdout=completed.stdout.strip(),
        stderr=completed.stderr.strip(),
        returncode=completed.returncode,
    )

