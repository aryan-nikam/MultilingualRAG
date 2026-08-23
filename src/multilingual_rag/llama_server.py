from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass

from multilingual_rag.config import Settings


@dataclass(frozen=True)
class LlamaServerResult:
    text: str


def generate_with_llama_server(
    settings: Settings,
    prompt: str,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> LlamaServerResult:
    payload = {
        "prompt": prompt,
        "n_predict": settings.llama_max_tokens if max_tokens is None else max_tokens,
        "temperature": settings.llama_temperature if temperature is None else temperature,
        "stop": ["</answer>"],
    }
    request = urllib.request.Request(
        f"{settings.llama_server_url.rstrip('/')}/completion",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=settings.llama_server_timeout_seconds,
        ) as response:
            raw_response = response.read().decode("utf-8")
            data = json.loads(raw_response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ConnectionError(
            f"llama-server returned HTTP {exc.code}: {detail[:500]}"
        ) from exc
    except socket.timeout as exc:
        raise TimeoutError(
            f"llama-server timed out after {settings.llama_server_timeout_seconds} seconds"
        ) from exc
    except urllib.error.URLError as exc:
        raise ConnectionError(
            f"llama-server is not reachable at {settings.llama_server_url}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError("llama-server returned invalid JSON") from exc

    text = data.get("content") or data.get("response") or ""
    return LlamaServerResult(text=str(text).strip())


def llama_server_health(settings: Settings, timeout_seconds: int = 2) -> bool:
    request = urllib.request.Request(
        f"{settings.llama_server_url.rstrip('/')}/health",
        method="GET",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return 200 <= response.status < 500
    except urllib.error.URLError:
        return False
