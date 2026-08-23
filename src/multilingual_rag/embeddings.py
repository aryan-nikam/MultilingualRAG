from __future__ import annotations

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=2)
def _load_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed_texts(model_name: str, texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    model = _load_model(model_name)
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    if isinstance(vectors, np.ndarray) and vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)

    return vectors.astype(float).tolist()


def embed_text(model_name: str, text: str) -> list[float]:
    return embed_texts(model_name, [text])[0]


def vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{value:.8f}" for value in vector) + "]"
