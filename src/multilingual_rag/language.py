from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from lingua import Language, LanguageDetectorBuilder


SUPPORTED_LANGUAGES = {
    Language.ENGLISH: "en",
    Language.HINDI: "hi",
    Language.MARATHI: "mr",
}


@dataclass(frozen=True)
class LanguageDetection:
    language: str
    confidence: float | None


@lru_cache(maxsize=1)
def _detector():
    return LanguageDetectorBuilder.from_languages(*SUPPORTED_LANGUAGES.keys()).build()


def detect_language(text: str) -> LanguageDetection:
    text = text.strip()
    if not text:
        return LanguageDetection(language="unknown", confidence=None)

    detector = _detector()
    detected = detector.detect_language_of(text)
    if detected is None:
        return LanguageDetection(language="unknown", confidence=None)

    confidence_values = detector.compute_language_confidence_values(text)
    confidence = next(
        (
            value.value
            for value in confidence_values
            if value.language == detected
        ),
        None,
    )

    return LanguageDetection(
        language=SUPPORTED_LANGUAGES.get(detected, "unknown"),
        confidence=confidence,
    )
