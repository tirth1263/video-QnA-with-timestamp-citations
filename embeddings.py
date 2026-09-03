"""Thin wrapper around Gemini `gemini-embedding-2-preview`.

This is the piece that makes the whole pipeline collapse into a single vector
search: the model is *natively multimodal*, so a text query and a video clip
land in the same embedding space. No transcription, no separate image encoder,
no hybrid fusion at query time.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

from google import genai
from google.genai import types

DEFAULT_MODEL = "gemini-embedding-2-preview"

# 128-3072 are valid; 768 / 1536 / 3072 are the recommended points. 1536 is a
# good accuracy/size trade-off and keeps the Weaviate index compact.
DEFAULT_DIM = 1536

# Anything under this goes inline in the request body. Larger clips are pushed
# through the Files API instead, which avoids blowing the request size limit.
INLINE_BYTE_LIMIT = 15 * 1024 * 1024

_MIME_BY_SUFFIX = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
    ".avi": "video/x-msvideo",
}


class EmbeddingError(RuntimeError):
    """Raised when the embedding API cannot produce a vector."""


@dataclass
class GeminiEmbedder:
    """Embeds text and video clips into one shared vector space."""

    api_key: str | None = None
    model: str = DEFAULT_MODEL
    output_dim: int = DEFAULT_DIM
    max_retries: int = 4

    def __post_init__(self) -> None:
        key = self.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise EmbeddingError(
                "GEMINI_API_KEY is not set. Create one at "
                "https://aistudio.google.com/apikey"
            )
        self._client = genai.Client(api_key=key)

    # ---------------------------------------------------------------- public

    def embed_text(self, text: str) -> list[float]:
        """Embed a natural-language query or caption."""
        if not text or not text.strip():
            raise EmbeddingError("Cannot embed empty text.")
        return self._embed([text])

    def embed_video(self, path: str | Path) -> list[float]:
        """Embed a video clip natively — pixels and audio, no transcript."""
        path = Path(path)
        if not path.exists():
            raise EmbeddingError(f"Clip not found: {path}")

        size = path.stat().st_size
        if size == 0:
            raise EmbeddingError(f"Clip is empty: {path}")

        mime = _MIME_BY_SUFFIX.get(path.suffix.lower(), "video/mp4")

        if size <= INLINE_BYTE_LIMIT:
            part = types.Part.from_bytes(data=path.read_bytes(), mime_type=mime)
        else:
            part = self._upload_part(path, mime)

        return self._embed([part])

    # --------------------------------------------------------------- private

    def _embed(self, contents: list) -> list[float]:
        """Call the API with bounded exponential backoff."""
        last_err: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                result = self._client.models.embed_content(
                    model=self.model,
                    contents=contents,
                    config=types.EmbedContentConfig(
                        output_dimensionality=self.output_dim
                    ),
                )
            except Exception as exc:  # noqa: BLE001 - surfaced after retries
                last_err = exc
                if not _is_retryable(exc) or attempt == self.max_retries - 1:
                    break
                time.sleep(2**attempt)
                continue

            values = _first_vector(result)
            if values is None:
                raise EmbeddingError(f"{self.model} returned no embedding values.")
            return values

        raise EmbeddingError(f"Embedding call failed: {last_err}") from last_err

    def _upload_part(self, path: Path, mime: str):
        """Push an oversized clip through the Files API and wait for ACTIVE."""
        handle = self._client.files.upload(file=str(path))

        for _ in range(60):
            state = getattr(getattr(handle, "state", None), "name", None) or getattr(
                handle, "state", None
            )
            if state == "ACTIVE":
                break
            if state == "FAILED":
                raise EmbeddingError(f"Files API failed to process {path.name}.")
            time.sleep(2)
            handle = self._client.files.get(name=handle.name)
        else:
            raise EmbeddingError(f"Files API timed out processing {path.name}.")

        return types.Part.from_uri(
            file_uri=handle.uri, mime_type=getattr(handle, "mime_type", mime)
        )


@dataclass
class GeminiDescriber:
    """Writes a short description of what happens in a clip.

    Retrieval never touches these strings - that is done entirely by the
    multimodal vectors. Descriptions exist because the answering model on
    Nebius is text-only: it needs to read *something* about the clips that
    the vector search surfaced. Gemini watches the clip directly, so this is
    still not a transcription service in the pipeline.
    """

    api_key: str | None = None
    model: str = "gemini-2.5-flash"
    max_words: int = 60

    def __post_init__(self) -> None:
        key = self.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise EmbeddingError("GEMINI_API_KEY is not set.")
        self._client = genai.Client(api_key=key)

    def describe(self, path: str | Path, timespan: str = "") -> str:
        path = Path(path)
        mime = _MIME_BY_SUFFIX.get(path.suffix.lower(), "video/mp4")

        prompt = (
            f"Describe this {timespan} video clip in at most {self.max_words} words. "
            "Cover what is shown on screen and what is said aloud. "
            "Be concrete: name objects, people, on-screen text, and spoken claims. "
            "Write plain prose with no preamble."
        )

        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=[
                    types.Part.from_bytes(data=path.read_bytes(), mime_type=mime),
                    prompt,
                ],
            )
        except Exception as exc:  # noqa: BLE001 - a missing description is survivable
            return f"(description unavailable: {type(exc).__name__})"

        return (getattr(response, "text", "") or "").strip()


def _first_vector(result) -> list[float] | None:
    """Pull the vector out of an EmbedContentResponse."""
    embeddings = getattr(result, "embeddings", None)
    if not embeddings:
        return None
    values = getattr(embeddings[0], "values", None)
    return list(values) if values else None


def _is_retryable(exc: Exception) -> bool:
    """Rate limits and transient server errors are worth another shot."""
    text = str(exc).lower()
    if any(tok in text for tok in ("429", "resource_exhausted", "rate limit")):
        return True
    return any(tok in text for tok in ("500", "502", "503", "504", "unavailable", "deadline"))
