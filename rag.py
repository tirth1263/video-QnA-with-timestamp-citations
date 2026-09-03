"""Retrieve segments, then have Nebius write an answer that cites timestamps.

Retrieval is a single `near_vector` call. Because Gemini put the text query and
the video clips in the same space, there is nothing to fuse: no BM25 leg, no
reranker, no reciprocal rank fusion. One search, one list of clips.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from openai import OpenAI

from embeddings import GeminiEmbedder
from weaviate_store import SegmentHit, SegmentStore, format_timestamp, parse_timestamp

DEFAULT_ANSWER_MODEL = "Qwen/Qwen3-235B-A22B"
NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
DEFAULT_TOP_K = 5

SYSTEM_PROMPT = """You answer questions about a single video.

You are given numbered excerpts. Each excerpt is one clip of the video and \
carries the time range it covers. The excerpt text was written by a model that \
watched that clip.

Rules:
- Answer only from the excerpts. Never invent events, quotes, or details.
- Cite the moment you are relying on inline, in square brackets, as [mm:ss] - \
for example: "She introduces the pricing tiers [01:20]."
- Use the START time of the excerpt you are citing. Cite as you go, not in a \
list at the end.
- Multiple moments can back one sentence: [00:40] [02:15].
- If the excerpts do not answer the question, say so plainly and point to the \
closest relevant moment instead. Do not pad.
- Be direct and specific. No preamble, no restating the question."""

TIMESTAMP_RE = re.compile(r"\[(\d{1,2}:\d{2}(?::\d{2})?)\]")


class RAGError(RuntimeError):
    """Raised when an answer cannot be produced."""


@dataclass
class Citation:
    """A [mm:ss] the model wrote, resolved back to a seekable position."""

    label: str
    seconds: float

    def __hash__(self) -> int:
        return hash(round(self.seconds, 2))


@dataclass
class Answer:
    question: str
    text: str
    hits: list[SegmentHit] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    model: str = DEFAULT_ANSWER_MODEL


class VideoRAG:
    """Query-side pipeline: embed -> search -> generate."""

    def __init__(
        self,
        store: SegmentStore,
        embedder: GeminiEmbedder,
        nebius_api_key: str | None = None,
        base_url: str | None = None,
        model: str = DEFAULT_ANSWER_MODEL,
    ):
        key = nebius_api_key or os.getenv("NEBIUS_API_KEY")
        if not key:
            raise RAGError(
                "NEBIUS_API_KEY is not set. Get one at https://dub.sh/nebius"
            )

        self.store = store
        self.embedder = embedder
        self.model = model
        self._client = OpenAI(
            base_url=base_url or os.getenv("NEBIUS_BASE_URL") or NEBIUS_BASE_URL,
            api_key=key,
        )

    # -------------------------------------------------------------- retrieval

    def retrieve(
        self, question: str, video_id: str | None = None, top_k: int = DEFAULT_TOP_K
    ) -> list[SegmentHit]:
        """One text embedding, one vector search. That is the whole retriever."""
        if not question.strip():
            raise RAGError("Ask a question first.")
        query_vector = self.embedder.embed_text(question)
        return self.store.search(vector=query_vector, video_id=video_id, limit=top_k)

    # ------------------------------------------------------------- generation

    def answer(
        self,
        question: str,
        video_id: str | None = None,
        top_k: int = DEFAULT_TOP_K,
        model: str | None = None,
        temperature: float = 0.2,
    ) -> Answer:
        model = model or self.model
        hits = self.retrieve(question, video_id=video_id, top_k=top_k)

        if not hits:
            return Answer(
                question=question,
                text=(
                    "Nothing is indexed for this video yet, so there is no "
                    "footage to search. Ingest a video first."
                ),
                model=model,
            )

        try:
            completion = self._client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"{build_context(hits)}\n\n"
                            f"Question: {question.strip()}\n\n"
                            "Answer using only the excerpts above, with [mm:ss] "
                            "citations."
                        ),
                    },
                ],
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI as-is
            raise RAGError(f"Nebius chat completion failed: {exc}") from exc

        text = _strip_reasoning((completion.choices[0].message.content or "").strip())

        return Answer(
            question=question,
            text=text,
            hits=hits,
            citations=extract_citations(text, hits),
            model=model,
        )


# ----------------------------------------------------------------- formatting


def build_context(hits: list[SegmentHit]) -> str:
    """Render retrieved clips as numbered, timestamped excerpts."""
    blocks = []
    for number, hit in enumerate(hits, start=1):
        body = hit.description.strip() or (
            "(no description was generated for this clip; it was retrieved on "
            "visual/audio similarity alone)"
        )
        blocks.append(f"[{number}] {hit.span} (start {hit.timestamp})\n{body}")
    return "Video excerpts:\n\n" + "\n\n".join(blocks)


def extract_citations(text: str, hits: list[SegmentHit] | None = None) -> list[Citation]:
    """Pull [mm:ss] markers out of an answer, in order, de-duplicated.

    A model occasionally cites a time a second or two off the clip boundary, so
    each citation is snapped to the nearest retrieved segment start when one is
    close enough. That keeps every button landing on real indexed footage.
    """
    starts = sorted({hit.start_sec for hit in hits}) if hits else []

    citations: list[Citation] = []
    seen: set[float] = set()

    for raw in TIMESTAMP_RE.findall(text):
        seconds = parse_timestamp(raw)
        if seconds is None:
            continue

        if starts:
            nearest = min(starts, key=lambda s: abs(s - seconds))
            if abs(nearest - seconds) <= 2.0:
                seconds = nearest

        if seconds in seen:
            continue
        seen.add(seconds)
        citations.append(Citation(label=format_timestamp(seconds), seconds=seconds))

    return citations


def _strip_reasoning(text: str) -> str:
    """Qwen3 and friends can emit <think> blocks; the user does not want them."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"^\s*<think>.*", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip() or text.strip()
