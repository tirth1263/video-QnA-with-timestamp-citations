"""Weaviate v4 client, schema, and vector search for video segments.

Vectors are bring-your-own: Gemini produces them, Weaviate only stores and
searches them. That keeps the vector space identical for clips and queries.

A dependency-free in-memory store implementing the same interface is included
so the app still runs where a Weaviate instance is not reachable (for example
on Streamlit Community Cloud, which has no Docker).
"""

from __future__ import annotations

import math
import os
import uuid
from dataclasses import dataclass, field
from typing import Protocol

COLLECTION = "VideoSegment"


@dataclass
class Segment:
    """One clip of a video, plus the vector Gemini produced for it."""

    video_id: str
    video_name: str
    clip_index: int
    start_sec: float
    end_sec: float
    vector: list[float]
    description: str = ""

    def properties(self) -> dict:
        return {
            "video_id": self.video_id,
            "video_name": self.video_name,
            "clip_index": self.clip_index,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "description": self.description,
        }


@dataclass
class SegmentHit:
    """A retrieved segment and how close it was to the query."""

    video_id: str
    video_name: str
    clip_index: int
    start_sec: float
    end_sec: float
    distance: float
    description: str = ""

    @property
    def timestamp(self) -> str:
        return format_timestamp(self.start_sec)

    @property
    def span(self) -> str:
        return f"{format_timestamp(self.start_sec)}-{format_timestamp(self.end_sec)}"


class SegmentStore(Protocol):
    """The surface that ingest.py and rag.py depend on."""

    backend: str

    def ensure_schema(self) -> None: ...
    def upsert_segments(self, segments: list[Segment]) -> int: ...
    def delete_video(self, video_id: str) -> None: ...
    def search(self, vector: list[float], video_id: str | None, limit: int) -> list[SegmentHit]: ...
    def close(self) -> None: ...


def format_timestamp(seconds: float) -> str:
    """Seconds to mm:ss, or h:mm:ss once past the hour mark."""
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def parse_timestamp(stamp: str) -> float | None:
    """mm:ss or h:mm:ss to seconds. Returns None if it is not a timestamp."""
    parts = stamp.strip().split(":")
    if not 2 <= len(parts) <= 3:
        return None
    try:
        values = [int(p) for p in parts]
    except ValueError:
        return None
    if len(values) == 2:
        minutes, secs = values
        return minutes * 60 + secs
    hours, minutes, secs = values
    return hours * 3600 + minutes * 60 + secs


# --------------------------------------------------------------------- Weaviate


class WeaviateStore:
    """Weaviate v4 backend."""

    backend = "weaviate"

    def __init__(self, url: str | None = None, api_key: str | None = None):
        import weaviate
        from weaviate.classes.init import Auth

        url = (url or os.getenv("WEAVIATE_URL") or "http://localhost:8080").rstrip("/")
        api_key = api_key or os.getenv("WEAVIATE_API_KEY") or None
        auth = Auth.api_key(api_key) if api_key else None

        if _is_cloud(url):
            self._client = weaviate.connect_to_weaviate_cloud(
                cluster_url=url,
                auth_credentials=auth,
                skip_init_checks=True,
            )
        else:
            host, port, secure = _split_url(url)
            self._client = weaviate.connect_to_custom(
                http_host=host,
                http_port=port,
                http_secure=secure,
                grpc_host=host,
                grpc_port=int(os.getenv("WEAVIATE_GRPC_PORT", "50051")),
                grpc_secure=secure,
                auth_credentials=auth,
                skip_init_checks=True,
            )
        self.url = url

    def ensure_schema(self) -> None:
        from weaviate.classes.config import Configure, DataType, Property

        if self._client.collections.exists(COLLECTION):
            return

        properties = [
            Property(name="video_id", data_type=DataType.TEXT),
            Property(name="video_name", data_type=DataType.TEXT),
            Property(name="clip_index", data_type=DataType.INT),
            Property(name="start_sec", data_type=DataType.NUMBER),
            Property(name="end_sec", data_type=DataType.NUMBER),
            Property(name="description", data_type=DataType.TEXT),
        ]

        # Newer clients express "I supply my own vectors" through vector_config;
        # older ones through vectorizer_config. Support both.
        try:
            self._client.collections.create(
                name=COLLECTION,
                properties=properties,
                vector_config=Configure.Vectors.self_provided(),
            )
        except (AttributeError, TypeError):
            self._client.collections.create(
                name=COLLECTION,
                properties=properties,
                vectorizer_config=Configure.Vectorizer.none(),
            )

    def upsert_segments(self, segments: list[Segment]) -> int:
        if not segments:
            return 0
        self.ensure_schema()
        collection = self._client.collections.get(COLLECTION)

        with collection.batch.dynamic() as batch:
            for seg in segments:
                batch.add_object(
                    properties=seg.properties(),
                    vector=seg.vector,
                    uuid=_deterministic_uuid(seg.video_id, seg.clip_index),
                )

        failed = collection.batch.failed_objects
        if failed:
            raise RuntimeError(
                f"Weaviate rejected {len(failed)} object(s): {failed[0].message}"
            )
        return len(segments)

    def delete_video(self, video_id: str) -> None:
        from weaviate.classes.query import Filter

        if not self._client.collections.exists(COLLECTION):
            return
        collection = self._client.collections.get(COLLECTION)
        collection.data.delete_many(where=Filter.by_property("video_id").equal(video_id))

    def search(
        self, vector: list[float], video_id: str | None, limit: int
    ) -> list[SegmentHit]:
        from weaviate.classes.query import Filter, MetadataQuery

        if not self._client.collections.exists(COLLECTION):
            return []
        collection = self._client.collections.get(COLLECTION)

        response = collection.query.near_vector(
            near_vector=vector,
            limit=limit,
            filters=Filter.by_property("video_id").equal(video_id) if video_id else None,
            return_metadata=MetadataQuery(distance=True),
        )

        hits = []
        for obj in response.objects:
            props = obj.properties
            hits.append(
                SegmentHit(
                    video_id=str(props.get("video_id", "")),
                    video_name=str(props.get("video_name", "")),
                    clip_index=int(props.get("clip_index", 0)),
                    start_sec=float(props.get("start_sec", 0.0)),
                    end_sec=float(props.get("end_sec", 0.0)),
                    distance=float(obj.metadata.distance or 0.0),
                    description=str(props.get("description", "") or ""),
                )
            )
        return hits

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:  # noqa: BLE001 - closing must never raise
            pass


# -------------------------------------------------------------------- fallback


@dataclass
class InMemoryStore:
    """Cosine-similarity store used when Weaviate is unreachable.

    Same interface, no persistence: the vectors live for the session only.
    """

    backend: str = "memory"
    _rows: list[Segment] = field(default_factory=list)

    def ensure_schema(self) -> None:
        return None

    def upsert_segments(self, segments: list[Segment]) -> int:
        incoming = {(s.video_id, s.clip_index) for s in segments}
        self._rows = [r for r in self._rows if (r.video_id, r.clip_index) not in incoming]
        self._rows.extend(segments)
        return len(segments)

    def delete_video(self, video_id: str) -> None:
        self._rows = [r for r in self._rows if r.video_id != video_id]

    def search(
        self, vector: list[float], video_id: str | None, limit: int
    ) -> list[SegmentHit]:
        candidates = [r for r in self._rows if video_id is None or r.video_id == video_id]
        scored = []
        for row in candidates:
            similarity = _cosine(vector, row.vector)
            scored.append(
                SegmentHit(
                    video_id=row.video_id,
                    video_name=row.video_name,
                    clip_index=row.clip_index,
                    start_sec=row.start_sec,
                    end_sec=row.end_sec,
                    # Weaviate reports cosine *distance*; mirror that convention.
                    distance=1.0 - similarity,
                    description=row.description,
                )
            )
        scored.sort(key=lambda h: h.distance)
        return scored[:limit]

    def close(self) -> None:
        return None


def connect(
    url: str | None = None,
    api_key: str | None = None,
    allow_fallback: bool = True,
) -> SegmentStore:
    """Connect to Weaviate, falling back to the in-memory store if permitted."""
    try:
        store = WeaviateStore(url=url, api_key=api_key)
        store.ensure_schema()
        return store
    except Exception:
        if not allow_fallback:
            raise
        return InMemoryStore()


# ----------------------------------------------------------------------- utils


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _deterministic_uuid(video_id: str, clip_index: int) -> str:
    """Stable id so re-ingesting a video overwrites instead of duplicating."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{video_id}/{clip_index}"))


def _is_cloud(url: str) -> bool:
    return "weaviate.network" in url or "weaviate.cloud" in url


def _split_url(url: str) -> tuple[str, int, bool]:
    from urllib.parse import urlparse

    parsed = urlparse(url if "://" in url else f"http://{url}")
    secure = parsed.scheme == "https"
    return parsed.hostname or "localhost", parsed.port or (443 if secure else 80), secure
