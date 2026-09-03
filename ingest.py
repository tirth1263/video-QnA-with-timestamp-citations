"""Ingestion: ffmpeg clip split -> Gemini embed -> Weaviate upsert.

The whole indexing story is three steps and no transcript:

    video --ffmpeg--> N-second clips --gemini--> vectors --weaviate--> index

Clips are re-encoded small (360p, low fps, mono audio) on purpose. The
embedding model does not need a pristine master, and a small clip keeps the
request inline and the call fast.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from embeddings import GeminiDescriber, GeminiEmbedder
from weaviate_store import Segment, SegmentStore, format_timestamp

DEFAULT_CLIP_SECONDS = 20

# Re-encode target for the clips that get shipped to the embedding API.
CLIP_HEIGHT = 360
CLIP_FPS = 8
CLIP_AUDIO_BITRATE = "64k"

ProgressFn = Callable[[int, int, str], None]


class IngestError(RuntimeError):
    """Raised when a video cannot be turned into indexed segments."""


@dataclass
class Clip:
    index: int
    start_sec: float
    end_sec: float
    path: Path

    @property
    def label(self) -> str:
        return f"{format_timestamp(self.start_sec)}-{format_timestamp(self.end_sec)}"


@dataclass
class IngestResult:
    video_id: str
    video_name: str
    duration_sec: float
    clip_seconds: int
    segments_indexed: int
    backend: str
    failures: list[str]


# ------------------------------------------------------------------- ffmpeg


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def probe_duration(video_path: str | Path) -> float:
    """Duration in seconds, via ffprobe."""
    proc = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            str(video_path),
        ]
    )
    if proc.returncode != 0:
        raise IngestError(f"ffprobe failed: {proc.stderr.strip() or 'unknown error'}")

    try:
        duration = float(json.loads(proc.stdout)["format"]["duration"])
    except (ValueError, KeyError, json.JSONDecodeError) as exc:
        raise IngestError(f"Could not read duration from ffprobe output: {exc}") from exc

    if duration <= 0:
        raise IngestError("Video reports a non-positive duration.")
    return duration


def split_video(
    video_path: str | Path,
    out_dir: str | Path,
    clip_seconds: int = DEFAULT_CLIP_SECONDS,
) -> list[Clip]:
    """Cut the video into fixed-length clips using ffmpeg's segment muxer."""
    if not ffmpeg_available():
        raise IngestError(
            "ffmpeg/ffprobe not found on PATH. Install ffmpeg "
            "(brew install ffmpeg / apt install ffmpeg) and retry."
        )
    if clip_seconds <= 0:
        raise IngestError("Clip length must be a positive number of seconds.")

    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    duration = probe_duration(video_path)
    pattern = str(out_dir / "clip_%04d.mp4")

    proc = _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"scale=-2:{CLIP_HEIGHT},fps={CLIP_FPS}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "30",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            CLIP_AUDIO_BITRATE,
            "-ac",
            "1",
            "-f",
            "segment",
            "-segment_time",
            str(clip_seconds),
            "-reset_timestamps",
            "1",
            "-force_key_frames",
            f"expr:gte(t,n_forced*{clip_seconds})",
            pattern,
        ]
    )
    if proc.returncode != 0:
        raise IngestError(f"ffmpeg failed to split the video: {proc.stderr.strip()}")

    files = sorted(out_dir.glob("clip_*.mp4"))
    if not files:
        raise IngestError("ffmpeg produced no clips - is the file a valid video?")

    clips: list[Clip] = []
    for index, path in enumerate(files):
        start = index * clip_seconds
        clips.append(
            Clip(
                index=index,
                start_sec=float(start),
                end_sec=float(min(start + clip_seconds, duration)),
                path=path,
            )
        )
    return clips


# ------------------------------------------------------------------ ingestion


def video_id_for(video_path: str | Path, clip_seconds: int) -> str:
    """Content-addressed id, so the same file at the same clip length reuses vectors."""
    video_path = Path(video_path)
    digest = hashlib.sha256()
    digest.update(video_path.name.encode("utf-8", "ignore"))
    digest.update(str(video_path.stat().st_size).encode())
    digest.update(str(clip_seconds).encode())

    # Sample the head of the file so different videos of equal size differ.
    with video_path.open("rb") as handle:
        digest.update(handle.read(1024 * 1024))

    return digest.hexdigest()[:16]


def ingest_video(
    video_path: str | Path,
    store: SegmentStore,
    embedder: GeminiEmbedder,
    clip_seconds: int = DEFAULT_CLIP_SECONDS,
    video_name: str | None = None,
    progress: ProgressFn | None = None,
    work_dir: str | Path | None = None,
    describer: GeminiDescriber | None = None,
) -> IngestResult:
    """Split, embed, and index a video. Returns a summary of what landed."""
    video_path = Path(video_path)
    if not video_path.exists():
        raise IngestError(f"Video not found: {video_path}")

    video_name = video_name or video_path.name
    video_id = video_id_for(video_path, clip_seconds)
    duration = probe_duration(video_path)

    owns_temp = work_dir is None
    work_dir = Path(work_dir or tempfile.mkdtemp(prefix="video_rag_"))

    def report(done: int, total: int, message: str) -> None:
        if progress:
            progress(done, total, message)

    try:
        report(0, 1, "Splitting video into clips with ffmpeg...")
        clips = split_video(video_path, work_dir / "clips", clip_seconds)
        total = len(clips)

        # Re-ingesting the same video replaces its old segments.
        store.delete_video(video_id)

        segments: list[Segment] = []
        failures: list[str] = []

        for done, clip in enumerate(clips, start=1):
            report(done - 1, total, f"Embedding clip {done}/{total} ({clip.label})")
            try:
                vector = embedder.embed_video(clip.path)
            except Exception as exc:  # noqa: BLE001 - one bad clip must not kill ingest
                failures.append(f"{clip.label}: {exc}")
                continue

            description = ""
            if describer is not None:
                report(done - 1, total, f"Describing clip {done}/{total} ({clip.label})")
                description = describer.describe(clip.path, timespan=clip.label)

            segments.append(
                Segment(
                    video_id=video_id,
                    video_name=video_name,
                    clip_index=clip.index,
                    start_sec=clip.start_sec,
                    end_sec=clip.end_sec,
                    vector=vector,
                    description=description,
                )
            )

        if not segments:
            detail = failures[0] if failures else "no clips were embedded"
            raise IngestError(f"Every clip failed to embed. First error - {detail}")

        report(total, total, f"Indexing {len(segments)} segments...")
        store.upsert_segments(segments)

        return IngestResult(
            video_id=video_id,
            video_name=video_name,
            duration_sec=duration,
            clip_seconds=clip_seconds,
            segments_indexed=len(segments),
            backend=getattr(store, "backend", "unknown"),
            failures=failures,
        )
    finally:
        if owns_temp:
            shutil.rmtree(work_dir, ignore_errors=True)


def estimate_clip_count(duration_sec: float, clip_seconds: int) -> int:
    """How many embedding calls an ingest will cost, before running it."""
    if clip_seconds <= 0:
        return 0
    return max(1, int(-(-duration_sec // clip_seconds)))


def iter_clip_labels(clips: Iterable[Clip]) -> list[str]:
    return [clip.label for clip in clips]
