"""Streamlit UI: upload a video, ask questions, click the timestamps.

Run with:  streamlit run main.py
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from embeddings import DEFAULT_DIM, DEFAULT_MODEL, GeminiDescriber, GeminiEmbedder
from ingest import (
    DEFAULT_CLIP_SECONDS,
    IngestError,
    estimate_clip_count,
    ffmpeg_available,
    ingest_video,
    probe_duration,
)
from rag import DEFAULT_ANSWER_MODEL, DEFAULT_TOP_K, RAGError, VideoRAG
from weaviate_store import InMemoryStore, WeaviateStore, format_timestamp

load_dotenv()

NEBIUS_MODELS = [
    "Qwen/Qwen3-235B-A22B",
    "Qwen/Qwen3-30B-A3B",
    "deepseek-ai/DeepSeek-V3",
    "meta-llama/Llama-3.3-70B-Instruct",
    "mistralai/Mistral-Nemo-Instruct-2407",
]

ACCEPTED_TYPES = ["mp4", "mov", "mkv", "webm"]

GEMINI_KEY_URL = "https://aistudio.google.com/apikey"

st.set_page_config(
    page_title="Video Q&A with Timestamp Citations",
    page_icon="🎬",
    layout="wide",
)


# ----------------------------------------------------------------- helpers


def secret(name: str, default: str = "") -> str:
    """Env var first, then Streamlit secrets - so both local and cloud work."""
    value = os.getenv(name)
    if value:
        return value
    try:
        return str(st.secrets[name])  # type: ignore[index]
    except Exception:
        return default


# Which keys this particular deployment ships. A public demo may reasonably
# provide the cheap answering key and ask visitors for the expensive one.
GEMINI_PRESET = secret("GEMINI_API_KEY")
NEBIUS_PRESET = secret("NEBIUS_API_KEY")


def key_field(label: str, preset: str, help_text: str, used_for: str) -> str:
    """Render a key input, or a confirmation when the deployment supplies it.

    A deployment can ship some keys and not others - shipping only the cheap
    one is a reasonable way to run a public demo. When a key is already
    present, asking for it again just makes visitors think it is missing.
    """
    if not preset:
        return st.text_input(label, value="", type="password", help=help_text)

    st.markdown(f"✅ **{label}** — provided, used for {used_for}.")
    if not st.checkbox(f"Use my own {label.split()[0]} key instead", key=f"own_{label}"):
        return preset
    return st.text_input(label, value="", type="password", help=help_text) or preset


def init_state() -> None:
    defaults = {
        "video_path": None,
        "video_name": None,
        "video_id": None,
        "duration": 0.0,
        "ingested": False,
        "clip_seconds": DEFAULT_CLIP_SECONDS,
        "segments": 0,
        "backend": None,
        "messages": [],
        "seek_to": 0,
        "warnings": [],
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


@st.cache_resource(show_spinner=False)
def get_store(url: str, api_key: str, allow_fallback: bool):
    """One store per (url, key) pair, kept alive across reruns."""
    try:
        store = WeaviateStore(url=url or None, api_key=api_key or None)
        store.ensure_schema()
        return store, None
    except Exception as exc:  # noqa: BLE001
        if not allow_fallback:
            raise
        return InMemoryStore(), str(exc)


def persist_upload(uploaded) -> str:
    """Write the upload somewhere ffmpeg and the video player can both reach."""
    suffix = Path(uploaded.name).suffix or ".mp4"
    tmp_dir = Path(tempfile.gettempdir()) / "video_rag_uploads"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    target = tmp_dir / f"{abs(hash(uploaded.name))}{suffix}"
    target.write_bytes(uploaded.getbuffer())
    return str(target)


def seek(seconds: float) -> None:
    st.session_state.seek_to = int(seconds)


# ------------------------------------------------------------------ sidebar

init_state()

with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    with st.expander("🔑 API keys", expanded=not GEMINI_PRESET):
        gemini_key = key_field(
            label="Gemini API key",
            preset=GEMINI_PRESET,
            help_text="Free key from https://aistudio.google.com/apikey",
            used_for="embedding clips and your questions",
        )
        nebius_key = key_field(
            label="Nebius API key",
            preset=NEBIUS_PRESET,
            help_text="Get one at https://dub.sh/nebius",
            used_for="writing the cited answer",
        )

    with st.expander("🗄️ Weaviate", expanded=False):
        weaviate_url = st.text_input(
            "Weaviate URL",
            value=secret("WEAVIATE_URL", "http://localhost:8080"),
        )
        weaviate_key = st.text_input(
            "Weaviate API key (cloud only)",
            value=secret("WEAVIATE_API_KEY"),
            type="password",
        )
        allow_fallback = st.checkbox(
            "Fall back to in-memory store",
            value=True,
            help=(
                "If Weaviate is unreachable, keep vectors in memory for this "
                "session instead of failing. Required on hosts without Docker."
            ),
        )

    st.divider()

    clip_seconds = st.slider(
        "Clip length (seconds)",
        min_value=5,
        max_value=60,
        value=DEFAULT_CLIP_SECONDS,
        step=5,
        help="Shorter clips give tighter timestamps. Longer clips cost fewer "
        "embedding calls.",
    )
    answer_model = st.selectbox("Nebius answer model", NEBIUS_MODELS, index=0)
    top_k = st.slider("Clips retrieved per question", 1, 12, DEFAULT_TOP_K)
    describe_clips = st.checkbox(
        "Describe clips during ingest",
        value=True,
        help=(
            "Retrieval is always pure vector search. Descriptions give the "
            "text-only answer model something to read. Turning this off makes "
            "ingest ~2x faster but answers much vaguer."
        ),
    )

    st.divider()
    st.caption(
        f"**Embeddings** `{DEFAULT_MODEL}` · {DEFAULT_DIM}d  \n"
        f"**Vector DB** Weaviate v4 (BYO vectors)  \n"
        f"**LLM** Nebius Token Factory"
    )


# --------------------------------------------------------------------- head

st.title("🎬 Video Q&A with Timestamp Citations")
st.markdown(
    "Ask questions about a video and get answers with **clickable timestamps**. "
    "One native multimodal embedding model handles both the clips and your "
    "question, so retrieval is a single vector search - no transcription "
    "service, no frame-level CLIP."
)

if NEBIUS_PRESET and not GEMINI_PRESET and not gemini_key:
    st.info(
        f"**One thing before you start: paste a Gemini API key in the sidebar.** "
        f"It is free at [aistudio.google.com/apikey]({GEMINI_KEY_URL}) and takes "
        "about a minute. That key embeds your video, which is the part that "
        "costs real money — so it runs on your quota, not this demo's. "
        "The answer model is already provided.",
        icon="🔑",
    )

if not ffmpeg_available():
    st.error(
        "**ffmpeg / ffprobe were not found on PATH.** Ingestion needs them to "
        "split the video. Install with `brew install ffmpeg` (macOS), "
        "`apt install ffmpeg` (Debian/Ubuntu), or `winget install ffmpeg` "
        "(Windows), then restart the app."
    )

left, right = st.columns([1, 1], gap="large")


# ------------------------------------------------------------------- ingest

with left:
    st.subheader("1 · Load a video")

    uploaded = st.file_uploader(
        "Upload a video",
        type=ACCEPTED_TYPES,
        label_visibility="collapsed",
    )

    if uploaded is not None:
        path = persist_upload(uploaded)
        if path != st.session_state.video_path:
            st.session_state.video_path = path
            st.session_state.video_name = uploaded.name
            st.session_state.ingested = False
            st.session_state.messages = []
            st.session_state.seek_to = 0
            try:
                st.session_state.duration = probe_duration(path)
            except IngestError:
                st.session_state.duration = 0.0

    if st.session_state.video_path:
        st.video(st.session_state.video_path, start_time=st.session_state.seek_to)

        duration = st.session_state.duration
        if duration:
            clips = estimate_clip_count(duration, clip_seconds)
            st.caption(
                f"`{st.session_state.video_name}` · {format_timestamp(duration)} · "
                f"~{clips} clips at {clip_seconds}s "
                f"({clips} embedding call{'s' if clips != 1 else ''})"
            )

        ingest_clicked = st.button(
            "⚡ Ingest video",
            type="primary",
            use_container_width=True,
            disabled=not ffmpeg_available(),
        )

        if ingest_clicked:
            if not gemini_key:
                st.error("Add your Gemini API key in the sidebar first.")
            else:
                os.environ["GEMINI_API_KEY"] = gemini_key
                store, fallback_reason = get_store(
                    weaviate_url, weaviate_key, allow_fallback
                )

                progress_bar = st.progress(0.0)
                status = st.empty()

                def on_progress(done: int, total: int, message: str) -> None:
                    progress_bar.progress(min(1.0, done / max(total, 1)))
                    status.write(message)

                try:
                    embedder = GeminiEmbedder(api_key=gemini_key)
                    describer = (
                        GeminiDescriber(api_key=gemini_key) if describe_clips else None
                    )
                    result = ingest_video(
                        st.session_state.video_path,
                        store=store,
                        embedder=embedder,
                        clip_seconds=clip_seconds,
                        video_name=st.session_state.video_name,
                        progress=on_progress,
                        describer=describer,
                    )
                except Exception as exc:  # noqa: BLE001
                    progress_bar.empty()
                    status.empty()
                    st.error(f"Ingestion failed: {exc}")
                else:
                    progress_bar.empty()
                    status.empty()

                    st.session_state.video_id = result.video_id
                    st.session_state.ingested = True
                    st.session_state.segments = result.segments_indexed
                    st.session_state.backend = result.backend
                    st.session_state.clip_seconds = result.clip_seconds
                    st.session_state.warnings = result.failures

                    st.success(
                        f"Indexed **{result.segments_indexed}** segments from "
                        f"`{result.video_name}`."
                    )
                    if fallback_reason:
                        st.info(
                            "Weaviate was unreachable, so vectors are held in "
                            "memory for this session only.",
                            icon="ℹ️",
                        )
                    if result.failures:
                        with st.expander(
                            f"{len(result.failures)} clip(s) failed to embed"
                        ):
                            for failure in result.failures:
                                st.write(f"- {failure}")
    else:
        st.info("Upload an mp4, mov, mkv, or webm file to get started.")


# ---------------------------------------------------------------------- ask

with right:
    st.subheader("2 · Ask about it")

    if not st.session_state.ingested:
        st.info("Ingest a video first, then ask anything about it here.")
    else:
        st.caption(
            f"Searching **{st.session_state.segments}** segments · "
            f"store: `{st.session_state.backend}` · "
            f"clip length: {st.session_state.clip_seconds}s"
        )

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                for index, citation in enumerate(message.get("citations", [])):
                    st.button(
                        f"▶ {citation['label']}",
                        key=f"cite-{message['id']}-{index}",
                        on_click=seek,
                        args=(citation["seconds"],),
                    )

                if message.get("hits"):
                    with st.expander("Retrieved clips"):
                        for hit in message["hits"]:
                            st.markdown(
                                f"**{hit['span']}** · distance `{hit['distance']:.4f}`"
                            )
                            if hit["description"]:
                                st.caption(hit["description"])

        question = st.chat_input("What happens in this video?")

        if question:
            if not nebius_key:
                st.error("Add your Nebius API key in the sidebar first.")
            elif not gemini_key:
                st.error("Add your Gemini API key in the sidebar first.")
            else:
                st.session_state.messages.append(
                    {
                        "role": "user",
                        "content": question,
                        "id": f"u{len(st.session_state.messages)}",
                    }
                )
                store, _ = get_store(weaviate_url, weaviate_key, allow_fallback)

                with st.chat_message("assistant"), st.spinner("Searching the video..."):
                    try:
                        pipeline = VideoRAG(
                            store=store,
                            embedder=GeminiEmbedder(api_key=gemini_key),
                            nebius_api_key=nebius_key,
                            model=answer_model,
                        )
                        answer = pipeline.answer(
                            question,
                            video_id=st.session_state.video_id,
                            top_k=top_k,
                            model=answer_model,
                        )
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Could not answer: {exc}")
                    else:
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": answer.text,
                                "id": f"a{len(st.session_state.messages)}",
                                "citations": [
                                    {"label": c.label, "seconds": c.seconds}
                                    for c in answer.citations
                                ],
                                "hits": [
                                    {
                                        "span": hit.span,
                                        "distance": hit.distance,
                                        "description": hit.description,
                                    }
                                    for hit in answer.hits
                                ],
                            }
                        )
                        st.rerun()

st.divider()
st.caption(
    "Built with Gemini `gemini-embedding-2-preview` · Weaviate v4 · "
    "Nebius Token Factory · Streamlit"
)
