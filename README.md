<div align="center">

# 🎬 Video Q&A with Timestamp Citations

### Ask a video anything. Get an answer that cites the exact second — and click to jump there.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](LIVE_APP_URL)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Gemini](https://img.shields.io/badge/Gemini-embedding--2--preview-4285F4?logo=google&logoColor=white)](https://ai.google.dev/gemini-api/docs/models/gemini-embedding-2-preview)
[![Weaviate](https://img.shields.io/badge/Weaviate-v4-00C9A7)](https://weaviate.io/)
[![Nebius](https://img.shields.io/badge/Nebius-Token%20Factory-1E88E5)](https://dub.sh/nebius)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**[🚀 Try the live app](LIVE_APP_URL)** · [How it works](#-how-it-works) · [Quickstart](#-quickstart) · [Architecture](#-architecture)

</div>

---

## The problem with asking questions about video

You have a 40-minute recording. Somewhere in it, someone said the thing you need. Scrubbing is hopeless, and "summarize this video" gives you a paragraph you can't verify.

The usual fix is a Rube Goldberg machine: run the audio through a transcription service, chunk the transcript, embed the chunks with a text model, run CLIP over sampled frames into a *second* index, then fuse the two rankings at query time and hope the timestamps survived the round trip.

**This project deletes that entire pipeline.**

Google's `gemini-embedding-2-preview` is natively multimodal — it embeds video (pixels *and* audio) and text into **one shared vector space**. So a plain-English question and a 20-second clip of footage are directly comparable. Retrieval collapses to a single `near_vector` call:

```python
query_vector = embedder.embed_text("what did she say about pricing?")
hits = store.search(vector=query_vector, video_id=video_id, limit=5)
```

No transcription service. No frame-level CLIP. No hybrid fusion. **One model, one index, one search.**

---

## ✨ What you get

| | |
|---|---|
| 🎯 **Answers that cite time, not vibes** | Every claim carries an inline `[mm:ss]` marker pointing at the footage it came from. |
| ▶️ **Click a timestamp, jump there** | Citations render as buttons that seek the embedded player to that exact second. |
| 🧠 **Native multimodal retrieval** | Text queries search video clips directly — same vector space, no translation layer. |
| 🔍 **Show your work** | Expand any answer to see the retrieved clips, their time spans, and cosine distances. |
| 🎚️ **Tunable precision** | 5s clips for surgical timestamps, 60s clips for cheap indexing. Your call, per video. |
| 🔄 **Swap the brain** | Any Nebius-served model answers — Qwen3-235B, DeepSeek-V3, Llama 3.3, Mistral. |
| 🪫 **Degrades gracefully** | No Weaviate reachable? Falls back to an in-memory vector store so the app still runs. |
| 🔐 **Bring your own keys** | Keys are entered in the sidebar and never persisted. Nothing is logged or stored. |

---

## 🧭 How it works

### Indexing

```
video.mp4
   │
   ├─ ffmpeg ──────────► 20s clips (360p, 8fps, mono audio)
   │                        │
   │                        ├─ gemini-embedding-2-preview ──► 1536-d vector
   │                        └─ gemini-2.5-flash ────────────► short description
   │                                                              │
   └──────────────────────────────────────► Weaviate ◄────────────┘
                                            (VideoSegment, BYO vectors)
```

Each clip becomes one `VideoSegment` object carrying `video_id`, `clip_index`, `start_sec`, `end_sec`, a description, and the vector Gemini produced. Vectors are **bring-your-own** — Weaviate stores and searches them, it never generates them.

### Answering

```
"what did she say about pricing?"
   │
   ├─ gemini-embedding-2-preview (text) ──► 1536-d vector
   │                                          │
   │                            near_vector search (top-k, scoped to video_id)
   │                                          │
   │                                    5 clips + time spans
   │                                          │
   └────────────────────► Nebius Qwen3-235B (chat completions)
                                              │
                          "She introduces the three tiers [01:20] and
                           notes the enterprise plan is custom-priced [02:05]."
                                              │
                                    ▶ 01:20   ▶ 02:05   ← clickable
```

### A note on honesty: why clips get described

The answering model on Nebius is **text-only** — it cannot watch video. So at ingest time, Gemini writes a short description of each clip (it watches the clip directly; this is not a transcription service in the pipeline).

**Retrieval never touches those descriptions.** Ranking is done entirely by the multimodal vectors. The descriptions exist purely so the text LLM has something concrete to read about the clips the vector search already chose. You can switch them off in the sidebar — ingest gets ~2× faster, and answers get correspondingly vaguer.

---

## 🚀 Quickstart

### Prerequisites

- **Python 3.10+**
- **ffmpeg** and **ffprobe** on your `PATH` — `brew install ffmpeg` · `apt install ffmpeg` · `winget install ffmpeg`
- A **[Google AI Studio](https://aistudio.google.com/apikey)** API key (free tier available)
- A **[Nebius Token Factory](https://dub.sh/nebius)** API key
- **Weaviate**, either local via Docker or a [Weaviate Cloud](https://console.weaviate.cloud/) cluster *(optional — the app falls back to an in-memory store)*

### 1. Start Weaviate locally

```bash
docker run -d --name weaviate -p 8080:8080 -p 50051:50051 \
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
  -e PERSISTENCE_DATA_PATH=/var/lib/weaviate \
  cr.weaviate.io/semitechnologies/weaviate:1.27.0
```

Or just use the bundled compose file, which also pins a persistent volume:

```bash
docker compose up -d
```

### 2. Install

```bash
git clone https://github.com/tirth1263/video-QnA-with-timestamp-citations.git
cd video-QnA-with-timestamp-citations

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# fill in NEBIUS_API_KEY, GEMINI_API_KEY,
# and optionally WEAVIATE_URL / WEAVIATE_API_KEY
```

### 3. Run

```bash
streamlit run main.py
```

1. **Upload a video** (`mp4` / `mov` / `mkv` / `webm`).
2. **Pick a clip length** (default 20s) and a Nebius answer model.
3. Click **⚡ Ingest video** — ffmpeg splits it, Gemini embeds each clip, Weaviate stores the vectors.
4. **Ask questions.** The pipeline embeds your query, searches Weaviate, and Nebius writes an answer with `[mm:ss]` citations.
5. **Click any timestamp button** to seek the embedded player to that moment.

> 💡 **First run tip:** start with a 2–3 minute video and 20s clips. That's ~9 embedding calls and finishes in well under a minute, so you can see the whole loop before committing to a long recording.

---

## 🏗 Architecture

```
Video ──► ffmpeg split (N-second clips) ──► Gemini gemini-embedding-2-preview (native video) ─┐
                                                                                              │
                                                             Weaviate (VideoSegment, BYO) ◄────┤
                                                                          │                   │
Query ──► Gemini gemini-embedding-2-preview (text) ──► near_vector search ◄────────────────────┘
                                                             │
                                                             ▼
                                            Nebius Qwen3-235B (chat completions)
                                                             │
                                             cited answer with [mm:ss] ◄──────────
```

Because text queries and video clips share the same embedding space, retrieval is a single vector search — no hybrid fusion.

### Project layout

```
video-QnA-with-timestamp-citations/
├── main.py               # Streamlit UI + embedded player that seeks to citations
├── ingest.py             # ffmpeg clip split + Gemini embed + Weaviate upsert
├── embeddings.py         # google-genai wrapper around gemini-embedding-2-preview
├── weaviate_store.py     # Weaviate v4 client + schema + search (+ in-memory fallback)
├── rag.py                # retrieve + Nebius chat completion with citations
├── docker-compose.yml    # local Weaviate
├── packages.txt          # apt deps for Streamlit Community Cloud (ffmpeg)
├── pyproject.toml
├── requirements.txt
└── .env.example
```

---

## ⚙️ Configuration

All settings are available in the sidebar at runtime, and every one of them can be seeded from the environment.

| Variable | Purpose | Default |
|---|---|---|
| `GEMINI_API_KEY` | Embeddings + clip descriptions | *required* |
| `NEBIUS_API_KEY` | Answer generation | *required* |
| `NEBIUS_BASE_URL` | Override for the legacy AI Studio endpoint | `https://api.tokenfactory.nebius.com/v1/` |
| `WEAVIATE_URL` | Local Docker or Weaviate Cloud cluster | `http://localhost:8080` |
| `WEAVIATE_API_KEY` | Weaviate Cloud auth | *empty* |
| `WEAVIATE_GRPC_PORT` | gRPC port for the v4 client | `50051` |

### Customization tips

- **Answer model** — swap `Qwen/Qwen3-235B-A22B` in the sidebar for any Nebius-served model.
- **Clip length** — shorter clips (10–15s) give tighter timestamps; longer clips (30–60s) cost fewer embedding calls.
- **Embedding dimensions** — `DEFAULT_DIM` in `embeddings.py` accepts 128–3072. Lower is smaller and faster; 1536 is the default trade-off. *Re-ingest after changing it — mixed dimensions break the index.*
- **Clip encoding** — `CLIP_HEIGHT`, `CLIP_FPS`, and `CLIP_AUDIO_BITRATE` in `ingest.py` control how aggressively clips are compressed before they go to the API.
- **Scope to one video** — multiple videos can be ingested; the UI scopes queries to the most recently ingested `video_id`.

---

## 🔬 Implementation details worth knowing

**Deterministic segment IDs.** Each segment's UUID is `uuid5(video_id + clip_index)`, so re-ingesting the same video overwrites its vectors instead of silently duplicating them.

**Content-addressed video IDs.** A video's ID hashes its name, size, first megabyte, *and* clip length — so the same file re-ingested at a different clip length is treated as a genuinely different index.

**Citation snapping.** LLMs cite `[01:21]` when the clip actually starts at `01:20`. Every extracted citation is snapped to the nearest retrieved segment start when it's within 2 seconds, so every button lands on real indexed footage rather than dead air.

**Partial-failure tolerance.** One clip failing to embed doesn't kill an ingest — failures are collected and reported, and the run continues with whatever succeeded.

**Retry with backoff.** Rate limits (`429`) and transient 5xx responses are retried with exponential backoff; genuine errors fail fast instead of burning four attempts.

**Files API escape hatch.** Clips under 15 MB go inline in the request. Anything larger is routed through the Gemini Files API automatically.

---

## 🩺 Troubleshooting

| Symptom | Fix |
|---|---|
| `GEMINI_API_KEY is not set` | Create one at [aistudio.google.com/apikey](https://aistudio.google.com/apikey), then add it to `.env` or the sidebar. |
| `weaviate.exceptions.WeaviateConnectionError` | Make sure Docker is running and `WEAVIATE_URL=http://localhost:8080`. Or leave the in-memory fallback enabled. |
| `ffmpeg: command not found` | Install ffmpeg and confirm both `ffmpeg` and `ffprobe` are on `PATH`. |
| Answers say "the excerpts do not cover that" | Retrieval found nothing relevant. Raise *clips retrieved per question*, or re-ingest with shorter clips. |
| Ingest is slow | Each clip is one embedding call plus one description call. Use longer clips, or switch off clip descriptions in the sidebar. |
| Timestamps feel imprecise | Lower the clip length. A 60s clip can only ever cite to within a minute. |

---

## 📜 License

MIT — see [LICENSE](LICENSE).

<div align="center">

Built with **[Gemini Embedding 2](https://ai.google.dev/gemini-api/docs/models/gemini-embedding-2-preview)** · **[Weaviate](https://weaviate.io/)** · **[Nebius Token Factory](https://dub.sh/nebius)** · **[Streamlit](https://streamlit.io/)**

⭐ If this saved you from building a three-stage transcription pipeline, consider starring the repo.

</div>
