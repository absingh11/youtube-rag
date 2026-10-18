# 🎬 YouTube RAG: Chat with any YouTube video

Paste a YouTube link and ask questions about the video. The answers come **only from the video's transcript** and include **clickable timestamps** that jump to the exact moment in the video where the answer is.

### 🎥 Demo
  
https://github.com/user-attachments/assets/c19c762a-12c0-464e-ae49-d003ce3767b9

Built with **LangChain (LCEL) · OpenAI · FAISS · FastAPI · Streamlit**.

---

## Architecture

```
                ┌────────────── INGESTION (once per video, cached) ───────────────┐
YouTube URL ──► extract_video_id ──► fetch transcript ──► timestamp-aware ──► OpenAI      ──► FAISS index
                                     (with start times)    chunking           embeddings      (saved to disk)
                └──────────────────────────────────────────────────────────────────┘

                ┌──────────────────────────── QUERY ──────────────────────────────┐
Question ──► MMR retriever (top-k) ──► format context with [mm:ss] ──► prompt ──► GPT ──► answer
                    │                                                                      + sources
                    └──────────────────── retrieved chunks ──────────────────────► (timestamp links)
                └──────────────────────────────────────────────────────────────────┘

Interfaces:  Streamlit chat UI (app.py) · REST API (FastAPI) · CLI
```

## Project structure

```
youtube-rag/
├── yt_rag/
│   ├── config.py       # all settings, overridable via .env
│   ├── transcript.py   # URL parsing, transcript fetch, timestamp-aware chunking
│   ├── indexer.py      # embeddings + FAISS, cached on disk per video
│   ├── rag.py          # LCEL RAG chain → answer + sources
│   ├── api.py          # FastAPI: /health, /ingest, /ask
│   └── cli.py          # terminal chat
├── app.py              # Streamlit chat UI
├── tests/              # pytest unit tests (no API key needed)
├── Dockerfile
├── requirements.txt
└── .env.example
```

## Quick start

```bash

python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # then add your OPENAI_API_KEY

streamlit run app.py                                   # chat UI → http://localhost:8501
uvicorn yt_rag.api:app --reload                        # API    → http://localhost:8000/docs
python -m yt_rag.cli "https://youtu.be/Gfr50f6ZBvo" -q "What is DeepMind?"
pytest -q                                              # run tests
```

### API example

```bash
curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"video": "https://youtu.be/Gfr50f6ZBvo", "question": "What is DeepMind?"}'
```
```json
{
  "video_id": "Gfr50f6ZBvo",
  "answer": "DeepMind is ... (see 2:14)",
  "sources": [{"timestamp": "2:14", "url": "https://www.youtube.com/watch?v=Gfr50f6ZBvo&t=134s", "text": "..."}]
}
```

### Docker
```bash
docker build -t yt-rag .
docker run -p 8501:8501 --env-file .env yt-rag
```

## Design decisions (the "why", useful in interviews)

| Decision | Reason |
|---|---|
| **Chunk on transcript-line boundaries** instead of raw text | Every chunk keeps an exact start/end time, so answers can cite and link timestamps. A plain `RecursiveCharacterTextSplitter` on the joined text loses this. |
| **1000-char chunks, 200 overlap** | Big enough to hold a complete thought, and the overlap keeps a sentence that crosses a boundary from being lost. |
| **MMR retrieval** (instead of plain similarity) | Overlapping chunks are near-duplicates, so plain top-k often returns the same passage 3 times. MMR balances relevance against diversity. |
| **Grounded prompt + "I couldn't find that"** | Reduces hallucination. The model is told to answer only from the context and to cite timestamps. |
| **Disk-cached FAISS index per video + embedding model** | A video is embedded only once, so repeat questions are fast and cheap. Including the model name in the cache key avoids stale vectors. |
| **Return sources with the answer** | Users can verify the answer, which is important for trust in any RAG system. |
| **Clear error handling** | Disabled captions, missing languages, and private videos all return clean 4xx errors instead of crashing. |

## Limitations & next steps
- Works only on videos that have captions. Next step: fall back to **Whisper** speech-to-text.
- Single-turn Q&A. Next step: add **conversation memory** with question rewriting for follow-ups.
- Evaluation: add a small Q&A test set and score it with **RAGAS** (faithfulness, answer relevance, context precision).
- Scaling: swap FAISS for a hosted vector DB (pgvector / Pinecone) and add a reranker (e.g. Cohere / cross-encoder).
- Stream tokens to the UI for faster perceived latency.
