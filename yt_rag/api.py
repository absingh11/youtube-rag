"""REST API. Run with:  uvicorn yt_rag.api:app --reload"""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .rag import VideoQA
from .transcript import TranscriptError

app = FastAPI(title="YouTube RAG API", version=__version__)


class IngestRequest(BaseModel):
    video: str = Field(..., examples=["https://www.youtube.com/watch?v=Gfr50f6ZBvo"])
    force_rebuild: bool = False


class AskRequest(BaseModel):
    video: str = Field(..., examples=["Gfr50f6ZBvo"])
    question: str = Field(..., min_length=1, examples=["What is DeepMind?"])


@lru_cache(maxsize=32)
def _get_qa(video: str) -> VideoQA:
    return VideoQA(video)


def _load(video: str, force: bool = False) -> VideoQA:
    try:
        if force:
            _get_qa.cache_clear()
            return VideoQA(video, force_rebuild=True)
        return _get_qa(video)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except TranscriptError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok", "version": __version__}


@app.post("/ingest")
def ingest(req: IngestRequest):
    qa = _load(req.video, req.force_rebuild)
    return {"video_id": qa.video_id, "chunks": qa.n_chunks}


@app.post("/ask")
def ask(req: AskRequest):
    qa = _load(req.video)
    result = qa.ask(req.question)
    return {
        "video_id": qa.video_id,
        "answer": result.answer,
        "sources": [s.__dict__ for s in result.sources],
    }
