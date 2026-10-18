"""Build (or load from disk) a FAISS index for a video."""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from .config import settings
from .transcript import chunk_segments, fetch_transcript

log = logging.getLogger(__name__)


def _embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=settings.embedding_model)


def _index_path(video_id: str) -> Path:
    # Include the embedding model so switching models never reuses stale vectors.
    model_tag = settings.embedding_model.replace("/", "_")
    return settings.index_dir / f"{video_id}__{model_tag}"


def build_or_load_index(video_id: str, force_rebuild: bool = False) -> tuple[FAISS, int]:
    """
    Return (vectorstore, n_chunks). Indexes are cached on disk, so a video is
    only transcribed and embedded once; later questions cost one query embedding.
    """
    path = _index_path(video_id)
    if path.exists() and not force_rebuild:
        log.info("Loading cached index for %s", video_id)
        store = FAISS.load_local(str(path), _embeddings(), allow_dangerous_deserialization=True)
        return store, store.index.ntotal

    log.info("Building index for %s", video_id)
    segments = fetch_transcript(video_id, settings.languages)
    chunks = chunk_segments(segments, settings.chunk_size, settings.chunk_overlap)
    docs = [
        Document(
            page_content=c.text,
            metadata={"video_id": video_id, "start": c.start, "end": c.end, "chunk_id": i},
        )
        for i, c in enumerate(chunks)
    ]
    store = FAISS.from_documents(docs, _embeddings())
    path.parent.mkdir(parents=True, exist_ok=True)
    store.save_local(str(path))
    return store, len(docs)
