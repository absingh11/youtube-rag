"""Central configuration. Every value can be overridden with an environment variable."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    temperature: float = float(os.getenv("TEMPERATURE", "0.2"))
    max_tokens: int = int(os.getenv("MAX_TOKENS", "500"))

    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    top_k: int = int(os.getenv("TOP_K", "4"))
    search_type: str = os.getenv("SEARCH_TYPE", "mmr")  # "similarity" or "mmr"

    languages: tuple = tuple(os.getenv("TRANSCRIPT_LANGUAGES", "en,en-US,en-GB,hi").split(","))
    index_dir: Path = Path(os.getenv("INDEX_DIR", "data/indexes"))


settings = Settings()
