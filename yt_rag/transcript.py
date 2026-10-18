"""Fetch YouTube transcripts and turn them into timestamped chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse

VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


class TranscriptError(Exception):
    """Raised when a transcript can't be fetched (disabled, missing, bad ID)."""


@dataclass
class Segment:
    text: str
    start: float  # seconds
    duration: float = 0.0


@dataclass
class Chunk:
    text: str
    start: float
    end: float


def extract_video_id(url_or_id: str) -> str:
    """Accept a raw ID or any common YouTube URL form and return the 11-char video ID."""
    value = url_or_id.strip()
    if VIDEO_ID_RE.match(value):
        return value

    parsed = urlparse(value if "://" in value else "https://" + value)
    host = (parsed.hostname or "").lower().removeprefix("www.").removeprefix("m.")

    candidate = None
    if host == "youtu.be":
        candidate = parsed.path.lstrip("/").split("/")[0]
    elif host in ("youtube.com", "music.youtube.com", "youtube-nocookie.com"):
        if parsed.path == "/watch":
            candidate = parse_qs(parsed.query).get("v", [None])[0]
        else:
            parts = [p for p in parsed.path.split("/") if p]
            if len(parts) >= 2 and parts[0] in ("embed", "shorts", "live", "v"):
                candidate = parts[1]

    if candidate and VIDEO_ID_RE.match(candidate):
        return candidate
    raise ValueError(f"Could not find a valid YouTube video ID in: {url_or_id!r}")


def format_timestamp(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def fetch_transcript(video_id: str, languages: tuple[str, ...] = ("en",)) -> list[Segment]:
    """Download the transcript, keeping per-line timestamps."""
    from youtube_transcript_api import (
        NoTranscriptFound,
        TranscriptsDisabled,
        VideoUnavailable,
        YouTubeTranscriptApi,
    )

    try:
        fetched = YouTubeTranscriptApi().fetch(video_id, languages=list(languages))
    except TranscriptsDisabled as e:
        raise TranscriptError("Captions are disabled for this video.") from e
    except NoTranscriptFound as e:
        raise TranscriptError(f"No transcript found in languages {list(languages)}.") from e
    except VideoUnavailable as e:
        raise TranscriptError("Video is unavailable (private, deleted or wrong ID).") from e
    except Exception as e:  # network errors, IP blocks, etc.
        raise TranscriptError(f"Failed to fetch transcript: {e}") from e

    return [
        Segment(text=s.text.replace("\n", " ").strip(), start=s.start, duration=s.duration)
        for s in fetched
        if s.text.strip()
    ]


def chunk_segments(segments: list[Segment], chunk_size: int = 1000, overlap: int = 200) -> list[Chunk]:
    """
    Group transcript lines into ~chunk_size-character chunks with ~overlap characters
    shared between neighbours. Splitting on line boundaries (instead of raw text)
    means every chunk keeps an exact start/end timestamp, so answers can cite
    "see 12:34" and link straight to that moment in the video.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[Chunk] = []
    i, n = 0, len(segments)
    while i < n:
        j, length = i, 0
        while j < n and (length == 0 or length + len(segments[j].text) + 1 <= chunk_size):
            length += len(segments[j].text) + 1
            j += 1

        window = segments[i:j]
        last = window[-1]
        chunks.append(
            Chunk(
                text=" ".join(s.text for s in window),
                start=window[0].start,
                end=last.start + last.duration,
            )
        )
        if j >= n:
            break

        # Step back over trailing lines to create the overlap, but always advance.
        back, k = 0, j
        while k - 1 > i and back + len(segments[k - 1].text) + 1 <= overlap:
            k -= 1
            back += len(segments[k].text) + 1
        i = max(k, i + 1)

    return chunks
