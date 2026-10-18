"""Unit tests for the parts that don't need an API key or network."""

import pytest

from yt_rag.transcript import Segment, chunk_segments, extract_video_id, format_timestamp

VID = "Gfr50f6ZBvo"


@pytest.mark.parametrize(
    "value",
    [
        VID,
        f"https://www.youtube.com/watch?v={VID}",
        f"https://www.youtube.com/watch?v={VID}&t=120s&list=abc",
        f"https://youtu.be/{VID}",
        f"https://youtu.be/{VID}?si=xyz",
        f"https://www.youtube.com/embed/{VID}",
        f"https://www.youtube.com/shorts/{VID}",
        f"https://m.youtube.com/watch?v={VID}",
        f"youtube.com/watch?v={VID}",
    ],
)
def test_extract_video_id(value):
    assert extract_video_id(value) == VID


@pytest.mark.parametrize("bad", ["", "hello", "https://example.com/watch?v=Gfr50f6ZBvo", "https://youtu.be/short"])
def test_extract_video_id_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        extract_video_id(bad)


def test_format_timestamp():
    assert format_timestamp(0) == "0:00"
    assert format_timestamp(75.9) == "1:15"
    assert format_timestamp(3725) == "1:02:05"


def _segments(n=100, words=8):
    return [Segment(text=" ".join(f"w{i}" for _ in range(words)), start=i * 3.0, duration=3.0) for i in range(n)]


def test_chunks_respect_size_and_cover_everything():
    segs = _segments()
    chunks = chunk_segments(segs, chunk_size=200, overlap=50)
    assert all(len(c.text) <= 200 for c in chunks)
    assert chunks[0].start == 0.0
    assert chunks[-1].end == segs[-1].start + segs[-1].duration
    joined = " ".join(c.text for c in chunks)
    assert all(s.text in joined for s in segs)  # no line lost


def test_chunks_overlap_and_timestamps_increase():
    chunks = chunk_segments(_segments(), chunk_size=200, overlap=50)
    for a, b in zip(chunks, chunks[1:]):
        assert b.start > a.start
        assert b.start < a.end  # neighbours share at least one line


def test_long_single_line_does_not_loop_forever():
    segs = [Segment(text="x" * 5000, start=0, duration=5), Segment(text="y", start=5, duration=1)]
    chunks = chunk_segments(segs, chunk_size=1000, overlap=200)
    assert len(chunks) == 2


def test_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_segments(_segments(), chunk_size=100, overlap=100)
