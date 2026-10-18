"""Command line usage:
    python -m yt_rag.cli "https://youtu.be/Gfr50f6ZBvo" -q "What is DeepMind?"
    python -m yt_rag.cli Gfr50f6ZBvo          # interactive chat
"""

import argparse
import logging

from .rag import VideoQA
from .transcript import TranscriptError


def main():
    p = argparse.ArgumentParser(description="Chat with a YouTube video")
    p.add_argument("video", help="YouTube URL or video ID")
    p.add_argument("-q", "--question", help="Ask one question and exit")
    p.add_argument("--rebuild", action="store_true", help="Ignore the cached index")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        qa = VideoQA(args.video, force_rebuild=args.rebuild)
    except (ValueError, TranscriptError) as e:
        raise SystemExit(f"Error: {e}")
    print(f"Indexed video {qa.video_id} ({qa.n_chunks} chunks)\n")

    def answer(q):
        res = qa.ask(q)
        print(res.answer)
        print("\nSources: " + ", ".join(s.timestamp for s in res.sources) + "\n")

    if args.question:
        answer(args.question)
        return
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"exit", "quit"}:
            break
        if q:
            answer(q)


if __name__ == "__main__":
    main()
