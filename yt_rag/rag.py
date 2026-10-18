"""The RAG chain: retrieve -> format context -> prompt -> LLM -> answer + sources."""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI

from .config import settings
from .indexer import build_or_load_index
from .transcript import extract_video_id, format_timestamp

SYSTEM_PROMPT = """You are an assistant that answers questions about a YouTube video using ONLY \
the transcript excerpts provided below. Each excerpt starts with its timestamp in [brackets].

Rules:
- Answer using the excerpts. If they only partly answer the question, share what they do say.
- only say "I couldn't find that in the video." if nothing in the excerpts is relevant.
- Transcripts are auto-generated and may misspell names; interpret them sensibly..

Transcript excerpts:
{context}"""

PROMPT = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", "{question}")])


def format_docs(docs: list[Document]) -> str:
    docs = sorted(docs, key=lambda d: d.metadata.get("start", 0))  # chronological reads better
    return "\n\n".join(f"[{format_timestamp(d.metadata.get('start', 0))}] {d.page_content}" for d in docs)


@dataclass
class Source:
    timestamp: str
    url: str
    text: str


@dataclass
class Answer:
    answer: str
    sources: list[Source] = field(default_factory=list)


class VideoQA:
    """Ask questions about one video. Build once, then call .ask() many times."""

    def __init__(self, url_or_id: str, force_rebuild: bool = False):
        self.video_id = extract_video_id(url_or_id)
        self.store, self.n_chunks = build_or_load_index(self.video_id, force_rebuild)

        search_kwargs = {"k": settings.top_k}
        if settings.search_type == "mmr":
            # MMR = relevant AND diverse chunks, avoids 4 near-duplicate overlapping chunks.
            search_kwargs["fetch_k"] = max(20, settings.top_k * 5)
        self.retriever = self.store.as_retriever(search_type=settings.search_type, search_kwargs=search_kwargs)

        llm = ChatOpenAI(model=settings.llm_model, temperature=settings.temperature, max_tokens=settings.max_tokens)

        # Same structure as the original script, but it also returns the retrieved
        # docs so the UI can show sources.
        retrieve = RunnableParallel(docs=self.retriever, question=RunnablePassthrough())
        generate = (
            RunnableLambda(lambda x: {"context": format_docs(x["docs"]), "question": x["question"]})
            | PROMPT
            | llm
            | StrOutputParser()
        )
        self.chain = retrieve | RunnablePassthrough.assign(answer=generate)

    def ask(self, question: str) -> Answer:
        if not question.strip():
            raise ValueError("Question must not be empty.")
        out = self.chain.invoke(question)
        sources = [
            Source(
                timestamp=format_timestamp(d.metadata["start"]),
                url=f"https://www.youtube.com/watch?v={self.video_id}&t={int(d.metadata['start'])}s",
                text=d.page_content,
            )
            for d in sorted(out["docs"], key=lambda d: d.metadata["start"])
        ]
        return Answer(answer=out["answer"], sources=sources)
