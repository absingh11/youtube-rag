"""Streamlit chat UI. Run with:  streamlit run app.py"""

import streamlit as st

from yt_rag.rag import VideoQA
from yt_rag.transcript import TranscriptError

st.set_page_config(page_title="Chat with YouTube", page_icon="🎬", layout="wide")
st.title("🎬 Chat with any YouTube video")
st.caption("Retrieval-Augmented Generation · LangChain · FAISS · OpenAI")


@st.cache_resource(show_spinner=False)
def load_qa(video: str) -> VideoQA:
    return VideoQA(video)


with st.sidebar:
    st.header("1. Load a video")
    video = st.text_input("YouTube URL or ID", value="https://www.youtube.com/watch?v=Gfr50f6ZBvo")
    if st.button("Load video", type="primary", use_container_width=True):
        try:
            with st.spinner("Fetching transcript and building index..."):
                qa = load_qa(video)
            st.session_state.qa = qa
            st.session_state.messages = []
        except (ValueError, TranscriptError) as e:
            st.error(str(e))

    if qa := st.session_state.get("qa"):
        st.success(f"Ready: {qa.n_chunks} chunks indexed")
        st.video(f"https://www.youtube.com/watch?v={qa.video_id}")

qa = st.session_state.get("qa")
if not qa:
    st.info("Paste a YouTube link in the sidebar and click **Load video** to start.")
    st.stop()

st.session_state.setdefault("messages", [])


def render_sources(sources):
    with st.expander("Sources from the video"):
        for s in sources:
            st.markdown(f"**[{s['timestamp']}]({s['url']})**: {s['text'][:300]}...")


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            render_sources(m["sources"])

if question := st.chat_input("Ask something about the video..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            res = qa.ask(question)
        sources = [s.__dict__ for s in res.sources]
        st.markdown(res.answer)
        render_sources(sources)
    st.session_state.messages.append({"role": "assistant", "content": res.answer, "sources": sources})
