from typing import Any

import streamlit as st
from backend.core import run_llm


def _format_sources(context_docs: list[Any]) -> list[str]:
    return [
        str(meta.get("source") or "Unknown")
        for doc in (context_docs or [])
        if (meta := (getattr(doc, "metadata", None) or {})) is not None
    ]


st.set_page_config(page_title="LangChain Documentation Helper", layout="centered")
st.title("LangChain Documentation Helper")

with st.sidebar:
    st.subheader("Session")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.pop("messages", None)
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me anything about LangChain docs. I’ll retrieve relevant context and cite sources.",
            "sources": [],
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- {s}")

if prompt := st.chat_input("Ask a question about LangChain..."):
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving docs and generating answer…"):
                result: dict[str, Any] = run_llm(prompt)
                answer = (
                    str(result.get("answer", "")).strip() or "(No answer returned.)"
                )
                sources = _format_sources(result.get("context", []))
                st.markdown(answer)

                if sources:
                    with st.expander("Sources"):
                        for s in sources:
                            st.markdown(f"- {s}")

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )

        except Exception as e:
            st.error("Failed to generate a response.")
            st.exception(e)
