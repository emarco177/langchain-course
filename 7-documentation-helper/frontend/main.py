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