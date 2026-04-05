"""Integration tests for MiniMax provider.

These tests make real API calls and require ``MINIMAX_API_KEY`` to be set.
Run with::

    MINIMAX_API_KEY=<key> pytest tests/test_integration_minimax.py -v
"""

import os
import sys

import pytest

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from utils.llm_provider import get_chat_model

pytestmark = pytest.mark.skipif(
    not os.getenv("MINIMAX_API_KEY"),
    reason="MINIMAX_API_KEY not set",
)


class TestMiniMaxIntegration:
    """Live integration tests against MiniMax API."""

    def test_basic_invoke(self):
        llm = get_chat_model(provider="minimax")
        response = llm.invoke("Say 'hello' and nothing else.")
        assert response.content
        # M2.7 may include <think> tags; strip them before checking length
        import re
        clean = re.sub(r"<think>.*?</think>\s*", "", response.content, flags=re.DOTALL)
        assert "hello" in clean.lower()

    def test_m25_highspeed_model(self):
        llm = get_chat_model(provider="minimax", model="MiniMax-M2.5-highspeed")
        response = llm.invoke("What is 2+2? Reply with just the number.")
        assert "4" in response.content

    def test_streaming(self):
        llm = get_chat_model(provider="minimax")
        chunks = list(llm.stream("Count from 1 to 3."))
        assert len(chunks) > 0
        full_text = "".join(c.content for c in chunks)
        assert "1" in full_text
