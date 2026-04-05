"""Multi-provider LLM factory for LangChain course projects.

Supports OpenAI, MiniMax, Anthropic, Google Gemini, and Ollama (local).
Drop this module into any project branch and call ``get_chat_model()`` to get
a configured LangChain chat model based on environment variables.

Usage
-----
Set one of the following environment variables to choose a provider:

    LLM_PROVIDER=openai        # default
    LLM_PROVIDER=minimax
    LLM_PROVIDER=anthropic
    LLM_PROVIDER=google
    LLM_PROVIDER=ollama

Each provider reads its own API key from the standard env var
(``OPENAI_API_KEY``, ``MINIMAX_API_KEY``, ``ANTHROPIC_API_KEY``, etc.).

Example::

    from utils.llm_provider import get_chat_model

    llm = get_chat_model()  # auto-detects provider from env
    response = llm.invoke("Hello!")
"""

from __future__ import annotations

import os
from typing import Any

from langchain_openai import ChatOpenAI

# Provider -> (default model, default temperature)
_PROVIDER_DEFAULTS: dict[str, dict[str, Any]] = {
    "openai": {"model": "gpt-4o", "temperature": 0},
    "minimax": {"model": "MiniMax-M2.7", "temperature": 0.01},
    "anthropic": {"model": "claude-sonnet-4-20250514", "temperature": 0},
    "google": {"model": "gemini-2.0-flash", "temperature": 0},
    "ollama": {"model": "llama3.2", "temperature": 0},
}

SUPPORTED_PROVIDERS = list(_PROVIDER_DEFAULTS.keys())


def _detect_provider() -> str:
    """Auto-detect provider from available API keys."""
    explicit = os.getenv("LLM_PROVIDER", "").strip().lower()
    if explicit and explicit in _PROVIDER_DEFAULTS:
        return explicit

    if os.getenv("MINIMAX_API_KEY"):
        return "minimax"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("GOOGLE_API_KEY"):
        return "google"
    return "ollama"


def _clamp_temperature(temperature: float, provider: str) -> float:
    """Clamp temperature to provider-specific valid range."""
    if provider == "minimax":
        return max(0.01, min(temperature, 1.0)) if temperature > 0 else 0.01
    return temperature


def get_chat_model(
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    **kwargs: Any,
) -> Any:
    """Create a LangChain chat model for the given provider.

    Parameters
    ----------
    provider : str, optional
        One of ``SUPPORTED_PROVIDERS``. If *None*, auto-detected from env.
    model : str, optional
        Model name override.  Defaults to provider-specific default.
    temperature : float, optional
        Sampling temperature.  Defaults to provider-specific default.
    **kwargs
        Extra keyword arguments passed to the underlying chat model class.

    Returns
    -------
    BaseChatModel
        A LangChain chat model instance ready to use.
    """
    provider = (provider or _detect_provider()).lower()
    if provider not in _PROVIDER_DEFAULTS:
        raise ValueError(
            f"Unsupported provider: {provider!r}. "
            f"Choose from: {SUPPORTED_PROVIDERS}"
        )

    defaults = _PROVIDER_DEFAULTS[provider]
    model = model or defaults["model"]
    temperature = temperature if temperature is not None else defaults["temperature"]
    temperature = _clamp_temperature(temperature, provider)

    if provider == "openai":
        return ChatOpenAI(model=model, temperature=temperature, **kwargs)

    if provider == "minimax":
        api_key = os.getenv("MINIMAX_API_KEY")
        if not api_key:
            raise ValueError(
                "MINIMAX_API_KEY environment variable is required for MiniMax provider"
            )
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            openai_api_key=api_key,
            openai_api_base="https://api.minimax.io/v1",
            **kwargs,
        )

    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError(
                "Install langchain-anthropic: pip install langchain-anthropic"
            )
        return ChatAnthropic(model=model, temperature=temperature, **kwargs)

    if provider == "google":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError(
                "Install langchain-google-genai: pip install langchain-google-genai"
            )
        return ChatGoogleGenerativeAI(model=model, temperature=temperature, **kwargs)

    # ollama
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        raise ImportError("Install langchain-ollama: pip install langchain-ollama")
    return ChatOllama(model=model, temperature=temperature, **kwargs)
