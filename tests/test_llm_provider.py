"""Unit tests for the multi-provider LLM factory."""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from utils.llm_provider import (
    SUPPORTED_PROVIDERS,
    _clamp_temperature,
    _detect_provider,
    get_chat_model,
)


# ── Provider detection ──────────────────────────────────────────────


class TestDetectProvider:
    """Tests for auto-detection logic."""

    def test_explicit_env_var_takes_priority(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "minimax", "OPENAI_API_KEY": "k"}):
            assert _detect_provider() == "minimax"

    def test_explicit_env_var_case_insensitive(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "MiniMax"}, clear=False):
            assert _detect_provider() == "minimax"

    def test_minimax_key_detected(self):
        env = {"MINIMAX_API_KEY": "mm-key"}
        with patch.dict(os.environ, env, clear=True):
            assert _detect_provider() == "minimax"

    def test_openai_key_detected(self):
        env = {"OPENAI_API_KEY": "sk-key"}
        with patch.dict(os.environ, env, clear=True):
            assert _detect_provider() == "openai"

    def test_anthropic_key_detected(self):
        env = {"ANTHROPIC_API_KEY": "ant-key"}
        with patch.dict(os.environ, env, clear=True):
            assert _detect_provider() == "anthropic"

    def test_google_key_detected(self):
        env = {"GOOGLE_API_KEY": "g-key"}
        with patch.dict(os.environ, env, clear=True):
            assert _detect_provider() == "google"

    def test_falls_back_to_ollama(self):
        with patch.dict(os.environ, {}, clear=True):
            assert _detect_provider() == "ollama"

    def test_minimax_has_higher_priority_than_openai(self):
        env = {"MINIMAX_API_KEY": "mm", "OPENAI_API_KEY": "sk"}
        with patch.dict(os.environ, env, clear=True):
            assert _detect_provider() == "minimax"

    def test_unknown_explicit_provider_falls_through(self):
        env = {"LLM_PROVIDER": "unknown_provider", "OPENAI_API_KEY": "sk"}
        with patch.dict(os.environ, env, clear=True):
            assert _detect_provider() == "openai"


# ── Temperature clamping ────────────────────────────────────────────


class TestClampTemperature:
    """Tests for temperature clamping per provider."""

    def test_minimax_zero_becomes_001(self):
        assert _clamp_temperature(0, "minimax") == 0.01

    def test_minimax_normal_value_unchanged(self):
        assert _clamp_temperature(0.7, "minimax") == 0.7

    def test_minimax_above_one_clamped(self):
        assert _clamp_temperature(1.5, "minimax") == 1.0

    def test_minimax_small_positive_stays(self):
        assert _clamp_temperature(0.01, "minimax") == 0.01

    def test_openai_zero_unchanged(self):
        assert _clamp_temperature(0, "openai") == 0

    def test_openai_value_unchanged(self):
        assert _clamp_temperature(0.8, "openai") == 0.8


# ── get_chat_model ──────────────────────────────────────────────────


class TestGetChatModel:
    """Tests for the model factory function."""

    @patch("utils.llm_provider.ChatOpenAI")
    def test_openai_provider(self, mock_cls):
        mock_cls.return_value = MagicMock()
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            model = get_chat_model(provider="openai")
        mock_cls.assert_called_once_with(model="gpt-4o", temperature=0)
        assert model is mock_cls.return_value

    @patch("utils.llm_provider.ChatOpenAI")
    def test_minimax_provider(self, mock_cls):
        mock_cls.return_value = MagicMock()
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "mm-test"}):
            model = get_chat_model(provider="minimax")
        mock_cls.assert_called_once_with(
            model="MiniMax-M2.7",
            temperature=0.01,
            openai_api_key="mm-test",
            openai_api_base="https://api.minimax.io/v1",
        )
        assert model is mock_cls.return_value

    @patch("utils.llm_provider.ChatOpenAI")
    def test_minimax_custom_model(self, mock_cls):
        mock_cls.return_value = MagicMock()
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "mm-test"}):
            get_chat_model(provider="minimax", model="MiniMax-M2.5-highspeed")
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["model"] == "MiniMax-M2.5-highspeed"

    @patch("utils.llm_provider.ChatOpenAI")
    def test_minimax_temp_clamped(self, mock_cls):
        mock_cls.return_value = MagicMock()
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "mm-test"}):
            get_chat_model(provider="minimax", temperature=0)
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["temperature"] == 0.01

    def test_minimax_requires_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="MINIMAX_API_KEY"):
                get_chat_model(provider="minimax")

    def test_unsupported_provider_raises(self):
        with pytest.raises(ValueError, match="Unsupported provider"):
            get_chat_model(provider="nonexistent")

    @patch("utils.llm_provider.ChatOpenAI")
    def test_custom_temperature(self, mock_cls):
        mock_cls.return_value = MagicMock()
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            get_chat_model(provider="openai", temperature=0.5)
        assert mock_cls.call_args[1]["temperature"] == 0.5

    @patch("utils.llm_provider.ChatOpenAI")
    def test_extra_kwargs_forwarded(self, mock_cls):
        mock_cls.return_value = MagicMock()
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            get_chat_model(provider="openai", max_tokens=100)
        assert mock_cls.call_args[1]["max_tokens"] == 100

    @patch("utils.llm_provider.ChatOpenAI")
    def test_auto_detect_minimax(self, mock_cls):
        mock_cls.return_value = MagicMock()
        env = {"MINIMAX_API_KEY": "mm-test"}
        with patch.dict(os.environ, env, clear=True):
            get_chat_model()
        assert mock_cls.call_args[1]["openai_api_base"] == "https://api.minimax.io/v1"


# ── Supported providers list ────────────────────────────────────────


class TestSupportedProviders:
    """Tests for the provider registry."""

    def test_all_expected_providers_present(self):
        expected = {"openai", "minimax", "anthropic", "google", "ollama"}
        assert set(SUPPORTED_PROVIDERS) == expected

    def test_minimax_in_supported_providers(self):
        assert "minimax" in SUPPORTED_PROVIDERS

    def test_supported_providers_is_list(self):
        assert isinstance(SUPPORTED_PROVIDERS, list)


# ── Anthropic / Google / Ollama import fallback ─────────────────────


class TestOptionalImports:
    """Tests that missing optional packages raise helpful ImportError."""

    def test_anthropic_import_error(self):
        with patch.dict("sys.modules", {"langchain_anthropic": None}):
            with pytest.raises(ImportError, match="langchain-anthropic"):
                get_chat_model(provider="anthropic")

    def test_google_import_error(self):
        with patch.dict("sys.modules", {"langchain_google_genai": None}):
            with pytest.raises(ImportError, match="langchain-google-genai"):
                get_chat_model(provider="google")

    def test_ollama_import_error(self):
        with patch.dict("sys.modules", {"langchain_ollama": None}):
            with pytest.raises(ImportError, match="langchain-ollama"):
                get_chat_model(provider="ollama")
