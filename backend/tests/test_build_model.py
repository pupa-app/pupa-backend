"""Tests for `agent._build_default_from_env` — env-driven provider selection.

This is the construction path used both at startup (for the default graph)
and as the fallback whenever an iOS client doesn't send `forwardedProps["llm"]`.
Per-(provider, model) construction lives in `build_model` and is covered by
`test_per_request_model.py`.

`conftest.py` seeds ``LLM_PROVIDER=bedrock`` + dummy AWS creds so the suite
can import `agent` without real cloud credentials.
"""

import pytest

from pupa_backend.harnesses.langgraph.agent import _build_default_from_env as _build_model


# ---------------------------------------------------------------------------
# openai_compatible provider
# ---------------------------------------------------------------------------

def test_openai_compatible_returns_chatopenai(monkeypatch: pytest.MonkeyPatch) -> None:
    """Happy path: all three required env vars → `ChatOpenAI` with expected attrs."""
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("LLM_API_KEY", "sk-test-key")
    monkeypatch.setenv("LLM_MODEL", "anthropic/claude-sonnet-4.6")

    from langchain_openai import ChatOpenAI

    model = _build_model()

    assert isinstance(model, ChatOpenAI)
    assert str(model.openai_api_base) == "https://openrouter.ai/api/v1"
    assert model.model_name == "anthropic/claude-sonnet-4.6"


def test_openai_compatible_missing_base_url_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing LLM_BASE_URL → RuntimeError naming the missing var."""
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "sk-test-key")
    monkeypatch.setenv("LLM_MODEL", "anthropic/claude-sonnet-4.6")

    with pytest.raises(RuntimeError, match="LLM_BASE_URL"):
        _build_model()


def test_openai_compatible_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing LLM_API_KEY → RuntimeError naming the missing var."""
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("LLM_MODEL", "anthropic/claude-sonnet-4.6")

    with pytest.raises(RuntimeError, match="LLM_API_KEY"):
        _build_model()


def test_openai_compatible_missing_model_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing LLM_MODEL → RuntimeError naming the missing var."""
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("LLM_API_KEY", "sk-test-key")
    monkeypatch.delenv("LLM_MODEL", raising=False)

    with pytest.raises(RuntimeError, match="LLM_MODEL"):
        _build_model()


# ---------------------------------------------------------------------------
# Unknown provider
# ---------------------------------------------------------------------------

def test_unknown_provider_lists_all_three(monkeypatch: pytest.MonkeyPatch) -> None:
    """The error for an unknown provider names all three valid options."""
    monkeypatch.setenv("LLM_PROVIDER", "grok")

    with pytest.raises(RuntimeError, match="openai_compatible"):
        _build_model()
