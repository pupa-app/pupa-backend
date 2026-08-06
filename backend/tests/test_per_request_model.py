"""Tests for per-request LLM selection via ``forwardedProps["llm"]``.

Covers the registry + cache layer in ``agent.py`` (``build_model``, ``get_model``,
``get_graph``) and the ``_resolve_per_request_graph`` helper in
``langgraph_harness.py`` that the FastAPI handler calls to translate the iOS
payload into a swap on ``self.graph``.

Backwards-compat is the load-bearing guarantee: an iOS client that sends no
``llm`` block (or sends an empty one) MUST hit the env-driven default graph,
the same as before per-request selection was introduced.
"""


import pytest

from pupa_backend.harnesses.langgraph.agent import (
    MODEL_REGISTRY,
    OPENROUTER_BASE_URL,
    PROVIDER_ANTHROPIC,
    PROVIDER_BEDROCK,
    PROVIDER_OPENROUTER,
    MissingCredentialsError,
    UnknownModelError,
    _MODEL_CACHE,
    build_model,
    get_model,
)
from pupa_backend.harnesses.langgraph.harness import LLMParams, _PerRequestModelError, _resolve_per_request_graph


# ---------------------------------------------------------------------------
# Registry shape
# ---------------------------------------------------------------------------

def test_registry_seeds_at_least_one_bedrock_and_one_anthropic_entry() -> None:
    """Sanity: the iOS catalog and the backend registry must overlap on at
    least one model per provider, else the iOS picker can't pick anything."""
    bedrock_keys = [m for (p, m) in MODEL_REGISTRY if p == PROVIDER_BEDROCK]
    anthropic_keys = [m for (p, m) in MODEL_REGISTRY if p == PROVIDER_ANTHROPIC]
    assert bedrock_keys, "MODEL_REGISTRY has no bedrock entries"
    assert anthropic_keys, "MODEL_REGISTRY has no anthropic entries"


# ---------------------------------------------------------------------------
# build_model — explicit (provider, model)
# ---------------------------------------------------------------------------

def test_build_model_unknown_pair_lists_known_combos() -> None:
    """Unknown (provider, model) raises with the full known list so the iOS
    user sees what's actually available."""
    with pytest.raises(UnknownModelError, match="claude-sonnet-4-6"):
        build_model(provider=PROVIDER_BEDROCK, model_id="does-not-exist")


def test_build_model_partial_input_is_rejected() -> None:
    """`provider` without `model` (or vice versa) is a client bug — fail loud
    rather than silently fall back to the default."""
    with pytest.raises(UnknownModelError, match="both be set or both omitted"):
        build_model(provider=PROVIDER_BEDROCK, model_id=None)
    with pytest.raises(UnknownModelError, match="both be set or both omitted"):
        build_model(provider=None, model_id="claude-sonnet-4-6")


def test_registry_seeds_openrouter_entries() -> None:
    """OpenRouter models must be pickable from the iOS catalog too."""
    openrouter_keys = [m for (p, m) in MODEL_REGISTRY if p == PROVIDER_OPENROUTER]
    assert openrouter_keys, "MODEL_REGISTRY has no openrouter entries"


def test_build_model_openrouter_without_creds_raises_missing_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Picking an OpenRouter model without OPENROUTER_API_KEY surfaces a clear
    MissingCredentialsError, same as the other providers."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(MissingCredentialsError, match="OPENROUTER_API_KEY"):
        build_model(provider=PROVIDER_OPENROUTER, model_id="glm-5.1")


def test_build_model_openrouter_builds_chatopenai_at_openrouter_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Happy path: key present → ChatOpenAI pointed at OpenRouter with the
    registry slug as the model name."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    from langchain_openai import ChatOpenAI

    model = build_model(provider=PROVIDER_OPENROUTER, model_id="glm-5.1")

    assert isinstance(model, ChatOpenAI)
    assert str(model.openai_api_base) == OPENROUTER_BASE_URL
    assert model.model_name == "z-ai/glm-5.1"


def test_build_model_anthropic_without_creds_raises_missing_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Picking an Anthropic-direct model on a Bedrock-only backend surfaces a
    clear MissingCredentialsError, which the AG-UI handler translates into a
    RunErrorEvent the iOS chat toasts."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(MissingCredentialsError, match="ANTHROPIC_API_KEY"):
        build_model(provider=PROVIDER_ANTHROPIC, model_id="claude-sonnet-4-6")


# ---------------------------------------------------------------------------
# get_model — cache
# ---------------------------------------------------------------------------

def test_get_model_caches_per_provider_model_pair() -> None:
    """Same (provider, model) → same instance; different pair → different
    instance. The cache keeps construction cost off the request path."""
    _MODEL_CACHE.clear()
    a = get_model(provider=PROVIDER_BEDROCK, model_id="claude-sonnet-4-6")
    b = get_model(provider=PROVIDER_BEDROCK, model_id="claude-sonnet-4-6")
    assert a is b

    default1 = get_model(None, None)
    default2 = get_model()
    assert default1 is default2
    # Bedrock-explicit and the env-default may both build a ChatBedrockConverse
    # but they're cached under different keys, so they're different instances.
    assert default1 is not a


# ---------------------------------------------------------------------------
# _resolve_per_request_graph — the function `_handle_stream_events` calls
# ---------------------------------------------------------------------------

class _StubGraph:
    """Marker used to assert which graph the resolver returned without
    instantiating a real LangGraph."""


@pytest.fixture
def stub_get_graph(monkeypatch: pytest.MonkeyPatch):
    """Replace `agent.get_graph` so we can assert the (provider, model) it was
    called with, without paying the cost of building a real graph."""
    calls: list[tuple[str | None, str | None]] = []
    sentinels: dict[tuple[str | None, str | None], _StubGraph] = {}

    def fake_get_graph(provider=None, model_id=None):
        calls.append((provider, model_id))
        sentinels.setdefault((provider, model_id), _StubGraph())
        return sentinels[(provider, model_id)]

    monkeypatch.setattr("pupa_backend.harnesses.langgraph.agent.get_graph", fake_get_graph)
    return calls, sentinels


def test_no_llm_block_returns_none_so_default_graph_is_used(stub_get_graph) -> None:
    """An empty or missing `llm` block must NOT call `get_graph` — the agent's
    pre-built default graph stays in place. This is the back-compat path."""
    calls, _ = stub_get_graph
    assert _resolve_per_request_graph({}) is None
    assert _resolve_per_request_graph(None) is None
    assert _resolve_per_request_graph({"command": {"resume": {}}}) is None
    assert calls == []


def test_well_formed_llm_block_resolves_to_matching_graph(stub_get_graph) -> None:
    calls, sentinels = stub_get_graph
    out = _resolve_per_request_graph({"llm": {"provider": "bedrock", "model": "claude-sonnet-4-6"}})
    assert out is sentinels[("bedrock", "claude-sonnet-4-6")]
    assert calls == [("bedrock", "claude-sonnet-4-6")]


def test_partial_llm_block_raises_per_request_model_error(stub_get_graph) -> None:
    """Either-only is a client bug — surface as a RunErrorEvent so the user
    sees the issue rather than silently getting the wrong model."""
    calls, _ = stub_get_graph
    with pytest.raises(_PerRequestModelError, match="both 'provider' and 'model'"):
        _resolve_per_request_graph({"llm": {"provider": "bedrock"}})
    assert calls == []


def test_malformed_llm_block_falls_back_to_default(stub_get_graph) -> None:
    """An old client that ships an unknown field shouldn't kill the request —
    warn and skip rather than fail the run."""
    calls, _ = stub_get_graph
    # `extra="forbid"` on LLMParams rejects unknown keys; the resolver swallows
    # the validation error and falls back to the default graph.
    assert _resolve_per_request_graph({"llm": {"unexpected_field": 1}}) is None
    assert calls == []


def test_unknown_model_surfaces_as_per_request_model_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A typoed model id from iOS becomes a RunErrorEvent, not a 500."""
    def fake_get_graph(provider=None, model_id=None):
        raise UnknownModelError("Unknown (provider, model) = ('bedrock', 'oops').")
    monkeypatch.setattr("pupa_backend.harnesses.langgraph.agent.get_graph", fake_get_graph)
    with pytest.raises(_PerRequestModelError, match="Unknown"):
        _resolve_per_request_graph({"llm": {"provider": "bedrock", "model": "oops"}})


def test_missing_credentials_surfaces_as_per_request_model_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Picking a provider the backend has no creds for becomes a RunErrorEvent."""
    def fake_get_graph(provider=None, model_id=None):
        raise MissingCredentialsError("ANTHROPIC_API_KEY is not set.")
    monkeypatch.setattr("pupa_backend.harnesses.langgraph.agent.get_graph", fake_get_graph)
    with pytest.raises(_PerRequestModelError, match="ANTHROPIC_API_KEY"):
        _resolve_per_request_graph({"llm": {"provider": "anthropic", "model": "claude-sonnet-4-6"}})


# ---------------------------------------------------------------------------
# LLMParams shape
# ---------------------------------------------------------------------------

def test_llmparams_rejects_extra_fields() -> None:
    """`extra="forbid"` keeps the wire schema honest — typos in the iOS payload
    don't silently parse and apply the wrong field."""
    with pytest.raises(Exception):
        LLMParams(provider="bedrock", model="claude-sonnet-4-6", temperature=0.7)
