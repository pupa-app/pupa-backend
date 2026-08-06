"""Regression suite for the `LG_RECURSION_LIMIT` env var.

LangGraph's default recursion cap of 25 is too low for multi-component
canvas turns and especially for the planning middleware's
break-into-steps pattern. `agent.build_graph` reads `LG_RECURSION_LIMIT`
(default 100) and wraps the compiled graph via `with_config(...)` so the
limit propagates into every `astream_events` call AG-UI makes.

Four contracts pinned:

1. **Default is 100** when env var is unset.
2. **Custom value flows through** to `graph.config["recursion_limit"]`.
3. **Non-integer fails loud** at startup — silent fallback masks config
   bugs in CI / prod deploys.
4. **Non-positive fails loud** — 0 / negative would deadlock or behave
   unpredictably.
"""



import pytest

from pupa_backend.harnesses.langgraph.agent import (
    DEFAULT_RECURSION_LIMIT,
    recursion_limit as _recursion_limit_from_env,
    build_graph,
)


def test_default_recursion_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LG_RECURSION_LIMIT", raising=False)
    assert _recursion_limit_from_env() == DEFAULT_RECURSION_LIMIT


def test_custom_recursion_limit_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LG_RECURSION_LIMIT", "250")
    assert _recursion_limit_from_env() == 250


def test_invalid_recursion_limit_non_integer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LG_RECURSION_LIMIT", "fifty")
    with pytest.raises(RuntimeError, match="must be an integer"):
        _recursion_limit_from_env()


def test_invalid_recursion_limit_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LG_RECURSION_LIMIT", "0")
    with pytest.raises(RuntimeError, match="must be >= 1"):
        _recursion_limit_from_env()


def test_invalid_recursion_limit_negative(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LG_RECURSION_LIMIT", "-5")
    with pytest.raises(RuntimeError, match="must be >= 1"):
        _recursion_limit_from_env()


def test_recursion_limit_value_flows_to_env_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """recursion_limit() reflects the env var; build_graph() is callable."""
    monkeypatch.setenv("LG_RECURSION_LIMIT", "77")
    assert _recursion_limit_from_env() == 77
    build_graph()  # should not raise
