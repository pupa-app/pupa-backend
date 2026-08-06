"""Per-request model selection for the Claude Code agent loop.

The loop runs the `claude` CLI under a Pro/Max subscription, so it can only run
Claude models. The iOS client picks one per turn via `forwardedProps.llm.model`
(stable claude-code aliases: `opus`/`sonnet`/`haiku`/`fable`); a non-Claude pick
falls back to the default Claude model (so imported `.pupa` bundles that remember
a model this backend can't run still work). Backwards-compat: no `llm` block → the
`CLAUDE_CODE_MODEL` / Opus 4.8 default, exactly as before.
"""

from __future__ import annotations

import uuid

import pytest
from ag_ui.core.types import RunAgentInput, UserMessage

from pupa_backend.harnesses.claude.endpoint import _resolve_loop_model
from pupa_backend.harnesses.claude.models import LOOP_MODEL_ALIASES, is_loop_model, loop_model_menu


def _mk_input(forwarded_props: dict | None) -> RunAgentInput:
    """Minimal AG-UI input — only `forwarded_props` matters for model resolution."""
    return RunAgentInput(
        thread_id=str(uuid.uuid4()),
        run_id=str(uuid.uuid4()),
        state={},
        messages=[UserMessage(id="u1", content="hi")],
        tools=[],
        context=[],
        forwarded_props=forwarded_props or {},
    )


# --------------------------------------------------------------------------- #
# _resolve_loop_model
# --------------------------------------------------------------------------- #

def test_per_request_alias_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    """A picked alias flows straight to `--model`; provider is ignored."""
    monkeypatch.delenv("CLAUDE_CODE_MODEL", raising=False)
    inp = _mk_input({"llm": {"provider": "claude_code", "model": "sonnet"}})
    assert _resolve_loop_model(inp) == "sonnet"


def test_full_claude_id_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pinned `claude-*` id (e.g. from a stale client) still passes through."""
    monkeypatch.delenv("CLAUDE_CODE_MODEL", raising=False)
    inp = _mk_input({"llm": {"provider": "anthropic", "model": "claude-opus-4-8"}})
    assert _resolve_loop_model(inp) == "claude-opus-4-8"


def test_no_llm_block_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Back-compat: no `llm` block and no env → Opus 4.8."""
    monkeypatch.delenv("CLAUDE_CODE_MODEL", raising=False)
    assert _resolve_loop_model(_mk_input({})) == "claude-opus-4-8"
    assert _resolve_loop_model(_mk_input(None)) == "claude-opus-4-8"


def test_env_default_used_when_no_per_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAUDE_CODE_MODEL", "claude-sonnet-4-6")
    assert _resolve_loop_model(_mk_input({})) == "claude-sonnet-4-6"


def test_per_request_wins_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Per-request selection overrides the `CLAUDE_CODE_MODEL` default."""
    monkeypatch.setenv("CLAUDE_CODE_MODEL", "claude-opus-4-8")
    inp = _mk_input({"llm": {"provider": "claude_code", "model": "haiku"}})
    assert _resolve_loop_model(inp) == "haiku"


def test_non_claude_pick_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """An OpenRouter slug (or anything non-Claude) — e.g. carried in from an
    imported `.pupa` bundle — must NOT break the turn. It falls back to the
    default Claude model instead of raising."""
    monkeypatch.delenv("CLAUDE_CODE_MODEL", raising=False)
    inp = _mk_input({"llm": {"provider": "openrouter", "model": "glm-5.1"}})
    assert _resolve_loop_model(inp) == "claude-opus-4-8"


def test_non_claude_pick_falls_back_to_env_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """The non-Claude fallback honours `CLAUDE_CODE_MODEL` when set."""
    monkeypatch.setenv("CLAUDE_CODE_MODEL", "claude-sonnet-4-6")
    inp = _mk_input({"llm": {"provider": "openrouter", "model": "glm-5.1"}})
    assert _resolve_loop_model(inp) == "claude-sonnet-4-6"


def test_empty_model_string_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty model value is treated as 'no selection', not an error."""
    monkeypatch.delenv("CLAUDE_CODE_MODEL", raising=False)
    inp = _mk_input({"llm": {"provider": "claude_code", "model": ""}})
    assert _resolve_loop_model(inp) == "claude-opus-4-8"


# --------------------------------------------------------------------------- #
# models.py helpers
# --------------------------------------------------------------------------- #

def test_is_loop_model_accepts_aliases_and_claude_ids() -> None:
    assert is_loop_model("opus")
    assert is_loop_model("fable")
    assert is_loop_model("claude-opus-4-8")
    assert not is_loop_model("glm-5.1")
    assert not is_loop_model("gpt-4o")


def test_loop_menu_shape_is_aliases_only() -> None:
    menu = loop_model_menu()
    aliases = {a for a, _ in LOOP_MODEL_ALIASES}
    assert {row["modelId"] for row in menu} == aliases
    # No pinned version ids leak into the menu — that's the whole point.
    assert all(not row["modelId"].startswith("claude-") for row in menu)
    assert all({"provider", "modelId", "label"} <= row.keys() for row in menu)


# --------------------------------------------------------------------------- #
# Harness model menus — each harness reports its own model list
# --------------------------------------------------------------------------- #

def test_claude_harness_reports_alias_menu() -> None:
    from pupa_backend.harnesses import ClaudeCodeHarness

    out = ClaudeCodeHarness().models()
    assert {row["modelId"] for row in out} == {a for a, _ in LOOP_MODEL_ALIASES}


def test_deepagents_harness_reports_full_registry() -> None:
    from pupa_backend.harnesses.langgraph.agent import MODEL_REGISTRY
    from pupa_backend.harnesses.langgraph.harness import DeepAgentsHarness

    out = DeepAgentsHarness().models()
    assert len(out) == len(MODEL_REGISTRY)
    # The OpenRouter/Bedrock models the Claude harness can't run are present here.
    assert any(row["modelId"] == "glm-5.1" for row in out)
