"""Per-request extended-thinking selection for the Claude Code agent loop.

The iOS client picks a thinking level per turn via `forwardedProps.llm.thinking`.
`resolve_thinking` maps that string to `ClaudeAgentOptions.thinking` config;
missing/unknown → None (option left unset, CLI default applies).
"""

from __future__ import annotations

import uuid

from ag_ui.core.types import RunAgentInput, UserMessage

from pupa_backend.harnesses.claude.thinking import (
    LOOP_THINKING_LEVELS,
    loop_thinking_menu,
    resolve_thinking,
)


def _mk_input(forwarded_props: dict | None) -> RunAgentInput:
    return RunAgentInput(
        thread_id=str(uuid.uuid4()),
        run_id=str(uuid.uuid4()),
        state={},
        messages=[UserMessage(id="u1", content="hi")],
        tools=[],
        context=[],
        forwarded_props=forwarded_props or {},
    )


def _mk(level: str | None) -> RunAgentInput:
    llm: dict = {"provider": "claude_code", "model": "sonnet"}
    if level is not None:
        llm["thinking"] = level
    return _mk_input({"llm": llm})


# --------------------------------------------------------------------------- #
# resolve_thinking
# --------------------------------------------------------------------------- #

def test_off_disables_thinking() -> None:
    assert resolve_thinking(_mk("off")) == {"thinking": {"type": "disabled"}}


def test_auto_is_adaptive() -> None:
    assert resolve_thinking(_mk("auto")) == {"thinking": {"type": "adaptive"}}


def test_levels_map_to_ascending_budgets() -> None:
    low = resolve_thinking(_mk("low"))["thinking"]
    med = resolve_thinking(_mk("medium"))["thinking"]
    high = resolve_thinking(_mk("high"))["thinking"]
    assert low["type"] == med["type"] == high["type"] == "enabled"
    assert low["budget_tokens"] < med["budget_tokens"] < high["budget_tokens"]


def test_missing_thinking_returns_none() -> None:
    """Back-compat: no `thinking` key → option left unset."""
    assert resolve_thinking(_mk(None)) is None
    assert resolve_thinking(_mk_input({})) is None
    assert resolve_thinking(_mk_input(None)) is None


def test_unknown_level_returns_none() -> None:
    """An unrecognised value (e.g. stale/imported bundle) falls back, not error."""
    assert resolve_thinking(_mk("ultra")) is None


def test_empty_string_returns_none() -> None:
    assert resolve_thinking(_mk("")) is None


def test_no_llm_block_returns_none() -> None:
    assert resolve_thinking(_mk_input({"llm": {"model": "sonnet"}})) is None


# --------------------------------------------------------------------------- #
# menu
# --------------------------------------------------------------------------- #

def test_thinking_menu_shape() -> None:
    menu = loop_thinking_menu()
    assert {row["level"] for row in menu} == {lvl for lvl, _ in LOOP_THINKING_LEVELS}
    assert all({"level", "label"} <= row.keys() for row in menu)
    # `auto` leads — it is the default UX.
    assert menu[0]["level"] == "auto"


# --------------------------------------------------------------------------- #
# Harness discovery — claude reports levels, deepagents reports none
# --------------------------------------------------------------------------- #

def test_claude_harness_reports_thinking_menu() -> None:
    from pupa_backend.harnesses import ClaudeCodeHarness

    out = ClaudeCodeHarness().thinking()
    assert {row["level"] for row in out} == {lvl for lvl, _ in LOOP_THINKING_LEVELS}


def test_deepagents_harness_has_no_thinking() -> None:
    """The route reads `thinking` via getattr; deepagents omits the method."""
    from pupa_backend.harnesses.langgraph.harness import DeepAgentsHarness

    assert getattr(DeepAgentsHarness(), "thinking", None) is None
