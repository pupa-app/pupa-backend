"""Token / cache usage log lines for the Claude loop.

`_pump` logs usage twice per turn — once per `AssistantMessage` (per-API-call,
the only place the cache read/write split shows) and once on the final
`ResultMessage` (turn totals + cost). Both lines are yellow.
"""

from __future__ import annotations

from pupa_backend.harnesses.claude import usage


class _Result:
    """Minimal stand-in for `ResultMessage` (only the fields `result_line` reads)."""

    def __init__(self, **kw) -> None:
        self.usage = kw.get("usage")
        self.total_cost_usd = kw.get("total_cost_usd")
        self.num_turns = kw.get("num_turns")
        self.duration_api_ms = kw.get("duration_api_ms")
        self.model_usage = kw.get("model_usage")


USAGE = {
    "input_tokens": 12,
    "output_tokens": 345,
    "cache_read_input_tokens": 18_000,
    "cache_creation_input_tokens": 2_000,
}


def test_format_usage_reports_cache_split_and_hit_rate() -> None:
    line = usage.format_usage(USAGE)
    assert "in=12" in line
    assert "out=345" in line
    assert "cache_read=18,000" in line
    assert "cache_write=2,000" in line
    # 18000 / (12 + 18000 + 2000) = 89%
    assert "(cache hit 89%)" in line


def test_format_usage_accepts_camel_case_model_usage_keys() -> None:
    line = usage.format_usage(
        {"inputTokens": 5, "outputTokens": 6, "cacheReadInputTokens": 7,
         "cacheCreationInputTokens": 8}
    )
    assert "in=5 out=6 cache_read=7 cache_write=8" in line


def test_format_usage_is_none_when_nothing_to_report() -> None:
    assert usage.format_usage(None) is None
    assert usage.format_usage({}) is None
    assert usage.format_usage({"input_tokens": 0, "output_tokens": 0}) is None


def test_message_line_is_yellow_and_tagged_with_thread() -> None:
    line = usage.message_line(USAGE, "thread-7")
    assert line.startswith("\033[33m")
    assert line.endswith("\033[0m")
    assert "claude_code tokens:" in line
    assert "(thread=thread-7)" in line


def test_message_line_skipped_when_usage_absent() -> None:
    assert usage.message_line(None, "t") is None


def test_result_line_adds_cost_turns_duration_and_per_model() -> None:
    line = usage.result_line(
        _Result(
            usage=USAGE,
            total_cost_usd=0.01234,
            num_turns=3,
            duration_api_ms=4210,
            model_usage={"claude-opus-4-8": {"inputTokens": 12, "outputTokens": 345}},
        ),
        "thread-7",
    )
    assert line.startswith("\033[33m") and line.endswith("\033[0m")
    assert "claude_code turn totals:" in line
    assert "cost=$0.0123" in line
    assert "turns=3" in line
    assert "api=4.2s" in line
    assert "claude-opus-4-8[in=12 out=345" in line


def test_result_line_survives_missing_usage() -> None:
    line = usage.result_line(_Result(num_turns=1), "t")
    assert "no token usage reported" in line
    assert "turns=1" in line


# --------------------------------------------------------------------------- #
# The pump actually emits both lines
# --------------------------------------------------------------------------- #

import logging

from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock

from pupa_backend.harnesses.claude import endpoint as cl_endpoint
from pupa_backend.harnesses.claude import registry


async def test_pump_logs_per_call_and_turn_totals(caplog) -> None:
    class _Client:
        async def receive_messages(self):
            yield AssistantMessage(
                content=[TextBlock(text="hi")],
                model="fake",
                message_id="m1",
                session_id="sdk-1",
                usage=USAGE,
            )
            yield ResultMessage(
                subtype="success",
                duration_ms=5000,
                duration_api_ms=4210,
                is_error=False,
                num_turns=2,
                session_id="sdk-1",
                total_cost_usd=0.01234,
                usage=USAGE,
            )

    session = registry.LiveSession(thread_id="t-usage")
    session.client = _Client()
    session.current_run_id = "run-1"

    with caplog.at_level(logging.INFO, logger="uvicorn.error"):
        await cl_endpoint._pump(session)

    logged = "\n".join(caplog.messages)
    assert "claude_code tokens:" in logged
    assert "claude_code turn totals:" in logged
    assert "cache_read=18,000" in logged
    assert "cost=$0.0123" in logged


async def test_pump_logs_one_token_line_per_message_not_per_block(caplog) -> None:
    """The SDK yields one `AssistantMessage` per content block, each carrying the
    same message-level usage — the pump must not print the identical line twice."""

    def _block(text):
        return AssistantMessage(
            content=[TextBlock(text=text)],
            model="fake",
            message_id="m1",  # same message, different blocks
            session_id="sdk-1",
            usage=USAGE,
        )

    class _Client:
        async def receive_messages(self):
            yield _block("part one")
            yield _block("part two")
            yield _block("part three")
            yield ResultMessage(
                subtype="success", duration_ms=1, duration_api_ms=1, is_error=False,
                num_turns=1, session_id="sdk-1", usage=USAGE,
            )

    session = registry.LiveSession(thread_id="t-dedupe")
    session.client = _Client()
    session.current_run_id = "run-1"

    with caplog.at_level(logging.INFO, logger="uvicorn.error"):
        await cl_endpoint._pump(session)

    assert sum("claude_code tokens:" in m for m in caplog.messages) == 1


# --------------------------------------------------------------------------- #
# Prompt-cache prefix diagnosis
# --------------------------------------------------------------------------- #

def _fp(**over):
    """A fingerprint of a plausible prefix; `over` perturbs one input."""
    base = dict(
        model="claude-opus-4-8",
        base_system="you are pupa",
        system="you are pupa\n\nambient: canvas empty",
        tool_specs=[("mcp__pupa_frontend__a", "does a", {"type": "object"})],
        permission_mode="default",
        thinking={},
        skills=None,
        cwd=None,
        context_pairs=[],
    )
    base.update(over)
    return usage.fingerprint(**base)


def test_unchanged_prefix_predicts_a_cache_read() -> None:
    usage.reset_fingerprints()
    first = usage.cache_line("t", _fp(), tool_count=1, system_chars=40)
    assert "first turn on this thread" in first
    assert "full cache write expected" in first

    second = usage.cache_line("t", _fp(), tool_count=1, system_chars=40)
    assert "prefix unchanged" in second
    assert "cache read expected" in second


def test_volatile_ambient_context_is_attributed_to_system_not_base() -> None:
    """The ambient block is appended to the system prompt every turn; when it
    moves, `system` changes while `base_system` holds — that distinction is the
    whole point of hashing both."""
    usage.reset_fingerprints()
    usage.cache_line("t", _fp(), tool_count=1, system_chars=40)
    line = usage.cache_line(
        "t", _fp(system="you are pupa\n\nambient: canvas has 3 items"),
        tool_count=1, system_chars=52,
    )
    assert "prefix changed [system]" in line
    assert "base_system" not in line
    assert "cache write expected" in line


def test_a_widened_tool_set_changes_set_order_and_schemas() -> None:
    usage.reset_fingerprints()
    usage.cache_line("t", _fp(), tool_count=1, system_chars=40)
    line = usage.cache_line(
        "t",
        _fp(tool_specs=[
            ("mcp__pupa_frontend__a", "does a", {"type": "object"}),
            ("mcp__pupa_frontend__b", "does b", {"type": "object"}),
        ]),
        tool_count=2, system_chars=40,
    )
    assert "tool_set" in line and "tool_order" in line and "tool_schemas" in line


def test_reordered_tools_break_the_prefix_even_with_the_same_set() -> None:
    """Same tools in a different order is still a different prompt — this mirrors
    the CLI's own "tool prompt/schema changed, same tool set" miss reason."""
    usage.reset_fingerprints()
    specs = [
        ("mcp__pupa_frontend__a", "does a", {"type": "object"}),
        ("mcp__pupa_frontend__b", "does b", {"type": "object"}),
    ]
    usage.cache_line("t", _fp(tool_specs=specs), tool_count=2, system_chars=40)
    line = usage.cache_line("t", _fp(tool_specs=list(reversed(specs))), tool_count=2, system_chars=40)
    assert "tool_order" in line
    assert "tool_set" not in line  # the set is identical; only the order moved


def test_model_and_permission_mode_flips_are_named() -> None:
    usage.reset_fingerprints()
    usage.cache_line("t", _fp(), tool_count=1, system_chars=40)
    line = usage.cache_line(
        "t", _fp(model="claude-sonnet-4-5", permission_mode="plan"),
        tool_count=1, system_chars=40,
    )
    assert "model" in line and "permission_mode" in line


def test_threads_are_tracked_independently() -> None:
    usage.reset_fingerprints()
    usage.cache_line("a", _fp(), tool_count=1, system_chars=40)
    assert "first turn on this thread" in usage.cache_line("b", _fp(), tool_count=1, system_chars=40)
    assert "prefix unchanged" in usage.cache_line("a", _fp(), tool_count=1, system_chars=40)


def test_fingerprint_tracking_is_bounded() -> None:
    usage.reset_fingerprints()
    for i in range(usage._MAX_TRACKED_THREADS + 50):
        usage.cache_line(f"t{i}", _fp(), tool_count=1, system_chars=40)
    assert len(usage._PREV_FINGERPRINT) == usage._MAX_TRACKED_THREADS


def test_cache_line_names_which_context_entry_moved() -> None:
    """A bare "system changed" doesn't say what the client sent differently."""
    usage.reset_fingerprints()
    canvas = "Live canvas state — thin enum. Shape: {components: [...]}"
    memories = "User memories — sandboxed markdown FileSystem persisted across sessions."
    usage.cache_line(
        "t", _fp(context_pairs=[(canvas, '{"components":[]}'), (memories, '{"paths":[]}')]),
        tool_count=1, system_chars=40,
    )
    line = usage.cache_line(
        "t",
        _fp(context_pairs=[(canvas, '{"components":[{"id":"a"}]}'), (memories, '{"paths":[]}')]),
        tool_count=1, system_chars=52,
    )
    assert "ctx.live-canvas-state.value" in line
    assert "ctx.user-memories.value" not in line   # that one held
    assert "ctx.live-canvas-state.desc" not in line  # the prose held; the payload moved


def test_cache_line_drops_redundant_system_when_an_entry_is_named() -> None:
    """`system` is base_system + every entry concatenated, so it always moves with
    them — printing both just pads the line."""
    usage.reset_fingerprints()
    pairs = [("Live canvas state — thin enum.", "{}")]
    usage.cache_line("t", _fp(context_pairs=pairs), tool_count=1, system_chars=40)
    line = usage.cache_line(
        "t", _fp(context_pairs=[("Live canvas state — thin enum.", '{"x":1}')]),
        tool_count=1, system_chars=44,
    )
    assert "ctx.live-canvas-state.value" in line
    assert "[system]" not in line and ", system" not in line


def test_context_label_survives_punctuation_and_empty_prose() -> None:
    assert usage.context_label("Live canvas state — thin enum.", 0) == "live-canvas-state"
    assert usage.context_label("Subagents —  Delegates in pupa/agents/", 1) == "subagents-delegates-in"
    assert usage.context_label("", 3) == "entry3"
