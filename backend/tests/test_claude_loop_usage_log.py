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
        async def receive_response(self):
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
