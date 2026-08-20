"""Yellow token / cache-usage log lines for the Claude loop.

The `claude` CLI reports usage twice per turn:

- every `AssistantMessage` carries the usage of **that one API call** — the only
  place a per-call cache read/write split is visible;
- the final `ResultMessage` carries the **turn totals** plus `total_cost_usd`,
  `num_turns`, and a per-model `model_usage` breakdown.

`_pump` logs both at INFO. Keys are read tolerantly: `usage` is snake_case
(`cache_read_input_tokens`) while `model_usage` entries are camelCase
(`cacheReadInputTokens`), and either may go missing as the CLI evolves.
"""

from __future__ import annotations

from typing import Any

_Y = "\033[33m"  # yellow — token stats
_X = "\033[0m"


def _pick(usage: dict[str, Any], *names: str) -> int:
    """First present key among `names`, coerced to int; 0 when absent."""
    for name in names:
        value = usage.get(name)
        if isinstance(value, (int, float)):
            return int(value)
    return 0


def _n(value: int) -> str:
    return f"{value:,}"


def summarize(usage: dict[str, Any] | None) -> dict[str, int] | None:
    """Flatten one usage dict to `{input, output, cache_read, cache_write}`.

    None when there is nothing to report, so callers can skip the log line
    entirely rather than print a row of zeros.
    """
    if not isinstance(usage, dict):
        return None
    stats = {
        "input": _pick(usage, "input_tokens", "inputTokens"),
        "output": _pick(usage, "output_tokens", "outputTokens"),
        "cache_read": _pick(usage, "cache_read_input_tokens", "cacheReadInputTokens"),
        "cache_write": _pick(
            usage, "cache_creation_input_tokens", "cacheCreationInputTokens"
        ),
    }
    return stats if any(stats.values()) else None


def format_usage(usage: dict[str, Any] | None) -> str | None:
    """`in=… out=… cache_read=… cache_write=… (cache hit …%)`, or None."""
    stats = summarize(usage)
    if stats is None:
        return None
    prompt = stats["input"] + stats["cache_read"] + stats["cache_write"]
    parts = [
        f"in={_n(stats['input'])}",
        f"out={_n(stats['output'])}",
        f"cache_read={_n(stats['cache_read'])}",
        f"cache_write={_n(stats['cache_write'])}",
    ]
    if prompt:
        parts.append(f"(cache hit {stats['cache_read'] * 100 // prompt}%)")
    return " ".join(parts)


def format_model_usage(model_usage: dict[str, Any] | None) -> str | None:
    """Per-model breakdown suffix, e.g. `claude-opus-4-8[in=… out=…]`."""
    if not isinstance(model_usage, dict) or not model_usage:
        return None
    rows = []
    for model, usage in model_usage.items():
        line = format_usage(usage)
        if line:
            rows.append(f"{model}[{line}]")
    return " ".join(rows) or None


def message_line(usage: dict[str, Any] | None, thread_id: str) -> str | None:
    """Log line for one API call (an `AssistantMessage`)."""
    line = format_usage(usage)
    if line is None:
        return None
    return f"{_Y}claude_code tokens: {line} (thread={thread_id}){_X}"


def result_line(msg: Any, thread_id: str) -> str:
    """Log line for the turn totals (a `ResultMessage`).

    Always returns a line — cost / duration / turn count are worth seeing even
    when the CLI omitted `usage`.
    """
    parts = [format_usage(getattr(msg, "usage", None)) or "no token usage reported"]
    cost = getattr(msg, "total_cost_usd", None)
    if isinstance(cost, (int, float)):
        parts.append(f"cost=${cost:.4f}")
    turns = getattr(msg, "num_turns", None)
    if isinstance(turns, int):
        parts.append(f"turns={turns}")
    api_ms = getattr(msg, "duration_api_ms", None)
    if isinstance(api_ms, (int, float)):
        parts.append(f"api={api_ms / 1000:.1f}s")
    per_model = format_model_usage(getattr(msg, "model_usage", None))
    if per_model:
        parts.append(per_model)
    return f"{_Y}claude_code turn totals: {' '.join(parts)} (thread={thread_id}){_X}"
