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

import hashlib
import json
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


# --------------------------------------------------------------------------- #
# Prompt-cache diagnosis
# --------------------------------------------------------------------------- #
#
# Anthropic caches on an exact prefix: `tools` → `system` → `messages`. The
# `claude` CLI sets the `cache_control` breakpoints itself (there is no knob on
# `ClaudeAgentOptions`, and a non-zero `cache_write` proves it is asking for a
# cache), so a run that writes on every turn and never reads is not a "we forgot
# to request caching" problem — it means the prefix we hand the CLI changed.
#
# The loop rebuilds `ClaudeAgentOptions` from scratch on every POST, so any of
# these drifting silently costs a full re-cache. We fingerprint the parts we
# control and log which ones moved since the thread's previous turn; the names
# mirror the CLI's own cache-miss reasons ("system prompt changed", "tools
# changed", "tool prompt/schema changed, same tool set", "model changed", …).

_MAX_TRACKED_THREADS = 256
_PREV_FINGERPRINT: dict[str, dict[str, str]] = {}


def _digest(value: Any) -> str:
    blob = json.dumps(value, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob, usedforsecurity=False).hexdigest()[:8]


def context_label(description: str, index: int) -> str:
    """Short stable name for one ambient-context entry, from its description.

    The descriptions are long prose ("Live canvas state — thin enum. …"), so we
    keep the first few words: enough to say *which* entry moved without dumping
    the block into the log. Falls back to the position when there's no prose.
    """
    words = "".join(c if c.isalnum() or c.isspace() else " " for c in description).split()
    slug = "-".join(w.lower() for w in words[:3])
    return slug or f"entry{index}"


def fingerprint(
    *,
    model: str | None,
    base_system: str,
    system: str,
    tool_specs: list[tuple[str, str, Any]],
    permission_mode: str | None,
    thinking: Any,
    skills: Any,
    cwd: str | None,
    context_pairs: list[tuple[str, str]] | None = None,
) -> dict[str, str]:
    """Hash every input that lands in the cacheable prefix.

    `tool_specs` is the ordered `(name, description, schema)` list for the whole
    advertised surface (frontend + config MCP). `base_system` is the loop prompt
    *without* the per-turn ambient-context block, so a drifting system prompt can
    be attributed to the volatile tail rather than the stable head.

    `context_pairs` are the ambient-context entries `(description, value)`,
    hashed one by one. A bare "system changed" doesn't say *what* the client sent
    differently; `ctx.live-canvas-state.value` does. The description (the app's
    policy prose / AGENTS.md) and the value (the JSON payload) are hashed apart
    because they drift for different reasons.
    """
    names = [name for name, _desc, _schema in tool_specs]
    per_entry: dict[str, str] = {}
    for index, (desc, val) in enumerate(context_pairs or []):
        label = context_label(desc, index)
        per_entry[f"ctx.{label}.desc"] = _digest(desc)
        per_entry[f"ctx.{label}.value"] = _digest(val)
    return {
        "model": model or "",
        "base_system": _digest(base_system),
        "system": _digest(system),
        **per_entry,
        "tool_set": _digest(sorted(names)),
        "tool_order": _digest(names),
        "tool_schemas": _digest(tool_specs),
        "permission_mode": permission_mode or "",
        "thinking": _digest(thinking),
        "skills": _digest(skills),
        "cwd": cwd or "",
    }


def cache_line(thread_id: str, fp: dict[str, str], tool_count: int, system_chars: int) -> str:
    """Yellow line naming what moved in the cacheable prefix since last turn.

    Also records `fp` as the thread's new baseline, so this is called exactly
    once per options build.
    """
    prev = _PREV_FINGERPRINT.get(thread_id)
    _PREV_FINGERPRINT[thread_id] = fp
    while len(_PREV_FINGERPRINT) > _MAX_TRACKED_THREADS:
        _PREV_FINGERPRINT.pop(next(iter(_PREV_FINGERPRINT)))

    shape = f"tools={tool_count} system={_n(system_chars)}ch"
    if prev is None:
        body = f"first turn on this thread, {shape} — full cache write expected"
    else:
        changed = sorted(k for k, v in fp.items() if prev.get(k) != v)
        # `system` is the concatenation of base_system + every ctx entry, so it
        # always moves with them; naming it too is noise. Keep it only when it
        # moved *alone* (the ambient block appearing/disappearing wholesale).
        if len(changed) > 1 and "system" in changed:
            changed.remove("system")
        if changed:
            body = f"prefix changed [{', '.join(changed)}], {shape} — cache write expected"
        else:
            body = f"prefix unchanged, {shape} — cache read expected"
    return f"{_Y}claude_code cache: {body} (thread={thread_id}){_X}"


def reset_fingerprints() -> None:
    """Drop all remembered prefixes (tests)."""
    _PREV_FINGERPRINT.clear()
