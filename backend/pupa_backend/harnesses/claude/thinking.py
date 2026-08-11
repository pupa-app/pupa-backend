"""Per-request extended-thinking selection for the Claude Code agent loop.

The iOS client picks a thinking level per turn via `forwardedProps.llm.thinking`
(a stable string level, alongside `llm.model`). We map it to the SDK's
`ClaudeAgentOptions.thinking` config:

  off    -> {"type": "disabled"}                 (no extended thinking)
  auto   -> {"type": "adaptive"}                 (model decides — the default UX)
  low    -> {"type": "enabled", "budget_tokens": 4000}
  medium -> {"type": "enabled", "budget_tokens": 12000}
  high   -> {"type": "enabled", "budget_tokens": 24000}

Back-compat: no `llm.thinking` (or an unknown value) → return None so the loop
leaves `thinking` unset and the CLI/subscription default applies, exactly as
before. Like `_resolve_loop_model`, this reads `forwarded_props` raw (no
snake_case normalisation) so the lowercase `llm`/`thinking` keys land as-is.

The provider is Claude-only here; other harnesses ignore an `llm.thinking` key.
"""

from __future__ import annotations

from typing import Any

# (level, label) in menu order. `auto` leads because it is the default UX — the
# model picks its own thinking budget. These strings are the wire contract with
# the iOS picker; keep in sync with the client's thinking catalog.
LOOP_THINKING_LEVELS: list[tuple[str, str]] = [
    ("auto", "Auto"),
    ("off", "Off"),
    ("low", "Low"),
    ("medium", "Medium"),
    ("high", "High"),
]

# Token budgets for the explicit `enabled` levels. Tuned coarse — a small step
# for a nudge, a large one for hard problems. `auto`/`off` don't use these.
_BUDGETS: dict[str, int] = {"low": 4000, "medium": 12000, "high": 24000}


def resolve_thinking(input: Any) -> dict[str, Any] | None:
    """`forwardedProps.llm.thinking` → `ClaudeAgentOptions` kwargs, or None.

    Returns a dict to spread into `ClaudeAgentOptions(...)` (currently just a
    `thinking=` config), or None when nothing was requested / the value is
    unrecognised — in which case the loop leaves the option unset.
    """
    fp = getattr(input, "forwarded_props", None) or {}
    llm = fp.get("llm") if isinstance(fp, dict) else None
    level = (llm.get("thinking") if isinstance(llm, dict) else None) or None
    if not level:
        return None
    if level == "off":
        return {"thinking": {"type": "disabled"}}
    if level == "auto":
        return {"thinking": {"type": "adaptive"}}
    if level in _BUDGETS:
        return {"thinking": {"type": "enabled", "budget_tokens": _BUDGETS[level]}}
    return None


def loop_thinking_menu() -> list[dict]:
    """`GET /harnesses` thinking rows for this harness: `{level, label}`."""
    return [{"level": level, "label": label} for level, label in LOOP_THINKING_LEVELS]
