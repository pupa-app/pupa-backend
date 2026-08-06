"""Model menu for the Claude Code agent loop — stable claude-code aliases.

The `claude` CLI has **no** list-models command, and pinning concrete version
ids (`claude-opus-4-8`, …) would go stale every time a new model ships. So the
loop's menu is claude-code's own stable **aliases**: `--model opus|sonnet|haiku|
fable` "an alias for the latest model" (per `claude --help`). Each alias always
resolves to the current model in its tier, so this list never needs touching as
versions change.

The iOS picker fetches these from `GET /harnesses` (under the `claude_code`
harness) and sends the chosen alias back in `forwardedProps.llm.model`; the loop
passes it straight to `ClaudeAgentOptions(model=...)`. A full `claude-*` id is
also accepted (see `is_loop_model`) so a pinned `CLAUDE_CODE_MODEL` or a stale
client still works.
"""

from __future__ import annotations

# (alias, label) in menu order. Aliases — not pinned ids — so the menu stays
# current as new model versions ship. `opus` leads because the loop default
# (Opus 4.8) is an Opus-tier model.
LOOP_MODEL_ALIASES: list[tuple[str, str]] = [
    ("opus", "Opus (latest)"),
    ("sonnet", "Sonnet (latest)"),
    ("haiku", "Haiku (latest)"),
    ("fable", "Fable (latest)"),
]

# The provider tag the loop reports to iOS. The loop ignores the provider on the
# way back in (it's Claude-only) — only the `model` alias is used — but the
# `/models` shape and the iOS `LLMParams` both carry a provider field.
LOOP_PROVIDER = "claude_code"


def loop_model_menu() -> list[dict]:
    """`GET /harnesses` model rows for this harness: `{provider, modelId, label}`."""
    return [
        {"provider": LOOP_PROVIDER, "modelId": alias, "label": label}
        for alias, label in LOOP_MODEL_ALIASES
    ]


def is_loop_model(model: str) -> bool:
    """True if `model` is something the subscription loop can run: one of the
    stable aliases, or a full `claude-*` id (so pinned configs / stale clients
    keep working). Anything else (e.g. an OpenRouter slug) is rejected."""
    return any(model == alias for alias, _ in LOOP_MODEL_ALIASES) or model.startswith("claude-")
