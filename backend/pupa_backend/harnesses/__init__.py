"""Agent-harness registry: serve several agent loops from one server.

A **harness** is a self-contained agent loop (LangGraph, Claude Code, …) that
owns an AG-UI SSE handler. Previously exactly one ran per process, chosen at
startup by `PUPA_AGENT_LOOP`; now every *enabled* harness is mounted at
`POST /harnesses/{id}` (and the default one also at `POST /`), so the iOS client
picks the harness per backend connection at request time.

Enabled harnesses come from `PUPA_HARNESSES` (JSON, emitted by `pupa_config`
from the config.yml `harnesses:` block), e.g.::

    {"deepagents": {"enabled": true, "default": true},
     "claude_code": {"enabled": true}}

Adding a harness today means adding an adapter to `_ADAPTERS` (fork-friendly);
a public plugin entry point is deferred until the backend package is published.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger("uvicorn.error")


@dataclass
class HarnessDeps:
    """Shared persistence/MCP handles passed to every harness at mount time."""

    checkpointer: Any = None
    store: Any = None
    mcp: Any = None


@runtime_checkable
class AgentHarness(Protocol):
    """A mountable, discoverable agent loop.

    `register` wires the AG-UI SSE handler at `path`; the discovery methods feed
    `GET /harnesses` so the client renders the right model list and permission
    controls without an app update.
    """

    id: str
    label: str

    def register(self, app: Any, path: str, deps: HarnessDeps) -> None: ...
    def models(self) -> list[dict]: ...
    def tools(self) -> list[dict]: ...
    def permission_schema(self) -> list[dict]: ...
    # Optional: extended-thinking levels this harness supports (`[]` = none). The
    # discovery route reads it via getattr, so a harness may omit the method.
    def thinking(self) -> list[dict]: ...


class ClaudeCodeHarness:
    """The Claude Code agent loop (subscription-billed) as a harness."""

    id = "claude_code"
    label = "Claude Code"

    def register(self, app: Any, path: str, deps: HarnessDeps) -> None:
        from pupa_backend.harnesses.claude import register_claude_loop_endpoint

        # No checkpointer/store: the loop keeps sessions in-process and the
        # Claude Code SDK owns its own history.
        register_claude_loop_endpoint(app, path=path, mcp=deps.mcp)

    def models(self) -> list[dict]:
        from pupa_backend.harnesses.claude.models import loop_model_menu

        return loop_model_menu()

    def thinking(self) -> list[dict]:
        from pupa_backend.harnesses.claude.thinking import loop_thinking_menu

        return loop_thinking_menu()

    def tools(self) -> list[dict]:
        # No static backend-tool registry — the loop's power is Claude Code's
        # native host tools, gated by the `claude_loop_native` scope control
        # below rather than an individual mute list.
        return []

    def permission_schema(self) -> list[dict]:
        # Keys read verbatim from RunAgentInput.state by claude_loop/gate.py.
        return [
            {
                "key": "claude_loop_native",
                "type": "choice",
                "label": "Host tools",
                "options": ["off", "read", "edit", "full"],
                "default": "full",
            },
            {
                "key": "claude_loop_auto_approve",
                "type": "bool",
                "label": "Run commands without asking",
                "default": False,
            },
        ]


# Built-in adapters, keyed by harness id. Fork-friendly extension point — add an
# adapter class here. `langgraph` is resolved lazily in `_adapter_for` so
# importing this module doesn't pull in `agent` (and its heavy deps).
_ADAPTERS: dict[str, type] = {
    "claude_code": ClaudeCodeHarness,
}


def _adapter_for(harness_id: str):
    if harness_id == "deepagents":
        from pupa_backend.harnesses.langgraph.harness import DeepAgentsHarness

        return DeepAgentsHarness
    adapter = _ADAPTERS.get(harness_id)
    if adapter is None:
        raise ValueError(
            f"Unknown harness {harness_id!r}. Known: deepagents, claude_code. "
            "Fix the `harnesses:` block in ~/.pupa-backend/config.yml (or "
            "PUPA_HARNESSES) to name one of those."
        )
    return adapter


@dataclass
class _Entry:
    harness: AgentHarness
    is_default: bool


class HarnessRegistry:
    """The enabled harnesses for this process, plus which one is the default."""

    def __init__(self, entries: list[_Entry]) -> None:
        self._entries = entries

    def enabled(self) -> list[AgentHarness]:
        return [e.harness for e in self._entries]

    def default(self) -> AgentHarness | None:
        for e in self._entries:
            if e.is_default:
                return e.harness
        return self._entries[0].harness if self._entries else None

    def ids(self) -> list[str]:
        return [e.harness.id for e in self._entries]


def _load_config() -> dict[str, dict]:
    """Parse `PUPA_HARNESSES` JSON; default to langgraph-only when unset."""
    raw = os.getenv("PUPA_HARNESSES")
    if not raw:
        return {"deepagents": {"enabled": True, "default": True}}
    try:
        data = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"PUPA_HARNESSES is not valid JSON: {exc}") from exc
    if not isinstance(data, dict) or not data:
        raise ValueError("PUPA_HARNESSES must be a non-empty JSON object.")
    return data


def build_registry() -> HarnessRegistry:
    """Instantiate every enabled harness from config. Empty enabled set is fatal."""
    config = _load_config()
    entries: list[_Entry] = []
    default_seen = False
    for harness_id, cfg in config.items():
        cfg = cfg or {}
        if not cfg.get("enabled", False):
            continue
        adapter_cls = _adapter_for(harness_id)
        is_default = bool(cfg.get("default", False))
        if is_default:
            default_seen = True
        entries.append(_Entry(harness=adapter_cls(), is_default=is_default))
    if not entries:
        raise ValueError(
            "No agent harness is enabled. Enable at least one in config.yml "
            "`harnesses:` (e.g. `deepagents: {enabled: true, default: true}`)."
        )
    if not default_seen:
        entries[0].is_default = True
    return HarnessRegistry(entries)


def claude_harness_enabled() -> bool:
    """True if the Claude Code harness is enabled — gates the credential scrub."""
    return bool((_load_config().get("claude_code") or {}).get("enabled", False))


def deepagents_harness_enabled() -> bool:
    """True if the deepagents harness is enabled.

    Gates the checkpointer/store lifespan and the `/db` router: both read and
    write this loop's LangGraph checkpoints, so a deploy without the harness
    has no data for them to serve.
    """
    return bool((_load_config().get("deepagents") or {}).get("enabled", False))
