"""Generic, config-driven MCP servers — attach arbitrary MCP servers to the agent.

This module wires in **any** number of MCP servers declared in config —
including browser automation via `@playwright/mcp`, which is an ordinary entry
here rather than a bespoke module. Operators list named
servers under `mcp_servers:` in `~/.pupa-backend/config.yml`; `pupa_config.py`
serialises that block to the `PUPA_MCP_SERVERS` env var (JSON), which this module
reads at startup. Each entry mirrors the `MultiServerMCPClient` connection shape
(and Claude Code's `.mcp.json` `mcpServers`):

    mcp_servers:
      atlassian:
        command: uvx
        args: [mcp-atlassian]
        transport: stdio          # default; omit for stdio
        env:
          CONFLUENCE_URL: https://your-domain.atlassian.net/wiki
          CONFLUENCE_USERNAME: you@corp.com
          CONFLUENCE_API_TOKEN: ${CONFLUENCE_API_TOKEN}   # ${VAR} from process env
      remote_http_example:
        url: https://host/mcp
        transport: streamable_http
        headers: { Authorization: "Bearer ${SOME_TOKEN}" }

Per-server `enabled: false` skips a server without deleting its block. Every
server's tools start hidden and are unlocked per-thread through a single
`get_tools(server=...)` gate tool (see `MCPServersLifecycle.build_gate_tool` and
`McpGateMiddleware`), so registering many servers keeps the model's tool list
lean. An optional per-server `description:` is surfaced in the gate tool's
listing and in `/backend-tools` discovery.

`langchain-mcp-adapters` is a core dependency (shipped by default), imported
lazily inside `mcp_servers_lifecycle()` to keep module import cheap when no
servers are configured.
"""

import json
import logging
import os
import re
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any, Awaitable, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AIMessage,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
from langgraph.config import get_config
from langgraph.prebuilt import InjectedState
from typing import Annotated

logger = logging.getLogger(__name__)

# Keys that are ours (control / meta), not part of a MultiServerMCPClient connection.
_META_KEYS = frozenset({"enabled", "description"})

_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z_0-9]*)\}")

# Per-thread activation: thread_id -> set of activated server names. A server's
# tools stay hidden from the model until `get_tools(server=...)` adds it here.
# Module-level (mirrors the old playwright gate): resets on restart, which is
# fine — the iOS app mints a fresh threadId per session, so a restart is a new
# session anyway.
_activated: dict[str, set[str]] = {}


def _tool_name(t: Any) -> str | None:
    if isinstance(t, dict):
        return t.get("name")
    return getattr(t, "name", None)


def _interpolate(value: Any) -> Any:
    """Recursively expand `${VAR}` placeholders from `os.environ`.

    A placeholder whose variable is unset is left literal so the
    misconfiguration is visible downstream rather than silently blanked.
    """
    if isinstance(value, str):
        return _VAR_RE.sub(lambda m: os.environ.get(m.group(1), m.group(0)), value)
    if isinstance(value, list):
        return [_interpolate(v) for v in value]
    if isinstance(value, dict):
        return {k: _interpolate(v) for k, v in value.items()}
    return value


def _load_mcp_config() -> dict[str, dict]:
    """Parse `PUPA_MCP_SERVERS` into `{name: connection}` for MultiServerMCPClient.

    Drops `enabled: false` entries, expands `${VAR}` placeholders, strips meta keys,
    and defaults each connection's `transport` to `"stdio"`. Returns `{}` when the
    env var is unset, empty, or malformed (a bad blob must not crash startup).
    """
    raw = os.getenv("PUPA_MCP_SERVERS")
    if not raw:
        return {}
    try:
        block = json.loads(raw)
    except (ValueError, TypeError):
        logger.warning("PUPA_MCP_SERVERS is not valid JSON; ignoring all MCP servers.")
        return {}
    if not isinstance(block, dict):
        return {}

    connections: dict[str, dict] = {}
    for name, entry in block.items():
        if not isinstance(entry, dict):
            logger.warning("MCP server %r is not a mapping; skipping.", name)
            continue
        if entry.get("enabled") is False:
            continue
        conn = {k: _interpolate(v) for k, v in entry.items() if k not in _META_KEYS}
        # stdio is the default transport when none is named (a `url` entry is HTTP/SSE).
        if "transport" not in conn:
            conn["transport"] = "streamable_http" if "url" in conn else "stdio"
        connections[name] = conn
    return connections


def _server_descriptions() -> dict[str, str]:
    """Map each enabled server to its optional `description:` (for the gate tool).

    Parsed from the raw `PUPA_MCP_SERVERS` block rather than from the stripped
    connections, since `description` is a meta key (see `_META_KEYS`). Servers
    without a description are simply absent; the gate falls back to a tool count.
    """
    raw = os.getenv("PUPA_MCP_SERVERS")
    if not raw:
        return {}
    try:
        block = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    if not isinstance(block, dict):
        return {}
    out: dict[str, str] = {}
    for name, entry in block.items():
        if not isinstance(entry, dict) or entry.get("enabled") is False:
            continue
        desc = entry.get("description")
        if isinstance(desc, str) and desc.strip():
            out[name] = desc.strip()
    return out


class MCPServersLifecycle:
    """Holds the tools loaded from every connected MCP server.

    Not instantiated directly — obtained via the `mcp_servers_lifecycle()` async
    context manager, which yields `None` when no servers are configured. Holds the
    loaded tools plus the metadata the gate needs: which tool belongs to which
    server, and each server's optional human description.
    """

    def __init__(
        self,
        tools: list[Any],
        server_tool_names: dict[str, frozenset[str]],
        server_descriptions: dict[str, str] | None = None,
    ) -> None:
        self._tools = tools
        self._server_tool_names = server_tool_names
        self._server_descriptions = server_descriptions or {}

    @property
    def tools(self) -> list[Any]:
        return self._tools

    @property
    def server_tool_names(self) -> dict[str, frozenset[str]]:
        return self._server_tool_names

    @property
    def server_descriptions(self) -> dict[str, str]:
        return self._server_descriptions

    def build_gate_tool(self) -> BaseTool:
        """Return the single `get_tools(server)` gate, bound to the loaded servers.

        The description enumerates every loaded server (with its `description:` or
        a tool count) so the model can discover what to unlock. Calling it marks
        `(thread_id, server)` activated; `McpGateMiddleware` reveals that server's
        tools from the next step onward.
        """
        servers = sorted(self._server_tool_names)
        valid = frozenset(servers)
        lines = []
        for s in servers:
            desc = self._server_descriptions.get(s)
            suffix = desc if desc else f"{len(self._server_tool_names.get(s, ()))} tool(s)"
            lines.append(f"  - {s}: {suffix}")
        listing = "\n".join(lines) if lines else "  (none available)"
        server_tool_names = self._server_tool_names

        @tool
        def get_tools(
            server: str,
            config: RunnableConfig,
            state: Annotated[dict, InjectedState],
        ) -> str:
            """placeholder — replaced by a dynamic description below."""
            thread_id = (config.get("configurable") or {}).get("thread_id", "")
            name = (server or "").strip()
            if name not in valid:
                allowed = ", ".join(servers) or "(none)"
                return f"Unknown MCP server {name!r}. Available servers: {allowed}."
            # Honour the Settings mute: a server disabled by the user (sent as
            # `mcp_<server>` in `disabled_tools`) can't be activated, and its
            # tools are stripped from the model's list by `ToolGatingMiddleware`
            # regardless — refuse here so the model doesn't waste a turn.
            disabled = {n for n in (state.get("disabled_tools") or []) if isinstance(n, str)}
            if f"mcp_{name}" in disabled:
                return (
                    f"MCP server {name!r} is disabled in Settings — its tools are "
                    "unavailable this session. Ask the user to enable it first."
                )
            _activated.setdefault(thread_id, set()).add(name)
            names = sorted(server_tool_names.get(name, ()))
            return (
                f"{name} activated — {len(names)} tool(s) now available: "
                f"{', '.join(names)}. They appear in your tool list from the next "
                "step onward."
            )

        get_tools.description = (
            "Activate an MCP server's tools for this session. Tools start hidden "
            "to keep context lean. Available servers:\n"
            f"{listing}\n"
            'Call get_tools(server="<name>") once; that server\'s tools appear '
            "from the next step onward. Do not call for an already-activated server."
        )
        return get_tools

    def build_gate_middleware(self) -> "McpGateMiddleware":
        return McpGateMiddleware(self._server_tool_names)


class McpGateMiddleware(AgentMiddleware):
    """Hide each MCP server's tools until `get_tools(server)` activates it.

    On every model call, for the current thread: keep non-MCP tools (including the
    `get_tools` gate) and the tools of every activated server; strip the rest. The
    gate itself is never an MCP tool, so it stays visible throughout — the model
    can unlock additional servers at any point.
    """

    def __init__(
        self,
        server_tool_names: dict[str, frozenset[str]],
        gate_tool_name: str = "get_tools",
    ) -> None:
        self._gate_tool_name = gate_tool_name
        # tool name -> owning server, for O(1) lookup while filtering.
        self._owner: dict[str, str] = {
            tool_name: server
            for server, names in server_tool_names.items()
            for tool_name in names
        }

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse | AIMessage | ExtendedModelResponse:
        try:
            thread_id = get_config().get("configurable", {}).get("thread_id", "")
        except Exception:
            thread_id = ""

        active = _activated.get(thread_id, frozenset())
        allowed = []
        for t in request.tools:
            owner = self._owner.get(_tool_name(t))
            if owner is None or owner in active:
                allowed.append(t)
        return await handler(request.override(tools=allowed))


def _tool_names(tools: list[Any]) -> frozenset[str]:
    return frozenset(
        name for t in tools if (name := getattr(t, "name", None)) is not None
    )


@asynccontextmanager
async def mcp_servers_lifecycle():
    """Async context manager that connects every configured MCP server.

    Yields an `MCPServersLifecycle` whose `.tools` is the concatenation of every
    server's tools, or `None` when no servers are configured (no import, no
    subprocess). A server that fails to start logs a warning and is skipped — one
    bad server must not blank the whole agent.

    Usage in app.py lifespan::

        async with mcp_servers_lifecycle() as mcp:
            graph = build_graph(..., mcp=mcp)
    """
    connections = _load_mcp_config()
    if not connections:
        yield None
        return

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
        from langchain_mcp_adapters.tools import load_mcp_tools
    except ImportError as exc:
        raise RuntimeError(
            "mcp_servers is configured but langchain-mcp-adapters failed to import "
            "(it is a core dependency). Re-sync deps: cd backend && uv sync"
        ) from exc

    client = MultiServerMCPClient(connections)
    all_tools: list[Any] = []
    server_tool_names: dict[str, frozenset[str]] = {}

    async with AsyncExitStack() as stack:
        for name in connections:
            try:
                session = await stack.enter_async_context(client.session(name))
                tools = await load_mcp_tools(session)
            except Exception:
                logger.warning("MCP server %r failed to start; skipping.", name, exc_info=True)
                continue
            all_tools.extend(tools)
            server_tool_names[name] = _tool_names(tools)
            logger.info("MCP server %r connected — %d tool(s).", name, len(tools))

        yield MCPServersLifecycle(
            tools=all_tools,
            server_tool_names=server_tool_names,
            server_descriptions=_server_descriptions(),
        )
