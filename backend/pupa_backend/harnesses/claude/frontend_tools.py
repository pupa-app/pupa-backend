"""Build an in-process SDK MCP server from the client-forwarded frontend tools.

Each descriptor in `RunAgentInput.tools` (the tools the iOS client advertises and
executes on-device) becomes one in-process `claude-agent-sdk` MCP tool. When
Claude calls it, the handler does **not** execute anything — it parks on a future
in the `LiveSession` (claimed by `(name, args)`), which the AG-UI resume POST
resolves with the on-device result. The model then continues as if the tool ran
locally.

Claude sees these tools namespaced as ``mcp__<SERVER_NAME>__<name>``. The endpoint
and gate map that back to the bare frontend name for the `on_interrupt` payload
and the permission allow-list.
"""

from __future__ import annotations

import logging
from typing import Any

from claude_agent_sdk import create_sdk_mcp_server, tool

from .registry import LiveSession, wait_timeout_for

logger = logging.getLogger("uvicorn.error")

# MCP server name the frontend tools live under. Kept short and stable; the
# `mcp__<server>__<tool>` prefix is derived from it.
SERVER_NAME = "pupa_frontend"
TOOL_PREFIX = f"mcp__{SERVER_NAME}__"


def qualified_name(bare: str) -> str:
    """Frontend tool name as Claude sees it (``mcp__pupa_frontend__<bare>``)."""
    return f"{TOOL_PREFIX}{bare}"


def bare_name(qualified: str) -> str:
    """Strip the MCP server prefix back to the bare frontend tool name."""
    return qualified[len(TOOL_PREFIX):] if qualified.startswith(TOOL_PREFIX) else qualified


def _descriptor_fields(descriptor: Any) -> tuple[str | None, str, dict[str, Any]]:
    """Pull (name, description, json-schema) out of an AG-UI tool descriptor.

    Tolerates both the `ag_ui.core.Tool` pydantic model and a plain dict, and the
    OpenAI-style ``{"function": {...}}`` nesting some clients use.
    """
    get = (lambda k: getattr(descriptor, k, None)) if not isinstance(descriptor, dict) else descriptor.get
    name = get("name")
    description = get("description") or ""
    schema = get("parameters") or get("input_schema") or get("inputSchema")
    if name is None and isinstance(descriptor, dict) and isinstance(descriptor.get("function"), dict):
        fn = descriptor["function"]
        name = fn.get("name")
        description = fn.get("description") or description
        schema = fn.get("parameters") or schema
    if not isinstance(schema, dict):
        schema = {"type": "object", "properties": {}}
    return name, description, schema


def build_frontend_mcp(tools: list[Any], session: LiveSession) -> tuple[Any, set[str]]:
    """Create the in-process MCP server for `tools` bound to `session`.

    Returns `(mcp_server_config, qualified_tool_names)`. `qualified_tool_names`
    feeds `allowed_tools` / the gate. An empty `tools` yields a server with no
    tools (the loop then runs as a plain assistant turn).
    """
    sdk_tools = []
    qualified: set[str] = set()

    for descriptor in tools or []:
        name, description, schema = _descriptor_fields(descriptor)
        if not name:
            logger.warning("claude_code loop: skipping frontend tool with no name: %r", descriptor)
            continue

        def _make_handler(tool_name: str):
            async def _handler(args: dict[str, Any]) -> dict[str, Any]:
                # Block until the AG-UI resume POST delivers this call's on-device
                # result (the live SDK may invoke this before or after resume). The
                # wait budget is per-tool: fast for CRUD, generous for subagents.
                return await session.claim_call(
                    tool_name, args or {}, timeout=wait_timeout_for(tool_name)
                )

            return _handler

        sdk_tool = tool(name, description, schema)(_make_handler(name))
        sdk_tools.append(sdk_tool)
        qualified.add(qualified_name(name))

    server = create_sdk_mcp_server(name=SERVER_NAME, tools=sdk_tools)
    return server, qualified


def frontend_tool_specs(tools: list[Any]) -> list[tuple[str, str, dict[str, Any]]]:
    """Ordered `(name, description, schema)` for `tools` — what the model sees.

    Used to fingerprint the cacheable prompt prefix (`usage.fingerprint`) without
    building a server.
    """
    specs: list[tuple[str, str, dict[str, Any]]] = []
    for descriptor in tools or []:
        name, description, schema = _descriptor_fields(descriptor)
        if name:
            specs.append((qualified_name(name), description, schema))
    return specs


def frontend_qualified_names(tools: list[Any]) -> set[str]:
    """Qualified (`mcp__pupa_frontend__*`) names for `tools`, without building a
    server. Lets a resume POST detect a gate-widened surface (new names vs the
    live client's `session.frontend_qualified`) cheaply — the actual in-process
    server for the widened set is only built if a continuation turn is armed.
    """
    names: set[str] = set()
    for descriptor in tools or []:
        name, _desc, _schema = _descriptor_fields(descriptor)
        if name:
            names.add(qualified_name(name))
    return names
