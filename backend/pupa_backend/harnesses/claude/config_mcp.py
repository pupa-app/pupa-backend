"""Bridge the backend's single shared MCP connection into the Claude Code loop.

The backend connects every operator-configured MCP server (config.yml
`mcp_servers:`) exactly once at startup via `mcp_servers.mcp_servers_lifecycle()`
— the same single shared connection the LangGraph path uses. This module wraps
those already-connected LangChain tools as **one in-process** `claude-agent-sdk`
MCP server, so every claude thread calls the *same* server instead of asking its
own `claude` subprocess to spin up a fresh copy.

Why in-process rather than the SDK's `--mcp-config` passthrough: an external
stdio/http server handed to the `claude` subprocess is only loaded if that
subprocess *trusts* it, and the loop runs with `setting_sources=[]` (no trust
source) — so those tools never surface to the model. In-process SDK servers need
no trust (this is exactly why the frontend tools work), and the tool actually
executes here in the backend process, against the one shared MCP session.

Claude sees these tools namespaced ``mcp__pupa_mcp__<tool>``; the loop's gate
(`gate._resolve_static`) already allows any `mcp__*` tool without a prompt.
"""

from __future__ import annotations

import logging
from typing import Any

from claude_agent_sdk import create_sdk_mcp_server, tool

logger = logging.getLogger("uvicorn.error")

# In-process MCP server name the config-driven tools live under. The
# `mcp__<server>__<tool>` prefix Claude sees is derived from it.
SERVER_NAME = "pupa_mcp"
TOOL_PREFIX = f"mcp__{SERVER_NAME}__"


def qualified_name(bare: str) -> str:
    """Config MCP tool name as Claude sees it (``mcp__pupa_mcp__<bare>``)."""
    return f"{TOOL_PREFIX}{bare}"


def _input_schema(lc_tool: Any) -> dict[str, Any]:
    """Best-effort JSON Schema for a LangChain tool's arguments.

    `langchain-mcp-adapters` sets `args_schema` from the MCP tool's `inputSchema`
    (a JSON-schema dict in recent versions; a pydantic model in older ones).
    """
    schema = getattr(lc_tool, "args_schema", None)
    if isinstance(schema, dict):
        return schema
    model_json_schema = getattr(schema, "model_json_schema", None)
    if callable(model_json_schema):
        try:
            return model_json_schema()
        except Exception:  # noqa: BLE001 — fall back to a permissive schema
            pass
    return {"type": "object", "properties": {}}


def _make_handler(lc_tool: Any):
    """SDK tool handler that runs the shared LangChain tool in-process.

    Unlike the frontend tools (which park on a future for the on-device result),
    this executes the tool against the single shared MCP session and returns the
    result inline. Errors are surfaced to the model as text rather than killing
    the loop.
    """

    async def _handler(args: dict[str, Any]) -> dict[str, Any]:
        try:
            result = await lc_tool.ainvoke(args or {})
        except Exception as exc:  # noqa: BLE001 — surface to the model, don't kill the loop
            return {"content": [{"type": "text", "text": f"Error calling tool: {exc}"}]}
        text = result if isinstance(result, str) else str(result)
        return {"content": [{"type": "text", "text": text}]}

    return _handler


def build_config_mcp(mcp: Any) -> tuple[Any, set[str]]:
    """Wrap the shared `MCPServersLifecycle` tools as one in-process SDK server.

    `mcp` is the `MCPServersLifecycle` yielded by `mcp_servers_lifecycle()` (or
    None when no servers are configured). Returns `(server, qualified_names)`, or
    `(None, set())` when there is nothing to expose. `qualified_names` feeds
    `allowed_tools`.
    """
    tools = list(getattr(mcp, "tools", None) or []) if mcp is not None else []
    if not tools:
        return None, set()

    sdk_tools = []
    qualified: set[str] = set()
    for lc_tool in tools:
        name = getattr(lc_tool, "name", None)
        if not name:
            logger.warning("claude_code loop: skipping config MCP tool with no name: %r", lc_tool)
            continue
        description = getattr(lc_tool, "description", "") or ""
        sdk_tool = tool(name, description, _input_schema(lc_tool))(_make_handler(lc_tool))
        sdk_tools.append(sdk_tool)
        qualified.add(qualified_name(name))

    if not sdk_tools:
        return None, set()

    server = create_sdk_mcp_server(name=SERVER_NAME, tools=sdk_tools)
    logger.info(
        "claude_code loop: bridged %d config MCP tool(s) into in-process server %r.",
        len(sdk_tools), SERVER_NAME,
    )
    return server, qualified
