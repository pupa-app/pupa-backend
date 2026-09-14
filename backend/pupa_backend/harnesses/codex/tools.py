"""Pupa frontend and configured-MCP tools as Codex dynamic tools."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from pupa_backend.agui.tools import (
    ToolSpec,
    descriptor_specs,
    invoke_langchain_tool,
    langchain_tool_specs,
)

FRONTEND_NAMESPACE = "pupa_frontend"
MCP_NAMESPACE = "pupa_mcp"
_VALID_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


@dataclass
class ToolSurface:
    dynamic_tools: list[dict[str, Any]]
    frontend: dict[str, ToolSpec] = field(default_factory=dict)
    mcp: dict[str, tuple[Any, ToolSpec]] = field(default_factory=dict)
    fingerprint: str = ""


def _namespace(name: str, description: str, specs: list[ToolSpec]) -> dict[str, Any]:
    return {
        "type": "namespace",
        "name": name,
        "description": description,
        "tools": [
            {
                "type": "function",
                "name": spec.name,
                "description": spec.description,
                "inputSchema": spec.input_schema,
            }
            for spec in specs
        ],
    }


def build_tool_surface(descriptors: list[Any], state: Any, mcp: Any) -> ToolSurface:
    disabled = set()
    if isinstance(state, dict) and isinstance(state.get("disabled_tools"), list):
        disabled = {str(name) for name in state["disabled_tools"]}

    frontend: dict[str, ToolSpec] = {}
    for spec in descriptor_specs(descriptors or []):
        if spec.name in disabled:
            continue
        if not _VALID_NAME.fullmatch(spec.name):
            raise ValueError(
                f"frontend tool name {spec.name!r} is not valid for Codex dynamic tools"
            )
        if spec.name in frontend:
            raise ValueError(f"duplicate frontend tool name {spec.name!r}")
        frontend[spec.name] = spec

    mcp_tools: dict[str, tuple[Any, ToolSpec]] = {}
    for tool, spec in langchain_tool_specs(mcp):
        if not _VALID_NAME.fullmatch(spec.name):
            continue
        mcp_tools.setdefault(spec.name, (tool, spec))

    dynamic: list[dict[str, Any]] = []
    if frontend:
        dynamic.append(
            _namespace(
                FRONTEND_NAMESPACE,
                "Tools provided and executed by the connected Pupa app.",
                list(frontend.values()),
            )
        )
    if mcp_tools:
        dynamic.append(
            _namespace(
                MCP_NAMESPACE,
                "Operator-configured tools executed by the Pupa backend.",
                [spec for _tool, spec in mcp_tools.values()],
            )
        )
    encoded = json.dumps(dynamic, sort_keys=True, separators=(",", ":"), default=str)
    return ToolSurface(
        dynamic_tools=dynamic,
        frontend=frontend,
        mcp=mcp_tools,
        fingerprint=hashlib.sha256(encoded.encode()).hexdigest(),
    )


async def invoke_mcp(surface: ToolSurface, name: str, arguments: dict[str, Any]) -> tuple[bool, str]:
    pair = surface.mcp.get(name)
    if pair is None:
        return False, f"Unknown Pupa MCP tool: {name}"
    tool, _spec = pair
    return await invoke_langchain_tool(tool, arguments)
