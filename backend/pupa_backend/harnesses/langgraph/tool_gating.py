"""LangGraph middleware that lets the client mute backend tools per turn.

The iOS client forwards a `disabled_tools: [name, ...]` list inside the AG-UI
`RunAgentInput.state` payload (mirrored from the Settings sheet's Developer
section). `ag_ui_langgraph.prepare_stream` filters `input.state` through the
graph's input schema before passing it to LangGraph — keys not in the schema
are dropped silently. We therefore declare a `ToolGatingState` schema with
`disabled_tools` so the key survives the filter and reaches
`request.state["disabled_tools"]` inside `awrap_model_call`. `create_agent`
unions every middleware's `state_schema` into the resolved agent state.

The middleware then filters `request.tools` to drop matching tool names
before delegating to the next handler — the model never sees the disabled
tools for that call, so it cannot emit a tool call for them.

Frontend (iOS-side) tool descriptors flow through the same `request.tools`
list, so this middleware will also hide frontend tools whose names appear in
`disabled_tools`. The Settings sheet only exposes backend tools today, but
the gating logic is name-based and applies uniformly.
"""



from typing import Any, Awaitable, Callable, NotRequired

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    AIMessage,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
)


class ToolGatingState(AgentState):
    """Agent state extension that surfaces the client-controlled tool gate."""

    disabled_tools: NotRequired[list[str]]


def _tool_name(tool: Any) -> str | None:
    if isinstance(tool, dict):
        return tool.get("name")
    return getattr(tool, "name", None)


class ToolGatingMiddleware(AgentMiddleware):
    """Drop tools whose names appear in `state["disabled_tools"]`.

    Some Settings entries advertise a disable id that differs from the real
    runtime tool name(s) — middleware specs (`subagents` → `task`) and MCP
    servers (`mcp_<server>` → that server's tools). `aliases` maps each such id
    to the real names; a disabled id with no alias gates a tool of the same
    name (the `tavily_search` / `shell` case and frontend tools).
    """

    state_schema = ToolGatingState

    def __init__(self, aliases: dict[str, set[str]] | None = None) -> None:
        super().__init__()
        self._aliases = aliases or {}

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse | AIMessage | ExtendedModelResponse:
        disabled_raw = request.state.get("disabled_tools") if request.state else None
        disabled: set[str] = set()
        for name in (disabled_raw or []):
            if not isinstance(name, str):
                continue
            disabled |= self._aliases.get(name, {name})
        if not disabled:
            return await handler(request)

        allowed = [tool for tool in request.tools if _tool_name(tool) not in disabled]
        if len(allowed) == len(request.tools):
            return await handler(request)

        return await handler(request.override(tools=allowed))
