"""Generic MCP gate — `get_tools(server)` + `McpGateMiddleware`.

Replaces the old per-server playwright gate. Contracts:

1. Before activation, no MCP server's tools are visible; the `get_tools` gate is.
2. Activating one server reveals only that server's tools on the next model
   call; other servers stay hidden; the gate stays visible (so the model can
   unlock further servers).
3. Activation is per-thread — a different thread sees nothing.
4. The `get_tools` tool activates a valid server and reports its tools, and
   rejects an unknown server name without activating anything.
5. The gate description lists every server with its optional description.

`MockChatModel` from conftest runs without AWS creds. We build the real
`MCPServersLifecycle` gate + middleware over fake MCP tools.
"""

from unittest.mock import patch

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

import pupa_backend.mcp_servers as mcp_servers
from pupa_backend.mcp_servers import MCPServersLifecycle, McpGateMiddleware

from .conftest import MockChatModel


# ---------------------------------------------------------------------------
# Fake MCP tools across two servers (stand-ins for real loaded MCP tools)
# ---------------------------------------------------------------------------

@tool
def browser_navigate(url: str) -> str:
    """Navigate to a URL."""
    return "navigated"


@tool
def browser_screenshot() -> str:
    """Take a screenshot."""
    return "screenshot"


@tool
def confluence_search(query: str) -> str:
    """Search Confluence."""
    return "results"


_TOOLS = [browser_navigate, browser_screenshot, confluence_search]
_SERVER_TOOL_NAMES = {
    "browser": frozenset(["browser_navigate", "browser_screenshot"]),
    "confluence": frozenset(["confluence_search"]),
}
_DESCRIPTIONS = {
    "browser": "Browser automation — navigate, click, screenshot",
    "confluence": "Confluence search + page read/write",
}


def _lifecycle() -> MCPServersLifecycle:
    return MCPServersLifecycle(
        tools=list(_TOOLS),
        server_tool_names=dict(_SERVER_TOOL_NAMES),
        server_descriptions=dict(_DESCRIPTIONS),
    )


def _name(t) -> str | None:
    return t.get("name") if isinstance(t, dict) else getattr(t, "name", None)


class _CollectingGate(McpGateMiddleware):
    """McpGateMiddleware that records the tool names of each model call."""

    def __init__(self, server_tool_names, sink):
        super().__init__(server_tool_names)
        self._sink = sink

    async def awrap_model_call(self, request, handler):
        async def capturing(req):
            self._sink.append([_name(t) for t in req.tools])
            return await handler(req)
        return await super().awrap_model_call(request, capturing)


def _build_agent(model, sink):
    lifecycle = _lifecycle()
    return create_agent(
        model=model,
        tools=[*lifecycle.tools, lifecycle.build_gate_tool()],
        middleware=[_CollectingGate(_SERVER_TOOL_NAMES, sink)],
        checkpointer=MemorySaver(),
        name="mcp_gate_test",
    )


async def _run(thread_id: str, sink: list):
    model = MockChatModel(responses=[AIMessage(content="ok", id="m1")])
    agent = _build_agent(model, sink)
    with patch(
        "pupa_backend.mcp_servers.get_config",
        return_value={"configurable": {"thread_id": thread_id}},
    ):
        await agent.ainvoke(
            {"messages": [HumanMessage(content="hi", id="h1")]},
            config={"configurable": {"thread_id": thread_id}},
        )


@pytest.fixture(autouse=True)
def _clear_activations():
    mcp_servers._activated.clear()
    yield
    mcp_servers._activated.clear()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

async def test_all_mcp_tools_hidden_before_activation():
    """No server's tools are visible until get_tools activates it; the gate is."""
    sink: list[list[str]] = []
    await _run("t_hidden", sink)

    assert len(sink) == 1
    visible = set(sink[0])
    assert "browser_navigate" not in visible
    assert "browser_screenshot" not in visible
    assert "confluence_search" not in visible
    assert "get_tools" in visible, "the gate tool must always be visible"


async def test_activated_server_revealed_others_hidden():
    """Activating one server reveals only its tools; the gate stays visible."""
    mcp_servers._activated["t_active"] = {"browser"}
    sink: list[list[str]] = []
    await _run("t_active", sink)

    visible = set(sink[0])
    assert "browser_navigate" in visible
    assert "browser_screenshot" in visible
    assert "confluence_search" not in visible, "non-activated server stays hidden"
    assert "get_tools" in visible, "gate stays visible to unlock more servers"


async def test_activation_is_per_thread():
    """Activation on one thread must not leak to another."""
    mcp_servers._activated["thread_a"] = {"browser"}
    sink: list[list[str]] = []
    await _run("thread_b", sink)

    visible = set(sink[0])
    assert "browser_navigate" not in visible
    assert "browser_screenshot" not in visible


def test_get_tools_activates_valid_and_rejects_unknown():
    gate = _lifecycle().build_gate_tool()

    msg = gate.invoke(
        {"server": "browser", "state": {"disabled_tools": []}},
        config={"configurable": {"thread_id": "tg"}},
    )
    assert "browser activated" in msg.lower()
    assert "browser_navigate" in msg and "browser_screenshot" in msg
    assert mcp_servers._activated.get("tg") == {"browser"}

    bad = gate.invoke(
        {"server": "nope", "state": {"disabled_tools": []}},
        config={"configurable": {"thread_id": "tg2"}},
    )
    assert "unknown mcp server" in bad.lower()
    assert "browser" in bad and "confluence" in bad, "lists the valid servers"
    assert "tg2" not in mcp_servers._activated, "unknown server activates nothing"


def test_get_tools_refuses_server_disabled_in_settings():
    """A server muted in Settings (sent as `mcp_<server>` in `disabled_tools`)
    can't be activated — the gate refuses and marks nothing activated."""
    gate = _lifecycle().build_gate_tool()

    msg = gate.invoke(
        {"server": "browser", "state": {"disabled_tools": ["mcp_browser"]}},
        config={"configurable": {"thread_id": "tg-disabled"}},
    )
    assert "disabled in settings" in msg.lower()
    assert "tg-disabled" not in mcp_servers._activated


def test_gate_description_lists_servers_with_descriptions():
    desc = _lifecycle().build_gate_tool().description
    assert "browser" in desc and "confluence" in desc
    assert "Browser automation" in desc
    assert "Confluence search" in desc
    assert 'get_tools(server="<name>")' in desc


def test_gate_description_falls_back_to_tool_count():
    """A server without a description shows a tool count instead."""
    lifecycle = MCPServersLifecycle(
        tools=list(_TOOLS),
        server_tool_names={"browser": _SERVER_TOOL_NAMES["browser"]},
        server_descriptions={},  # no description for browser
    )
    desc = lifecycle.build_gate_tool().description
    assert "browser: 2 tool(s)" in desc
