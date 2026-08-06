"""Regression suite for the Settings-driven backend tool gate.

Pins two contracts on the real `create_agent` + `ToolGatingMiddleware`
pipeline:

1. **Tools whose names appear in `state["disabled_tools"]` never reach the
   model** — `ToolGatingMiddleware.awrap_model_call` rebinds `request.tools`
   per call. A `CollectingMiddleware` subclass snapshots the post-filter
   tool list each call; we assert `tavily_search` is absent when the user
   has disabled it and present otherwise.

2. **The agent's input JSON schema exposes `disabled_tools`** — load-bearing
   piece of the iOS fix. `ag_ui_langgraph.prepare_stream` filters
   `RunAgentInput.state` through the graph's input schema before passing
   it to LangGraph, so without `ToolGatingMiddleware.state_schema`
   declaring `disabled_tools` the key gets dropped on the wire and the
   gate never fires.
"""



from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

from pupa_backend.harnesses.langgraph.tool_gating import ToolGatingMiddleware

from .conftest import MockChatModel


@tool
def tavily_search(query: str) -> str:
    """Pretend web search."""
    return "ok"


@tool
def other_tool(x: str) -> str:
    """A second backend tool we never disable — used as a control."""
    return "ok"


def _build_agent(model: MockChatModel, middleware):
    return create_agent(
        model=model,
        tools=[tavily_search, other_tool],
        middleware=[middleware],
        checkpointer=MemorySaver(),
        name="tool_gating_test",
    )


async def test_disabled_tool_is_dropped_from_request_tools():
    """When `disabled_tools` contains a name, the middleware filters it
    out of `request.tools` before the handler runs — both for backend
    tools that the agent already has and for the round-trip into the
    model. The negative control (no disabled_tools) leaves the list
    untouched.
    """
    collected: list[list[str]] = []

    class CollectingMiddleware(ToolGatingMiddleware):
        async def awrap_model_call(self, request, handler):
            async def capturing_handler(req):
                collected.append([
                    (t.get("name") if isinstance(t, dict) else getattr(t, "name", None))
                    for t in req.tools
                ])
                return await handler(req)

            return await super().awrap_model_call(request, capturing_handler)

    model = MockChatModel(responses=[
        AIMessage(content="a", id="m1"),
        AIMessage(content="b", id="m2"),
    ])
    agent = _build_agent(model, CollectingMiddleware())

    await agent.ainvoke(
        {
            "messages": [HumanMessage(content="hi", id="h1")],
            "disabled_tools": ["tavily_search"],
        },
        config={"configurable": {"thread_id": "disabled"}},
    )
    await agent.ainvoke(
        {"messages": [HumanMessage(content="hi", id="h2")]},
        config={"configurable": {"thread_id": "enabled"}},
    )

    assert len(collected) == 2
    disabled_call, enabled_call = collected
    assert "tavily_search" not in disabled_call
    assert "other_tool" in disabled_call
    assert set(enabled_call) == {"tavily_search", "other_tool"}


@tool
def task(prompt: str) -> str:
    """Stand-in for the subagent `task` tool."""
    return "ok"


@tool
def browser_click(selector: str) -> str:
    """Stand-in for an MCP server tool."""
    return "ok"


def _collecting_cls():
    """Fresh CollectingMiddleware + the list it appends post-filter tool names
    to, so each test gets an isolated capture buffer."""
    collected: list[list[str]] = []

    class CollectingMiddleware(ToolGatingMiddleware):
        async def awrap_model_call(self, request, handler):
            async def capturing_handler(req):
                collected.append([
                    (t.get("name") if isinstance(t, dict) else getattr(t, "name", None))
                    for t in req.tools
                ])
                return await handler(req)

            return await super().awrap_model_call(request, capturing_handler)

    return CollectingMiddleware, collected


async def test_alias_disable_id_drops_real_tool_name():
    """A disable id that differs from the bound tool name (e.g. `subagents` →
    `task`) drops the real tool via the alias map, not the literal id."""
    CollectingMiddleware, collected = _collecting_cls()
    model = MockChatModel(responses=[AIMessage(content="a", id="m1")])
    mw = CollectingMiddleware(aliases={"subagents": {"task"}})
    agent = create_agent(
        model=model,
        tools=[task, other_tool],
        middleware=[mw],
        checkpointer=MemorySaver(),
        name="alias_gating_test",
    )

    await agent.ainvoke(
        {
            "messages": [HumanMessage(content="hi", id="h1")],
            "disabled_tools": ["subagents"],
        },
        config={"configurable": {"thread_id": "alias"}},
    )

    assert collected[0] is not None
    assert "task" not in collected[0]
    assert "other_tool" in collected[0]


async def test_mcp_alias_drops_only_that_servers_tools():
    """Muting `mcp_<server>` drops every tool that server owns while other
    tools survive — the regression behind the dead MCP toggles."""
    CollectingMiddleware, collected = _collecting_cls()
    model = MockChatModel(responses=[AIMessage(content="a", id="m1")])
    mw = CollectingMiddleware(aliases={"mcp_demo": {"browser_click"}})
    agent = create_agent(
        model=model,
        tools=[browser_click, other_tool],
        middleware=[mw],
        checkpointer=MemorySaver(),
        name="mcp_alias_gating_test",
    )

    await agent.ainvoke(
        {
            "messages": [HumanMessage(content="hi", id="h1")],
            "disabled_tools": ["mcp_demo"],
        },
        config={"configurable": {"thread_id": "mcp-alias"}},
    )

    assert "browser_click" not in collected[0]
    assert "other_tool" in collected[0]


def test_agent_input_schema_exposes_disabled_tools():
    """The AG-UI bridge filters `RunAgentInput.state` through the graph's
    input schema (`filter_object_by_schema_keys` in
    `ag_ui_langgraph.utils`). If `disabled_tools` doesn't appear here,
    the key gets silently dropped on the wire — exactly the bug surfaced
    when this feature was first wired up. Pins the schema contract
    independently of any AG-UI version.
    """
    model = MockChatModel(responses=[AIMessage(content="ok")])
    agent = _build_agent(model, ToolGatingMiddleware())

    schema = agent.get_input_jsonschema()
    keys = set(schema.get("properties", {}).keys())
    assert "disabled_tools" in keys, (
        f"expected `disabled_tools` in input schema, got {keys!r}. "
        "Did `ToolGatingMiddleware.state_schema` regress?"
    )
