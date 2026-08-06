"""Regression suite for `TodoListMiddleware` adoption.

`TodoListMiddleware` is the langchain "planning" middleware — it injects a
`write_todos` tool + a planning system-prompt fragment, and adds a
`todos: list[Todo]` field to agent state. It was deferred in
[`docs/langgraph-middleware-research.md`](../../docs/langgraph-middleware-research.md)
§3.3 as 🔴 "drift vs. iOS workspace" but the iOS surface has no todo tools
today, so the drift risk was hypothetical.

Three contracts pinned:

1. **`write_todos` is bound as a backend tool.** Captured via a
   `CollectingMiddleware` subclass of `ToolGatingMiddleware` that snapshots
   `request.tools` per call (same pattern as `test_tool_gating.py`).
2. **`todos` appears in the agent's output schema.** Regression-pin against
   `state_schema` merging — if it drops, any future iOS surface that wants
   to read the plan back would see nothing.
3. **Compositional ordering.** The chain `[CopilotKit, TodoList, ToolGating]`
   runs `ToolGating` innermost on `awrap_model_call`, so it sees the merged
   tool list and can mute `write_todos` by name. Same path the iOS Settings
   sheet uses for `tavily_search`.
"""



from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from pupa_backend.harnesses.langgraph.tool_gating import ToolGatingMiddleware

from .conftest import MockChatModel


def _build_agent(model: MockChatModel, middleware: list):
    return create_agent(
        model=model,
        tools=[],
        middleware=middleware,
        checkpointer=MemorySaver(),
        name="planning_test",
    )


async def test_write_todos_is_bound_as_backend_tool():
    """`TodoListMiddleware.tools = [write_todos]` is set in `__init__`;
    `create_agent` reads it at graph-build time and merges it into the
    agent's tool list. The tool must appear in every captured
    `request.tools` snapshot during a model call.
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

    model = MockChatModel(responses=[AIMessage(content="ok", id="m1")])
    agent = _build_agent(
        model,
        [TodoListMiddleware(), CollectingMiddleware()],
    )

    await agent.ainvoke(
        {"messages": [HumanMessage(content="hi", id="h1")]},
        config={"configurable": {"thread_id": "todo-bound"}},
    )

    assert len(collected) == 1
    assert "write_todos" in collected[0]


def test_todos_appears_in_agent_state_schema():
    """`TodoListMiddleware.state_schema = PlanningState` adds a `todos`
    field to the unioned agent state. If schema merging regresses, the
    field silently drops — and any future iOS surface that wants to read
    the planning state back would see nothing.
    """
    model = MockChatModel(responses=[AIMessage(content="ok")])
    agent = _build_agent(model, [TodoListMiddleware()])

    schema = agent.get_output_jsonschema()
    keys = set(schema.get("properties", {}).keys())
    assert "todos" in keys, (
        f"expected `todos` in output schema, got {keys!r}. "
        "Did TodoListMiddleware.state_schema regress?"
    )


async def test_tool_gating_can_mute_write_todos():
    """Compositional check: with the chain `[TodoList, ToolGating]`,
    `disabled_tools: ["write_todos"]` should drop `write_todos` from the
    tool snapshot the inner handler sees, while leaving it bound on calls
    that don't disable it. Proves `ToolGating` (innermost) sees the
    post-TodoList tool list.
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
    agent = _build_agent(
        model,
        [TodoListMiddleware(), CollectingMiddleware()],
    )

    await agent.ainvoke(
        {
            "messages": [HumanMessage(content="hi", id="h1")],
            "disabled_tools": ["write_todos"],
        },
        config={"configurable": {"thread_id": "muted"}},
    )
    await agent.ainvoke(
        {"messages": [HumanMessage(content="hi", id="h2")]},
        config={"configurable": {"thread_id": "enabled"}},
    )

    assert len(collected) == 2
    muted_call, enabled_call = collected
    assert "write_todos" not in muted_call
    assert "write_todos" in enabled_call
