"""Regression suite for `CustomCopilotKitMiddleware`.

Pins the wire contract between the backend and AGUIKit for frontend
tools:

1. **Batched pause** — when the model emits N frontend tool_calls in one
   AIMessage, the graph pauses on a *single* interrupt carrying every
   call, with `{id, name, args}` per call.
2. **Resume injects ToolMessages** — re-invoking with
   `Command(resume={"tool_results": [...]})` appends one ToolMessage per
   result (adjacent to the AIMessage) and the model is re-invoked with
   the full tool history visible.
3. **Mixed response** — frontend + backend tool_calls in the same
   AIMessage: backend calls flow through the standard ToolNode in the
   same turn, frontend calls are answered from the resume payload.
4. **Missing result placeholder** — if the client drops a result on the
   floor, the middleware synthesises a `missing_tool_result` ToolMessage
   so the AIMessage's tool_calls stay paired and Bedrock doesn't reject
   the conversation.
5. **No frontend tools advertised** — the middleware is a no-op when
   `state["copilotkit"]["actions"]` is empty, so backend-only graphs
   keep their existing tools_condition behaviour.
"""



from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from pupa_backend.harnesses.langgraph.frontend_interrupt import CustomCopilotKitMiddleware

from .conftest import MockChatModel


FRONTEND_TOOLS = [
    {
        "name": "addItem",
        "description": "append one item to the tracker",
        "parameters": {
            "type": "object",
            "properties": {"item": {"type": "object"}},
            "required": ["item"],
        },
    },
]


def _build_graph(model: MockChatModel, checkpointer: MemorySaver, *, backend_tools=None):
    return create_agent(
        model=model,
        tools=backend_tools or [],
        middleware=[CustomCopilotKitMiddleware()],
        checkpointer=checkpointer,
        name="frontend_interrupt_test",
    )


# ---------------------------------------------------------------------------
# 1. Batched pause: N frontend tool_calls → one interrupt with all of them.
# ---------------------------------------------------------------------------


async def test_n_frontend_tool_calls_pause_in_one_batched_interrupt():
    cp = MemorySaver()
    config = {"configurable": {"thread_id": "batched-pause"}}
    model = MockChatModel(responses=[
        AIMessage(
            id="ai1",
            content="",
            tool_calls=[
                {"name": "addItem", "args": {"item": {"name": "apple"}}, "id": "call_A"},
                {"name": "addItem", "args": {"item": {"name": "pear"}}, "id": "call_B"},
                {"name": "addItem", "args": {"item": {"name": "plum"}}, "id": "call_C"},
            ],
        ),
    ])
    graph = _build_graph(model, cp)

    await graph.ainvoke(
        {
            "messages": [HumanMessage(id="h1", content="add three items")],
            "copilotkit": {"actions": FRONTEND_TOOLS},
        },
        config=config,
    )

    state = graph.get_state(config)
    assert state.tasks, "expected a pending task on interrupt"
    interrupts = state.tasks[0].interrupts
    assert len(interrupts) == 1, f"expected exactly one batched interrupt, got {len(interrupts)}"

    value = interrupts[0].value
    assert isinstance(value, dict)
    calls = value.get("frontend_tool_calls")
    assert isinstance(calls, list)
    assert len(calls) == 3, f"expected three calls in one batch, got {len(calls)}"
    assert [c["id"] for c in calls] == ["call_A", "call_B", "call_C"]
    assert [c["name"] for c in calls] == ["addItem", "addItem", "addItem"]
    assert calls[0]["args"] == {"item": {"name": "apple"}}


# ---------------------------------------------------------------------------
# 2. Resume injects one ToolMessage per result and the model continues.
# ---------------------------------------------------------------------------


async def test_resume_appends_tool_messages_and_model_continues():
    cp = MemorySaver()
    config = {"configurable": {"thread_id": "batched-resume"}}
    model = MockChatModel(responses=[
        AIMessage(
            id="ai1",
            content="",
            tool_calls=[
                {"name": "addItem", "args": {"item": {"name": "apple"}}, "id": "call_A"},
                {"name": "addItem", "args": {"item": {"name": "pear"}}, "id": "call_B"},
            ],
        ),
        AIMessage(id="ai2", content="Done — added two items."),
    ])
    graph = _build_graph(model, cp)

    await graph.ainvoke(
        {
            "messages": [HumanMessage(id="h1", content="add two items")],
            "copilotkit": {"actions": FRONTEND_TOOLS},
        },
        config=config,
    )
    await graph.ainvoke(
        Command(resume={"tool_results": [
            {"toolCallId": "call_A", "content": '{"ok":true,"totalItems":1}'},
            {"toolCallId": "call_B", "content": '{"ok":true,"totalItems":2}'},
        ]}),
        config=config,
    )

    state = graph.get_state(config)
    assert not state.tasks or all(not t.interrupts for t in state.tasks)

    messages = state.values["messages"]
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    assert {m.tool_call_id for m in tool_messages} == {"call_A", "call_B"}
    contents = {m.tool_call_id: m.content for m in tool_messages}
    assert "totalItems\":1" in contents["call_A"]
    assert "totalItems\":2" in contents["call_B"]

    ai_messages = [m for m in messages if isinstance(m, AIMessage)]
    final = next((m for m in reversed(ai_messages) if m.content), None)
    assert final is not None
    assert final.content == "Done — added two items."


# ---------------------------------------------------------------------------
# 3. Mixed response: backend ToolNode still runs on the same turn.
# ---------------------------------------------------------------------------


@tool
def _backend_echo(text: str) -> str:
    """A trivial backend tool that echoes its argument back."""
    return f"echoed:{text}"


async def test_mixed_frontend_and_backend_tool_calls_in_one_turn():
    cp = MemorySaver()
    config = {"configurable": {"thread_id": "mixed-turn"}}
    model = MockChatModel(responses=[
        AIMessage(
            id="ai1",
            content="",
            tool_calls=[
                {"name": "_backend_echo", "args": {"text": "hello"}, "id": "call_be"},
                {"name": "addItem", "args": {"item": {"name": "apple"}}, "id": "call_fe"},
            ],
        ),
        AIMessage(id="ai2", content="Both done."),
    ])
    graph = _build_graph(model, cp, backend_tools=[_backend_echo])

    await graph.ainvoke(
        {
            "messages": [HumanMessage(id="h1", content="do both")],
            "copilotkit": {"actions": FRONTEND_TOOLS},
        },
        config=config,
    )
    # First ainvoke pauses on the frontend call. Resume with the result.
    await graph.ainvoke(
        Command(resume={"tool_results": [
            {"toolCallId": "call_fe", "content": '{"ok":true}'},
        ]}),
        config=config,
    )

    state = graph.get_state(config)
    assert not state.tasks or all(not t.interrupts for t in state.tasks)

    messages = state.values["messages"]
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    by_id = {m.tool_call_id: m.content for m in tool_messages}
    # Frontend result came from the client; backend result from ToolNode.
    assert "ok\":true" in by_id["call_fe"]
    assert "echoed:hello" in by_id["call_be"]

    final_ai = next((m for m in reversed([m for m in messages if isinstance(m, AIMessage)]) if m.content), None)
    assert final_ai is not None
    assert final_ai.content == "Both done."


# ---------------------------------------------------------------------------
# 4. Missing-result placeholder keeps Bedrock-pairable history.
# ---------------------------------------------------------------------------


async def test_missing_tool_result_synthesises_placeholder():
    cp = MemorySaver()
    config = {"configurable": {"thread_id": "missing-result"}}
    model = MockChatModel(responses=[
        AIMessage(
            id="ai1",
            content="",
            tool_calls=[
                {"name": "addItem", "args": {"item": {"name": "apple"}}, "id": "call_A"},
                {"name": "addItem", "args": {"item": {"name": "pear"}}, "id": "call_B"},
            ],
        ),
        AIMessage(id="ai2", content="continued anyway"),
    ])
    graph = _build_graph(model, cp)

    await graph.ainvoke(
        {
            "messages": [HumanMessage(id="h1", content="add two")],
            "copilotkit": {"actions": FRONTEND_TOOLS},
        },
        config=config,
    )
    # Client only sent back one result; the other should be filled in.
    await graph.ainvoke(
        Command(resume={"tool_results": [
            {"toolCallId": "call_A", "content": '{"ok":true}'},
        ]}),
        config=config,
    )

    state = graph.get_state(config)
    messages = state.values["messages"]
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    by_id = {m.tool_call_id: m.content for m in tool_messages}
    assert set(by_id) == {"call_A", "call_B"}
    assert "ok\":true" in by_id["call_A"]
    assert "missing_tool_result" in by_id["call_B"]


# ---------------------------------------------------------------------------
# 5. No frontend tools advertised → middleware is a no-op.
# ---------------------------------------------------------------------------


async def test_noop_when_no_frontend_tools_advertised():
    cp = MemorySaver()
    config = {"configurable": {"thread_id": "noop"}}
    model = MockChatModel(responses=[
        AIMessage(id="ai1", content="hi"),
    ])
    graph = _build_graph(model, cp)

    await graph.ainvoke(
        {"messages": [HumanMessage(id="h1", content="hello")]},
        config=config,
    )

    state = graph.get_state(config)
    assert not state.tasks or all(not t.interrupts for t in state.tasks)
    messages = state.values["messages"]
    ai_messages = [m for m in messages if isinstance(m, AIMessage)]
    assert ai_messages and ai_messages[-1].content == "hi"


# ---------------------------------------------------------------------------
# 6. Standalone usability: middleware works with `tools=[]` (no `tool_node`).
# ---------------------------------------------------------------------------


async def test_middleware_works_standalone_with_no_backend_tools_and_no_other_middleware():
    """Pins the contract that ``CustomCopilotKitMiddleware`` is
    usable as a drop-in middleware for an agent that has ONLY frontend tools —
    no backend tools registered, no other middleware in the chain.

    ``@hook_config(can_jump_to=["model"])`` on ``after_model`` is the
    load-bearing decorator. With it, LangChain's ``_add_middleware_edge``
    (factory.py:1835) installs a conditional edge on the no-tools-but-has-
    after_model branch (factory.py:1568-1577) that honours
    ``state["jump_to"]``. Without the decorator, the edge falls through to
    ``exit_node`` and the run silently ends after every resume — model is
    never re-invoked, so the final ``"Resumed and continued."`` AIMessage
    never appears.

    This makes the middleware viable as a candidate for upstreaming to
    ``copilotkit`` as a replacement for the strip/restore-around-after_agent
    dance the current package uses.
    """
    cp = MemorySaver()
    config = {"configurable": {"thread_id": "standalone"}}
    model = MockChatModel(responses=[
        AIMessage(
            id="ai1",
            content="",
            tool_calls=[
                {"name": "addItem", "args": {"item": {"name": "apple"}}, "id": "call_A"},
            ],
        ),
        AIMessage(id="ai2", content="Resumed and continued."),
    ])
    # NOTE: `tools=[]` — no backend tools at all. Only this middleware.
    graph = create_agent(
        model=model,
        tools=[],
        middleware=[CustomCopilotKitMiddleware()],
        checkpointer=cp,
        name="standalone_test",
    )

    await graph.ainvoke(
        {
            "messages": [HumanMessage(id="h1", content="add one")],
            "copilotkit": {"actions": FRONTEND_TOOLS},
        },
        config=config,
    )
    # Interrupt is parked — model has been called exactly once.
    assert model.call_count == 1
    assert graph.get_state(config).tasks[0].interrupts

    await graph.ainvoke(
        Command(resume={"tool_results": [
            {"toolCallId": "call_A", "content": '{"ok":true}'},
        ]}),
        config=config,
    )

    # The model is re-invoked after resume — proving the jump-to-model
    # routing works even though no `tool_node` exists in the graph.
    assert model.call_count == 2, (
        "Model should be re-invoked after resume. If this fails, "
        "@hook_config(can_jump_to=['model']) was likely removed or "
        "{'jump_to': 'model'} dropped from the all-frontend return value."
    )
    state = graph.get_state(config)
    assert not state.tasks or all(not t.interrupts for t in state.tasks)
    ai_messages = [m for m in state.values["messages"] if isinstance(m, AIMessage)]
    final = next((m for m in reversed(ai_messages) if m.content), None)
    assert final is not None and final.content == "Resumed and continued."
