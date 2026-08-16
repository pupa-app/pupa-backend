"""An abandoned frontend interrupt must not poison the thread.

The iOS client can park on a frontend tool call and never POST the resume — the
app is backgrounded or killed mid-dispatch, and on wake it starts a fresh turn
on the same thread instead (see pupa-app/pupa#258). The checkpoint is then left
holding an `AIMessage` whose `tool_calls` have no answering `ToolMessage`.

Sent to the model as-is that is a hard API error — Anthropic and Bedrock both
reject a `tool_use` with no corresponding `tool_result` — and because it lives
in the checkpointer it would poison *every* later turn on the thread, not just
the next one. The Claude Code loop had the same class of failure and needed a
real fix (pupa-app/pupa-backend#11).

The deepagents loop survives it for free: `CopilotKitMiddleware`'s
`_fix_messages_for_bedrock` runs on every model call and strips unanswered
tool_calls, and langchain-anthropic then drops the assistant message it emptied.
That is upstream behaviour this repo does not own, so these tests pin it — if a
copilotkit upgrade drops that cleanup, the deepagents loop silently acquires the
bug and this suite says so.
"""

from __future__ import annotations

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

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


def _dangling_tool_call_ids(messages: list) -> list[str]:
    """tool_call ids an AIMessage emitted with no ToolMessage answering them."""
    answered = {
        m.tool_call_id for m in messages if isinstance(m, ToolMessage) and m.tool_call_id
    }
    emitted: list[str] = []
    for m in messages:
        if isinstance(m, AIMessage):
            emitted += [c.get("id") for c in (m.tool_calls or []) if c.get("id")]
    return [i for i in emitted if i not in answered]


def _build_graph(model: MockChatModel, checkpointer: MemorySaver):
    return create_agent(
        model=model,
        tools=[],
        middleware=[CustomCopilotKitMiddleware()],
        checkpointer=checkpointer,
        name="abandoned_interrupt_test",
    )


async def _park_on_interrupt(graph, config) -> None:
    await graph.ainvoke(
        {
            "messages": [HumanMessage(id="h1", content="add an item")],
            "copilotkit": {"actions": FRONTEND_TOOLS},
        },
        config=config,
    )


def _parked_model() -> MockChatModel:
    return MockChatModel(responses=[
        # Turn 1 — one frontend call, which pauses the graph.
        AIMessage(id="ai1", content="", tool_calls=[
            {"name": "addItem", "args": {"item": {"name": "apple"}}, "id": "call_A"},
        ]),
        # Turn 2 — the reply to the user's new message.
        AIMessage(id="ai2", content="Sure, doing that instead."),
    ])


async def test_abandoned_interrupt_leaves_a_dangling_tool_call_in_the_checkpoint() -> None:
    """Baseline: the hazard is real, so the guard below is load-bearing."""
    graph = _build_graph(_parked_model(), MemorySaver())
    config = {"configurable": {"thread_id": "abandoned-baseline"}}

    await _park_on_interrupt(graph, config)

    state = graph.get_state(config)
    assert state.tasks, "expected the graph to be parked on an interrupt"
    assert _dangling_tool_call_ids(state.values["messages"]) == ["call_A"]


async def test_new_turn_after_an_abandoned_interrupt_sends_no_dangling_tool_call() -> None:
    """The turn the app starts on wake must reach the model well-formed.

    A `tool_use` with no `tool_result` is rejected outright by Anthropic and
    Bedrock, so this is the assertion that stands between an abandoned interrupt
    and a thread that errors on every subsequent turn.
    """
    model = _parked_model()
    graph = _build_graph(model, MemorySaver())
    config = {"configurable": {"thread_id": "abandoned-new-turn"}}

    await _park_on_interrupt(graph, config)
    # No resume — the app gave up and sent a fresh user message instead.
    await graph.ainvoke(
        {
            "messages": [HumanMessage(id="h2", content="actually, do this instead")],
            "copilotkit": {"actions": FRONTEND_TOOLS},
        },
        config=config,
    )

    second_call = model.stream_call_args_list[-1]
    assert _dangling_tool_call_ids(second_call) == [], (
        "the new turn sent the model a tool_use with no tool_result — the "
        "abandoned interrupt poisons the thread"
    )
    # The user's new message did get through, and the turn produced a reply.
    assert any(
        isinstance(m, HumanMessage) and "do this instead" in str(m.content)
        for m in second_call
    )


async def test_the_abandoned_interrupt_is_cleared_from_the_checkpoint() -> None:
    """Healing is persisted, so turn 3 and beyond are clean too — the failure
    mode this guards against is a checkpointed one, not a per-request one."""
    graph = _build_graph(_parked_model(), MemorySaver())
    config = {"configurable": {"thread_id": "abandoned-cleared"}}

    await _park_on_interrupt(graph, config)
    await graph.ainvoke(
        {
            "messages": [HumanMessage(id="h2", content="actually, do this instead")],
            "copilotkit": {"actions": FRONTEND_TOOLS},
        },
        config=config,
    )

    state = graph.get_state(config)
    assert _dangling_tool_call_ids(state.values["messages"]) == []
    assert not state.tasks, "the abandoned interrupt is still pending"


async def test_a_resumed_interrupt_keeps_its_tool_call_paired() -> None:
    """Guard the guard: the cleanup must not strip a call that WAS answered."""
    from langgraph.types import Command

    model = MockChatModel(responses=[
        AIMessage(id="ai1", content="", tool_calls=[
            {"name": "addItem", "args": {"item": {"name": "apple"}}, "id": "call_A"},
        ]),
        AIMessage(id="ai2", content="Added it."),
    ])
    graph = _build_graph(model, MemorySaver())
    config = {"configurable": {"thread_id": "abandoned-control"}}

    await _park_on_interrupt(graph, config)
    await graph.ainvoke(
        Command(resume={"tool_results": [{"toolCallId": "call_A", "content": "ok"}]}),
        config=config,
    )

    messages = graph.get_state(config).values["messages"]
    assert _dangling_tool_call_ids(messages) == []
    # The pairing survived rather than being stripped: both halves are present.
    assert any(
        isinstance(m, AIMessage) and any(c.get("id") == "call_A" for c in (m.tool_calls or []))
        for m in messages
    ), "the answered tool_call was stripped — results would be orphaned"
    assert any(isinstance(m, ToolMessage) and m.tool_call_id == "call_A" for m in messages)
