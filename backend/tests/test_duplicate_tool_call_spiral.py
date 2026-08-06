"""Regression suite for the duplicate-tool-call spiral.

See docs/bug-duplicate-tool-call-spiral.md for the full causal chain. These
tests pin three contracts on the real `CopilotKitMiddleware` +
`MemorySaver` + `create_agent` pipeline:

1. **Happy path** — when the LLM emits text alongside `tool_calls`, the
   iOS-side AIMessage carries the wire `messageId` and round 2 settles in
   exactly one extra model call. No spiral.

2. **Spiral when iOS mints a fresh UUID** (the live bug). When the LLM
   emits `tool_calls` without text, iOS today synthesises a new AIMessage
   with a fresh UUID. Round 2 then carries a duplicate AIMessage that
   confuses `_fix_messages_for_bedrock` into stripping the older one's
   `tool_calls` — leaving a ghost turn that drives the model to re-emit.
   This test demonstrates the bug *exists today* by simulating that
   iOS-side ID mismatch and asserting the model is invoked more than
   twice. It is the negative control for the iOS-side fix.

3. **Spiral is closed when `parentMessageId` is preserved** (the iOS
   fix's contract). Same scenario as (2) but the iOS-side simulation
   reuses the round-1 AIMessage id (as the fix in AgentSession.swift
   does). The model is invoked exactly twice — no spiral.
"""



from typing import Iterable

import pytest
from copilotkit import CopilotKitMiddleware
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

from .conftest import MockChatModel


# Frontend tool descriptor as our iOS clients send it in `RunAgentInput.tools`.
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


def _ai_with_tool_calls(*, msg_id: str, content: str = "") -> AIMessage:
    """Build an AIMessage that requests two parallel addItem calls."""
    return AIMessage(
        id=msg_id,
        content=content,
        tool_calls=[
            {"name": "addItem", "args": {"item": {"name": "apple"}}, "id": "call_A"},
            {"name": "addItem", "args": {"item": {"name": "pear"}}, "id": "call_B"},
        ],
    )


def _ios_tool_answers() -> list[ToolMessage]:
    return [
        ToolMessage(tool_call_id="call_A", content='{"ok":true,"totalItems":1}'),
        ToolMessage(tool_call_id="call_B", content='{"ok":true,"totalItems":2}'),
    ]


def _build_graph(model: MockChatModel, checkpointer: MemorySaver):
    """Mirror backend/agent.py exactly except for the model — keep
    the same `tools=[]`, the same middleware, the same checkpointer wiring.
    """
    return create_agent(
        model=model,
        tools=[],
        middleware=[CopilotKitMiddleware()],
        checkpointer=checkpointer,
        name="pupa_agent",
    )


def _initial_state(human_text: str) -> dict:
    return {
        "messages": [HumanMessage(id="h1", content=human_text)],
        "copilotkit": {"actions": FRONTEND_TOOLS},
    }


def _ios_simulated_round2(
    *,
    round1_ai_id: str,
    ios_ai_id: str,
    extra_messages: Iterable = (),
) -> dict:
    """Build the messages an iOS client would POST for round 2.

    `round1_ai_id` is the AIMessage id from the backend checkpoint; iOS
    *should* reuse it. `ios_ai_id` is what iOS actually uses — pass the
    same string to simulate the post-fix behaviour, or a different one to
    simulate the pre-fix (fresh-UUID) behaviour.
    """
    return {
        "messages": [
            HumanMessage(id="h1", content="add a couple of items"),
            _ai_with_tool_calls(msg_id=ios_ai_id),
            *_ios_tool_answers(),
            *extra_messages,
        ],
        "copilotkit": {"actions": FRONTEND_TOOLS},
    }


# ---------------------------------------------------------------------------
# 1. Happy path: text + tool_calls → IDs align, no spiral.
# ---------------------------------------------------------------------------


async def test_happy_path_text_plus_tool_calls():
    cp = MemorySaver()
    config = {"configurable": {"thread_id": "happy"}}

    model = MockChatModel(responses=[
        _ai_with_tool_calls(msg_id="ai_round1", content="Sure, I'll add a couple."),
        AIMessage(id="ai_round2", content="Done."),
    ])
    graph = _build_graph(model, cp)

    await graph.ainvoke(_initial_state("add a couple of items"), config=config)

    # iOS round 2: text streamed, so iOS attaches tool_calls to the wire id.
    await graph.ainvoke(
        _ios_simulated_round2(round1_ai_id="ai_round1", ios_ai_id="ai_round1"),
        config=config,
    )

    assert model.call_count == 2, (
        f"Expected exactly 2 model calls (one per round); got {model.call_count}"
    )


# ---------------------------------------------------------------------------
# 2. Bug repro: tool-call-only round + iOS mints a fresh UUID → spiral.
# ---------------------------------------------------------------------------


async def test_spiral_when_tool_calls_without_text_and_fresh_uuid():
    """Negative control. Demonstrates that the spiral exists today when iOS
    fails to propagate `parentMessageId` (the pre-fix AgentSession.swift
    behaviour). We hand the model 6 spiral rounds of canned responses; if the
    bug is alive, the model gets invoked through several of them before any
    settlement signal could appear.
    """
    cp = MemorySaver()
    config = {"configurable": {"thread_id": "spiral"}}

    # Round 1 emits ONLY tool_calls (content=""), the trigger condition. The
    # model is then handed five more re-emit responses so the spiral has
    # somewhere to go before our `recursion_limit` cap stops the run.
    canned_spiral = [_ai_with_tool_calls(msg_id=f"ai_round{n}") for n in range(1, 7)]
    model = MockChatModel(responses=canned_spiral)
    graph = _build_graph(model, cp)

    await graph.ainvoke(_initial_state("add a couple of items"), config=config)
    assert model.call_count == 1

    # iOS mints a fresh UUID for each round's synthesised AIMessage. The
    # spiral adds one invocation per iOS round and never settles — we
    # confirmed this empirically against `maxRounds = 6` here (production
    # caps at 8 via AgentSession.swift). If the bug ever regresses to
    # convergence after one extra round, this assertion fires.
    for round_n in range(2, 7):
        await graph.ainvoke(
            _ios_simulated_round2(
                round1_ai_id=f"ai_round{round_n - 1}",
                ios_ai_id=f"ios_uuid_round{round_n - 1}",  # mismatch!
            ),
            config=config,
        )

    assert model.call_count >= 6, (
        "Spiral must drive a fresh model invocation every iOS round when "
        f"the iOS-side AIMessage id diverges. Got {model.call_count}."
    )


# ---------------------------------------------------------------------------
# 3. Fix contract: tool-call-only round + iOS preserves parentMessageId → no spiral.
# ---------------------------------------------------------------------------


async def test_no_spiral_when_parent_message_id_is_preserved():
    """The contract the iOS fix locks in. Same scenario as test 2 but the
    iOS-side simulation reuses the round-1 AIMessage id (as the post-fix
    `AgentSession.swift` does via `parentMessageId`). Round 2 settles in
    one extra model call.
    """
    cp = MemorySaver()
    config = {"configurable": {"thread_id": "fixed"}}

    model = MockChatModel(responses=[
        _ai_with_tool_calls(msg_id="ai_round1", content=""),
        AIMessage(id="ai_round2", content="Done."),
    ])
    graph = _build_graph(model, cp)

    await graph.ainvoke(_initial_state("add a couple of items"), config=config)
    assert model.call_count == 1

    # iOS preserves the backend's AIMessage id — the post-fix contract.
    await graph.ainvoke(
        _ios_simulated_round2(round1_ai_id="ai_round1", ios_ai_id="ai_round1"),
        config=config,
    )

    assert model.call_count == 2, (
        "With parentMessageId preserved, the iOS-side AIMessage merges into "
        "the checkpoint and round 2 must settle in exactly one extra model "
        f"call. Got {model.call_count}."
    )


# ---------------------------------------------------------------------------
# 4. In-turn loop trap — depends on graph topology, not on iOS.
# ---------------------------------------------------------------------------
#
# The duplicate-tool-call spiral has two flavours:
#
#   (a) Cross-round — covered by tests 1-3 above. Triggered by an iOS-side
#       AIMessage id mismatch and bounded by `AgentSession.maxRounds`.
#   (b) In-turn — triggered purely by graph topology. When ANY backend tool
#       is passed to `create_agent`, langchain installs a conditional
#       `after_model → model` edge (factory.py:1492-1537). On the next
#       iteration the model sees an AIMessage with its frontend tool_call
#       stripped by `CopilotKitMiddleware.after_model` and may re-emit. This
#       cannot fire today because we ship `tools=[]` in agent.py — but it
#       becomes live the moment a backend tool is added.
#
# These two tests pin the topology contract so a future "let's also add a
# backend tool" PR breaks loudly here instead of silently re-introducing
# the spiral one layer deeper.


# Single, intentionally weak backend tool used to flip create_agent into the
# "install model-loop edges" branch. Its execution path isn't what we care
# about — only that registering it changes the compiled graph topology.
@tool
def _trivial_backend_tool(x: str) -> str:
    """No-op backend tool used to materialise the `tools` node + loop edges."""
    return f"echoed: {x}"


_FRONTEND_TOOLS_FOR_TOPOLOGY_TESTS = [
    {
        "name": "addItem",
        "description": "frontend tool",
        "parameters": {
            "type": "object",
            "properties": {"item": {"type": "object"}},
        },
    },
]


def _build_graph_with_tools(model: MockChatModel, tools: list, checkpointer: MemorySaver):
    return create_agent(
        model=model,
        tools=tools,
        middleware=[CopilotKitMiddleware()],
        checkpointer=checkpointer,
        name="pupa_agent",
    )


async def test_in_turn_loop_exists_when_a_backend_tool_is_registered():
    """With a backend tool registered, the conditional `after_model → model`
    edge is installed and the model is invoked multiple times within a
    single `agent.ainvoke()`. The mock emits a mixed `tool_calls` set
    (frontend + backend); after `after_model` strips the frontend call,
    `tools_condition` sees the surviving backend call, ToolNode runs, and
    control loops back to the model. On the next iteration the AIMessage
    no longer carries the frontend call — exactly the state that drives
    the in-turn variant of the spiral.

    This test is independent of iOS: it pins the upstream graph topology.
    """
    cp = MemorySaver()
    config = {"configurable": {"thread_id": "backend-tools"}}

    model = MockChatModel(responses=[
        # Iteration 1: frontend + backend tool_calls. after_model strips
        # the frontend; ToolNode runs the backend; loop back to model.
        AIMessage(id="ai_iter1", content="", tool_calls=[
            {"name": "addItem", "args": {"item": {"name": "a"}}, "id": "call_fe1"},
            {"name": "_trivial_backend_tool", "args": {"x": "ping"}, "id": "call_be1"},
        ]),
        # Iteration 2: model "didn't see" its own frontend call (it was
        # stripped), so it re-emits.
        AIMessage(id="ai_iter2", content="", tool_calls=[
            {"name": "addItem", "args": {"item": {"name": "a"}}, "id": "call_fe2"},
        ]),
        # Iteration 3: settles.
        AIMessage(id="ai_iter3", content="done"),
        # Spare responses in case the loop runs further than expected;
        # MockChatModel raises IndexError otherwise.
        AIMessage(id="ai_iter4", content="extra"),
    ])
    graph = _build_graph_with_tools(model, [_trivial_backend_tool], cp)

    await graph.ainvoke(
        {
            "messages": [HumanMessage(id="h1", content="add and process")],
            "copilotkit": {"actions": _FRONTEND_TOOLS_FOR_TOPOLOGY_TESTS},
        },
        config=config,
    )

    assert model.call_count >= 2, (
        "When a backend tool is registered, `create_agent` installs the "
        "after_model → model loop edges and the model is invoked more "
        f"than once per ainvoke(). Got {model.call_count} — the loop "
        "edges may have been removed upstream."
    )

    # The actual bug condition: on the second model invocation the AIMessage
    # in scrollback should still carry the frontend tool_call the model just
    # emitted. CopilotKitMiddleware.after_model strips it before the loop
    # routes back to the model — so the model sees a history that contradicts
    # reality and may re-issue the call. We pin this here so a future
    # upstream change (e.g. synthesising an optimistic ToolMessage instead of
    # stripping the call) would flip this assertion and tell us the bug shape
    # has changed.
    iter2_input = model.stream_call_args_list[1]
    iter2_ai_messages = [m for m in iter2_input if isinstance(m, AIMessage)]
    assert iter2_ai_messages, "Expected at least one AIMessage in iteration 2 history"
    iter2_fe_tool_call_names = {
        tc.get("name")
        for m in iter2_ai_messages
        for tc in (getattr(m, "tool_calls", None) or [])
    }
    assert "addItem" not in iter2_fe_tool_call_names, (
        "On the 2nd model invocation the AIMessage history must NOT carry "
        "the frontend tool_call the model emitted in iteration 1 — that's "
        "the in-turn bug: after_model strips the call before the loop "
        "back, so the model sees a history that doesn't reflect what it "
        "just did. If this assertion flips, CopilotKitMiddleware has been "
        "fixed upstream (or the strip behaviour has changed) — update the "
        "diagnosis doc to match. Got tool_calls in iter-2 history: "
        f"{iter2_fe_tool_call_names}"
    )


async def test_in_turn_loop_does_not_exist_without_backend_tools():
    """With `tools=[]` (the production config), `create_agent` skips the
    `after_model → model` loop edges entirely (factory.py:1492 branch),
    so the model is invoked exactly once per `agent.ainvoke()` regardless
    of what it emits. This is what mechanically rules out the in-turn
    variant of the spiral in production today.

    Cross-round behaviour (the iOS-side trigger) is covered by tests 1-3.
    """
    cp = MemorySaver()
    config = {"configurable": {"thread_id": "no-backend-tools"}}

    # Identical first response to the previous test (mixed tool_calls) —
    # but with no backend tool registered, after_model strips the frontend
    # call and there is nothing left to dispatch. The graph terminates.
    model = MockChatModel(responses=[
        AIMessage(id="ai_iter1", content="", tool_calls=[
            {"name": "addItem", "args": {"item": {"name": "a"}}, "id": "call_fe1"},
        ]),
        AIMessage(id="ai_iter2", content="should not be reached"),
    ])
    graph = _build_graph_with_tools(model, [], cp)

    await graph.ainvoke(
        {
            "messages": [HumanMessage(id="h1", content="add an item")],
            "copilotkit": {"actions": _FRONTEND_TOOLS_FOR_TOPOLOGY_TESTS},
        },
        config=config,
    )

    assert model.call_count == 1, (
        "With tools=[], the compiled graph has no `after_model → model` "
        "edge — the model must be invoked exactly once per ainvoke(). "
        f"Got {model.call_count}; the loop edge may have been re-added "
        "upstream, which would re-open the in-turn spiral."
    )
