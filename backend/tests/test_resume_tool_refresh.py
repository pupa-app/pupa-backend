"""Regression for the mid-turn tool-surface refresh bug.

The iOS client recomputes its advertised tool list on every round,
including the resume POST that follows a frontend interrupt. The wire
payload itself is correct (``AgentSession.runLoop`` re-evaluates the
``toolFilter`` closure each iteration before building the next
``RunAgentInput``).

The trouble is that ``ag_ui_langgraph.LangGraphAGUIAgent.prepare_stream``
takes the ``Command(resume=...)`` branch on a resume POST and discards
the freshly-merged ``state`` (which would carry the new
``copilotkit.actions``). Without intervention, the next model call after
the interrupt would read the checkpointed (stale) actions and the model
would never see tools unlocked mid-turn (e.g. by an
``addComponent(kind:"checklist")`` call earlier in the same turn).

The fix (Shape A): the iOS client embeds its fresh descriptor list in
``forwardedProps.command.resume.tools_after_round`` alongside
``tool_results``, and ``CustomCopilotKitMiddleware``
parses it and writes ``state["copilotkit"]["actions"] = new_tools`` in
its ``after_model`` return dict. ``CopilotKitMiddleware.awrap_model_call``
then merges the new descriptors into ``request.tools`` on the next
model call — same path it already uses for the round-1 surface, just
with a refreshed source.

This test drives the actual ``LangGraphAGUIAgent`` bridge end-to-end:

1. POST 1: ``RunAgentInput(tools=[tool_A])``. Model emits a ``tool_A``
   call. The middleware pauses on an interrupt.
2. POST 2: ``RunAgentInput(tools=[tool_A, tool_B],
   forwarded_props={"command": {"resume": {"tool_results": [...],
   "tools_after_round": [...]}}})``. Tools have been widened by
   ``tool_B``; the descriptor list is mirrored in
   ``tools_after_round`` so our middleware can pick it up despite
   ``ag_ui_langgraph`` dropping ``RunAgentInput.tools`` on the resume
   path.
3. Capture ``request.tools`` on every model call.
4. Assert: round 2's bound tool list contains ``tool_B``.
"""



import uuid

from ag_ui.core.types import Context, RunAgentInput, Tool, UserMessage
from copilotkit import LangGraphAGUIAgent
from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

from pupa_backend.harnesses.langgraph.frontend_interrupt import CustomCopilotKitMiddleware

from .conftest import MockChatModel


TOOL_A = Tool(
    name="renderTracker",
    description="render a tracker (round 1 surface)",
    parameters={"type": "object", "properties": {"title": {"type": "string"}}},
)

TOOL_B = Tool(
    name="renderChecklist",
    description="render a checklist (widened in round 2 after addComponent)",
    parameters={"type": "object", "properties": {"title": {"type": "string"}}},
)


class ToolCapturingMiddleware(AgentMiddleware):
    """Snapshot `request.tools` (by name) on every model call.

    Lets the test assert what the model actually got bound to in each
    round, independent of what was on the wire.
    """

    def __init__(self) -> None:
        super().__init__()
        self.captured: list[list[str]] = []

    async def awrap_model_call(self, request, handler):
        names = [
            (t.get("name") if isinstance(t, dict) else getattr(t, "name", None))
            for t in (request.tools or [])
        ]
        self.captured.append([n for n in names if n])
        return await handler(request)


def _build_bridge(model: MockChatModel, capturing: ToolCapturingMiddleware) -> LangGraphAGUIAgent:
    """Mirror `app.py`'s wiring: same middlewares, MemorySaver checkpointer."""
    graph = create_agent(
        model=model,
        tools=[],
        middleware=[
            CustomCopilotKitMiddleware(),
            capturing,
        ],
        checkpointer=MemorySaver(),
        name="resume_tool_refresh_test",
    )
    return LangGraphAGUIAgent(
        name="resume_tool_refresh_test",
        description="end-to-end bridge for the resume tool-refresh bug",
        graph=graph,
    )


async def _drain(stream) -> None:
    """Pull every event from the bridge so the run actually executes."""
    async for _ in stream:
        pass


async def test_resume_post_widens_tool_surface_for_next_model_call():
    thread_id = str(uuid.uuid4())
    # Round 1: model emits a frontend tool_call for renderTracker → interrupt fires.
    # Round 2 (post-resume): model wraps up with a final text reply. We do
    # NOT make the model emit any tool here — we only care about what
    # `request.tools` looked like when this 2nd call happened.
    model = MockChatModel(responses=[
        AIMessage(
            id="ai-round1",
            content="",
            tool_calls=[{
                "name": "renderTracker",
                "args": {"title": "Books"},
                "id": "call_render",
            }],
        ),
        AIMessage(id="ai-round2", content="Done."),
    ])
    capturing = ToolCapturingMiddleware()
    bridge = _build_bridge(model, capturing)

    # --- POST 1: initial run, tools=[renderTracker] -----------------------
    post1 = RunAgentInput(
        thread_id=thread_id,
        run_id=str(uuid.uuid4()),
        state={},
        messages=[UserMessage(id="u1", content="show me my books")],
        tools=[TOOL_A],
        context=[],
        forwarded_props={},
    )
    await _drain(bridge.run(post1))

    # The interrupt should be parked — the model was called exactly once so far.
    assert len(capturing.captured) == 1, capturing.captured
    assert "renderTracker" in capturing.captured[0]
    assert "renderChecklist" not in capturing.captured[0]

    # --- POST 2: resume, tools widened to [renderTracker, renderChecklist] ---
    # Mirrors what AgentSession.runLoop POSTs after dispatching the
    # frontend interrupt locally and updating its MyAppStore.
    # The descriptor list embedded in `tools_after_round` mirrors what
    # `RunAgentInput.tools` carries. `ag_ui_langgraph` drops the latter on
    # the resume branch, so the middleware reads from `tools_after_round`
    # to refresh `state["copilotkit"]["actions"]` for the next model call.
    tools_after_round_payload = [
        TOOL_A.model_dump(by_alias=True),
        TOOL_B.model_dump(by_alias=True),
    ]
    post2 = RunAgentInput(
        thread_id=thread_id,
        run_id=str(uuid.uuid4()),
        state={},
        messages=[UserMessage(id="u1", content="show me my books")],
        tools=[TOOL_A, TOOL_B],
        context=[],
        forwarded_props={
            "command": {
                "resume": {
                    "tool_results": [
                        {"toolCallId": "call_render", "content": '{"ok":true}'},
                    ],
                    "tools_after_round": tools_after_round_payload,
                },
            },
        },
    )
    await _drain(bridge.run(post2))

    # The model must have been called a second time (post-resume).
    assert len(capturing.captured) == 2, capturing.captured

    # THE CONTRACT: round-2's bound tool list reflects the widened wire
    # payload via `tools_after_round`. Without the middleware's mid-turn
    # rewrite of `state["copilotkit"]["actions"]`, this would still hold
    # round 1's `[renderTracker]` because `ag_ui_langgraph` drops
    # `RunAgentInput.tools` on the resume branch.
    assert "renderChecklist" in capturing.captured[1], (
        f"resume POST sent tools={{renderTracker, renderChecklist}} but the "
        f"model's round-2 request.tools was {capturing.captured[1]!r}. "
        "The widened surface is being dropped on the resume path."
    )
    assert "renderTracker" in capturing.captured[1]


async def test_resume_without_tools_after_round_keeps_checkpointed_actions():
    """Older clients (or any caller that omits ``tools_after_round``) must
    keep working — the middleware leaves ``state["copilotkit"]["actions"]``
    alone in that case, so the next model call sees whatever the
    checkpoint already had. Negative control on the mid-turn rewrite.
    """
    thread_id = str(uuid.uuid4())
    model = MockChatModel(responses=[
        AIMessage(
            id="ai-round1",
            content="",
            tool_calls=[{
                "name": "renderTracker",
                "args": {"title": "Books"},
                "id": "call_render",
            }],
        ),
        AIMessage(id="ai-round2", content="Done."),
    ])
    capturing = ToolCapturingMiddleware()
    bridge = _build_bridge(model, capturing)

    post1 = RunAgentInput(
        thread_id=thread_id,
        run_id=str(uuid.uuid4()),
        state={},
        messages=[UserMessage(id="u1", content="show me my books")],
        tools=[TOOL_A],
        context=[],
        forwarded_props={},
    )
    await _drain(bridge.run(post1))

    # Resume without `tools_after_round`. The middleware should leave
    # `copilotkit.actions` untouched and the model should still see the
    # round-1 surface (no crash, no widening).
    post2 = RunAgentInput(
        thread_id=thread_id,
        run_id=str(uuid.uuid4()),
        state={},
        messages=[UserMessage(id="u1", content="show me my books")],
        tools=[TOOL_A],
        context=[],
        forwarded_props={
            "command": {
                "resume": {
                    "tool_results": [
                        {"toolCallId": "call_render", "content": '{"ok":true}'},
                    ],
                },
            },
        },
    )
    await _drain(bridge.run(post2))

    assert len(capturing.captured) == 2, capturing.captured
    assert capturing.captured[1] == ["renderTracker"]
