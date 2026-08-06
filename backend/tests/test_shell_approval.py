"""Contracts for ShellApprovalMiddleware.

Four contracts:

1. **Interrupt fires** — a shell call from an unapproved command pauses the
   graph with a ``request_shell_approval`` frontend_tool_calls payload.

2. **Approve once** — resuming with ``approved=True, remember=False`` lets
   the shell command run; the same command triggers another interrupt on
   the next turn.

3. **Approve + remember** — ``remember=True`` adds the command to the
   in-memory allowlist; the second turn skips the interrupt entirely.

4. **Deny** — ``approved=False`` injects a "denied" ToolMessage and
   the shell command never executes.

Thread-id isolation is implicit: each test uses a distinct
``thread_id`` so remembered approvals from one scenario cannot bleed
into another.
"""



import json
import os
import pytest
import asyncio

os.environ.setdefault("SHELL_TOOL_ENABLED", "1")

from langchain.agents import create_agent
from langchain.agents.middleware.shell_tool import ShellToolMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool as lc_tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from pupa_backend.harnesses.langgraph.shell_approval import ShellApprovalMiddleware

from .conftest import MockChatModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@lc_tool
def shell(command: str) -> str:
    """Execute a shell command (mock — returns a fixed string)."""
    return f"ok: {command}"


def _approval_middleware() -> ShellApprovalMiddleware:
    return ShellApprovalMiddleware()


def _build_agent(model: MockChatModel, approval_mw: ShellApprovalMiddleware):
    return create_agent(
        model=model,
        tools=[shell],
        middleware=[approval_mw, ShellToolMiddleware()],
        checkpointer=MemorySaver(),
    )


def _approval_resume(tc_id: str, *, approved: bool, remember: bool) -> Command:
    return Command(resume={
        "tool_results": [
            {
                "toolCallId": tc_id,
                "content": json.dumps({"approved": approved, "remember": remember}),
            }
        ]
    })


async def _collect_interrupt(agent, messages, config) -> dict | None:
    """Run agent until the first interrupt; return the interrupt value or None."""
    async for ev in agent.astream({"messages": messages}, config):
        if "__interrupt__" in ev:
            return ev["__interrupt__"][0].value
    return None


async def _collect_tool_messages(agent, command: Command, config) -> list[ToolMessage]:
    """Resume agent and collect ToolMessages from the tools node."""
    tool_msgs: list[ToolMessage] = []
    async for ev in agent.astream(command, config):
        for msg in ev.get("tools", {}).get("messages", []):
            if isinstance(msg, ToolMessage):
                tool_msgs.append(msg)
    return tool_msgs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_interrupt_fires_for_unapproved_command():
    """shell call on an unknown command pauses with request_shell_approval."""
    model = MockChatModel(responses=[
        AIMessage(content="", tool_calls=[
            {"id": "tc1", "name": "shell", "args": {"command": "ls"}, "type": "tool_call"}
        ]),
        AIMessage(content="done"),
    ])
    mw = _approval_middleware()
    agent = _build_agent(model, mw)
    config = {"configurable": {"thread_id": "t-interrupt"}}

    iv = await _collect_interrupt(agent, [HumanMessage(content="ls")], config)

    assert iv is not None, "expected an interrupt"
    assert "frontend_tool_calls" in iv
    calls = iv["frontend_tool_calls"]
    assert len(calls) == 1
    assert calls[0]["name"] == "request_shell_approval"
    assert calls[0]["args"]["command"] == "ls"


@pytest.mark.asyncio
async def test_approve_once_executes_then_interrupts_again():
    """approve-once runs the command but the next call interrupts again."""
    tc_id = "tc-once"
    model = MockChatModel(responses=[
        AIMessage(content="", tool_calls=[
            {"id": tc_id, "name": "shell", "args": {"command": "pwd"}, "type": "tool_call"}
        ]),
        AIMessage(content="done turn 1"),
        AIMessage(content="", tool_calls=[
            {"id": "tc-once-2", "name": "shell", "args": {"command": "pwd"}, "type": "tool_call"}
        ]),
        AIMessage(content="done turn 2"),
    ])
    mw = _approval_middleware()
    agent = _build_agent(model, mw)
    config = {"configurable": {"thread_id": "t-once"}}

    # Turn 1: interrupt, then approve once
    await _collect_interrupt(agent, [HumanMessage(content="pwd")], config)
    await _collect_tool_messages(agent, _approval_resume(tc_id, approved=True, remember=False), config)

    # Turn 2: same command → should interrupt again (not pre-approved)
    iv2 = await _collect_interrupt(agent, [HumanMessage(content="pwd again")], config)
    assert iv2 is not None, "expected second interrupt for approve-once command"


@pytest.mark.asyncio
async def test_approve_remember_skips_subsequent_interrupt():
    """approve + remember stores the command; next call bypasses interrupt."""
    tc_id = "tc-rem"
    model = MockChatModel(responses=[
        AIMessage(content="", tool_calls=[
            {"id": tc_id, "name": "shell", "args": {"command": "date"}, "type": "tool_call"}
        ]),
        AIMessage(content="done turn 1"),
        AIMessage(content="", tool_calls=[
            {"id": "tc-rem-2", "name": "shell", "args": {"command": "date"}, "type": "tool_call"}
        ]),
        AIMessage(content="done turn 2"),
    ])
    mw = _approval_middleware()
    agent = _build_agent(model, mw)
    config = {"configurable": {"thread_id": "t-remember"}}

    # Turn 1: approve + remember
    await _collect_interrupt(agent, [HumanMessage(content="date")], config)
    msgs1 = await _collect_tool_messages(agent, _approval_resume(tc_id, approved=True, remember=True), config)
    assert any("ok: date" in m.content for m in msgs1), "shell should have executed"

    # Turn 2: same command → no interrupt, shell runs directly
    iv2 = await _collect_interrupt(agent, [HumanMessage(content="date again")], config)
    assert iv2 is None, "second call should not interrupt — command was remembered"


@pytest.mark.asyncio
async def test_deny_injects_refusal_tool_message():
    """deny returns a ToolMessage with denial content; shell never executes."""
    tc_id = "tc-deny"
    model = MockChatModel(responses=[
        AIMessage(content="", tool_calls=[
            {"id": tc_id, "name": "shell", "args": {"command": "rm -rf /"}, "type": "tool_call"}
        ]),
        AIMessage(content="ok, sorry"),
    ])
    mw = _approval_middleware()
    agent = _build_agent(model, mw)
    config = {"configurable": {"thread_id": "t-deny"}}

    await _collect_interrupt(agent, [HumanMessage(content="dangerous")], config)
    tool_msgs = await _collect_tool_messages(
        agent, _approval_resume(tc_id, approved=False, remember=False), config
    )

    assert tool_msgs, "expected a ToolMessage from the denial"
    content = tool_msgs[0].content.lower()
    assert "denied" in content
    assert "at this time" in content, "denial should signal it's not a permanent block"
    assert "rm -rf /" not in tool_msgs[0].content  # shell never ran


@pytest.mark.asyncio
async def test_client_can_disable_approval_per_turn():
    """state[shell_approval_disabled]=True bypasses the interrupt entirely."""
    tc_id = "tc-disabled"
    model = MockChatModel(responses=[
        AIMessage(content="", tool_calls=[
            {"id": tc_id, "name": "shell", "args": {"command": "id"}, "type": "tool_call"}
        ]),
        AIMessage(content="done"),
    ])
    mw = _approval_middleware()
    agent = _build_agent(model, mw)
    config = {"configurable": {"thread_id": "t-disabled"}}

    # Run with shell_approval_disabled=True — middleware should pass through
    iv = None
    async for ev in agent.astream(
        {"messages": [HumanMessage(content="id")], "shell_approval_disabled": True},
        config,
    ):
        if "__interrupt__" in ev:
            iv = ev["__interrupt__"][0].value
            break

    assert iv is None, "interrupt should NOT fire when shell_approval_disabled=True"


@pytest.mark.asyncio
async def test_remember_is_isolated_per_thread():
    """allowlist remembered in thread A does not affect thread B."""
    tc_id = "tc-iso"
    cmd = "whoami"

    def make_model():
        return MockChatModel(responses=[
            AIMessage(content="", tool_calls=[
                {"id": tc_id, "name": "shell", "args": {"command": cmd}, "type": "tool_call"}
            ]),
            AIMessage(content="done"),
        ])

    mw = _approval_middleware()  # shared middleware instance (as in production)

    # Thread A: approve + remember
    agent_a = _build_agent(make_model(), mw)
    cfg_a = {"configurable": {"thread_id": "t-iso-A"}}
    await _collect_interrupt(agent_a, [HumanMessage(content=cmd)], cfg_a)
    await _collect_tool_messages(agent_a, _approval_resume(tc_id, approved=True, remember=True), cfg_a)
    assert cmd in mw._approved.get("t-iso-A", set()), "should be in A's allowlist"

    # Thread B: same command → still interrupts (not in B's allowlist)
    agent_b = _build_agent(make_model(), mw)
    cfg_b = {"configurable": {"thread_id": "t-iso-B"}}
    iv = await _collect_interrupt(agent_b, [HumanMessage(content=cmd)], cfg_b)
    assert iv is not None, "thread B should still interrupt — its allowlist is empty"


@pytest.mark.asyncio
async def test_parallel_shell_calls_produce_single_batched_interrupt():
    """Two shell calls in one turn → ONE interrupt listing both; resuming both works.

    Regression test for the RuntimeError raised by LangGraph when multiple
    ``interrupt()`` calls fire in parallel (one per ``awrap_tool_call`` task)
    and the client resumes without specifying an ``interrupt_id``.

    The fix moves shell-approval interrupts to ``after_model``, which fires once
    per model turn and can batch all pending shell calls into a single interrupt.
    """
    model = MockChatModel(responses=[
        AIMessage(content="", tool_calls=[
            {"id": "tc-par1", "name": "shell", "args": {"command": "ls"}, "type": "tool_call"},
            {"id": "tc-par2", "name": "shell", "args": {"command": "pwd"}, "type": "tool_call"},
        ]),
        AIMessage(content="done"),
    ])
    mw = _approval_middleware()
    agent = _build_agent(model, mw)
    config = {"configurable": {"thread_id": "t-parallel"}}

    iv = await _collect_interrupt(agent, [HumanMessage(content="ls and pwd")], config)

    assert iv is not None, "expected an interrupt"
    assert "frontend_tool_calls" in iv
    calls = iv["frontend_tool_calls"]
    assert len(calls) == 2, (
        f"expected one batched interrupt with 2 approval requests, got {len(calls)}: {calls}. "
        "This likely means two separate interrupts fired (one per parallel awrap_tool_call), "
        "which would cause a RuntimeError on resume."
    )
    assert all(c["name"] == "request_shell_approval" for c in calls)
    assert {c["args"]["command"] for c in calls} == {"ls", "pwd"}

    # Resuming both in one shot must not raise RuntimeError
    resume = Command(resume={
        "tool_results": [
            {"toolCallId": "tc-par1", "content": json.dumps({"approved": True, "remember": False})},
            {"toolCallId": "tc-par2", "content": json.dumps({"approved": True, "remember": False})},
        ]
    })
    tool_msgs = await _collect_tool_messages(agent, resume, config)
    assert len(tool_msgs) == 2
    assert {m.content for m in tool_msgs} == {"ok: ls", "ok: pwd"}
