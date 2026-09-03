"""Background work must outlive the turn that started it.

Claude Code background subagents (`Agent` + `run_in_background`) and background
shell jobs keep running inside the CLI child after the turn's `ResultMessage`.
Disposing the session there kills them and the next turn can only report them
lost. These tests pin the lifecycle that keeps them alive: park instead of
dispose, defer the injected turn the CLI runs when a task reports in, and feed
the next user turn into the *same* live client.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    TaskNotificationMessage,
    TaskStartedMessage,
    TaskUpdatedMessage,
    TextBlock,
    ToolUseBlock,
)
from fastapi import FastAPI

from pupa_backend.agui.background import BackgroundWork
from pupa_backend.harnesses.claude import endpoint as cl_endpoint
from pupa_backend.harnesses.claude import env as cl_env
from pupa_backend.harnesses.claude import registry
from pupa_backend.harnesses.claude.frontend_tools import qualified_name

THREAD = "thread-bg"


# --------------------------------------------------------------------------- #
# The tracker itself (harness-neutral)
# --------------------------------------------------------------------------- #

def test_tracker_clears_on_any_terminal_status() -> None:
    work = BackgroundWork()
    work.start("t1", "build docs")
    work.start("t2", "run tests")
    assert work.active
    assert not work.update("t1", "running")
    assert work.update("t1", "completed")
    assert work.update("t2", "killed")  # the `task_updated` vocabulary
    assert not work.active


def test_tracker_learns_a_task_it_never_saw_start() -> None:
    work = BackgroundWork()
    assert not work.update("late", "running")
    assert work.active
    assert work.update("late", "stopped")
    assert not work.active


def test_hold_is_a_wall_not_an_idle_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUPA_BACKGROUND_HOLD", "60")
    work = BackgroundWork()
    assert work.hold(now=1000.0)
    assert work.holding
    assert not work.hold_expired(now=1059.0)
    assert work.hold_expired(now=1060.0)
    work.release()
    assert not work.holding
    assert not work.hold_expired(now=99999.0)


def test_hold_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUPA_BACKGROUND_HOLD", "0")
    work = BackgroundWork()
    assert work.hold() is False
    assert not work.holding


# --------------------------------------------------------------------------- #
# Registry: the sweeper respects the hold
# --------------------------------------------------------------------------- #

async def test_sweeper_keeps_a_held_session_and_evicts_an_expired_one() -> None:
    registry._REGISTRY.clear()
    held = registry.create("thread-held")
    held.last_activity = 0.0  # ancient by the idle clock
    held.background.start("t1", "long job")
    held.background.hold()

    assert await registry.sweep_idle(timeout=1.0) == 0
    assert registry.get("thread-held") is held

    held.background.hold_until = 0.0  # wall passed
    assert await registry.sweep_idle(timeout=1.0) == 1
    assert registry.get("thread-held") is None


# --------------------------------------------------------------------------- #
# End-to-end through the endpoint, with a fake CLI child
# --------------------------------------------------------------------------- #

def _task_started(task_id: str, description: str) -> TaskStartedMessage:
    return TaskStartedMessage(
        subtype="task_started",
        data={},
        task_id=task_id,
        description=description,
        uuid="u-1",
        session_id="sdk-bg",
    )


def _task_done(task_id: str) -> TaskNotificationMessage:
    return TaskNotificationMessage(
        subtype="task_notification",
        data={},
        task_id=task_id,
        status="completed",
        output_file="",
        summary="done",
        uuid="u-2",
        session_id="sdk-bg",
    )


def _assistant(text: str, message_id: str) -> AssistantMessage:
    return AssistantMessage(
        content=[TextBlock(text=text)], model="fake",
        message_id=message_id, session_id="sdk-bg",
    )


def _result(origin=None, is_error: bool = False) -> ResultMessage:
    return ResultMessage(
        subtype="success", duration_ms=1, duration_api_ms=1,
        is_error=is_error, num_turns=1, session_id="sdk-bg", origin=origin,
    )


class _FakeSDKClient:
    """A CLI child whose message stream spans every turn, like the real one.

    `script` is a list of per-`query()` message lists: each turn the endpoint
    sends, the client pushes that turn's messages onto the stream. The test can
    also `inject()` messages between turns — that is what a background task
    reporting in looks like.
    """

    instances: list["_FakeSDKClient"] = []
    script: list[list] = []

    def __init__(self, options=None, transport=None):
        self.options = options
        self.queries: list = []
        self.disconnected = False
        self.inbox: asyncio.Queue = asyncio.Queue()
        _FakeSDKClient.instances.append(self)

    async def connect(self, prompt=None):
        return None

    async def query(self, prompt, session_id="default"):
        self.queries.append(prompt)
        turn = _FakeSDKClient.script.pop(0) if _FakeSDKClient.script else []
        for msg in turn:
            await self.inbox.put(msg)

    async def interrupt(self):
        return None

    async def disconnect(self):
        self.disconnected = True

    def inject(self, *messages) -> None:
        for msg in messages:
            self.inbox.put_nowait(msg)

    async def receive_messages(self):
        while True:
            yield await self.inbox.get()


def _sse(text: str) -> list[dict]:
    return [
        json.loads(line[len("data: "):])
        for line in text.splitlines()
        if line.startswith("data: ")
    ]


def _body(run_id: str, text: str, tools=None, state=None) -> dict:
    return {
        "thread_id": THREAD,
        "run_id": run_id,
        "messages": [{"id": run_id + "-u", "role": "user", "content": text}],
        "tools": tools or [],
        "state": state or {},
        "context": [],
        "forwardedProps": {},
    }


@pytest.fixture
def loop_app(monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    for var in cl_env.FORBIDDEN_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(
        cl_endpoint, "assert_subscription_billing", lambda: {"authMethod": "oauth_token"}
    )
    monkeypatch.setattr(cl_endpoint, "ClaudeSDKClient", _FakeSDKClient)
    monkeypatch.setenv("PUPA_CLAUDE_RETIRE_DRAIN", "0.05")  # keep the suite snappy
    _FakeSDKClient.instances = []
    _FakeSDKClient.script = []
    registry._REGISTRY.clear()
    registry._SESSION_IDS.clear()
    app = FastAPI()
    cl_endpoint.register_claude_loop_endpoint(app)
    return app


async def _settle(predicate, timeout: float = 2.0) -> None:
    """Yield to the pump until `predicate()` holds."""
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        assert asyncio.get_running_loop().time() < deadline, "pump never got there"
        await asyncio.sleep(0)


async def test_turn_parks_instead_of_disposing_when_a_task_is_in_flight(loop_app) -> None:
    _FakeSDKClient.script = [[
        _task_started("bg-1", "sleep then write"),
        _assistant("launched", "m-1"),
        _result(),
    ]]
    transport = httpx.ASGITransport(app=loop_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        types = [e["type"] for e in _sse((await client.post("/", json=_body("run-1", "go"))).text)]

    assert types[0] == "RUN_STARTED"
    assert types[-1] == "RUN_FINISHED"  # the user's run still ends cleanly
    session = registry.get(THREAD)
    assert session is not None, "the session was disposed — background work would be killed"
    assert session.background.active
    assert session.background.holding
    assert _FakeSDKClient.instances[0].disconnected is False


async def test_injected_turn_is_deferred_then_delivered_on_the_next_run(loop_app) -> None:
    _FakeSDKClient.script = [
        [_task_started("bg-1", "sleep then write"), _assistant("launched", "m-1"), _result()],
        [_assistant("all done", "m-3"), _result()],
    ]
    transport = httpx.ASGITransport(app=loop_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/", json=_body("run-1", "go"))
        session = registry.get(THREAD)
        assert session is not None
        child = _FakeSDKClient.instances[0]

        # The background task reports in: the CLI runs a turn nobody asked for.
        child.inject(
            _assistant("the background agent finished", "m-2"),
            _task_done("bg-1"),
            _result(origin={"kind": "task-notification"}),
        )
        await _settle(lambda: bool(session.deferred) and not session.background.active)

        # No terminal event was emitted for it — nothing was listening.
        assert session.queue.qsize() == 0
        assert any(
            getattr(e, "delta", None) == "the background agent finished"
            or getattr(e, "content", None) == "the background agent finished"
            for e in session.deferred
        ), session.deferred

        r2 = await client.post("/", json=_body("run-2", "what happened?"))
        events = _sse(r2.text)

    types = [e["type"] for e in events]
    assert types[0] == "RUN_STARTED"
    assert types[-1] == "RUN_FINISHED"
    text = "".join(e.get("delta", "") for e in events if e["type"] == "TEXT_MESSAGE_CONTENT")
    assert "the background agent finished" in text  # the deferred turn rode run 2
    assert "all done" in text
    # Same child: reusing it is what kept the task alive.
    assert len(_FakeSDKClient.instances) == 1
    assert len(_FakeSDKClient.instances[0].queries) == 2
    # Task terminal + turn finished cleanly → the session is disposable again.
    assert registry.get(THREAD) is None


async def test_a_turn_needing_new_tools_starts_fresh_rather_than_reusing(loop_app) -> None:
    """A widened tool surface can't run on the frozen live client — start fresh
    (and say so) rather than pretend the background work survives."""
    _FakeSDKClient.script = [
        [_task_started("bg-1", "long job"), _assistant("launched", "m-1"), _result()],
        [_assistant("fresh", "m-2"), _result()],
    ]
    transport = httpx.ASGITransport(app=loop_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/", json=_body("run-1", "go"))
        tools = [{"name": "renderTracker", "description": "r", "parameters": {"type": "object"}}]
        r2 = await client.post("/", json=_body("run-2", "now render", tools=tools))
        assert "RUN_FINISHED" in [e["type"] for e in _sse(r2.text)]

    assert len(_FakeSDKClient.instances) == 2
    assert _FakeSDKClient.instances[0].disconnected is True


async def test_scope_change_starts_fresh(loop_app) -> None:
    _FakeSDKClient.script = [
        [_task_started("bg-1", "long job"), _assistant("launched", "m-1"), _result()],
        [_assistant("fresh", "m-2"), _result()],
    ]
    transport = httpx.ASGITransport(app=loop_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/", json=_body("run-1", "go", state={"claude_loop_native": "full"}))
        r2 = await client.post(
            "/", json=_body("run-2", "again", state={"claude_loop_native": "read"})
        )
        assert "RUN_FINISHED" in [e["type"] for e in _sse(r2.text)]

    assert len(_FakeSDKClient.instances) == 2


async def test_frontend_call_from_an_injected_turn_is_rejected_not_parked(loop_app) -> None:
    """The device isn't listening between runs: fail the call instead of hanging
    the loop on a result that can never arrive."""
    _FakeSDKClient.script = [[
        _task_started("bg-1", "long job"), _assistant("launched", "m-1"), _result(),
    ]]
    transport = httpx.ASGITransport(app=loop_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post(
            "/",
            json=_body(
                "run-1", "go",
                tools=[{"name": "renderTracker", "description": "r", "parameters": {"type": "object"}}],
            ),
        )
    session = registry.get(THREAD)
    assert session is not None
    _FakeSDKClient.instances[0].inject(
        AssistantMessage(
            content=[ToolUseBlock(id="call-bg", name=qualified_name("renderTracker"), input={})],
            model="fake", message_id="m-2", session_id="sdk-bg",
        ),
    )
    await _settle(lambda: "call-bg" in session.pending and not session.has_unresolved_pending())

    assert json.loads(session.pending["call-bg"].result) == {
        "ok": False, "error": "app_not_attached",
    }
    assert registry.INTERRUPT not in list(session.queue._queue)


async def test_no_background_work_still_disposes(loop_app) -> None:
    """The ordinary turn is unchanged: nothing to hold open, session goes away."""
    _FakeSDKClient.script = [[_assistant("hi", "m-1"), _result()]]
    transport = httpx.ASGITransport(app=loop_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/", json=_body("run-1", "hello"))
    assert registry.get(THREAD) is None


async def test_errored_turn_does_not_park_on_background_work(loop_app) -> None:
    """A wedged child must not be kept alive on the strength of tasks it may
    never report."""
    _FakeSDKClient.script = [[
        _task_started("bg-1", "long job"),
        _result(is_error=True),
    ]]
    transport = httpx.ASGITransport(app=loop_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        types = [e["type"] for e in _sse((await client.post("/", json=_body("run-1", "go"))).text)]
    assert "RUN_ERROR" in types
    assert registry.get(THREAD) is None


async def test_task_updated_patch_status_is_terminal(loop_app) -> None:
    """A background task's terminal state can arrive only as `task_updated`."""
    _FakeSDKClient.script = [[
        _task_started("bg-1", "long job"), _assistant("launched", "m-1"), _result(),
    ]]
    transport = httpx.ASGITransport(app=loop_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/", json=_body("run-1", "go"))
    session = registry.get(THREAD)
    assert session is not None and session.background.active
    _FakeSDKClient.instances[0].inject(
        TaskUpdatedMessage(
            subtype="task_updated", data={}, task_id="bg-1",
            patch={"status": "killed", "end_time": 1}, status=None, session_id="sdk-bg",
        )
    )
    await _settle(lambda: not session.background.active)

    # A trailing patch with no status at all must not read as "running again".
    _FakeSDKClient.instances[0].inject(
        TaskUpdatedMessage(
            subtype="task_updated", data={}, task_id="bg-1",
            patch={"end_time": 2}, status=None, session_id="sdk-bg",
        )
    )
    for _ in range(50):
        await asyncio.sleep(0)
    assert not session.background.active


# --------------------------------------------------------------------------- #
# Regressions from the first review round
# --------------------------------------------------------------------------- #

async def test_permission_prompt_between_runs_is_deferred_not_queued(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gate must route like the pump does.

    Emitting straight onto the queue with no run open puts the prompt — and an
    `INTERRUPT` sentinel — ahead of the next run's `RunStarted`, so that run's
    SSE starts mid-message and ends on the previous run's id.
    """
    from pupa_backend.harnesses.claude import gate

    monkeypatch.setenv("PUPA_CLAUDE_LOOP_NATIVE", "edit")
    monkeypatch.setenv("PUPA_CLAUDE_LOOP_REQUIRE_APPROVAL", "1")
    monkeypatch.delenv("PUPA_CLAUDE_LOOP_AUTO_APPROVE", raising=False)
    session = registry.LiveSession(thread_id="t-perm-bg")
    session.current_run_id = "r1"
    session.run_open = False  # held open for background work, no SSE attached
    hook = gate.make_pre_tool_use_hook({}, session)
    task = asyncio.ensure_future(
        hook({"tool_name": "Bash", "tool_input": {"command": "ls"}}, "tid", None)
    )
    await _settle(lambda: session.pending_decision is not None)

    assert session.queue.qsize() == 0, "the prompt jumped ahead of the next RunStarted"
    assert len(session.deferred) == 3  # start / content / end
    session.pending_decision.set_result(False)
    await asyncio.wait_for(task, timeout=1)


def test_a_status_less_patch_does_not_resurrect_a_finished_task() -> None:
    """`task_updated` may carry only `end_time`; the SDK parses that as
    `status=None`. Reading absence-of-status as "running" pins the session."""
    work = BackgroundWork()
    work.start("bg-1", "job")
    assert work.update("bg-1", "completed")
    assert not work.active
    assert not work.update("bg-1", None)
    work.start("bg-1", "job")  # a late `task_started` echo must not revive it
    assert not work.active


async def test_model_change_starts_fresh(loop_app) -> None:
    """A live client is frozen on its model — honour the user's new pick."""
    _FakeSDKClient.script = [
        [_task_started("bg-1", "long job"), _assistant("launched", "m-1"), _result()],
        [_assistant("fresh", "m-2"), _result()],
    ]
    transport = httpx.ASGITransport(app=loop_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first = _body("run-1", "go")
        first["forwardedProps"] = {"llm": {"model": "haiku"}}
        await client.post("/", json=first)
        second = _body("run-2", "again")
        second["forwardedProps"] = {"llm": {"model": "opus"}}
        await client.post("/", json=second)

    assert len(_FakeSDKClient.instances) == 2
    assert _FakeSDKClient.instances[1].options.model == "opus"


async def test_a_continued_turn_re_delivers_changed_ambient_context(loop_app) -> None:
    """The live client's system prompt froze last turn's context, so the new one
    has to ride the query or the model answers on a stale snapshot."""
    _FakeSDKClient.script = [
        [_task_started("bg-1", "long job"), _assistant("launched", "m-1"), _result()],
        [_assistant("ok", "m-2"), _result()],
    ]
    transport = httpx.ASGITransport(app=loop_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first = _body("run-1", "go")
        first["context"] = [{"description": "canvas", "value": "one box"}]
        await client.post("/", json=first)
        second = _body("run-2", "and now?")
        second["context"] = [{"description": "canvas", "value": "two boxes"}]
        await client.post("/", json=second)

    assert len(_FakeSDKClient.instances) == 1  # continued on the live client
    sent = _FakeSDKClient.instances[0].queries[1]
    assert "two boxes" in sent
    assert sent.endswith("and now?")


async def test_declining_to_continue_still_delivers_the_background_report(loop_app) -> None:
    """The held output is the whole point — it must not die with the session."""
    _FakeSDKClient.script = [
        [_task_started("bg-1", "long job"), _assistant("launched", "m-1"), _result()],
        [_assistant("fresh", "m-2"), _result()],
    ]
    transport = httpx.ASGITransport(app=loop_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/", json=_body("run-1", "go"))
        session = registry.get(THREAD)
        assert session is not None
        _FakeSDKClient.instances[0].inject(
            _assistant("BACKGROUND REPORT: found 3 bugs", "m-bg"),
            _task_done("bg-1"),
            _result(origin={"kind": "task-notification"}),
        )
        await _settle(lambda: bool(session.deferred))

        # A tool the frozen client can't expose forces a fresh session.
        tools = [{"name": "renderTracker", "description": "r", "parameters": {"type": "object"}}]
        events = _sse((await client.post("/", json=_body("run-2", "now render", tools=tools))).text)

    assert len(_FakeSDKClient.instances) == 2
    types = [e["type"] for e in events]
    assert types[0] == "RUN_STARTED"
    text = "".join(e.get("delta", "") for e in events if e["type"] == "TEXT_MESSAGE_CONTENT")
    assert "BACKGROUND REPORT: found 3 bugs" in text


async def test_resume_post_on_a_background_held_session_does_not_hang(loop_app) -> None:
    """Nothing is parked on a tool call, so attaching would block on a queue the
    idle pump never feeds. Answer like any stale resume."""
    _FakeSDKClient.script = [[
        _task_started("bg-1", "long job"), _assistant("launched", "m-1"), _result(),
    ]]
    transport = httpx.ASGITransport(app=loop_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/", json=_body("run-1", "go"))
        resume = _body("run-2", "")
        resume["forwardedProps"] = {
            "command": {"resume": {"tool_results": [{"toolCallId": "nope", "content": "x"}]}}
        }
        events = await asyncio.wait_for(
            client.post("/", json=resume), timeout=5
        )

    types = [e["type"] for e in _sse(events.text)]
    assert types == ["RUN_ERROR"]
    assert registry.get(THREAD) is not None  # the hold survives the stray resume


async def test_pruned_rejects_do_not_poison_a_later_identical_call() -> None:
    """A rejected between-runs call keeps an `app_not_attached` result nobody
    consumed; `claim_call` matches on (name, args), so the next real call would
    take it and the device's answer would never be seen."""
    session = registry.LiveSession(thread_id="t-prune")
    await session.register_pending("call-bg", "renderTracker", {}, run_id=None)
    await session.reject_pending(["call-bg"], "app_not_attached")

    session.open_run("run-2", object())
    assert "call-bg" not in session.pending

    await session.register_pending("call-user", "renderTracker", {}, run_id="run-2")
    await session.resolve_results([{"toolCallId": "call-user", "content": "real result"}])
    got = await asyncio.wait_for(session.claim_call("renderTracker", {}, timeout=1), timeout=2)
    assert got["content"][0]["text"] == "real result"


def test_deferred_trim_keeps_a_frame_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dropping the oldest events must not leave the backlog starting on a
    `TEXT_MESSAGE_CONTENT` whose `START` was trimmed away."""
    from ag_ui.core import EventType
    from ag_ui.core.events import (
        TextMessageContentEvent,
        TextMessageEndEvent,
        TextMessageStartEvent,
    )

    monkeypatch.setattr(registry, "_MAX_DEFERRED", 4)
    session = registry.LiveSession(thread_id="t-trim")
    for i in range(3):
        session.defer(TextMessageStartEvent(
            type=EventType.TEXT_MESSAGE_START, message_id=f"m{i}", role="assistant"))
        session.defer(TextMessageContentEvent(
            type=EventType.TEXT_MESSAGE_CONTENT, message_id=f"m{i}", delta="x"))
        session.defer(TextMessageEndEvent(
            type=EventType.TEXT_MESSAGE_END, message_id=f"m{i}"))

    assert session.deferred, "everything was trimmed away"
    assert session.deferred[0].type == EventType.TEXT_MESSAGE_START


async def test_harness_sweep_evicts_an_expired_hold() -> None:
    """`sweep_idle` runs on a timer, not only when the next turn happens to
    arrive — otherwise the hold is not a bound at all."""
    from pupa_backend.harnesses import ClaudeCodeHarness, sweep_harnesses

    registry._REGISTRY.clear()
    session = registry.create("thread-expired")
    session.background.start("t1", "job")
    session.background.hold_until = 0.0

    class _Registry:
        def enabled(self):
            return [ClaudeCodeHarness()]

    task = asyncio.ensure_future(sweep_harnesses(_Registry(), interval=0.01))
    try:
        await _settle(lambda: registry.get("thread-expired") is None, timeout=3.0)
    finally:
        task.cancel()
