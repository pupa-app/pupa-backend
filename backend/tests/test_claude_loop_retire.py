"""A new turn must wind the parked session down, not yank its transport.

When a new-turn POST reuses a thread whose session is parked mid-tool-call,
`create()` fired `dispose()` and moved on. `dispose()` cancels the pump and
closes the SDK transport immediately, which rejects the CLI child's in-flight
`hook_0` (PreToolUse) / permission control requests:

    Error in hook callback hook_0: ... error: Stream closed
          at sendRequest (/$bunfs/root/src/entrypoints/cli.js:2876:133)

and leaves the SDK session interrupted for the next turn to resume.

`registry.retire()` is the graceful path: release the parked handlers, ask the
CLI to `interrupt()`, give the pump a bounded window to reach its
`ResultMessage`, and only then dispose. Bounded so a wedged child can't stall
the user's send.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
from fastapi import FastAPI

from pupa_backend.harnesses.claude import registry


class _FakeClient:
    """Records the teardown order the CLI child would observe."""

    def __init__(self) -> None:
        self.events: list[str] = []
        self.interrupted = asyncio.Event()

    async def interrupt(self) -> None:
        self.events.append("interrupt")
        self.interrupted.set()

    async def disconnect(self) -> None:
        self.events.append("disconnect")


def _parked_session(thread_id: str, client: _FakeClient) -> registry.LiveSession:
    session = registry.create(thread_id)
    session.client = client
    return session


# --------------------------------------------------------------------------- #
# retire(): ordering and bounds
# --------------------------------------------------------------------------- #


async def test_retire_interrupts_before_disconnecting() -> None:
    """The transport must not close until the child has been asked to stop."""
    client = _FakeClient()
    session = _parked_session("t-retire-order", client)

    async def _pump() -> None:
        await client.interrupted.wait()
        session.mark_error()

    session.pump_task = asyncio.ensure_future(_pump())

    await registry.retire("t-retire-order")

    assert client.events == ["interrupt", "disconnect"]
    assert registry.get("t-retire-order") is None


async def test_retire_waits_for_the_pump_to_settle() -> None:
    """Disconnect happens after the pump reached its terminal, not before."""
    client = _FakeClient()
    session = _parked_session("t-retire-wait", client)
    settled = False

    async def _pump() -> None:
        nonlocal settled
        await client.interrupted.wait()
        await asyncio.sleep(0.05)  # the child taking a moment to wind down
        settled = True
        session.mark_error()

    session.pump_task = asyncio.ensure_future(_pump())

    await registry.retire("t-retire-wait")

    assert settled, "retire() disconnected before the pump settled"


async def test_retire_releases_parked_tool_handlers_before_interrupting() -> None:
    """A handler blocked in `claim_call` must get a result, not a closed stream."""
    client = _FakeClient()
    session = _parked_session("t-retire-release", client)
    await session.register_pending("call-1", "lsMemories", {})

    claimed: asyncio.Future = asyncio.get_running_loop().create_future()

    async def _handler() -> None:
        claimed.set_result(await session.claim_call("lsMemories", {}, timeout=5.0))

    handler = asyncio.ensure_future(_handler())
    await asyncio.sleep(0)

    async def _pump() -> None:
        await client.interrupted.wait()
        session.mark_error()

    session.pump_task = asyncio.ensure_future(_pump())

    await registry.retire("t-retire-release")

    result = await asyncio.wait_for(claimed, timeout=1.0)
    assert "superseded" in str(result), f"handler got {result!r}, not a superseded marker"
    await handler


async def test_retire_is_bounded_when_the_pump_never_settles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wedged child must not stall the user's next send."""
    monkeypatch.setenv("PUPA_CLAUDE_RETIRE_DRAIN", "0.1")
    client = _FakeClient()
    session = _parked_session("t-retire-bounded", client)
    session.pump_task = asyncio.ensure_future(asyncio.Event().wait())

    await asyncio.wait_for(registry.retire("t-retire-bounded"), timeout=2.0)

    assert client.events == ["interrupt", "disconnect"]
    assert session.pump_task.cancelled() or session.pump_task.done()
    assert registry.get("t-retire-bounded") is None


async def test_retire_survives_a_client_that_cannot_interrupt() -> None:
    """No `interrupt()` on the client → fall through to the normal teardown."""

    class _NoInterrupt:
        def __init__(self) -> None:
            self.events: list[str] = []

        async def disconnect(self) -> None:
            self.events.append("disconnect")

    client = _NoInterrupt()
    session = _parked_session("t-retire-nointerrupt", client)
    session.pump_task = asyncio.ensure_future(asyncio.sleep(0))

    await registry.retire("t-retire-nointerrupt")

    assert client.events == ["disconnect"]
    assert registry.get("t-retire-nointerrupt") is None


async def test_retire_on_an_unknown_thread_is_a_noop() -> None:
    await registry.retire("t-retire-absent")  # must not raise


async def test_retire_on_an_already_disposed_session_is_a_noop() -> None:
    client = _FakeClient()
    session = _parked_session("t-retire-disposed", client)
    await session.dispose()

    await registry.retire("t-retire-disposed")

    assert "interrupt" not in client.events


# --------------------------------------------------------------------------- #
# Endpoint: a new-turn POST retires the parked session first
# --------------------------------------------------------------------------- #


class _RecordingSDKClient:
    """Fake `ClaudeSDKClient` that parks forever — stands in for a session waiting
    on an on-device tool result when the user sends something new."""

    instances: list["_RecordingSDKClient"] = []

    def __init__(self, options=None, transport=None):
        self.options = options
        self.events: list[str] = []
        self.parked = asyncio.Event()
        _RecordingSDKClient.instances.append(self)

    async def connect(self, prompt=None):
        return None

    async def query(self, prompt, session_id="default"):
        return None

    async def interrupt(self):
        self.events.append("interrupt")
        self.parked.set()

    async def disconnect(self):
        self.events.append("disconnect")

    # Bounded so an unfixed backend fails the assertion below instead of hanging
    # the suite: the turn winds itself up shortly after the interrupt would have
    # arrived, whether or not one did.
    park_timeout = 0.3

    async def receive_response(self):
        from claude_agent_sdk import ResultMessage

        try:
            await asyncio.wait_for(self.parked.wait(), timeout=self.park_timeout)
        except asyncio.TimeoutError:
            pass
        yield ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=True,
            result="interrupted",
            num_turns=1,
            session_id="sdk-s",
        )


def _app(monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    from pupa_backend.harnesses.claude import endpoint as cl_endpoint
    from pupa_backend.harnesses.claude import env as cl_env

    for var in cl_env.FORBIDDEN_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(
        cl_endpoint, "assert_subscription_billing", lambda: {"authMethod": "oauth_token"}
    )
    monkeypatch.setattr(cl_endpoint, "ClaudeSDKClient", _RecordingSDKClient)

    app = FastAPI()
    cl_endpoint.register_claude_loop_endpoint(app)
    return app


def _body(thread_id: str, run_id: str) -> dict:
    return {
        "thread_id": thread_id,
        "run_id": run_id,
        "messages": [{"id": run_id, "role": "user", "content": "hi"}],
        "tools": [],
        "state": {},
        "context": [],
        "forwardedProps": {},
    }


async def test_new_turn_retires_the_parked_session_gracefully(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wake-up path: a new message on a thread with a live session must
    interrupt that session before closing its transport."""
    _RecordingSDKClient.instances.clear()
    app = _app(monkeypatch)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first = asyncio.ensure_future(client.post("/", json=_body("wake-thread", "r1")))
        await asyncio.sleep(0.05)  # let the first turn park in receive_response
        assert registry.get("wake-thread") is not None

        await asyncio.wait_for(
            client.post("/", json=_body("wake-thread", "r2")), timeout=5.0
        )
        await asyncio.wait_for(first, timeout=5.0)

    parked = _RecordingSDKClient.instances[0]
    assert parked.events == ["interrupt", "disconnect"], (
        f"parked client torn down as {parked.events} — the transport closed "
        "without asking the child to stop"
    )

    await registry.remove("wake-thread")
