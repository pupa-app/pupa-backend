"""Repros for two frontend-tool round-trip defects (see GH issue).

Both target the real `registry.LiveSession` park/resume machinery used by the
claude_loop harness.

1. **Dead-client 300s wall** — a handler parked in `claim_call` only ends on
   wall-clock timeout or an explicit `dispose()`. A vanished SSE/HTTP client is
   invisible, so the handler waits the full timeout (300s in prod). `dispose()`
   *does* unblock fast — the fix is to wire client-disconnect to it.

2. **Partial resume clobbers still-pending siblings** — `resolve_results`
   synthesises `missing_tool_result` for EVERY unresolved slot in the session,
   not just the run being resumed. When parallel frontend calls come back
   incrementally (slow subagents), a resume carrying the finished ones nukes the
   siblings that are still legitimately in flight.
"""

from __future__ import annotations

import asyncio
import json

from pupa_backend.harnesses.claude import registry


async def test_parked_handler_has_no_fast_fail_on_dead_client() -> None:
    """Repro #1: a parked `claim_call` waits the FULL timeout; nothing detects a
    gone client. `dispose()` is the only fast path and it is NOT wired to a
    client disconnect today."""
    loop = asyncio.get_event_loop()
    session = registry.LiveSession(thread_id="t-deadclient")
    await session.register_pending("c1", "addTrackerItems", {"x": 1})

    start = loop.time()
    raised = False
    try:
        # No resume, no dispose — simulate the client vanishing mid-call.
        await session.claim_call("addTrackerItems", {"x": 1}, timeout=0.4)
    except asyncio.TimeoutError as e:
        raised = True
        assert "no frontend tool result delivered" in str(e)
    elapsed = loop.time() - start

    assert raised, "expected a TimeoutError"
    # The defect: it burns the whole wall (~0.4s here, 300s in prod) with no
    # earlier exit even though the client is gone.
    assert elapsed >= 0.4, f"expected full-timeout wall, got {elapsed:.3f}s"


async def test_dispose_unblocks_parked_handler_fast() -> None:
    """Shows the fast-fail mechanism EXISTS (dispose) — the fix for #1 is to
    trigger it on disconnect. Parked handler returns immediately once disposed,
    long before its timeout."""
    loop = asyncio.get_event_loop()
    session = registry.LiveSession(thread_id="t-dispose")
    await session.register_pending("c1", "addTrackerItems", {"x": 1})

    task = asyncio.ensure_future(
        session.claim_call("addTrackerItems", {"x": 1}, timeout=30.0)
    )
    await asyncio.sleep(0.02)
    assert not task.done()

    start = loop.time()
    await session.dispose()
    result = await asyncio.wait_for(task, timeout=1.0)
    elapsed = loop.time() - start

    payload = json.loads(result["content"][0]["text"])
    assert payload == {"ok": False, "error": "session_disposed"}
    assert elapsed < 0.5, "dispose should unblock the handler promptly"


async def test_resume_does_not_clobber_other_runs_pending_call() -> None:
    """Repro #2 (fixed): a resume for run r2 must not synth-error a call still in
    flight from run r1. Previously `resolve_results` swept the whole session and
    filled r1's unresolved slot with `missing_tool_result`."""
    session = registry.LiveSession(thread_id="t-crossrun")
    # r1 emitted a call that is still legitimately in flight on-device.
    await session.register_pending("call_A", "invoke_agent", {"prompt": "A"}, run_id="r1")
    # r2 emitted its own call and the client answers r2's batch.
    await session.register_pending("call_B", "invoke_agent", {"prompt": "B"}, run_id="r2")
    await session.resolve_results([{"toolCallId": "call_B", "content": "resultB"}])

    pc_a = session.pending["call_A"]
    assert pc_a.result is registry._UNSET, (
        f"r1's still-pending call was clobbered by r2's resume: {pc_a.result!r}"
    )

    # r1's own resume later delivers the real result; the handler gets it.
    await session.resolve_results([{"toolCallId": "call_A", "content": "resultA"}])
    result_a = await asyncio.wait_for(
        session.claim_call("invoke_agent", {"prompt": "A"}, timeout=1.0), timeout=1.0
    )
    assert result_a == {"content": [{"type": "text", "text": "resultA"}]}


async def test_same_batch_dropped_call_still_synthesised() -> None:
    """Guardrail: within ONE batch (same run), a call the client omits from its
    resume IS genuinely dropped and must still be synth-errored so its handler
    never hangs."""
    session = registry.LiveSession(thread_id="t-samebatch")
    await session.register_pending("call_A", "invoke_agent", {"prompt": "A"}, run_id="r1")
    await session.register_pending("call_B", "invoke_agent", {"prompt": "B"}, run_id="r1")
    # Client answers only A; B was dropped from the same batch.
    await session.resolve_results([{"toolCallId": "call_A", "content": "resultA"}])

    result_b = await asyncio.wait_for(
        session.claim_call("invoke_agent", {"prompt": "B"}, timeout=1.0), timeout=1.0
    )
    payload = json.loads(result_b["content"][0]["text"])
    assert payload == {"ok": False, "error": "missing_tool_result"}


async def test_wait_timeout_configurable_via_env(monkeypatch) -> None:
    """Fix #1 (partial): the park wall is env-tunable so an abandoned turn can
    fail sooner than the default without touching slow-tool headroom."""
    loop = asyncio.get_event_loop()
    monkeypatch.setenv("PUPA_FRONTEND_WAIT_TIMEOUT", "0.3")
    session = registry.LiveSession(thread_id="t-envtimeout")
    await session.register_pending("c1", "addTrackerItems", {"x": 1}, run_id="r1")

    start = loop.time()
    try:
        await session.claim_call("addTrackerItems", {"x": 1})  # no explicit timeout
    except asyncio.TimeoutError:
        pass
    elapsed = loop.time() - start
    assert 0.3 <= elapsed < 1.0, f"env timeout not honoured: {elapsed:.3f}s"


def test_per_tool_wait_budget() -> None:
    """Per-tool wall mechanism is wired (fast vs slow buckets + knobs). Defaults
    stay generous (300s) until the liveness heartbeat makes shortening
    safe; env knobs let ops tune each bucket meanwhile."""
    assert registry.wait_timeout_for("addTrackerItems") == registry._FAST_WAIT_DEFAULT
    assert registry.wait_timeout_for("get_tools_tracker") == registry._FAST_WAIT_DEFAULT
    assert registry.wait_timeout_for("addCalendarEvent") == registry._FAST_WAIT_DEFAULT
    assert registry.wait_timeout_for("invoke_agent") == registry._SLOW_WAIT_DEFAULT
    assert registry._FAST_WAIT_DEFAULT <= registry._SLOW_WAIT_DEFAULT


def test_per_tool_wait_budget_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("PUPA_FRONTEND_WAIT_TIMEOUT", "12")
    monkeypatch.setenv("PUPA_FRONTEND_WAIT_TIMEOUT_SLOW", "600")
    assert registry.wait_timeout_for("addTrackerItems") == 12.0
    assert registry.wait_timeout_for("invoke_agent") == 600.0
