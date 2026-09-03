"""Per-thread live-session registry for the Claude Code agent loop.

Claude's single tool-calling loop cannot serialize mid-tool-call, so when the
model calls a *frontend* tool (executed on the iOS device) we keep the
`ClaudeSDKClient` **alive** across the two HTTP requests of the AG-UI
interrupt/resume contract. This module owns that parked state.

## Shape

`LiveSession` holds: the SDK client, the background **pump** task draining the
SDK message stream into AG-UI events, an `asyncio.Queue` those events land in, a
`pending` map of in-flight frontend calls (each a `PendingCall` with a future the
model's tool result will be resolved through), and bookkeeping (sdk session id,
last activity). A module-level `dict[thread_id, LiveSession]` is the registry.

## Frontend-call correlation (subtle)

The in-process SDK MCP tool handler receives only the call *arguments* — not the
tool_use id, and not the SDK's internal MCP request id. So we cannot key the
handler's result by call id directly. Instead the **pump** registers a
`PendingCall` per frontend `ToolUseBlock` (it has the id + input), the resume POST
fills in each call's **result slot** (keyed by id), and each handler **claims** an
unconsumed `PendingCall` matching its `(name, args)` *whose result is already
available*, blocking until it is. For two identical same-name/same-args calls the
pairing is interchangeable (the results are equivalent), so this is correct in
every distinguishable case without relying on the SDK's opaque request→handler
binding.

Crucially the handler waits on the **result**, not on a future the resume
resolves: the live SDK does not invoke the tool handler until the pump pulls past
the interrupt, which races with the resume POST. Keying on a result slot makes the
handoff order-independent — the handler works whether it runs before or after the
resume delivers results. A condition variable wakes waiters on both register and
resolve.

## Draining

`attach()` returns an async generator that drains the queue, yielding AG-UI
events until an interrupt boundary or run finish/error, then returns — ending the
SSE response while the pump (and SDK client) stay parked for the resume POST.

## Two reasons a session stays parked

1. A **frontend tool call** is out on the device (INTERRUPT) — the resume POST
   brings its result back into the same live turn.
2. **Background work** the loop started outlives the turn (PARK) — Claude Code
   background subagents keep running in the CLI child and report completion
   later through an injected turn. Disposing at turn end kills them, so the
   session stays connected and the *next* user turn is fed into it rather than
   into a fresh subprocess. Bounded by `PUPA_BACKGROUND_HOLD`; see
   `pupa_backend.agui.background`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

from ag_ui.core import EventType

from pupa_backend.agui.background import BackgroundWork

logger = logging.getLogger("uvicorn.error")


# Frontend tools that legitimately run long and so keep the generous wall. Every
# other on-device tool is sub-second CRUD (tracker/calendar/memory writes,
# addComponent, get_tools_*), so an abandoned turn on one of those should fail in
# tens of seconds, not minutes. `invoke_agent` runs a subagent sub-session that
# can take minutes; keep it on the slow wall.
_SLOW_FRONTEND_TOOLS = frozenset({"invoke_agent"})

# Defaults stay generous (300s) on purpose: while parked the SSE is closed, so a
# backgrounded-but-alive app must be able to return and deliver a parked tool
# result without the wall cutting it off. Shortening this safely needs the
# liveness heartbeat — until then, only ops may lower it via env. The
# per-tool split + knobs are kept so the heartbeat work can differentiate later.
_FAST_WAIT_DEFAULT = 300.0  # seconds — CRUD tools
_SLOW_WAIT_DEFAULT = 300.0  # seconds — subagent-class tools (invoke_agent)


def _env_timeout(var: str, fallback: float) -> float:
    raw = os.environ.get(var)
    if raw:
        try:
            val = float(raw)
            if val > 0:
                return val
        except ValueError:
            logger.warning("claude_code loop: bad %s=%r; using %s", var, raw, fallback)
    return fallback


def wait_timeout_for(name: str | None) -> float:
    """Park-wait budget for a frontend tool, by name.

    Both walls default to 300s (`PUPA_FRONTEND_WAIT_TIMEOUT` /
    `PUPA_FRONTEND_WAIT_TIMEOUT_SLOW`): while parked the SSE is already closed,
    so a backgrounded-but-alive app must be able to return and deliver a parked
    result — shortening the fast wall safely needs the liveness heartbeat. A
    parked handler ends only on this wall, an explicit `dispose()`, or the
    resume POST.
    """
    if name in _SLOW_FRONTEND_TOOLS:
        return _env_timeout("PUPA_FRONTEND_WAIT_TIMEOUT_SLOW", _SLOW_WAIT_DEFAULT)
    return _env_timeout("PUPA_FRONTEND_WAIT_TIMEOUT", _FAST_WAIT_DEFAULT)


def _default_wait_timeout() -> float:
    """Fallback wait when a caller gives no tool name (fast wall)."""
    return _env_timeout("PUPA_FRONTEND_WAIT_TIMEOUT", _FAST_WAIT_DEFAULT)


# Liveness grace: a parked handler fails this long after the client's
# last keepalive ping, decoupling dead-app detection from tool duration. Only
# applies once a client has pinged at least once this park (older clients keep
# the full wall) and is suspended while the client reports itself backgrounded.
_LIVENESS_GRACE_DEFAULT = 30.0  # seconds


def liveness_grace() -> float:
    return _env_timeout("PUPA_FRONTEND_LIVENESS_GRACE", _LIVENESS_GRACE_DEFAULT)

# Queue markers. AG-UI events ride the queue as themselves; these sentinels mark
# where an `attach()` drain should stop and hand control back to the HTTP layer.
INTERRUPT = object()  # batch of frontend calls emitted; park for the resume POST
FINISH = object()  # run completed; session is disposable
ERROR = object()  # run errored; a RunErrorEvent precedes this on the queue
# Run completed but the session must stay alive: background work the loop
# started is still in flight and will report through this same live session.
# Ends the SSE exactly like INTERRUPT, without evicting the session.
PARK = object()

# Default idle timeout before the sweeper evicts a parked session (seconds).
_DEFAULT_IDLE_TIMEOUT = 900.0

# Cap on events held between runs (a background task's injected turn). Bounds a
# session that keeps reporting while nobody ever sends another message.
_MAX_DEFERRED = 2000

# The only events the backlog trim may drop: the *body* of a frame. Dropping a
# `START` orphans every delta after it, and dropping an `END` leaves the client's
# message open forever — either is a protocol violation, not merely lossy. So an
# overflowing backlog loses its oldest text, never the structure around it.
_TRIMMABLE_EVENTS = frozenset({
    EventType.TEXT_MESSAGE_CONTENT,
    EventType.THINKING_TEXT_MESSAGE_CONTENT,
    EventType.TOOL_CALL_ARGS,
})

# Frame ids the trim dropped the `START` of, remembered so the rest of their
# deltas go too. Bounded: a frame whose `END` never arrives would otherwise stay
# forever.
_MAX_DROPPED_FRAMES = 256

# How far over the cap the backlog may run before a trim. Trimming on every
# `defer` would rebuild the list per streamed delta (and log a line per event);
# a batch keeps it amortised.
_TRIM_SLACK = 200


def _stale_pending_after() -> float:
    """How long an unclaimed pending call is kept before it is assumed abandoned.

    A margin past the longest park wall: once that has elapsed its handler has
    raised `TimeoutError`, so nobody can still be waiting on the slot.
    """
    return max(wait_timeout_for(None), wait_timeout_for("invoke_agent")) + 60.0

# Events that open and close a frame. A body event whose `START` was dropped is
# unrenderable, so the trim drops frames whole rather than by position.
_FRAME_START_EVENTS = frozenset({
    EventType.TEXT_MESSAGE_START,
    EventType.THINKING_TEXT_MESSAGE_START,
    EventType.TOOL_CALL_START,
})
_FRAME_END_EVENTS = frozenset({
    EventType.TEXT_MESSAGE_END,
    EventType.THINKING_TEXT_MESSAGE_END,
    EventType.TOOL_CALL_END,
})


def _frame_id(event: Any) -> str | None:
    """The id tying one frame's events together, or None for a standalone event."""
    return getattr(event, "message_id", None) or getattr(event, "tool_call_id", None)

# How long a new `attach()` waits for the consumer it displaced to hand the
# queue over. Bounded: a generator nobody is iterating (client gone, close not
# yet delivered) must not stall the request that replaced it.
_ATTACH_HANDOVER_TIMEOUT = 2.0


@dataclass
class _Attachment:
    """One `attach()` generator's hold on a session queue. `stop` asks it to let
    go; `done` reports that it has (set even on GeneratorExit)."""

    stop: asyncio.Event = field(default_factory=asyncio.Event)
    done: asyncio.Event = field(default_factory=asyncio.Event)

# Window a retiring session gets to wind its turn down before the transport is
# closed under it. Long enough for the CLI child to answer an `interrupt()` with
# a `ResultMessage`, short enough that a wedged child can't stall the user's next
# send. Tunable via `PUPA_CLAUDE_RETIRE_DRAIN`.
_RETIRE_DRAIN_DEFAULT = 2.0


def retire_drain_timeout() -> float:
    return _env_timeout("PUPA_CLAUDE_RETIRE_DRAIN", _RETIRE_DRAIN_DEFAULT)


def _args_key(args: Any) -> str:
    """Stable JSON key for matching handler args against a recorded ToolUseBlock."""
    try:
        return json.dumps(args, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return repr(args)


_UNSET = object()  # result-slot sentinel: "no result delivered yet"


@dataclass
class PendingCall:
    """A frontend tool call the model emitted, awaiting the on-device result.

    `result` is `_UNSET` until the resume POST delivers it; `consumed` flips once a
    handler has taken it (so two same-(name,args) calls each claim exactly one).
    """

    call_id: str
    name: str
    args: dict[str, Any]
    result: Any = _UNSET
    consumed: bool = False
    run_id: str | None = None  # HTTP run that emitted this call — scopes synth
    seq: int = 0  # the session's run counter when registered — see `prune_pending`
    registered_at: float = field(default_factory=time.monotonic)
    rejected: bool = False  # synthesised failure, not an on-device result


@dataclass
class LiveSession:
    thread_id: str
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    pending: dict[str, PendingCall] = field(default_factory=dict)
    cond: asyncio.Condition = field(default_factory=asyncio.Condition)
    client: Any = None  # ClaudeSDKClient — typed Any to avoid importing the SDK here
    pump_task: asyncio.Task | None = None
    sdk_session_id: str | None = None
    sdk_model: str | None = None  # model the SDK reported on the first assistant msg
    current_run_id: str | None = None  # run_id of the in-flight HTTP request
    # Qualified (`mcp__pupa_frontend__*`) tool names the CURRENT live SDK client's
    # in-process MCP server exposes. An in-process server's tool list is frozen at
    # `connect()` (the SDK advertises tools without `listChanged` and the CLI
    # refuses `mcp_toggle` for SDK servers), so a gate tool that unlocks more tools
    # can't widen this client — instead the endpoint arms a continuation turn
    # (fresh client, `resume=sdk_session_id`, widened tools). Updated whenever a
    # client is built (`endpoint._options_for`).
    frontend_qualified: set[str] = field(default_factory=set)
    # Set by the resume POST when it advertises frontend tools not on the live
    # client (a gate unlock). The pump consumes it at the turn's ResultMessage to
    # start a continuation turn that actually exposes them. See
    # `endpoint._start_continuation`.
    pending_widen_descriptors: list[Any] | None = None
    # Captured on the new-turn POST so the pump can rebuild options for a
    # continuation (same state/model/config-MCP) without the original request in
    # scope.
    turn_input: Any = None
    turn_mcp: Any = None
    # A native-command permission request the gate parked, awaiting the user's
    # next chat message (yes/no). Resolved by the endpoint; see gate.py.
    pending_decision: asyncio.Future | None = None
    # Whether that question actually went out on a live run. The endpoint reads
    # the user's next message as the answer, so an ask still sitting in the
    # backlog must be denied rather than answered blind. See `gate._ask_user`.
    pending_decision_delivered: bool = False
    # Once the user approves with "always", every later command in this thread
    # runs without a prompt (set by the endpoint from the approval reply).
    auto_approve: bool = False
    last_activity: float = field(default_factory=time.monotonic)
    disposed: bool = False
    # The generator currently draining this session's queue, or None. A new
    # attach displaces it and waits for the handover. See `attach`.
    attachment: "_Attachment | None" = None
    # Events a displaced consumer had in hand when it stopped. Drained ahead of
    # the queue so the handover doesn't reorder the turn.
    pushback: list = field(default_factory=list)
    # Client liveness heartbeat. None until the first ping this park —
    # clients that never ping keep the full-wall behaviour.
    last_keepalive: float | None = None
    # The client told us it is backgrounded (iOS suspends its timers): suspend
    # the liveness grace and fall back to the absolute per-tool wall.
    client_backgrounded: bool = False
    # Out-of-band work this session started that outlives the turn (Claude Code
    # background subagents / background shell tasks). While it is `active` the
    # turn parks instead of disposing, and the next turn on this thread is routed
    # into this same live session. See `pupa_backend.agui.background`.
    background: BackgroundWork = field(default_factory=BackgroundWork)
    # True between an HTTP run's `run_started` and its terminal event. While it
    # is False nothing is listening: the loop is between turns, and anything it
    # produces on its own (a background task's injected turn) is held in
    # `deferred` until the next run opens. See `emit` / `open_run`.
    run_open: bool = False
    # Runs opened on this session, so a pending call can be aged in runs rather
    # than wall-clock. See `prune_pending`.
    run_seq: int = 0
    deferred: list = field(default_factory=list)
    # Frames the backlog trim dropped the `START` of. Their remaining deltas are
    # dropped too — a body event with no `START` is unrenderable. An id leaves the
    # set when its `END` goes by.
    dropped_frames: dict = field(default_factory=dict)
    # Model + thinking level the live client was built with. Both are frozen at
    # `connect()`, so a turn wanting different ones needs a fresh client.
    client_shape: tuple | None = None
    # Ambient context (`input.context`) the live client's system prompt carries.
    # A continued turn re-delivers it in the query when it has changed.
    context_pairs: list | None = None
    # The permission-relevant slice of the state the live client was built with
    # (`endpoint._gate_state`). A turn wanting a different one can't reuse this
    # client — the gate's hooks closed over the old state — so it starts fresh.
    gate_state: tuple | None = None

    def touch(self) -> None:
        self.last_activity = time.monotonic()

    async def keepalive(self, backgrounded: bool = False) -> None:
        """Record a client liveness ping and wake parked handlers to extend
        their deadline. `backgrounded=True` is the client's scene-phase notice."""
        async with self.cond:
            self.last_keepalive = time.monotonic()
            self.client_backgrounded = backgrounded
            self.cond.notify_all()
        self.touch()

    # -- event production (called by the pump) ---------------------------------

    def emit(self, event: Any) -> None:
        self.queue.put_nowait(event)

    def defer(self, event: Any) -> None:
        """Hold an event the loop produced with no run open.

        A background task reporting in makes the CLI run a turn nobody asked
        for: its output belongs to the conversation but to no HTTP run, and no
        SSE is attached to carry it. Held here and released by the next
        `open_run`, so it reaches the user on their next message.
        """
        self.deferred.append(event)
        # Trim in batches, not on every event: `defer` is called one event at a
        # time, so trimming at the cap would rebuild the whole list per delta and
        # log a line each time.
        if len(self.deferred) <= _MAX_DEFERRED + _TRIM_SLACK:
            return
        before = len(self.deferred)
        # Drop the oldest *body* events only. One long message can be most of the
        # backlog, so trimming by position would take its `START` with it and
        # orphan everything after — losing the whole report and leaving the next
        # delta at the head of a backlog with no `START`.
        budget = before - _MAX_DEFERRED
        survivors = []
        for held in self.deferred:
            if budget > 0 and getattr(held, "type", None) in _TRIMMABLE_EVENTS:
                budget -= 1
                continue
            survivors.append(held)
        self.deferred = survivors
        if len(self.deferred) > _MAX_DEFERRED:
            # Not enough body to give: a tool-call loop is `START`/`ARGS`/`END`,
            # only a third of it trimmable, so body-trimming alone would fall
            # further behind on every batch and never converge. Give up whole
            # frames instead, oldest first — the backlog is back under the cap
            # when this returns, so the trim always converges.
            self._drop_oldest_frames(len(self.deferred) - _MAX_DEFERRED)
        if before != len(self.deferred):
            logger.warning(
                "claude_code loop: deferred backlog full thread_id=%s — dropped %d "
                "of the oldest event(s)", self.thread_id, before - len(self.deferred),
            )

    def _drop_oldest_frames(self, count: int) -> None:
        """Drop `count` oldest events, then everything they orphaned.

        Keeps the backlog's invariant: it never starts on — or contains — a body
        event whose `START` is missing.
        """
        for held in self.deferred[:count]:
            frame = _frame_id(held)
            # `None` must never enter the set — it would swallow every standalone
            # event (a `CustomEvent`, a terminal) from then on.
            if frame is not None and getattr(held, "type", None) in _FRAME_START_EVENTS:
                self.dropped_frames[frame] = None
        del self.deferred[:count]
        self.deferred = [held for held in self.deferred if not self._is_orphan(held)]
        while len(self.dropped_frames) > _MAX_DROPPED_FRAMES:
            self.dropped_frames.pop(next(iter(self.dropped_frames)), None)

    def _is_orphan(self, event: Any) -> bool:
        """True for an event whose frame lost its `START` to a trim.

        Its `END` closes the frame out of the drop set — the id can be reused
        after that, and a later frame carrying it must not be swallowed too.
        """
        frame = _frame_id(event)
        # `None` must never match — it would swallow every standalone event (a
        # `CustomEvent`, a terminal) from then on.
        if frame is None or frame not in self.dropped_frames:
            return False
        if getattr(event, "type", None) in _FRAME_END_EVENTS:
            self.dropped_frames.pop(frame, None)
        return True

    def route(self, event: Any) -> bool:
        """Emit onto the open run's SSE, or hold it for the next one.

        The single chokepoint for anything produced *by the loop* (the pump, the
        permission gate): between runs there is no drain and no run the event can
        belong to. Teardown events bypass this deliberately — they exist to
        unblock a drain that may be attached right now.

        Returns whether the event went out live. The permission gate needs that
        answer: an ask the user cannot have seen must not be answered by their
        next message. Events orphaned by a backlog trim are dropped here rather
        than in `defer`, so a trimmed frame's later deltas can't reach a live SSE
        either once a run opens.
        """
        if self._is_orphan(event):
            return False
        if self.run_open:
            self.emit(event)
            return True
        self.defer(event)
        return False

    def take_deferred(self) -> list:
        """Hand the backlog to whoever will deliver it, and forget it here.

        Used when this session is being torn down but its held background output
        should still reach the user — the replacement session inherits it.
        """
        held, self.deferred = self.deferred, []
        return held

    def open_run(self, run_id: str, started: Any) -> None:
        """Start an HTTP run: emit `started`, then release the deferred backlog.

        Ordering is the point — `run_started` must be the first frame of the SSE,
        and anything the loop produced between turns follows it as part of this
        run.
        """
        self.current_run_id = run_id
        self.run_open = True
        self.run_seq += 1
        self.background.release()  # a live run is in flight; re-armed at the next park
        self.prune_pending()
        self._drop_stale_sentinels()
        self.queue.put_nowait(started)
        for event in self.deferred:
            self.queue.put_nowait(event)
        if self.deferred:
            logger.info(
                "claude_code loop: released %d deferred event(s) from background "
                "work thread_id=%s", len(self.deferred), self.thread_id,
            )
            self.deferred.clear()
        if self.pending_decision is not None and not self.pending_decision.done():
            # Its text has now reached the client — either live when it was
            # asked, or in the backlog this run just released. From here the
            # user's next message really is the answer. (The endpoint denies an
            # ask it could not deliver *before* opening a run, so anything still
            # parked here is one the user is about to see.)
            self.pending_decision_delivered = True
        self.touch()

    def _drop_stale_sentinels(self) -> None:
        """Discard terminal sentinels queued while nothing was attached.

        A sentinel belongs to the run that produced it. One left on the queue —
        a turn that parked or finished with no SSE attached — would be the first
        thing the *next* `attach` sees, so that run's response would end on it
        having delivered nothing, not even its `RunStarted`. Every empty-SSE bug
        this loop has had reached the user through that gap, so close the gap
        rather than relying on nothing ever queueing a stray sentinel again.
        """
        kept, dropped = [], 0
        while not self.queue.empty():
            item = self.queue.get_nowait()
            if item is INTERRUPT or item is PARK or item is FINISH or item is ERROR:
                dropped += 1
                continue
            kept.append(item)
        for item in kept:
            self.queue.put_nowait(item)
        if dropped:
            logger.info(
                "claude_code loop: dropped %d terminal sentinel(s) left by an "
                "unattached run thread_id=%s", dropped, self.thread_id,
            )

    def mark_interrupt(self) -> None:
        self.run_open = False
        self.queue.put_nowait(INTERRUPT)

    def mark_finish(self) -> None:
        self.run_open = False
        self.queue.put_nowait(FINISH)

    def mark_park(self) -> None:
        """End the run's SSE but keep the session registered and connected."""
        self.run_open = False
        self.queue.put_nowait(PARK)

    def mark_error(self) -> None:
        self.run_open = False
        self.queue.put_nowait(ERROR)

    async def register_pending(
        self, call_id: str, name: str, args: dict[str, Any], run_id: str | None = None
    ) -> None:
        """Record a frontend call from the model and wake any waiting handler."""
        async with self.cond:
            self.pending[call_id] = PendingCall(
                call_id=call_id, name=name, args=args, run_id=run_id, seq=self.run_seq
            )
            self.cond.notify_all()

    # -- frontend-tool handler side -------------------------------------------

    async def claim_call(
        self, name: str, args: dict[str, Any], timeout: float | None = None
    ) -> dict[str, Any]:
        """Block until a matching `PendingCall` has a result; consume and return it.

        Matches on `(name, args)`, waiting for both the pump to record the block and
        the resume POST to deliver the result — in either order. Returns the MCP
        tool-result the in-process SDK tool hands back to the model.
        """
        if timeout is None:
            timeout = _default_wait_timeout()
        absolute = time.monotonic() + timeout  # per-tool wall — hard cap
        grace = liveness_grace()
        async with self.cond:
            while True:
                matches = [
                    pc for pc in self.pending.values()
                    if not pc.consumed
                    and pc.name == name
                    and _args_key(pc.args) == _args_key(args)
                ]
                pick = next(
                    (pc for pc in matches if pc.result is not _UNSET and not pc.rejected),
                    None,
                )
                if pick is None:
                    # Only fall back to a synthesised rejection when no genuine
                    # call of this shape is still outstanding. A rejected slot
                    # lingers for a run so its *own* handler can take it — but a
                    # handler that is waiting for the device (or the model's
                    # retry of the same call) must block for the real answer, not
                    # be handed a stale `app_not_attached`.
                    outstanding = any(pc.result is _UNSET for pc in matches)
                    if not outstanding:
                        pick = next((pc for pc in matches if pc.result is not _UNSET), None)
                if pick is not None:
                    pick.consumed = True
                    return _mcp_tool_result(pick.result)
                # Effective deadline: once the client has pinged, wait
                # only `grace` past its last ping — a dead app fails fast no
                # matter how long the tool budget is. A backgrounded client
                # suspends the grace (its timers are frozen); the absolute wall
                # still bounds everything. Recomputed every loop so a ping that
                # lands mid-park extends the deadline in place.
                now = time.monotonic()
                deadline = absolute
                liveness_bound = (
                    self.last_keepalive is not None and not self.client_backgrounded
                )
                if liveness_bound:
                    deadline = min(absolute, self.last_keepalive + grace)
                remaining = deadline - now
                if remaining <= 0:
                    if liveness_bound and deadline < absolute:
                        raise asyncio.TimeoutError(
                            f"no frontend tool result for {name!r}: client liveness "
                            f"lost (no keepalive for {grace:.0f}s)"
                        )
                    raise asyncio.TimeoutError(
                        f"no frontend tool result delivered for {name!r} within {timeout:.0f}s"
                    )
                try:
                    await asyncio.wait_for(self.cond.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    # Deadline may have been extended by a ping racing the wait
                    # timeout — loop to recompute; the top of the loop raises if
                    # it genuinely expired.
                    continue

    # -- resume side ----------------------------------------------------------

    async def resolve_results(self, results: list[dict[str, Any]]) -> None:
        """Fill the pending result slots from the iOS resume payload.

        `results` is the normalised `[{toolCallId, content}, ...]` from
        `agui.tool_results.parse_tool_results`. A call the client dropped from
        *its own batch* (cancel/crash) gets a synthesised error so the SDK's tool
        handler never hangs. Does **not** clear `pending` — the handlers may not
        have run yet; they consume the slots once they do.

        The synth is **scoped to the batch this resume answers**: iOS POSTs one
        resume per interrupt batch, carrying every result for that batch at once,
        so any batch call absent from `results` was genuinely dropped. Calls from
        a *different* run still legitimately in flight (parallel/slow tools whose
        resume hasn't landed) are left untouched — previously they were clobbered
        with `missing_tool_result` by an unrelated resume.
        """
        async with self.cond:
            answered_run_ids: set[str | None] = set()
            for r in results:
                call_id = r.get("toolCallId")
                pc = self.pending.get(str(call_id)) if call_id is not None else None
                if pc is not None:
                    answered_run_ids.add(pc.run_id)
                    if pc.result is _UNSET:
                        pc.result = r.get("content", "")
            # Which runs' unresolved calls may we synth as missing? Only those this
            # resume actually addressed. If it addressed none we can't identify the
            # batch — fall back to synth-all ONLY when a single run is in flight (the
            # original no-hang guarantee), never across multiple runs.
            unresolved_run_ids = {pc.run_id for pc in self.pending.values() if pc.result is _UNSET}
            if not answered_run_ids and len(unresolved_run_ids) <= 1:
                answered_run_ids = unresolved_run_ids
            for pc in self.pending.values():
                if pc.result is _UNSET and pc.run_id in answered_run_ids:
                    pc.result = json.dumps({"ok": False, "error": "missing_tool_result"})
            self.cond.notify_all()
        self.touch()

    def prune_pending(self) -> None:
        """Drop pending calls no handler can still legitimately claim.

        One pump now spans every turn of a live session, so `pending` would grow
        for the session's whole life. Worse, a call rejected between runs
        (`reject_pending`) keeps an `app_not_attached` result nobody consumed: a
        later identical `(name, args)` call would claim *that* slot and never see
        the device's real answer.

        A rejected slot survives one whole run before it goes, though. The
        handler that should take it may not have been scheduled yet when the run
        opens — `_continue_live_turn` awaits the query first, so a rejection can
        land in that window — and pruning it out from under that handler would
        block the handler for the full park wall.

        A slot past the park wall goes regardless: its handler can only have
        raised `TimeoutError` by then, so nobody is waiting on it, and leaving it
        makes `_cannot_continue_live` decline for a reason that is no longer true.
        That age is measured in **wall-clock, not runs** — runs can be opened
        milliseconds apart (a permission reply, a resume), so counting them would
        prune a slot whose handler is still inside `claim_call`, and the device's
        real answer would then have no slot to land in.
        """
        now = time.monotonic()
        stale = [
            call_id for call_id, pc in self.pending.items()
            if pc.consumed
            or (pc.run_id is None and pc.seq <= self.run_seq - 2)
            or now - pc.registered_at > _stale_pending_after()
        ]
        for call_id in stale:
            self.pending.pop(call_id, None)

    def has_unresolved_pending(self) -> bool:
        """True while any frontend call is still waiting for its on-device result."""
        return any(pc.result is _UNSET for pc in self.pending.values())

    async def reject_pending(self, call_ids: list[str], error: str) -> None:
        """Fail exactly these calls' result slots.

        Used when the loop calls a frontend tool with no run open — an injected
        background-task turn, say. The device isn't listening, so the handler
        must be given an error now rather than blocking on the park wall.
        """
        async with self.cond:
            for call_id in call_ids:
                pc = self.pending.get(call_id)
                if pc is not None and pc.result is _UNSET:
                    pc.result = json.dumps({"ok": False, "error": error})
                    pc.rejected = True
            self.cond.notify_all()

    # -- teardown -------------------------------------------------------------

    async def release_pending(self, error: str) -> None:
        """Hand every still-unresolved call a synthesised error so its handler
        returns instead of blocking the SDK, and deny a parked approval.

        Called before any teardown: a handler still sitting in `claim_call` when
        the transport closes leaves the CLI child waiting on a control response
        that will never come.
        """
        async with self.cond:
            for pc in self.pending.values():
                if pc.result is _UNSET:
                    pc.result = json.dumps({"ok": False, "error": error})
            self.cond.notify_all()
        if self.pending_decision is not None and not self.pending_decision.done():
            self.pending_decision.set_result(False)  # deny on teardown

    async def dispose(self) -> None:
        """End the run, unblock waiting handlers, cancel the pump, drop the client.

        The terminal `RunError` + `ERROR` sentinel go on the queue **first**: a
        teardown mid-turn (thread reused by a fresh POST, idle sweep) otherwise
        leaves an attached `attach()` drain blocked on `queue.get()` forever, so
        the SSE ends with no `RunFinished`/`RunError` and the app sits silent. If
        the pump already queued its own terminal, that one is drained first and
        this pair is never seen.
        """
        if self.disposed:
            return
        self.disposed = True
        # Local import: `events` → `frontend_tools` → `registry` would cycle.
        from . import events as cl_events

        self.emit(cl_events.run_error(
            "the Claude Code session ended before the turn completed "
            "(replaced by a new turn or evicted while idle)"
        ))
        self.mark_error()
        # Unblock any handler still waiting on a result so it doesn't hang the
        # SDK teardown — hand it a synthesised error rather than nothing.
        await self.release_pending("session_disposed")
        if self.pump_task is not None and not self.pump_task.done():
            self.pump_task.cancel()
        client = self.client
        if client is not None:
            try:
                await client.disconnect()
            except Exception:  # noqa: BLE001 — best-effort teardown
                logger.debug("claude_code loop: client disconnect failed", exc_info=True)


def _mcp_tool_result(content: Any) -> dict[str, Any]:
    """Wrap an on-device result string as an MCP `CallToolResult`-shaped dict."""
    text = content if isinstance(content, str) else json.dumps(content, default=str)
    return {"content": [{"type": "text", "text": text}]}


# Module-level registry keyed by thread_id.
_REGISTRY: dict[str, LiveSession] = {}

# SDK session ids survive a finished session so a later user turn on the same
# thread can `resume=<sdk_session_id>` for cross-turn conversation continuity.
_SESSION_IDS: dict[str, str] = {}


def remembered_session_id(thread_id: str) -> str | None:
    return _SESSION_IDS.get(thread_id)


def remember_session_id(thread_id: str, sdk_session_id: str | None) -> None:
    if sdk_session_id:
        _SESSION_IDS[thread_id] = sdk_session_id


def get(thread_id: str) -> LiveSession | None:
    return _REGISTRY.get(thread_id)


def create(thread_id: str) -> LiveSession:
    """Create (replacing any stale entry for this thread) and register a session."""
    existing = _REGISTRY.get(thread_id)
    if existing is not None:
        # Don't leak the old parked task if a fresh turn reuses the thread.
        asyncio.ensure_future(existing.dispose())
    session = LiveSession(thread_id=thread_id)
    _REGISTRY[thread_id] = session
    return session


async def remove(thread_id: str, session: LiveSession | None = None) -> None:
    """Evict and dispose the session registered for `thread_id`.

    Pass `session` to make the removal identity-checked: a stale session whose
    drain finishes *after* a fresh POST already claimed the thread disposes only
    itself and leaves the newcomer registered.
    """
    current = _REGISTRY.get(thread_id)
    if session is not None and current is not session:
        await session.dispose()
        return
    _REGISTRY.pop(thread_id, None)
    if current is not None:
        await current.dispose()


def note_reattach(thread_id: str) -> None:
    """A re-attach POST landed for `thread_id` — treat it as a liveness ping.

    Registered with `sse_replay.register_reattach_observer`, which serves
    re-attaches by short-circuit and so never reaches this loop. Without it a
    client that backgrounded (suspending the grace in `claim_call`) stayed marked
    backgrounded, on a stale last-ping clock, at the exact moment it proved it
    was back.

    Synchronous by necessity — the observer runs on the middleware's path — so
    the actual `keepalive()` is scheduled rather than awaited.
    """
    session = _REGISTRY.get(thread_id)
    if session is None or session.disposed:
        return
    asyncio.ensure_future(session.keepalive(backgrounded=False))


async def retire(thread_id: str) -> None:
    """Wind down the session on `thread_id` before a new turn claims the thread.

    A parked session is mid-tool-call as far as the CLI child is concerned.
    `dispose()` alone cancels the pump and closes the SDK transport immediately,
    which rejects the child's in-flight PreToolUse / permission control requests
    — the `Stream closed` errors from `hook_0` — and leaves the SDK session
    interrupted for the next turn to resume, which the CLI answers with a
    `Continue from where you left off.` no-op instead of the user's prompt.

    So wind down in order: release the parked handlers, ask the child to
    `interrupt()`, wait a bounded `retire_drain_timeout()` for the pump to reach
    its terminal, then dispose as usual. Every step is best-effort — a thread
    must always end up free for the newcomer.
    """
    session = _REGISTRY.get(thread_id)
    if session is None or session.disposed:
        return

    await session.release_pending("superseded_by_new_turn")

    interrupt = getattr(session.client, "interrupt", None)
    if callable(interrupt):
        try:
            await interrupt()
        except Exception:  # noqa: BLE001 — a child that won't interrupt still gets torn down
            logger.debug("claude_code loop: retire interrupt failed", exc_info=True)
        else:
            pump = session.pump_task
            if pump is not None and not pump.done():
                # `asyncio.wait` (not `wait_for(shield(...))`): it neither cancels
                # the pump on timeout nor re-raises its outcome. A pump that was
                # cancelled elsewhere would otherwise surface CancelledError —
                # a BaseException that escapes `except Exception` and would fail
                # the user's POST on what is meant to be best-effort teardown.
                done, _pending = await asyncio.wait(
                    {pump}, timeout=retire_drain_timeout()
                )
                if not done:
                    logger.info(
                        "claude_code loop: retire drain timed out thread_id=%s — "
                        "tearing down anyway",
                        thread_id,
                    )

    await remove(thread_id, session)


async def attach(session: LiveSession):
    """Async generator: drain the session queue until interrupt / finish / error.

    Yields AG-UI event objects (the endpoint encodes them). On INTERRUPT the
    generator returns leaving the session parked; on FINISH/ERROR it removes the
    session from the registry after the trailing events have been yielded.

    One consumer at a time. A second POST can land on a live session — a resume
    whose response was lost is re-sent while the pump is still draining — and two
    generators on one queue each take a *share* of the events: the turn is split
    across two SSE responses, and both append into the same replay log through
    independent task chains, so the log's frame order stops matching the turn's.
    A new attach therefore displaces the old one, which stops at its next wake
    and hands back anything it had in flight.
    """
    previous = session.attachment
    mine = _Attachment()
    session.attachment = mine
    if previous is not None:
        previous.stop.set()
        try:
            await asyncio.wait_for(previous.done.wait(), timeout=_ATTACH_HANDOVER_TIMEOUT)
        except asyncio.TimeoutError:
            logger.info(
                "claude_code loop: attach handover timed out thread_id=%s — "
                "proceeding; the displaced consumer stops at its next wake",
                session.thread_id,
            )
    session.touch()
    try:
        while True:
            if session.pushback:
                item = session.pushback.pop(0)
            else:
                get = asyncio.ensure_future(session.queue.get())
                displaced = asyncio.ensure_future(mine.stop.wait())
                done, _pending = await asyncio.wait(
                    {get, displaced}, return_when=asyncio.FIRST_COMPLETED
                )
                if displaced in done:
                    # Both can complete on the same wake. The item belongs to the
                    # new consumer, and it is OLDER than anything still queued —
                    # hand it back at the front rather than dropping or
                    # re-queueing it behind newer frames.
                    if get in done:
                        session.pushback.insert(0, get.result())
                    else:
                        get.cancel()
                    return
                displaced.cancel()
                item = get.result()
            if item is INTERRUPT or item is PARK:
                session.touch()
                return
            if item is FINISH or item is ERROR:
                await remove(session.thread_id, session)
                return
            yield item
            session.touch()
    finally:
        # Unblocks the displacing attach — including on GeneratorExit, when the
        # client disconnected and FastAPI closed this generator.
        mine.done.set()
        # Only clear the slot if it is still ours: a displacing attach has
        # already installed its own.
        if session.attachment is mine:
            session.attachment = None


def _evictable(session: LiveSession, now: float, timeout: float) -> bool:
    """Idle past the wall — unless it is held open for background work.

    A session parked on in-flight background work keeps its subprocess so the
    work can report; it is evicted only once the hold expires
    (`PUPA_BACKGROUND_HOLD`), whatever the idle clock says.
    """
    if session.background.holding:
        pump = session.pump_task
        if pump is not None and pump.done():
            return True  # nothing is left to receive the work it is held for
        return session.background.hold_expired(now)
    return now - session.last_activity > timeout


async def sweep_idle(timeout: float = _DEFAULT_IDLE_TIMEOUT) -> int:
    """Evict sessions idle longer than `timeout`. Returns the number evicted."""
    now = time.monotonic()
    stale = [tid for tid, s in _REGISTRY.items() if _evictable(s, now, timeout)]
    for tid in stale:
        session = _REGISTRY.get(tid)
        if session is not None and session.background.holding:
            logger.info(
                "claude_code loop: background hold expired thread_id=%s — dropping %s",
                tid, session.background.summary(),
            )
        else:
            logger.info("claude_code loop: evicting idle session thread_id=%s", tid)
        await remove(tid)
    return len(stale)
