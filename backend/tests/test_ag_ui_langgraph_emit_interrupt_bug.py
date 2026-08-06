"""Proves the `ag-ui-langgraph` emit-path bug behind the render* silent-stop.

Symptom (client): the agent calls a frontend `render*` tool, the chat
silently stops, and it resumes only when the user sends another message.

Mechanism: `LangGraphAgent._handle_stream_events` collects the interrupts it
emits as `on_interrupt` from **only `state.tasks[0]`** (right after
`aget_state(config)`), while the recovery path in `prepare_stream` uses
`_collect_interrupts` over **all** tasks. When a frontend interrupt parks on a
non-first task, the in-run emit drops it and the run ends with only
`RUN_FINISHED` — byte-for-byte identical to a clean finish, so the client
can't tell and silently stops. The next user message hits the recovery path,
which *does* see the parked interrupt, so the turn "continues".

The maintainers already fixed the recovery path (docstring on
`_collect_interrupts`: "collect from ALL tasks, not just tasks[0] … #1409
where parallel tool calls could have interrupts on tasks other than the first
one") but the in-run emit path is unpatched in the pinned release (0.0.42);
the fix is only on upstream `main`.

These tests document the bug in the installed version. The second is a CANARY:
when upstream ships the emit-path fix, it fails — signalling that AGUIKit's
dropped-interrupt self-heal (AgentSession settle branch) is no longer
load-bearing and both it and this canary can be removed.
"""

import inspect

from ag_ui_langgraph.agent import LangGraphAgent


class _FakeInterrupt:
    def __init__(self, value):
        self.value = value


class _FakeTask:
    """Minimal stand-in for a LangGraph PregelTask (only `.interrupts` used)."""

    def __init__(self, interrupts):
        self.interrupts = tuple(interrupts)


def test_collect_interrupts_recovery_path_sees_a_nonfirst_task_interrupt():
    """The recovery path uses the real library helper and catches an interrupt
    parked on a task other than the first — this is why the chat resumes on the
    next user message even though the in-run emit dropped it."""
    frontend = _FakeInterrupt(
        {"frontend_tool_calls": [{"id": "call_A", "name": "renderTracker", "args": {}}]}
    )
    tasks = (_FakeTask([]), _FakeTask([frontend]))  # interrupt on the 2nd task

    # `prepare_stream` (agent.py) collects from ALL tasks — finds it.
    assert LangGraphAgent._collect_interrupts(tasks) == [frontend]

    # The in-run emit path reads only `tasks[0].interrupts` — misses it.
    emit_path_interrupts = tasks[0].interrupts if tasks else []
    assert list(emit_path_interrupts) == []


def test_installed_emit_path_still_reads_only_first_task_canary():
    """CANARY. The pinned `ag-ui-langgraph` still drops non-first-task
    interrupts at emit time. When this fails, upstream has shipped the fix
    (main uses `_collect_interrupts(state.tasks)` here) and the AGUIKit
    self-heal + this canary can be retired."""
    src = inspect.getsource(LangGraphAgent._handle_stream_events)
    # Narrow to the in-run emit region: post-stream `aget_state` → the
    # `RunFinishedEvent` the method always yields afterwards.
    emit_region = src.split("aget_state(config)", 1)[1].split("RunFinishedEvent", 1)[0]

    assert "tasks[0].interrupts" in emit_region, (
        "ag-ui-langgraph emit path no longer reads tasks[0]. The upstream fix "
        "has shipped — retire the AGUIKit dropped-interrupt self-heal and this "
        "canary."
    )
    assert "_collect_interrupts(state.tasks)" not in emit_region, (
        "emit path now collects from all tasks — bug fixed upstream; retire the "
        "client self-heal and this canary."
    )
