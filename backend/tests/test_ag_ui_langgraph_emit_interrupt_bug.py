"""The `ag-ui-langgraph` emit-path interrupt bug — now fixed upstream.

Symptom it caused (client): the agent called a frontend `render*` tool, the
chat silently stopped, and it resumed only when the user sent another message.

Mechanism: `LangGraphAgent._handle_stream_events` used to collect the
interrupts it emits as `on_interrupt` from **only `state.tasks[0]`** (right
after `aget_state(config)`), while the recovery path in `prepare_stream` used
`_collect_interrupts` over **all** tasks. A frontend interrupt parked on a
non-first task was dropped by the in-run emit, and the run ended with only
`RUN_FINISHED` — byte-for-byte identical to a clean finish, so the client
couldn't tell and silently stopped. The next user message hit the recovery
path, which *did* see the parked interrupt, so the turn "continued".

`0.0.43` ships the fix: the emit path now calls
`_collect_interrupts(state.tasks)` too. A canary here used to assert the bug
was still present so we'd notice this moment; it fired on the upgrade and is
now inverted into a pin on the fixed behaviour, so a downgrade or a regression
is caught rather than silently re-introducing the silent-stop.

**Follow-up in the client repo:** AGUIKit's dropped-interrupt self-heal
(`AgentSession` settle branch) exists only to paper over this bug and is no
longer load-bearing. It's harmless — it now never triggers — but it can be
retired.
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


def test_collect_interrupts_sees_a_nonfirst_task_interrupt():
    """The helper both paths now use catches an interrupt parked on a task
    other than the first."""
    frontend = _FakeInterrupt(
        {"frontend_tool_calls": [{"id": "call_A", "name": "renderTracker", "args": {}}]}
    )
    tasks = (_FakeTask([]), _FakeTask([frontend]))  # interrupt on the 2nd task

    assert LangGraphAgent._collect_interrupts(tasks) == [frontend]
    # What the old emit path did, for contrast: it read this and found nothing.
    assert list(tasks[0].interrupts) == []


def test_installed_emit_path_collects_from_all_tasks():
    """Pin the fix. A regression here is invisible from the client's side — a
    dropped frontend interrupt looks exactly like a clean finish — so it has to
    be caught in the dependency, not in behaviour."""
    src = inspect.getsource(LangGraphAgent._handle_stream_events)
    # Narrow to the in-run emit region: post-stream `aget_state` → the
    # `RunFinishedEvent` the method always yields afterwards.
    emit_region = src.split("aget_state(config)", 1)[1].split("RunFinishedEvent", 1)[0]

    assert "_collect_interrupts(state.tasks)" in emit_region, (
        "ag-ui-langgraph emit path no longer collects from all tasks — the "
        "non-first-task interrupt bug is back, and it presents as a silent stop "
        "mid-turn in the client."
    )
    assert "tasks[0].interrupts" not in emit_region, (
        "emit path reads tasks[0] again — the pre-0.0.43 bug."
    )
