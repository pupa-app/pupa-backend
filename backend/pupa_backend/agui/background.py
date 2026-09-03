"""Out-of-band work a harness session starts that outlives the turn.

Some agent loops can start work that keeps running *after* the turn that
launched it ends — Claude Code's background subagents (`Agent` /
`run_in_background`) and background shell tasks are the first case. The loop
reports such a task's completion later, on its own, as an **injected turn**
nobody prompted for.

A harness that tears its session down at turn end kills that work, and the next
turn (a fresh process) can only report the task as lost. So the lifecycle rule,
for **any** harness with this dynamic:

1. Track every unit of out-of-band work the loop starts (`start`), and clear it
   on a terminal status (`update`).
2. At turn end, when `active`, **park** instead of disposing: end the HTTP run
   (the client still gets its `RunFinished`) but keep the loop's process alive
   and its message stream drained, so the injected turn is received.
3. Route the *next* turn on that thread into the **same** live session — a fresh
   process cannot see the previous one's tasks.
4. Bound the hold (`hold` / `hold_expired`): work that never reports must not pin
   a subprocess forever.

This module is bookkeeping plus the policy knob. Each harness maps its own
lifecycle messages onto `start` / `update` and makes the park-vs-dispose call
itself — nothing here knows about any particular SDK.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field

logger = logging.getLogger("uvicorn.error")

# Statuses that mean a task is over. Spans both vocabularies the Claude CLI uses
# (`task_notification` says `stopped`, `task_updated` says the raw `killed`) plus
# the obvious synonyms a future harness might report.
TERMINAL_STATUSES: frozenset[str] = frozenset(
    {"completed", "complete", "failed", "error", "stopped", "killed", "cancelled",
     "canceled", "timed_out", "timeout"}
)

# How long a session may stay alive purely because background work is
# outstanding, measured from the end of the turn that parked it. A hard wall: a
# task that never reports a terminal status must not pin a subprocess for the
# life of the process.
_HOLD_DEFAULT = 1800.0  # seconds


def hold_window() -> float:
    """`PUPA_BACKGROUND_HOLD` seconds (default 1800). <= 0 disables the hold —
    a session then never stays alive for background work."""
    raw = os.environ.get("PUPA_BACKGROUND_HOLD")
    if raw:
        try:
            return float(raw)
        except ValueError:
            logger.warning("background work: bad PUPA_BACKGROUND_HOLD=%r; using %s",
                           raw, _HOLD_DEFAULT)
    return _HOLD_DEFAULT


@dataclass
class BackgroundWork:
    """The out-of-band tasks one live session has outstanding.

    `tasks` maps the loop's own task id to a short description (logs only). The
    ids are opaque strings — this class never interprets them.
    """

    tasks: dict[str, str] = field(default_factory=dict)
    # Wall-clock deadline for the current park, or None when not holding.
    hold_until: float | None = None

    # -- the loop reports lifecycle -------------------------------------------

    def start(self, task_id: str | None, description: str = "") -> None:
        if not task_id:
            return
        self.tasks.setdefault(str(task_id), description or "")

    def update(self, task_id: str | None, status: str | None) -> bool:
        """Apply a status. Returns True when it was terminal (task cleared).

        An update for an unknown id is still honoured — a harness may learn of a
        task only when it ends.
        """
        if not task_id:
            return False
        if status is not None and str(status).lower() in TERMINAL_STATUSES:
            self.tasks.pop(str(task_id), None)
            return True
        self.tasks.setdefault(str(task_id), "")
        return False

    # -- the harness asks --------------------------------------------------

    @property
    def active(self) -> bool:
        return bool(self.tasks)

    def summary(self) -> str:
        """`2 task(s): a1b2 (build docs), c3d4` — for a log line, never the wire."""
        if not self.tasks:
            return "none"
        parts = [f"{tid}{f' ({d})' if d else ''}" for tid, d in self.tasks.items()]
        return f"{len(self.tasks)} task(s): " + ", ".join(parts)

    # -- the hold ------------------------------------------------------------

    def hold(self, now: float | None = None) -> bool:
        """Arm (or re-arm) the retention wall for a park. False if disabled.

        Called at each turn end that parks, so the wall measures *time since the
        last turn*, not since the task started.
        """
        window = hold_window()
        if window <= 0:
            self.hold_until = None
            return False
        self.hold_until = (time.monotonic() if now is None else now) + window
        return True

    def release(self) -> None:
        self.hold_until = None

    @property
    def holding(self) -> bool:
        return self.hold_until is not None

    def hold_expired(self, now: float | None = None) -> bool:
        if self.hold_until is None:
            return False
        return (time.monotonic() if now is None else now) >= self.hold_until
