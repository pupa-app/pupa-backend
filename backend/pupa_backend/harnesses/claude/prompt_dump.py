"""Dump the exact prompt prefix the loop hands the CLI, and diff it turn to turn.

Opt-in via ``PUPA_CLAUDE_PROMPT_DUMP=<dir>``. **Off by default and never for a
shared deploy**: the payload contains whatever the app sends as ambient context
— live canvas state, memories, AGENTS.md — plus every tool schema, so it is user
data at rest. Point it at a scratch dir, reproduce, delete.

`usage.cache_line` says *which key* moved; this says *what the bytes were*. Each
options build writes `<dump-dir>/<thread>/NNN.json`, and from the second one on a
`NNN.diff` against its predecessor — that unified diff is the definitive answer
to "the app changed nothing, why did the cache re-write?".

Long text (system prompt, context values, tool descriptions) is stored as arrays
of lines so `indent=2` puts one source line per JSON line and the diff lands on
the line that actually moved instead of one multi-kilobyte blob.
"""

from __future__ import annotations

import difflib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger("uvicorn.error")

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


def dump_dir() -> Path | None:
    """`PUPA_CLAUDE_PROMPT_DUMP` as a path, or None when the dump is off."""
    raw = (os.getenv("PUPA_CLAUDE_PROMPT_DUMP") or "").strip()
    return Path(raw).expanduser() if raw else None


def _safe(name: str) -> str:
    """Thread ids come from the client — never let one escape the dump dir."""
    return _UNSAFE.sub("_", name)[:64] or "unknown"


def _lines(text: Any) -> list[str]:
    return str(text or "").splitlines()


def build_payload(
    *,
    thread_id: str,
    model: str | None,
    base_system: str,
    system: str,
    context_pairs: list[tuple[str, str]],
    tool_specs: list[tuple[str, str, Any]],
    permission_mode: str | None,
    thinking: Any,
    skills: Any,
    cwd: str | None,
    fingerprint: dict[str, str],
) -> dict[str, Any]:
    """The whole cacheable prefix, in the order the model sees it."""
    from .usage import context_label

    return {
        "thread_id": thread_id,
        "model": model,
        "permission_mode": permission_mode,
        "thinking": thinking,
        "skills": skills,
        "cwd": cwd,
        "fingerprint": fingerprint,
        # `tools` first, then `system`, then the transcript — the prefix order
        # Anthropic caches on. The transcript isn't ours to dump (the CLI rebuilds
        # it from the resumed session), so it stops at the system prompt.
        "tools": [
            {"name": name, "description": _lines(desc), "schema": schema}
            for name, desc, schema in tool_specs
        ],
        "base_system": _lines(base_system),
        "context": [
            {
                "label": context_label(desc, i),
                "description": _lines(desc),
                "value": _lines(val),
            }
            for i, (desc, val) in enumerate(context_pairs)
        ],
        "system": _lines(system),
    }


def _render(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n"


def write(thread_id: str, payload: dict[str, Any]) -> Path | None:
    """Write this turn's prefix and its diff vs the previous turn.

    Returns the JSON path, or None when the dump is off. Never raises — a broken
    dump must not cost the user their turn.
    """
    root = dump_dir()
    if root is None:
        return None
    try:
        thread_dir = root / _safe(thread_id)
        thread_dir.mkdir(parents=True, exist_ok=True)
        previous = sorted(thread_dir.glob("[0-9]*.json"))
        path = thread_dir / f"{len(previous):03d}.json"
        current = _render(payload)
        path.write_text(current, encoding="utf-8")

        if previous:
            prior = previous[-1]
            diff = "".join(
                difflib.unified_diff(
                    prior.read_text(encoding="utf-8").splitlines(keepends=True),
                    current.splitlines(keepends=True),
                    fromfile=prior.name,
                    tofile=path.name,
                )
            )
            diff_path = path.with_suffix(".diff")
            diff_path.write_text(diff or "(identical)\n", encoding="utf-8")
            changed = sum(
                1 for line in diff.splitlines()
                if line[:1] in "+-" and not line.startswith(("+++", "---"))
            )
            logger.info(
                "claude_code prompt dump: %s (%s vs %s)",
                diff_path, f"{changed} changed line(s)" if changed else "identical", prior.name,
            )
        else:
            logger.info("claude_code prompt dump: %s (first turn on this thread)", path)
        return path
    except Exception:  # noqa: BLE001 — diagnostics must never break a turn
        logger.exception("claude_code prompt dump: failed")
        return None
