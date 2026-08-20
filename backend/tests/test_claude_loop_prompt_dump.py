"""`PUPA_CLAUDE_PROMPT_DUMP` — dump the cacheable prompt prefix and diff turns.

The fingerprint line names *which* key moved; this dump answers *what the bytes
were*, so an "app changed nothing" turn that still re-writes cache can be settled
from a unified diff instead of a guess.
"""

from __future__ import annotations

import json

import pytest

from pupa_backend.harnesses.claude import prompt_dump


def _payload(canvas_value: str = '{"components":[]}', **over):
    base = dict(
        thread_id="t-1",
        model="claude-haiku-4-5",
        base_system="you are pupa",
        system=f"you are pupa\n\nLive canvas state\n{canvas_value}",
        context_pairs=[("Live canvas state — thin enum.", canvas_value)],
        tool_specs=[("mcp__pupa_frontend__a", "does a", {"type": "object"})],
        permission_mode="default",
        thinking={},
        skills=None,
        cwd=None,
        fingerprint={"model": "abc"},
    )
    base.update(over)
    return prompt_dump.build_payload(**base)


@pytest.fixture
def dump_root(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PUPA_CLAUDE_PROMPT_DUMP", str(tmp_path / "dumps"))
    return tmp_path / "dumps"


def test_dump_is_off_unless_the_env_var_is_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PUPA_CLAUDE_PROMPT_DUMP", raising=False)
    assert prompt_dump.dump_dir() is None
    assert prompt_dump.write("t-1", _payload()) is None


def test_first_turn_writes_json_and_no_diff(dump_root) -> None:
    path = prompt_dump.write("t-1", _payload())
    assert path == dump_root / "t-1" / "000.json"
    assert not path.with_suffix(".diff").exists()

    written = json.loads(path.read_text())
    # Long text is stored line-by-line so the diff lands on the moved line.
    assert written["system"] == ["you are pupa", "", "Live canvas state", '{"components":[]}']
    assert written["context"][0]["label"] == "live-canvas-state"
    assert written["tools"][0]["name"] == "mcp__pupa_frontend__a"


def test_second_turn_diffs_against_the_first(dump_root) -> None:
    prompt_dump.write("t-1", _payload())
    second = prompt_dump.write("t-1", _payload(canvas_value='{"components":[{"id":"a"}]}'))

    assert second.name == "001.json"
    diff = second.with_suffix(".diff").read_text()
    assert "--- 000.json" in diff and "+++ 001.json" in diff
    assert '+        "{\\"components\\":[{\\"id\\":\\"a\\"}]}"' in diff
    # The untouched tool block is not in the diff.
    assert "mcp__pupa_frontend__a" not in diff


def test_an_unchanged_turn_records_that_it_was_identical(dump_root) -> None:
    """The decisive negative result: if the prefix really didn't move, the diff
    says so — and a cache re-write then has to be blamed on the transcript."""
    prompt_dump.write("t-1", _payload())
    second = prompt_dump.write("t-1", _payload())
    assert second.with_suffix(".diff").read_text() == "(identical)\n"


def test_threads_dump_into_separate_folders(dump_root) -> None:
    prompt_dump.write("t-1", _payload())
    other = prompt_dump.write("t-2", _payload())
    assert other == dump_root / "t-2" / "000.json"
    assert not other.with_suffix(".diff").exists()  # first turn for THIS thread


def test_a_hostile_thread_id_cannot_escape_the_dump_dir(dump_root) -> None:
    path = prompt_dump.write("../../etc/passwd", _payload())
    assert path is not None
    assert dump_root in path.parents
    assert path.parent.name == ".._.._etc_passwd"


def test_a_broken_dump_never_costs_the_turn(dump_root, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prompt_dump, "_render", lambda payload: 1 / 0)
    assert prompt_dump.write("t-1", _payload()) is None
