"""Tests for the agent-harness registry (`harnesses.py`) and the config →
`PUPA_HARNESSES` resolution (`pupa_config._resolve_harnesses`)."""

import json

import pytest

from pupa_backend.harnesses import (
    ClaudeCodeHarness,
    build_registry,
    claude_harness_enabled,
    deepagents_harness_enabled,
)
from pupa_backend.pupa_config import _resolve_harnesses


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

def test_default_registry_is_langgraph(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PUPA_HARNESSES", raising=False)
    reg = build_registry()
    assert reg.ids() == ["deepagents"]
    assert reg.default().id == "deepagents"


def test_both_enabled_default_langgraph(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "PUPA_HARNESSES",
        json.dumps(
            {
                "deepagents": {"enabled": True, "default": True},
                "claude_code": {"enabled": True},
            }
        ),
    )
    reg = build_registry()
    assert set(reg.ids()) == {"deepagents", "claude_code"}
    assert reg.default().id == "deepagents"


def test_disabled_harness_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "PUPA_HARNESSES",
        json.dumps(
            {
                "deepagents": {"enabled": False},
                "claude_code": {"enabled": True, "default": True},
            }
        ),
    )
    reg = build_registry()
    assert reg.ids() == ["claude_code"]


def test_no_explicit_default_picks_first_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "PUPA_HARNESSES",
        json.dumps({"claude_code": {"enabled": True}, "deepagents": {"enabled": True}}),
    )
    reg = build_registry()
    assert reg.default().id == "claude_code"  # first enabled


def test_empty_enabled_set_is_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUPA_HARNESSES", json.dumps({"deepagents": {"enabled": False}}))
    with pytest.raises(ValueError, match="No agent harness is enabled"):
        build_registry()


def test_unknown_harness_id_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUPA_HARNESSES", json.dumps({"made_up": {"enabled": True}}))
    with pytest.raises(ValueError, match="Unknown harness"):
        build_registry()


def test_invalid_json_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUPA_HARNESSES", "{not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        build_registry()


def test_claude_harness_enabled_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PUPA_HARNESSES", raising=False)
    assert claude_harness_enabled() is False
    monkeypatch.setenv("PUPA_HARNESSES", json.dumps({"claude_code": {"enabled": True}}))
    assert claude_harness_enabled() is True


def test_deepagents_harness_enabled_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gates the checkpointer/store lifespan and the `/db` router."""
    monkeypatch.delenv("PUPA_HARNESSES", raising=False)
    assert deepagents_harness_enabled() is True  # default config is deepagents-only
    monkeypatch.setenv("PUPA_HARNESSES", json.dumps({"claude_code": {"enabled": True}}))
    assert deepagents_harness_enabled() is False
    monkeypatch.setenv("PUPA_HARNESSES", json.dumps({"deepagents": {"enabled": False}}))
    assert deepagents_harness_enabled() is False


async def test_persistence_lifespan_skipped_without_langgraph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Claude-only deploy opens no database.

    The savers and `/db` both serve LangGraph checkpoints, so there is nothing
    to persist — and `PUPA_REQUIRE_DB_SCHEME` must not demand a database no
    harness in the process would read. Pinned here because the failure mode is
    a deploy that boots fine and then dies on an unrelated env var.
    """
    from pupa_backend.app import _persistence_lifespan

    monkeypatch.setenv("PUPA_HARNESSES", json.dumps({"claude_code": {"enabled": True}}))
    monkeypatch.setenv("PUPA_REQUIRE_DB_SCHEME", "postgresql")
    monkeypatch.delenv("DATABASE_URL", raising=False)

    async with _persistence_lifespan() as (checkpointer, store):
        assert (checkpointer, store) == (None, None)


async def test_persistence_lifespan_opens_db_for_langgraph(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from pupa_backend.app import _persistence_lifespan

    monkeypatch.delenv("PUPA_HARNESSES", raising=False)
    monkeypatch.delenv("PUPA_REQUIRE_DB_SCHEME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("DATABASE_URL", raising=False)

    async with _persistence_lifespan() as (checkpointer, store):
        assert checkpointer is not None and store is not None


# --------------------------------------------------------------------------- #
# Discovery shape
# --------------------------------------------------------------------------- #

def test_claude_harness_permission_schema() -> None:
    keys = {c["key"] for c in ClaudeCodeHarness().permission_schema()}
    assert keys == {"claude_loop_native", "claude_loop_auto_approve"}


# --------------------------------------------------------------------------- #
# Config resolution
# --------------------------------------------------------------------------- #

def test_resolve_harnesses_emits_json_and_claude_env() -> None:
    data = {
        "harnesses": {
            "deepagents": {"enabled": True, "default": True},
            "claude_code": {"enabled": True, "native": "read", "auto_approve": True},
        }
    }
    out = _resolve_harnesses(data)
    assert "PUPA_HARNESSES" in out
    assert json.loads(out["PUPA_HARNESSES"]) == data["harnesses"]
    # Nested claude knobs flatten onto the legacy env vars.
    assert out["PUPA_CLAUDE_LOOP_NATIVE"] == "read"
    assert out["PUPA_CLAUDE_LOOP_AUTO_APPROVE"] == "1"


def test_resolve_harnesses_absent_is_empty() -> None:
    assert _resolve_harnesses({}) == {}
