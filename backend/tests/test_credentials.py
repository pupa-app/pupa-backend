"""Tests for the credential stash (`credentials.py`).

The stash lets the LangGraph and Claude Code harnesses coexist: when the Claude
harness is enabled, billing-diverting vars are moved out of `os.environ` (so the
`claude` subprocess can't inherit them) into a private in-process dict that the
LangGraph model builders read via `get_credential`.
"""

import os

import pytest

import pupa_backend.credentials as credentials
from pupa_backend.harnesses.claude.env import FORBIDDEN_ENV_VARS


@pytest.fixture(autouse=True)
def _clear_stash():
    """Each test starts with an empty stash and leaves os.environ as it found it.

    `stash_forbidden_credentials` calls `del os.environ[k]` directly (not via
    monkeypatch), so we snapshot and restore the forbidden vars here to avoid
    leaking a deletion into later tests (e.g. the shared conftest AWS creds)."""
    saved = {k: os.environ.get(k) for k in FORBIDDEN_ENV_VARS}
    credentials._STASH.clear()
    yield
    credentials._STASH.clear()
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def test_stash_moves_present_vars_out_of_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA-test")

    moved = credentials.stash_forbidden_credentials()

    assert set(moved) >= {"ANTHROPIC_API_KEY", "AWS_ACCESS_KEY_ID"}
    # Gone from the process env — the `claude` subprocess can't inherit them.
    assert "ANTHROPIC_API_KEY" not in os.environ
    assert "AWS_ACCESS_KEY_ID" not in os.environ
    # But retrievable via the stash.
    assert credentials.get_credential("ANTHROPIC_API_KEY") == "sk-secret"
    assert credentials.get_credential("AWS_ACCESS_KEY_ID") == "AKIA-test"


def test_stash_skips_absent_and_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in FORBIDDEN_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")  # empty == absent

    moved = credentials.stash_forbidden_credentials()

    assert moved == []
    assert credentials._STASH == {}


def test_get_credential_reads_through_env_when_not_stashed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No scrub (Claude harness disabled) → get_credential reads live env."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-live")
    # Stash never populated.
    assert credentials.get_credential("ANTHROPIC_API_KEY") == "sk-live"


def test_get_credential_missing_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    assert credentials.get_credential("AWS_PROFILE") is None


def test_stash_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    first = credentials.stash_forbidden_credentials()
    second = credentials.stash_forbidden_credentials()
    assert "ANTHROPIC_API_KEY" in first
    assert second == []  # already moved
    assert credentials.get_credential("ANTHROPIC_API_KEY") == "sk-secret"
