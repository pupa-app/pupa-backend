"""Tests for the startup auth-state banner in `app._log_auth_state`.

The banner is the operator's one signal that they're in an unusable state
(e.g. fresh `make backend` with neither bootstrap key nor disabled flag),
so the four branches need test coverage to stay honest.
"""

import logging
from pathlib import Path

import pytest

from pupa_backend.auth.devices import reset_for_testing


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("PUPA_API_KEY", raising=False)
    monkeypatch.delenv("PUPA_AUTH_DISABLED", raising=False)
    reset_for_testing(tmp_path / "devices.json")


async def _run_banner(caplog: pytest.LogCaptureFixture) -> str:
    """Run the banner with INFO+ captured; return the concatenated message text."""
    from pupa_backend.app import _log_auth_state  # local import — app.py runs at module load

    # Banner uses uvicorn.error so its messages flow through uvicorn's
    # pre-configured handler at runtime.
    caplog.set_level(logging.INFO, logger="uvicorn.error")
    caplog.clear()
    await _log_auth_state()
    return "\n".join(record.getMessage() for record in caplog.records)


@pytest.mark.asyncio
async def test_disabled_env_warns_about_open_backend(
    monkeypatch: pytest.MonkeyPatch, clean_env: None, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("PUPA_AUTH_DISABLED", "1")
    text = await _run_banner(caplog)
    assert "DISABLED" in text
    assert any(r.levelno >= logging.WARNING for r in caplog.records)


@pytest.mark.asyncio
async def test_api_key_set_no_devices_invites_first_pair(
    monkeypatch: pytest.MonkeyPatch, clean_env: None, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("PUPA_API_KEY", "boot")
    text = await _run_banner(caplog)
    assert "no paired devices" in text
    assert "make pair" in text


@pytest.mark.asyncio
async def test_api_key_set_with_devices_suggests_unsetting(
    monkeypatch: pytest.MonkeyPatch, clean_env: None, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("PUPA_API_KEY", "boot")
    await reset_for_testing().issue(label="iphone")  # adds one paired device
    text = await _run_banner(caplog)
    assert "paired device" in text
    assert "unset" in text.lower() or "Once you trust" in text


@pytest.mark.asyncio
async def test_no_api_key_with_devices_running_pure_pair(
    clean_env: None, caplog: pytest.LogCaptureFixture
) -> None:
    await reset_for_testing().issue(label="iphone")
    text = await _run_banner(caplog)
    assert "paired device" in text
    assert "bootstrap key not set" in text


@pytest.mark.asyncio
async def test_nothing_configured_warns_loudly(
    clean_env: None, caplog: pytest.LogCaptureFixture
) -> None:
    """The dangerous state: backend running, nothing configured. Every gated
    route 401s. The banner is the only feedback the operator gets, so it has
    to be a WARNING not an INFO."""
    text = await _run_banner(caplog)
    assert "nothing is configured" in text
    assert any(r.levelno >= logging.WARNING for r in caplog.records)
    assert "PUPA_API_KEY" in text
    assert "PUPA_AUTH_DISABLED" in text
