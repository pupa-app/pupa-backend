"""The cloudflared *quick* tunnel must not inherit `~/.cloudflared/config.yml`.

`cloudflared tunnel --url …` reads the operator's global config by default. A
machine that has been through `pupa-backend setup`'s named-tunnel path has one,
and its required catch-all is `http_status:404` — so every request to the random
`*.trycloudflare.com` host (whose Host header matches no `hostname:` rule) is
answered by cloudflared itself with a bodiless 404 and never reaches uvicorn.

That failure is invisible from the backend log (no request arrives) and the iOS
client reports it as "pairing code unknown or expired", because a 404 on
`POST /auth/pair` is indistinguishable from a spent code.

So: pass an isolated `--config`, and warn when a global config exists.

`pupa_backend.app` is imported inside the tests, never at module scope: importing
it runs `stash_forbidden_credentials()`, which moves `AWS_*` out of `os.environ`
and would strip conftest's dummy credentials for the whole session.
"""

import logging
import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _restore_env():
    """Undo the credential stash the `app` import performs on first import."""
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


class _FakeProc:
    """Enough of `subprocess.Popen` for `_start_cloudflared_tunnel`'s reader."""

    def __init__(self, lines: list[str]) -> None:
        self.stdout = iter(lines)
        self.terminated = False

    def terminate(self) -> None:  # pragma: no cover - only on the timeout path
        self.terminated = True


def _patch_popen(monkeypatch, tmp_path: Path):
    """Capture argv, stub out cloudflared, and keep writes inside tmp_path."""
    from pupa_backend import app as app_mod

    calls: list[list[str]] = []

    def _fake_popen(argv, **kwargs):
        calls.append(list(argv))
        return _FakeProc(["Your quick Tunnel https://fake-tunnel-xyz.trycloudflare.com\n"])

    monkeypatch.setattr(app_mod.shutil, "which", lambda _: "/usr/local/bin/cloudflared")
    monkeypatch.setattr(app_mod.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(app_mod, "_TUNNEL_URL_FILE", tmp_path / "tunnel_url")
    monkeypatch.setattr(app_mod, "_QUICK_TUNNEL_CONFIG", tmp_path / "cloudflared-quick.yml")
    return app_mod, calls


def test_quick_tunnel_passes_isolated_config(monkeypatch, tmp_path: Path) -> None:
    app_mod, calls = _patch_popen(monkeypatch, tmp_path)
    monkeypatch.setattr(app_mod, "_CLOUDFLARED_USER_CONFIG", tmp_path / "absent.yml")

    assert app_mod._start_cloudflared_tunnel() is not None

    argv = calls[0]
    assert "--config" in argv, f"quick tunnel must isolate its config: {argv}"
    cfg = Path(argv[argv.index("--config") + 1])
    # `--config` is a global flag: it has to precede the `tunnel` subcommand.
    assert argv.index("--config") < argv.index("tunnel")
    assert cfg == tmp_path / "cloudflared-quick.yml"
    # cloudflared rejects a zero-byte config ("Configuration file … was empty").
    assert cfg.exists() and cfg.read_text().strip()


def test_quick_tunnel_warns_when_global_config_would_be_shadowed(
    monkeypatch, tmp_path: Path, caplog
) -> None:
    app_mod, _ = _patch_popen(monkeypatch, tmp_path)
    user_cfg = tmp_path / "config.yml"
    user_cfg.write_text(
        "tunnel: 0000\ningress:\n  - hostname: api.example.test\n"
        "    service: http://localhost:8004\n  - service: http_status:404\n"
    )
    monkeypatch.setattr(app_mod, "_CLOUDFLARED_USER_CONFIG", user_cfg)

    with caplog.at_level(logging.WARNING, logger=app_mod.logger.name):
        assert app_mod._start_cloudflared_tunnel() is not None

    warning = "\n".join(r.getMessage() for r in caplog.records)
    assert str(user_cfg) in warning
    assert "cloudflared" in warning.lower()


def test_named_tunnel_keeps_the_global_config(monkeypatch, tmp_path: Path) -> None:
    """The named tunnel *needs* those ingress rules — don't isolate that one."""
    app_mod, calls = _patch_popen(monkeypatch, tmp_path)

    assert app_mod._start_named_tunnel("pupa-backend", "api.example.test") is not None
    assert "--config" not in calls[0]
