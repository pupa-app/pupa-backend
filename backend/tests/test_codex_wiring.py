"""Registry, config, setup, and service wiring for the Codex harness."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from pupa_backend.harnesses import CodexHarness, build_registry
from pupa_backend.pupa_config import _resolve_harnesses, known_env_vars
from pupa_backend.scripts import service, setup


def test_registry_builds_codex_harness(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "PUPA_HARNESSES",
        json.dumps({"codex": {"enabled": True, "default": True}}),
    )
    registry = build_registry()
    assert registry.ids() == ["codex"]
    assert registry.default().id == "codex"


def test_codex_permission_schema() -> None:
    controls = {row["key"]: row for row in CodexHarness().permission_schema()}
    assert controls["codex_loop_native"]["options"] == ["read", "workspace", "full"]
    assert controls["codex_loop_native"]["default"] == "workspace"
    assert controls["codex_loop_auto_approve"]["default"] is False


def test_codex_nested_config_flattens_to_environment() -> None:
    output = _resolve_harnesses(
        {
            "harnesses": {
                "codex": {
                    "enabled": True,
                    "native": "read",
                    "auto_approve": True,
                    "model": "codex-model",
                    "workspace": "/work",
                    "config_dir": "/config",
                    "binary": "/bin/codex",
                }
            }
        }
    )
    assert output["PUPA_CODEX_LOOP_NATIVE"] == "read"
    assert output["PUPA_CODEX_LOOP_AUTO_APPROVE"] == "1"
    assert output["PUPA_CODEX_MODEL"] == "codex-model"
    assert output["PUPA_CODEX_WORKSPACE"] == "/work"
    assert output["PUPA_CODEX_CONFIG_DIR"] == "/config"
    assert output["PUPA_CODEX_BIN"] == "/bin/codex"
    assert known_env_vars()["PUPA_CODEX_BIN"] == "harnesses.codex.binary"
    assert known_env_vars()["PUPA_DEFAULT_MODEL"] == "default_model"


def test_service_path_includes_both_cli_directories(monkeypatch: pytest.MonkeyPatch) -> None:
    paths = {"claude": "/opt/claude/bin/claude", "codex": "/opt/codex/bin/codex"}
    monkeypatch.setattr(service.shutil, "which", paths.get)
    result = service._service_path({"PATH": "/usr/bin:/bin"}).split(":")
    assert result[:2] == ["/opt/claude/bin", "/opt/codex/bin"]
    assert result[2:] == ["/usr/bin", "/bin"]


def test_setup_codex_preflight_accepts_chatgpt_login(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(setup.shutil, "which", lambda name: "/usr/bin/codex")
    monkeypatch.setattr(
        setup.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout="Logged in using ChatGPT\n", stderr=""
        ),
    )
    ok, message = setup._check_codex_subscription()
    assert ok is True
    assert "ChatGPT" in message


def test_setup_codex_preflight_rejects_api_login(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(setup.shutil, "which", lambda name: "/usr/bin/codex")
    monkeypatch.setattr(
        setup.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout="Logged in using an API key\n", stderr=""
        ),
    )
    assert setup._check_codex_subscription()[0] is False
