"""Tests for the OS-service unit generator in `scripts/service.py`.

These guard the two regressions that made the systemd user service
crash-loop and silently drop MCP config:

- `_systemd_env_line` must quote values so systemd does not split a
  value containing spaces (e.g. the PUPA_MCP_SERVERS JSON blob) on
  whitespace, and must escape embedded quotes/backslash/percent so JSON
  round-trips intact.
- `_service_path` must carry the installing user's PATH (plus the dir
  the `claude` binary lives in) into the unit — otherwise the service
  starts with a minimal PATH, `claude` is not found, and the
  claude_code agent loop aborts startup.
"""

import os
from pathlib import Path

import pytest

import pupa_backend.scripts.service as service
from pupa_backend.pupa_config import known_env_vars


def test_env_line_simple_value_is_quoted():
    assert service._systemd_env_line("A", "b") == 'Environment="A=b"\n'


def test_env_line_value_with_spaces_stays_one_token():
    # An unquoted `Environment=K=x y` makes systemd treat `y` as a second
    # assignment. The whole thing must be wrapped in quotes.
    line = service._systemd_env_line("DESC", "x y z")
    assert line == 'Environment="DESC=x y z"\n'


def test_env_line_json_quotes_are_escaped():
    value = '{"k": "v v"}'
    line = service._systemd_env_line("PUPA_MCP_SERVERS", value)
    # Inner double-quotes escaped so systemd unquotes back to the original JSON.
    assert line == 'Environment="PUPA_MCP_SERVERS={\\"k\\": \\"v v\\"}"\n'


def test_env_line_backslash_and_percent_are_escaped():
    assert service._systemd_env_line("A", "a\\b") == 'Environment="A=a\\\\b"\n'
    # `%` is a systemd specifier char — must be doubled to survive literally.
    assert service._systemd_env_line("A", "50%") == 'Environment="A=50%%"\n'


def test_service_path_prepends_claude_dir(monkeypatch):
    monkeypatch.setattr(service.shutil, "which", lambda name: "/opt/tools/claude")
    path = service._service_path({"PATH": "/usr/bin:/bin"})
    parts = path.split(os.pathsep)
    assert parts[0] == "/opt/tools"
    assert "/usr/bin" in parts and "/bin" in parts


def test_service_path_no_duplicate_claude_dir(monkeypatch):
    monkeypatch.setattr(service.shutil, "which", lambda name: "/usr/local/bin/claude")
    path = service._service_path({"PATH": "/usr/local/bin:/usr/bin"})
    assert path.split(os.pathsep).count("/usr/local/bin") == 1


def test_service_path_falls_back_to_process_env(monkeypatch):
    monkeypatch.setattr(service.shutil, "which", lambda name: None)
    monkeypatch.setenv("PATH", "/from/process/env")
    # No PATH in the config dict → fall back to the installing process PATH.
    assert service._service_path({}) == "/from/process/env"


# ---------------------------------------------------------------------------
# Shell-only credential guard
# ---------------------------------------------------------------------------
#
# `pupa-backend run` inherits the installing shell's environment; a launchd
# agent / systemd unit does not. A key exported in `.zshrc` therefore works
# interactively and vanishes under the service, which crash-loops at startup
# with an opaque MissingCredentialsError. The guard catches that at install
# time, while the operator is still looking at the terminal.
#
# The var list is not maintained here — it is derived from
# `pupa_config.known_env_vars()`, and operators extend it with
# `service.check_env` in config.yml.


@pytest.fixture
def clean_env(monkeypatch):
    """Drop every candidate var so the developer's own shell can't leak in."""
    for name in known_env_vars():
        monkeypatch.delenv(name, raising=False)
    for name in ("MY_CUSTOM_TOKEN", "AWS_ACCESS_KEY_ID"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("PUPA_SERVICE_ALLOW_SHELL_ONLY", raising=False)
    return monkeypatch


def test_shell_only_secrets_flags_var_missing_from_config(clean_env):
    clean_env.setenv("OPENROUTER_API_KEY", "sk-or-xxx")
    assert service._shell_only_secrets({}) == ["OPENROUTER_API_KEY"]


def test_shell_only_secrets_ignores_var_present_in_config(clean_env):
    clean_env.setenv("OPENROUTER_API_KEY", "sk-or-xxx")
    assert service._shell_only_secrets({"OPENROUTER_API_KEY": "sk-or-yyy"}) == []


def test_shell_only_secrets_ignores_blank_config_value(clean_env):
    # A key present but empty in config.yml would still fail at startup.
    clean_env.setenv("OPENROUTER_API_KEY", "sk-or-xxx")
    assert service._shell_only_secrets({"OPENROUTER_API_KEY": ""}) == ["OPENROUTER_API_KEY"]


def test_shell_only_secrets_ignores_blank_shell_value(clean_env):
    clean_env.setenv("OPENROUTER_API_KEY", "")
    assert service._shell_only_secrets({}) == []


def test_shell_only_secrets_ignores_vars_outside_the_config_schema(clean_env):
    clean_env.setenv("EDITOR", "vim")
    assert service._shell_only_secrets({}) == []


def test_shell_only_secrets_reports_every_offender_sorted(clean_env):
    clean_env.setenv("TAVILY_API_KEY", "tvly-x")
    clean_env.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    assert service._shell_only_secrets({}) == ["ANTHROPIC_API_KEY", "TAVILY_API_KEY"]


def test_shell_only_secrets_honours_operator_supplied_extra_names(clean_env):
    """`service.check_env` extends the check to vars the schema can't name —
    raw AWS keys, an MCP server's token — which live under `env:` in config.yml."""
    clean_env.setenv("MY_CUSTOM_TOKEN", "abc")
    assert service._shell_only_secrets({}) == []
    assert service._shell_only_secrets({}, extra=["MY_CUSTOM_TOKEN"]) == ["MY_CUSTOM_TOKEN"]


def test_check_env_names_reads_config_block():
    assert service._check_env_names({"service": {"check_env": ["A", "B"]}}) == ["A", "B"]


def test_check_env_names_absent_block_is_empty():
    assert service._check_env_names({}) == []


def test_check_env_names_ignores_malformed_block():
    assert service._check_env_names({"service": {"check_env": "A"}}) == []


def test_assert_no_shell_only_secrets_exits_with_actionable_message(clean_env):
    clean_env.setenv("OPENROUTER_API_KEY", "sk-or-xxx")
    with pytest.raises(SystemExit) as exc:
        service._assert_no_shell_only_secrets({})
    msg = str(exc.value)
    assert "OPENROUTER_API_KEY" in msg
    assert "config.yml" in msg
    assert "PUPA_SERVICE_ALLOW_SHELL_ONLY=1" in msg
    # Never echo the secret itself into the terminal.
    assert "sk-or-xxx" not in msg


def test_assert_message_hint_comes_from_the_config_schema(clean_env):
    clean_env.setenv("OPENROUTER_API_KEY", "sk-or-xxx")
    with pytest.raises(SystemExit) as exc:
        service._assert_no_shell_only_secrets({})
    assert known_env_vars()["OPENROUTER_API_KEY"] in str(exc.value)


def test_assert_message_points_extra_names_at_the_env_block(clean_env):
    clean_env.setenv("MY_CUSTOM_TOKEN", "abc")
    with pytest.raises(SystemExit) as exc:
        service._assert_no_shell_only_secrets({}, extra=["MY_CUSTOM_TOKEN"])
    assert "env.MY_CUSTOM_TOKEN" in str(exc.value)


def test_assert_no_shell_only_secrets_passes_when_config_has_them(clean_env):
    clean_env.setenv("OPENROUTER_API_KEY", "sk-or-xxx")
    service._assert_no_shell_only_secrets({"OPENROUTER_API_KEY": "sk-or-yyy"})


def test_assert_no_shell_only_secrets_bypass(clean_env):
    clean_env.setenv("OPENROUTER_API_KEY", "sk-or-xxx")
    clean_env.setenv("PUPA_SERVICE_ALLOW_SHELL_ONLY", "1")
    service._assert_no_shell_only_secrets({})


# ---------------------------------------------------------------------------
# Thin units — config.yml is read at runtime, not snapshotted at install
# ---------------------------------------------------------------------------
#
# `app.py` calls `load_pupa_config()` at import, so the service process reads
# config.yml itself. Baking those values into the unit bought nothing and cost
# two real bugs: the plist is written 0644 (config.yml is 0600), so install
# copied every secret into a world-readable file; and the snapshot froze at
# install time, so editing config.yml did nothing until a reinstall.
#
# PATH is the exception — launchd/systemd start with a minimal PATH and the
# backend cannot reconstruct the operator's, so it stays in the unit.

@pytest.fixture
def fake_config(monkeypatch):
    import pupa_backend.pupa_config as pupa_config

    monkeypatch.setattr(
        pupa_config, "load_pupa_config",
        lambda apply=True: {"PUPA_API_KEY": "super-secret", "LLM_PROVIDER": "openrouter"},
    )
    monkeypatch.setattr(service.shutil, "which", lambda name: None)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")


def test_launchd_plist_carries_path(fake_config, tmp_path):
    plist = service._launchd_plist(tmp_path, Path("/usr/bin/python3"))
    assert "<key>PATH</key>" in plist
    assert "/usr/bin:/bin" in plist


def test_launchd_plist_does_not_embed_secrets(fake_config, tmp_path):
    plist = service._launchd_plist(tmp_path, Path("/usr/bin/python3"))
    assert "super-secret" not in plist
    assert "PUPA_API_KEY" not in plist


def test_systemd_unit_carries_path(fake_config, tmp_path):
    unit = service._systemd_unit(tmp_path, Path("/usr/bin/python3"))
    assert 'Environment="PATH=/usr/bin:/bin"' in unit


def test_systemd_unit_does_not_embed_secrets(fake_config, tmp_path):
    unit = service._systemd_unit(tmp_path, Path("/usr/bin/python3"))
    assert "super-secret" not in unit
    assert "PUPA_API_KEY" not in unit
