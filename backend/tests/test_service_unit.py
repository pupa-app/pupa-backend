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

import pupa_backend.scripts.service as service


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
