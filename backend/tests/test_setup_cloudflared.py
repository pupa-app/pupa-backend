"""Tests for the Cloudflare named-tunnel onboarding helpers in `scripts/setup.py`
plus the `cloudflared.*` → env-var mapping in `pupa_config`.

The interactive wizard flow itself isn't exercised here; we cover the pure /
parsing pieces that the full-auto path depends on:
  - the generated ~/.cloudflared/config.yml content (round-trips through YAML)
  - UUID parsing from `cloudflared tunnel create` output
  - existing-tunnel lookup (skips soft-deleted tunnels)
  - DNS routing treating an already-existing route as success
  - config.yml `cloudflared:` block → PUPA_CLOUDFLARED_* env vars
"""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml

from pupa_backend.scripts import setup  # noqa: E402
from pupa_backend.pupa_config import _yaml_to_env_dict  # noqa: E402


# ---------------------------------------------------------------------------
# _cloudflared_config_yaml
# ---------------------------------------------------------------------------

def test_cloudflared_config_yaml_roundtrips() -> None:
    text = setup._cloudflared_config_yaml(
        "31753954-382f-481f-a2c6-2e84e0fad0dd",
        "/home/u/.cloudflared/31753954-382f-481f-a2c6-2e84e0fad0dd.json",
        "api.example.test",
    )
    data = yaml.safe_load(text)
    assert data["tunnel"] == "31753954-382f-481f-a2c6-2e84e0fad0dd"
    assert data["credentials-file"].endswith(".json")
    # First ingress rule routes the hostname to the local backend; last is the
    # required catch-all.
    assert data["ingress"][0] == {
        "hostname": "api.example.test",
        "service": "http://localhost:8004",
    }
    assert data["ingress"][-1] == {"service": "http_status:404"}


# ---------------------------------------------------------------------------
# _create_tunnel — UUID parsing
# ---------------------------------------------------------------------------

def test_create_tunnel_parses_uuid_from_output(monkeypatch) -> None:
    out = (
        "Tunnel credentials written to "
        "/home/u/.cloudflared/31753954-382f-481f-a2c6-2e84e0fad0dd.json.\n"
        "Created tunnel pupa-backend with id 31753954-382f-481f-a2c6-2e84e0fad0dd\n"
    )
    monkeypatch.setattr(
        setup.subprocess, "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout=out, stderr=""),
    )
    assert setup._create_tunnel("pupa-backend") == "31753954-382f-481f-a2c6-2e84e0fad0dd"


def test_create_tunnel_returns_none_on_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        setup.subprocess, "run",
        lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr="boom"),
    )
    assert setup._create_tunnel("pupa-backend") is None


# ---------------------------------------------------------------------------
# _find_tunnel_id
# ---------------------------------------------------------------------------

def test_find_tunnel_id_matches_name_and_skips_deleted(monkeypatch) -> None:
    listing = (
        '[{"id": "aaaa1111-382f-481f-a2c6-2e84e0fad0dd", "name": "pupa-backend",'
        ' "deleted_at": "2026-01-01T00:00:00Z"},'
        ' {"id": "bbbb2222-382f-481f-a2c6-2e84e0fad0dd", "name": "pupa-backend"}]'
    )
    monkeypatch.setattr(
        setup.subprocess, "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout=listing, stderr=""),
    )
    assert setup._find_tunnel_id("pupa-backend") == "bbbb2222-382f-481f-a2c6-2e84e0fad0dd"


def test_find_tunnel_id_treats_zero_time_as_live(monkeypatch) -> None:
    """cloudflared reports a LIVE tunnel's deleted_at as the Go zero-time, not
    null — it must still be found (regression: it was treated as deleted)."""
    listing = (
        '[{"id": "31753954-382f-481f-a2c6-2e84e0fad0dd", "name": "pupa-backend",'
        ' "deleted_at": "0001-01-01T00:00:00Z", "connections": []}]'
    )
    monkeypatch.setattr(
        setup.subprocess, "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout=listing, stderr=""),
    )
    assert setup._find_tunnel_id("pupa-backend") == "31753954-382f-481f-a2c6-2e84e0fad0dd"


def test_find_tunnel_id_missing_binary_returns_none(monkeypatch) -> None:
    def _raise(*a, **k):
        raise FileNotFoundError
    monkeypatch.setattr(setup.subprocess, "run", _raise)
    assert setup._find_tunnel_id("pupa-backend") is None


# ---------------------------------------------------------------------------
# _route_tunnel_dns
# ---------------------------------------------------------------------------

def test_route_tunnel_dns_success(monkeypatch) -> None:
    monkeypatch.setattr(
        setup.subprocess, "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout="Added CNAME", stderr=""),
    )
    assert setup._route_tunnel_dns("pupa-backend", "api.example.test") is True


def test_route_tunnel_dns_already_exists_counts_as_success(monkeypatch) -> None:
    monkeypatch.setattr(
        setup.subprocess, "run",
        lambda *a, **k: SimpleNamespace(
            returncode=1, stdout="", stderr="record api.example.test already exists",
        ),
    )
    assert setup._route_tunnel_dns("pupa-backend", "api.example.test") is True


def test_route_tunnel_dns_other_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        setup.subprocess, "run",
        lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr="auth error"),
    )
    assert setup._route_tunnel_dns("pupa-backend", "api.example.test") is False


# ---------------------------------------------------------------------------
# pupa_config — cloudflared.* mapping
# ---------------------------------------------------------------------------

def test_cloudflared_block_maps_to_env() -> None:
    data = {
        "connectivity": "cloudflared",
        "cloudflared": {"hostname": "api.example.test", "tunnel": "pupa-backend"},
    }
    env = _yaml_to_env_dict(data)
    assert env["PUPA_CONNECTIVITY"] == "cloudflared"
    assert env["PUPA_CLOUDFLARED_HOSTNAME"] == "api.example.test"
    assert env["PUPA_CLOUDFLARED_TUNNEL"] == "pupa-backend"


def test_no_cloudflared_block_omits_env() -> None:
    env = _yaml_to_env_dict({"connectivity": "cloudflared"})
    assert "PUPA_CLOUDFLARED_HOSTNAME" not in env
    assert "PUPA_CLOUDFLARED_TUNNEL" not in env
