"""`tailscale serve` proxy: gating, argv, teardown, and bind-host choice."""

from __future__ import annotations

import subprocess

import pytest

from pupa_backend import tailscale_proxy as tp


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    for var in ("PUPA_CONNECTIVITY", "PUPA_TAILSCALE_SERVE", "PUPA_HOST"):
        monkeypatch.delenv(var, raising=False)


class _Runner:
    """Records argv and returns a canned CompletedProcess."""

    def __init__(self, returncode: int = 0):
        self.calls: list[list[str]] = []
        self.returncode = returncode

    def __call__(self, argv, **kwargs):
        self.calls.append(list(argv))
        return subprocess.CompletedProcess(argv, self.returncode, stdout="", stderr="")


def test_disabled_without_tailscale_connectivity(monkeypatch):
    monkeypatch.setattr(tp, "tailscale_bin", lambda: "/usr/local/bin/tailscale")
    assert tp.should_proxy() is False

    monkeypatch.setenv("PUPA_CONNECTIVITY", "cloudflared")
    assert tp.should_proxy() is False


def test_enabled_for_tailscale_connectivity(monkeypatch):
    monkeypatch.setenv("PUPA_CONNECTIVITY", "tailscale")
    monkeypatch.setattr(tp, "tailscale_bin", lambda: "/usr/local/bin/tailscale")
    assert tp.should_proxy() is True


def test_opt_out_env_wins(monkeypatch):
    monkeypatch.setenv("PUPA_CONNECTIVITY", "tailscale")
    monkeypatch.setenv("PUPA_TAILSCALE_SERVE", "0")
    monkeypatch.setattr(tp, "tailscale_bin", lambda: "/usr/local/bin/tailscale")
    assert tp.should_proxy() is False


def test_disabled_when_cli_missing(monkeypatch):
    monkeypatch.setenv("PUPA_CONNECTIVITY", "tailscale")
    monkeypatch.setattr(tp, "tailscale_bin", lambda: None)
    assert tp.should_proxy() is False


def test_start_registers_raw_tcp_forward_and_stop_removes_it(monkeypatch):
    monkeypatch.setenv("PUPA_CONNECTIVITY", "tailscale")
    monkeypatch.setattr(tp, "tailscale_bin", lambda: "/usr/local/bin/tailscale")
    monkeypatch.setattr(tp, "cert_domain", lambda: None)  # tailnet HTTPS off
    runner = _Runner()
    monkeypatch.setattr(tp, "_run", runner)

    proxy = tp.start_serve_proxy(8004)
    assert proxy is not None
    assert proxy.mode == "tcp"
    assert proxy.terminates_tls is False
    assert proxy.public_url is None
    assert runner.calls[-1] == [
        "/usr/local/bin/tailscale", "serve", "--bg", "--yes",
        "--tcp=8004", "tcp://127.0.0.1:8004",
    ]

    proxy.stop()
    assert runner.calls[-1] == [
        "/usr/local/bin/tailscale", "serve", "--tcp=8004", "off",
    ]


def test_https_mode_when_tailnet_has_certs(monkeypatch):
    monkeypatch.setenv("PUPA_CONNECTIVITY", "tailscale")
    monkeypatch.setattr(tp, "tailscale_bin", lambda: "/usr/local/bin/tailscale")
    monkeypatch.setattr(tp, "cert_domain", lambda: "host.tail1234.ts.net")
    runner = _Runner()
    monkeypatch.setattr(tp, "_run", runner)

    proxy = tp.start_serve_proxy(8004)
    assert proxy is not None
    assert proxy.terminates_tls is True
    assert proxy.public_url == "https://host.tail1234.ts.net"
    assert runner.calls[-1] == [
        "/usr/local/bin/tailscale", "serve", "--bg", "--yes",
        "--https=443", "http://127.0.0.1:8004",
    ]

    proxy.stop()
    assert runner.calls[-1] == [
        "/usr/local/bin/tailscale", "serve", "--https=443", "off",
    ]


def test_forced_tcp_mode_ignores_available_certs(monkeypatch):
    monkeypatch.setenv("PUPA_CONNECTIVITY", "tailscale")
    monkeypatch.setenv("PUPA_TAILSCALE_SERVE", "tcp")
    monkeypatch.setattr(tp, "tailscale_bin", lambda: "/usr/local/bin/tailscale")
    monkeypatch.setattr(tp, "cert_domain", lambda: "host.tail1234.ts.net")
    monkeypatch.setattr(tp, "_run", _Runner())

    proxy = tp.start_serve_proxy(8004)
    assert proxy is not None and proxy.mode == "tcp"


def test_start_returns_none_when_serve_fails(monkeypatch):
    monkeypatch.setenv("PUPA_CONNECTIVITY", "tailscale")
    monkeypatch.setattr(tp, "tailscale_bin", lambda: "/usr/local/bin/tailscale")
    monkeypatch.setattr(tp, "cert_domain", lambda: None)
    monkeypatch.setattr(tp, "_run", _Runner(returncode=1))

    assert tp.start_serve_proxy(8004) is None


def test_bind_host_prefers_loopback_behind_the_proxy(monkeypatch):
    assert tp.bind_host(proxied=False) == "0.0.0.0"
    assert tp.bind_host(proxied=True) == "127.0.0.1"

    monkeypatch.setenv("PUPA_HOST", "192.168.1.5")
    assert tp.bind_host(proxied=True) == "192.168.1.5"


def test_local_backend_url_is_http_when_tailscaled_terminates_tls(monkeypatch):
    """A configured tls.cert must not make `pair` post https:// at a backend
    that serves plain HTTP behind `tailscale serve --https`."""
    from pupa_backend import cli

    monkeypatch.setenv("PUPA_TLS_CERT", "/tmp/server.crt")
    monkeypatch.setenv("PUPA_CONNECTIVITY", "tailscale")
    monkeypatch.setattr(tp, "tailscale_bin", lambda: "/usr/local/bin/tailscale")

    monkeypatch.setattr(tp, "cert_domain", lambda: "host.tail1234.ts.net")
    assert cli._local_backend_url() == "http://localhost:8004"

    # No tailnet certs → raw-TCP passthrough, backend serves its own TLS again.
    monkeypatch.setattr(tp, "cert_domain", lambda: None)
    assert cli._local_backend_url() == "https://localhost:8004"


def test_local_backend_url_honours_port(monkeypatch):
    from pupa_backend import cli

    monkeypatch.delenv("PUPA_TLS_CERT", raising=False)
    monkeypatch.setenv("PORT", "9000")
    assert cli._local_backend_url() == "http://localhost:9000"
