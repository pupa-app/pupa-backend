"""Publish the loopback-bound backend to the tailnet via `tailscale serve`.

macOS 15+ gates connections to wildcard-bound (`0.0.0.0`) sockets behind the
Local Network privacy grant. When that grant is missing — common right after a
Tailscale install — nothing reaches the backend, not even `pupa-backend pair`
on the same machine: SYNs stall in `SYN_RCVD`. Loopback-bound sockets are
never gated.

So with `connectivity: tailscale` the backend binds `127.0.0.1` and tailscaled
(which holds its own network permission) forwards the tailnet into it, in one
of two modes:

- **https** (preferred, needs HTTPS enabled for the tailnet) — tailscaled
  terminates TLS on :443 with an auto-renewing Let's Encrypt cert for the
  node's MagicDNS name and proxies plain HTTP to the backend. The client sees
  an ordinary trusted cert: no fingerprint pinning, and no self-signed cert for
  iOS's App Transport Security to refuse.
- **tcp** (fallback) — raw TCP passthrough on the backend's own port, so the
  backend keeps serving its self-signed cert end-to-end and the client pins the
  fingerprint.

Opt out entirely with `PUPA_TAILSCALE_SERVE=0` (backend falls back to
`0.0.0.0`); force a mode with `PUPA_TAILSCALE_SERVE=tcp` / `=https`.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess

logger = logging.getLogger("uvicorn.error")

# macOS app bundle ships the CLI here; the standalone/Homebrew install is on PATH.
_APP_BUNDLE_BIN = "/Applications/Tailscale.app/Contents/MacOS/Tailscale"

_OFF_VALUES = {"0", "false", "no", "off"}


def tailscale_bin() -> str | None:
    return shutil.which("tailscale") or (
        _APP_BUNDLE_BIN if os.path.exists(_APP_BUNDLE_BIN) else None
    )


def should_proxy() -> bool:
    """True when this deploy reaches devices over Tailscale and the CLI is here."""
    if os.getenv("PUPA_CONNECTIVITY", "").strip().lower() != "tailscale":
        return False
    if os.getenv("PUPA_TAILSCALE_SERVE", "").strip().lower() in _OFF_VALUES:
        return False
    return tailscale_bin() is not None


def _run(argv: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, text=True, timeout=20)


def cert_domain() -> str | None:
    """The node's cert domain, set only when the tailnet has HTTPS enabled.

    Empty `CertDomains` means the admin console's HTTPS switch is off, so
    tailscaled cannot mint a Let's Encrypt cert and `--https` would fail.
    """
    bin_path = tailscale_bin()
    if bin_path is None:
        return None
    try:
        result = _run([bin_path, "status", "--json"])
        domains = json.loads(result.stdout).get("CertDomains") or []
    except (OSError, subprocess.SubprocessError, ValueError, AttributeError):
        return None
    return domains[0] if domains else None


class ServeProxy:
    """A registered `tailscale serve` forward; `stop()` removes it."""

    def __init__(self, bin_path: str, port: int, mode: str, domain: str | None = None):
        self._bin = bin_path
        self.port = port
        self.mode = mode  # "https" | "tcp"
        self.domain = domain

    @property
    def terminates_tls(self) -> bool:
        """True when tailscaled owns TLS, so the backend serves plain HTTP."""
        return self.mode == "https"

    @property
    def public_url(self) -> str | None:
        """URL a tailnet device should use — https mode only (no port suffix)."""
        return f"https://{self.domain}" if self.mode == "https" and self.domain else None

    @property
    def _serve_args(self) -> list[str]:
        if self.mode == "https":
            return ["--https=443", f"http://127.0.0.1:{self.port}"]
        return [f"--tcp={self.port}", f"tcp://127.0.0.1:{self.port}"]

    @property
    def _off_flag(self) -> str:
        return "--https=443" if self.mode == "https" else f"--tcp={self.port}"

    def stop(self) -> None:
        try:
            _run([self._bin, "serve", self._off_flag, "off"])
        except (OSError, subprocess.SubprocessError) as exc:  # best-effort teardown
            logger.warning("tailscale serve teardown failed: %s", exc)


def _forced_mode() -> str | None:
    forced = os.getenv("PUPA_TAILSCALE_SERVE", "").strip().lower()
    return forced if forced in {"https", "tcp"} else None


def start_serve_proxy(port: int) -> ServeProxy | None:
    """Register the tailnet→loopback forward. Returns None when not applicable."""
    if not should_proxy():
        return None
    bin_path = tailscale_bin()
    assert bin_path is not None  # should_proxy() checked it

    forced = _forced_mode()
    domain = cert_domain()
    if forced == "tcp":
        mode = "tcp"
    elif forced == "https" or domain:
        mode = "https"
    else:
        mode = "tcp"
        logger.warning(
            "tailnet HTTPS is off, falling back to raw-TCP passthrough with the "
            "self-signed cert — iOS may refuse it. Enable HTTPS at "
            "https://login.tailscale.com/admin/dns and restart for a trusted cert."
        )

    proxy = ServeProxy(bin_path, port, mode, domain)
    try:
        result = _run([bin_path, "serve", "--bg", "--yes", *proxy._serve_args])
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("tailscale serve failed (%s) — binding 0.0.0.0 instead", exc)
        return None
    if result.returncode != 0:
        logger.warning(
            "tailscale serve failed (exit %s: %s) — binding 0.0.0.0 instead",
            result.returncode,
            (result.stderr or result.stdout or "").strip(),
        )
        return None
    if proxy.mode == "https":
        logger.info(
            "tailscale serve: %s (trusted cert, TLS terminated by tailscaled) "
            "→ http://127.0.0.1:%s",
            proxy.public_url,
            port,
        )
    else:
        logger.info("tailscale serve: tailnet tcp/%s → tcp://127.0.0.1:%s", port, port)
    return proxy


def bind_host(proxied: bool) -> str:
    """Bind address for uvicorn. Explicit `PUPA_HOST` always wins."""
    explicit = os.getenv("PUPA_HOST", "").strip()
    if explicit:
        return explicit
    return "127.0.0.1" if proxied else "0.0.0.0"
