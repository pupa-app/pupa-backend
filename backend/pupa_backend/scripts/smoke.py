"""End-to-end smoke test for a running Pupa backend.

Runs a battery of HTTP checks against the live backend (local or Railway)
to verify auth + per-route authorization match the contract documented in
[docs/architecture.md](../docs/architecture.md):

- ``/auth/config`` public probe shape (authRequired, methods, version).
- Middleware: missing/wrong bearer → 401, valid bearer → 200.
- ``PUPA_API_KEY`` identity bypasses every scope check.
- Full-scope device: passes scope-gated routes (``/db/threads/*``,
  ``/backend-tools``); blocked from operator-only routes
  (``/auth/devices``) with a clean 403.
- Limited-scope device (``agent``+``db`` only): blocked from
  ``/backend-tools`` (no ``tools`` scope) but still passes
  ``/db/threads/*`` and gets the same 403 on operator-only routes.

Mints two short-lived device tokens via ``/auth/pair/begin`` and revokes
them in a ``finally`` so the device store stays clean even when an
assertion fails. Exits non-zero on any failure so CI can use it.

Usage::

    # Local — reads ~/.pupa-backend/config.yml for the bootstrap key.
    python backend/scripts/smoke.py

    # Railway (or any remote) — pass both explicitly.
    python backend/scripts/smoke.py \\
        --base-url https://<your-service>.up.railway.app \\
        --api-key  "$PUPA_API_KEY"

    # Self-signed local HTTPS:
    python backend/scripts/smoke.py --base-url https://localhost:8004 --insecure

Stdlib-only (urllib + yaml-or-grep) so it runs on a fresh laptop without
backend deps installed.
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_BASE = "http://localhost:8004"
CONFIG_PATH = Path.home() / ".pupa-backend" / "config.yml"


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


def _api_key_from_config() -> str | None:
    """Pull ``auth.api_key`` out of ``~/.pupa-backend/config.yml`` if present.

    PyYAML is an optional dep here — fall back to a tiny grep for the line
    so the script runs on a fresh clone without ``uv sync`` having run.
    """
    if not CONFIG_PATH.exists():
        return None
    try:
        import yaml  # type: ignore[import]

        data = yaml.safe_load(CONFIG_PATH.read_text()) or {}
        key = ((data.get("auth") or {}).get("api_key") or "").strip()
        return key or None
    except ImportError:
        for line in CONFIG_PATH.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("api_key:"):
                value = stripped.split(":", 1)[1].strip().strip("'\"")
                return value or None
        return None


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def _request(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    json_body: Any = None,
    insecure: bool = False,
) -> tuple[int, str]:
    """Send a request, return ``(status_code, body_text)``. HTTPError
    responses (4xx/5xx) come back through the same return — never raised."""
    data = json.dumps(json_body).encode("utf-8") if json_body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)

    ctx = None
    if url.startswith("https://") and insecure:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

    try:
        with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")
    except urllib.error.URLError as exc:
        return 0, f"connection failed: {exc.reason}"


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------


@dataclass
class Runner:
    base_url: str
    api_key: str
    insecure: bool = False
    passed: int = 0
    failed: int = 0
    failures: list[str] = field(default_factory=list)

    def check(
        self,
        label: str,
        method: str,
        path: str,
        *,
        token: str | None = None,
        json_body: Any = None,
        expect_status: int,
    ) -> tuple[int, str]:
        headers = _bearer(token) if token else None
        code, body = _request(
            method, self.base_url + path, headers=headers, json_body=json_body,
            insecure=self.insecure,
        )
        ok = code == expect_status
        symbol = "\x1b[32m✓\x1b[0m" if ok else "\x1b[31m✗\x1b[0m"
        print(f"  {symbol} {label:<66} {code} (want {expect_status})")
        if ok:
            self.passed += 1
        else:
            self.failed += 1
            self.failures.append(f"{label}: got {code}, expected {expect_status} — {body[:200]}")
        return code, body

    def section(self, title: str) -> None:
        print(f"\n\x1b[1m{title}\x1b[0m")


# ---------------------------------------------------------------------------
# Pairing helpers
# ---------------------------------------------------------------------------


def _mint_device(
    runner: Runner, label: str, scopes: list[str] | None = None
) -> tuple[str, str]:
    """Mint a pairing code and redeem it. Returns ``(device_id, token)``."""
    body: dict[str, Any] = {"label": label}
    if scopes is not None:
        body["scopes"] = scopes
    code, raw = _request(
        "POST", runner.base_url + "/auth/pair/begin",
        headers=_bearer(runner.api_key), json_body=body, insecure=runner.insecure,
    )
    if code != 200:
        raise RuntimeError(f"/auth/pair/begin failed ({code}): {raw}")
    pairing_code = json.loads(raw)["code"]

    code, raw = _request(
        "POST", runner.base_url + "/auth/pair",
        json_body={"code": pairing_code, "label": label}, insecure=runner.insecure,
    )
    if code != 200:
        raise RuntimeError(f"/auth/pair redeem failed ({code}): {raw}")
    resp = json.loads(raw)
    return resp["deviceId"], resp["token"]


def _revoke_device(runner: Runner, device_id: str) -> None:
    """Best-effort revoke — failures are logged, never raised, so cleanup
    doesn't mask a real test failure."""
    code, raw = _request(
        "DELETE", runner.base_url + f"/auth/devices/{device_id}",
        headers=_bearer(runner.api_key), insecure=runner.insecure,
    )
    if code not in (200, 404):
        print(f"  ⚠ revoke {device_id} returned {code}: {raw[:100]}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------


def run(runner: Runner) -> int:
    # ---- 1. Public probe -------------------------------------------------
    runner.section("1. /auth/config (public probe)")
    code, raw = _request(
        "GET", runner.base_url + "/auth/config", insecure=runner.insecure,
    )
    if code == 200:
        cfg = json.loads(raw)
        ok = cfg.get("authRequired") is True and "api_key" in cfg.get("methods", [])
        runner.check(
            f"shape: authRequired=True, 'api_key' in methods, version={cfg.get('version')!r}",
            "GET", "/auth/config", expect_status=200,
        )
        if not ok:
            print(f"    body: {raw[:200]}")
    else:
        runner.check("GET /auth/config", "GET", "/auth/config", expect_status=200)
        print(f"    body: {raw[:200]}")
        return 1  # can't proceed if /auth/config is unreachable

    # ---- 2. Middleware ---------------------------------------------------
    runner.section("2. Middleware: bearer presence/correctness")
    runner.check(
        "no bearer → 401",
        "GET", "/backend-tools", expect_status=401,
    )
    runner.check(
        "wrong bearer → 401",
        "GET", "/backend-tools", token="not-a-real-key", expect_status=401,
    )

    # ---- 3. API key bypasses every scope ---------------------------------
    runner.section("3. API key identity (operator) bypasses scope checks")
    runner.check(
        "/backend-tools → 200",
        "GET", "/backend-tools", token=runner.api_key, expect_status=200,
    )
    runner.check(
        "/db/threads/{id}/messages → 200 (empty thread)",
        "GET", "/db/threads/smoke-nonexistent/messages",
        token=runner.api_key, expect_status=200,
    )
    runner.check(
        "/auth/devices → 200 (operator can list)",
        "GET", "/auth/devices", token=runner.api_key, expect_status=200,
    )

    # ---- 4. Per-device scope matrix --------------------------------------
    full_id = None
    lim_id = None
    try:
        runner.section("4. Device-bearer scope matrix")
        full_id, full_tok = _mint_device(runner, "smoke-full")
        lim_id, lim_tok = _mint_device(runner, "smoke-limited", scopes=["agent", "db"])

        # Full-scope device
        runner.check(
            "[full] /backend-tools (has 'tools') → 200",
            "GET", "/backend-tools", token=full_tok, expect_status=200,
        )
        runner.check(
            "[full] /db/threads/{id}/messages (has 'agent') → 200",
            "GET", "/db/threads/smoke-nonexistent/messages",
            token=full_tok, expect_status=200,
        )
        runner.check(
            "[full] /auth/devices (operator-only) → 403",
            "GET", "/auth/devices", token=full_tok, expect_status=403,
        )

        # Limited device (agent + db only)
        runner.check(
            "[lim ] /backend-tools (no 'tools') → 403",
            "GET", "/backend-tools", token=lim_tok, expect_status=403,
        )
        runner.check(
            "[lim ] /db/threads/{id}/messages (has 'agent') → 200",
            "GET", "/db/threads/smoke-nonexistent/messages",
            token=lim_tok, expect_status=200,
        )
        runner.check(
            "[lim ] /auth/devices (operator-only) → 403",
            "GET", "/auth/devices", token=lim_tok, expect_status=403,
        )
    finally:
        if full_id is not None:
            _revoke_device(runner, full_id)
        if lim_id is not None:
            _revoke_device(runner, lim_id)

    # ---- Summary ---------------------------------------------------------
    total = runner.passed + runner.failed
    color = "\x1b[32m" if runner.failed == 0 else "\x1b[31m"
    print(f"\n{color}{runner.passed}/{total} checks passed\x1b[0m")
    for failure in runner.failures:
        print(f"  - {failure}")
    return 0 if runner.failed == 0 else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("BACKEND_URL", DEFAULT_BASE),
        help=f"Backend URL (default: $BACKEND_URL or {DEFAULT_BASE}).",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("PUPA_API_KEY") or _api_key_from_config(),
        help="Operator API key (default: $PUPA_API_KEY or auth.api_key in "
             "~/.pupa-backend/config.yml).",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Skip TLS verification — for self-signed local HTTPS only.",
    )
    return parser


def main() -> int:
    args = _build_argparser().parse_args()
    if not args.api_key:
        print(
            "error: no API key found. Pass --api-key, export PUPA_API_KEY, "
            "or set auth.api_key in ~/.pupa-backend/config.yml.",
            file=sys.stderr,
        )
        return 2
    base = args.base_url.rstrip("/")
    print(f"smoke @ {base}  (key: {args.api_key[:8]}…)")
    runner = Runner(base_url=base, api_key=args.api_key, insecure=args.insecure)
    return run(runner)


if __name__ == "__main__":
    sys.exit(main())
