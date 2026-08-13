"""Operator CLI: mint a short-lived pairing code for a new device.

Run via `make pair` (which sets `BACKEND_URL` + reads `PUPA_API_KEY`).
Posts to `/auth/pair/begin` and prints the code in big letters so the
operator can show it to whoever's pairing. When the `qrcode` package is
installed (via `uv pip install -e '.[setup]'`), also renders a QR code
encoding a `pupa-pair://` URL so the iOS user can scan to pair
without typing the code.

Standalone Python (stdlib `urllib`) so it works wherever `make` works — no
extra deps, no `httpx` / `requests`. QR output is opt-in via the `qrcode`
extra.
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# Load ~/.pupa-backend/config.yml (or .env legacy) so os.getenv() works below.
from pupa_backend.pupa_config import load_pupa_config  # type: ignore[import]
load_pupa_config()


def _default_backend_url() -> str:
    return os.environ.get("BACKEND_URL", "http://localhost:8004").rstrip("/")


def _api_key() -> str | None:
    key = os.environ.get("PUPA_API_KEY", "").strip()
    return key or None


def _unverified_ctx() -> ssl.SSLContext:
    """Unverified context for the self-signed local backend cert."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _post(url: str, body: dict, api_key: str | None) -> dict:
    payload = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=payload, method="POST")
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    ctx = _unverified_ctx() if url.startswith("https://") else None
    try:
        with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            sys.exit(
                "401 from backend. The /auth/pair/begin route requires an existing\n"
                "credential. Either export PUPA_API_KEY=... before running\n"
                "`make pair`, or pair from a device that's already authenticated.\n"
            )
        sys.exit(f"HTTP {exc.code} from {url}: {exc.read().decode('utf-8', errors='replace')}")
    except urllib.error.URLError as exc:
        sys.exit(f"could not reach {url}: {exc.reason}\n(is `make backend` running?)")


def _cert_fingerprint() -> str | None:
    return os.environ.get("PUPA_CERT_FINGERPRINT") or None


_TUNNEL_URL_FILE = Path.home() / ".pupa-backend" / "tunnel_url"


def _tailnet_https_url() -> str | None:
    """The `https://<magicdns>` URL when `tailscale serve` is terminating TLS.

    In that mode the cert is a real, publicly-trusted one held by tailscaled, so
    the device needs no fingerprint — and the URL carries no port (:443).
    """
    from pupa_backend.tailscale_proxy import cert_domain, should_proxy

    if not should_proxy():
        return None
    domain = cert_domain()
    return f"https://{domain}" if domain else None


def _backend_public_url() -> str:
    """Derive the URL the phone should use based on the configured connectivity."""
    connectivity = os.environ.get("PUPA_CONNECTIVITY", "")

    if connectivity == "cloudflared":
        # Named tunnel (custom domain) → stable URL straight from config; no need
        # to wait for the backend to write a quick-tunnel URL file.
        named_host = os.environ.get("PUPA_CLOUDFLARED_HOSTNAME")
        if named_host:
            return f"https://{named_host}"
        if not _TUNNEL_URL_FILE.exists():
            sys.exit(
                "cloudflared tunnel URL not found.\n"
                "Start the backend first (`pupa-backend run`) — it auto-starts the tunnel."
            )
        return _TUNNEL_URL_FILE.read_text().strip()

    # tailscale or localhost: prefer stored hostname, fall back to live tailscale IP
    hostname = os.environ.get("PUPA_HOSTNAME")
    if not hostname:
        try:
            result = subprocess.run(
                ["tailscale", "ip", "-4"],
                capture_output=True, text=True, timeout=3,
            )
            hostname = result.stdout.strip() if result.returncode == 0 else ""
        except (FileNotFoundError, subprocess.TimeoutExpired):
            hostname = ""
    hostname = hostname or "localhost"
    scheme = "https" if os.environ.get("PUPA_TLS_CERT") else "http"
    return f"{scheme}://{hostname}:8004"


def _print_qr(qr_url: str) -> None:
    """Print a terminal QR code for qr_url.
    ``print_tty`` uses ANSI control characters and only works on a real
    terminal — if stdout is piped (CI, ``| tail``, capturing in scripts)
    it raises ``OSError: Not a tty``. Fall back to ``print_ascii`` in that
    case so the QR still renders, just with `#`/space cells instead of
    half-blocks.
    """
    import qrcode  # type: ignore
    import qrcode.constants  # type: ignore


    qr = qrcode.QRCode(
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=1,
        border=1,
    )
    qr.add_data(qr_url)
    qr.make(fit=True)
    print()
    try:
        qr.print_tty()
    except OSError:
        qr.print_ascii(tty=False)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mint a pairing code for a new Pupa device.",
    )
    parser.add_argument(
        "--label",
        help="Suggested device label (e.g. \"User's iPhone\"). The device can override.",
        default=None,
    )
    parser.add_argument(
        "--code-ttl",
        type=int,
        default=None,
        metavar="SECONDS",
        help="How long the 8-char bootstrap code stays valid (seconds). Default: 300 (5 min).",
    )
    parser.add_argument(
        "--device-ttl",
        type=int,
        default=0,
        metavar="SECONDS",
        help=(
            "Lifetime of the issued device token (seconds). "
            "Default: 0 (never expires) — the pairing is meant for your own "
            "device. Pass a finite value (e.g. 259200 = 3 days) when sharing "
            "with someone else."
        ),
    )
    parser.add_argument(
        "--public-url",
        default=None,
        metavar="URL",
        help=(
            "Absolute URL the iOS app should connect to (e.g. "
            "https://<your-service>.up.railway.app). When set, "
            "overrides both the POST target (replacing BACKEND_URL) and "
            "the URL embedded in the QR code / deep link. Use this when "
            "pairing a phone against a remote deployment — the default "
            "auto-derivation (tailscale / cloudflared / localhost) "
            "assumes the backend is on your network."
        ),
    )
    args = parser.parse_args()

    backend_url = (args.public_url or _default_backend_url()).rstrip("/")
    url = backend_url + "/auth/pair/begin"
    body: dict = {}
    if args.label:
        body["label"] = args.label
    if args.code_ttl is not None:
        body["codeTtlSeconds"] = args.code_ttl
    if args.device_ttl:  # 0 means no expiry — omit the field
        body["deviceTokenTtlSeconds"] = args.device_ttl
    result = _post(url, body, _api_key())

    code = result["code"]
    expires = result["expiresAt"]
    scopes = ", ".join(result.get("scopes") or [])

    # Build the pupa-pair:// URL the iOS app can scan. An explicit
    # --public-url (e.g. a Railway hostname) overrides the
    # tailscale/cloudflared/localhost auto-derivation; in that case we also
    # skip the local TLS cert fingerprint — remote backends serve a
    # publicly-trusted cert that iOS validates the normal way.
    if args.public_url:
        public_url = args.public_url.rstrip("/")
        fp: str | None = None
    elif tailnet_url := _tailnet_https_url():
        # tailscaled terminates TLS with a trusted cert — nothing to pin.
        public_url = tailnet_url
        fp = None
    else:
        public_url = _backend_public_url()
        fp = _cert_fingerprint()
    qr_params: dict[str, str] = {
        "url": public_url,
        "code": code,
    }
    if fp:
        qr_params["fp"] = fp
    pair_url = "pupa-pair://?" + urllib.parse.urlencode(qr_params)

    print("")
    print("  ┌────────────────────────────────────┐")
    print(f"  │       Pairing code: {code:<14} │")
    print("  └────────────────────────────────────┘")
    print("")
    print(f"  backend url : {public_url}")
    print(f"  scopes      : {scopes}")
    print(f"  expires at  : {expires}")
    if fp:
        print(f"  cert SHA-256: {fp}")
    print("")
    fields = "the URL, the code, and the cert fingerprint" if fp else "the URL and the code"
    print("  Option A — scan QR (easiest):")
    print("    iOS Settings → Backend → Pair via QR, then scan. Fills in")
    print(f"    {fields} together.")
    _print_qr(pair_url)
    print("")
    print("  Option B — manual entry:")
    print("    iOS Settings → Backend → Edit → fill in URL + code + cert fingerprint.")
    print("")
    print("  (The `pupa-pair://` scheme is deliberately unregistered in the app, so")
    print("   sending this link to the device and tapping it does nothing — the QR")
    print("   is the only automatic path.)")
    print("")


if __name__ == "__main__":
    main()
