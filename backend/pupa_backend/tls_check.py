"""Sanity-check the configured TLS cert against Apple's client rules.

Its own module, not part of `app.py`: importing `app` applies the operator's
`config.yml` to `os.environ` as a side effect, which no test wants just to
check a certificate.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("uvicorn.error")


def warn_unusable_cert(cert_path: str) -> None:
    """Flag a self-signed cert that Apple clients will refuse outright.

    iOS/macOS reject TLS server certs valid for more than 398 days, and reject
    any cert past its notAfter — both *before* the client's fingerprint pinning
    runs, so the app just reports "the backend refused a secure connection".
    Certs minted by older setup runs were 10-year, hence unusable. Silent when
    `cryptography` isn't installed (it ships with the `[setup]` extra).
    """
    try:
        from cryptography import x509
    except ImportError:
        return
    try:
        cert = x509.load_pem_x509_certificate(Path(cert_path).read_bytes())
    except (OSError, ValueError) as exc:
        logger.warning("could not read TLS cert %s: %s", cert_path, exc)
        return
    from datetime import datetime, timezone

    lifetime = (cert.not_valid_after_utc - cert.not_valid_before_utc).days
    remaining = (cert.not_valid_after_utc - datetime.now(timezone.utc)).days
    if lifetime > 398:
        logger.warning(
            "TLS cert is valid for %s days — Apple clients refuse anything over "
            "398, so pairing will fail with a certificate error. Re-run "
            "`pupa-backend setup` to mint a compliant cert, then re-pair.",
            lifetime,
        )
    elif remaining < 0:
        logger.warning("TLS cert expired %s days ago — re-run `pupa-backend setup`.", -remaining)
    elif remaining < 30:
        logger.warning(
            "TLS cert expires in %s days — re-run `pupa-backend setup` and re-pair "
            "before then.",
            remaining,
        )
