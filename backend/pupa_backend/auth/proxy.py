"""Whether to believe `X-Forwarded-*` on this deployment.

Forwarded headers are written by the *client* unless something in front
overwrites them. Trusting them unconditionally makes every per-client control
that reads them — the rate limiter's bucket key, the HTTPS check — bypassable
by anyone who can set a header. Trusting them never breaks the proxied
deployments, where the process only ever sees `127.0.0.1`.

So it has to be a deployment fact, not a guess from the request:

- `PUPA_TRUSTED_PROXY` (config `transport.trusted_proxy`) is the explicit
  answer and always wins, either way.
- Otherwise it's inferred from `PUPA_CONNECTIVITY`: with `tailscale` or
  `cloudflared` this process starts the proxy itself, so it knows one is there.
- Otherwise **false** — a direct bind (`0.0.0.0`, the LAN/offline default)
  has nothing in front, so the peer address is the honest one.

Getting this wrong in the safe direction (a real proxy, flag unset) collapses
callers into one bucket and can over-throttle failed attempts. Getting it
wrong the other way voids the limits entirely, which is why the default is
off and `deploy/cloud-config.yml` sets it explicitly for Railway.
"""

import os

from .devices import truthy

# Connectivity modes where this process launches the proxy in front of itself.
_PROXIED_CONNECTIVITY = frozenset({"tailscale", "cloudflared"})


def starts_its_own_proxy() -> bool:
    """Whether this process launches the proxy that fronts it.

    The same fact the inference below rests on, read by `app.main()` to bind
    the listener to loopback. The two have to agree: believing `X-Forwarded-*`
    while the listener is also reachable directly lets anyone who can route to
    it write their own hop — which is the whole hole the trust gate closes.
    """
    return os.getenv("PUPA_CONNECTIVITY", "").strip().lower() in _PROXIED_CONNECTIVITY


def trust_forwarded_headers() -> bool:
    explicit = os.getenv("PUPA_TRUSTED_PROXY")
    if explicit is not None and explicit.strip() != "":
        return truthy(explicit)
    return starts_its_own_proxy()


def forwarded_values(headers, name: str) -> list[str]:
    """Split a comma-separated forwarded header into its hops, or []."""
    raw = headers.get(name)
    if not raw:
        return []
    return [p.strip() for p in raw.split(",") if p.strip()]
