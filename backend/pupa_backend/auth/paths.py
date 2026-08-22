"""Path predicates shared by the auth middlewares.

A leaf module so `transport.py` and `middleware.py` can agree on what a probe
is without one importing the other — they sit at different depths of the same
middleware stack.
"""


def is_health_probe(path: str) -> bool:
    """A platform/liveness probe, exempt from both the auth and the HTTPS
    guard. The AGUI helper registers `GET {path}/health` — with `path="/"` that
    serialises to `//health` until Starlette normalises it, so match both."""
    return path.endswith("/health")
