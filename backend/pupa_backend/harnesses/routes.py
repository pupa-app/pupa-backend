"""FastAPI router exposing the enabled agent harnesses to the iOS client.

`GET /harnesses` — one round-trip the app fetches on connect / backend switch, so
the harness list, each harness's model menu, its toggleable tools, and its
permission-control schema are always in sync with the backend (no app update
needed). Replaces the old `/models` and `/backend-tools` routes, which each only
described the single active loop.

Response shape per harness::

    {"id", "label", "isDefault",
     "models":      [{"provider", "modelId", "label"}],
     "thinking":    [{"level", "label"}],   # extended-thinking levels ([] = none)
     "tools":       [{"name", "description", "enabledByEnv"}],
     "permissions": [{"key", "type", "label", ...}]}

`POST /harnesses/{id}` (the SSE run streams) are mounted separately in app.py's
lifespan — this router only serves the GET discovery document.
"""

from __future__ import annotations

import os

from fastapi import APIRouter, Depends, Request

from pupa_backend.auth import require_scope
from pupa_backend.harnesses import build_registry

router = APIRouter()

_DEFAULT_MODEL_ID = "gpt-5.6-sol"


def ordered_models(models: list[dict]) -> list[dict]:
    """Put Pupa's preferred initial model first without changing harness logic."""
    preferred = (os.getenv("PUPA_DEFAULT_MODEL") or _DEFAULT_MODEL_ID).strip().lower()
    return sorted(
        list(models),
        key=lambda model: str(model.get("modelId") or "").lower() != preferred,
    )


@router.get("", dependencies=[Depends(require_scope("agent"))])
async def list_harnesses(request: Request) -> list[dict]:
    registry = getattr(request.app.state, "harness_registry", None) or build_registry()
    default = registry.default()
    default_id = default.id if default is not None else None
    return [
        {
            "id": h.id,
            "label": h.label,
            "isDefault": h.id == default_id,
            "models": ordered_models(h.models()),
            # Optional per-harness capability — deepagents omits it.
            "thinking": getattr(h, "thinking", list)() or [],
            "tools": h.tools(),
            "permissions": h.permission_schema(),
        }
        for h in registry.enabled()
    ]
