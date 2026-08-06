"""WebSocket signalling endpoint for screen-share.

Single endpoint `WS /screenshare/ws?role=publisher|viewer[&share_id=<uuid>]`:

- Publisher connects first (share_id required); broker holds the session.
  Sending anything before a viewer arrives is dropped on the floor.
- Viewer connects; `share_id` is optional — if omitted the broker
  auto-resolves to the sole active publisher. Publisher receives
  `{"type": "viewer_joined"}` so it knows to create an SDP offer
  (publisher-as-offerer, plan §3).
- Subsequent JSON messages are relayed verbatim in both directions. The
  broker never inspects SDP / ICE payloads.
- On either side disconnecting, the broker sends `{"type": "bye"}` to the
  surviving peer and (for a publisher loss) closes the viewer cleanly.

Auth: FastAPI's `app.middleware("http")` is NOT invoked for WebSocket
upgrades — Starlette dispatches WS via a separate ASGI scope. This handler
verifies the `Authorization: Bearer` header itself. Three accepted tokens:
paired-device token (screenshare scope), bootstrap `PUPA_API_KEY`,
or the per-process sidecar token written to a temp file at startup.

WS close codes used:
- 4400 — bad / missing query params
- 4401 — auth required, missing or wrong Bearer
- 4404 — viewer for unknown / publisher-less share_id (or no active publisher
         when connecting without share_id)
- 4409 — duplicate publisher or second concurrent viewer (single-viewer v1)

For 4404 and 4409 we *first* send an `{"type":"error","code":N,"reason":...}`
JSON message and *then* close. URLSessionWebSocketTask on iOS doesn't
surface application close codes (4xxx) reliably — they get mapped to
generic `.abnormalClosure` (1006). The pre-close JSON gives clients a
human-readable reason regardless of how URLSession reports the close.
"""

import hmac
import logging
import os

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from pupa_backend.auth.devices import get_store as get_device_store, truthy

from .broker import ShareSession, broker
from .sidecar_token import validate as validate_sidecar_token

logger = logging.getLogger(__name__)
router = APIRouter()


async def _authorised(websocket: WebSocket) -> bool:
    """WebSocket auth check — accepts:
    - Paired-device tokens with the `screenshare` scope.
    - The bootstrap `PUPA_API_KEY` (while set).
    - The per-process sidecar token (publisher role only): generated on
      backend startup, written to a temp file read by `pupa-backend screenshare`.
      Safer than a loopback-IP check because local reverse proxies
      (ngrok, cloudflared) also connect from 127.0.0.1.
    Mirrors the HTTP middleware's default (auth required unless
    `PUPA_AUTH_DISABLED=1`).
    """
    if truthy(os.getenv("PUPA_AUTH_DISABLED")):
        return True

    header = websocket.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return False

    # Per-process sidecar token — valid only for the publisher role.
    if websocket.query_params.get("role") == "publisher" and validate_sidecar_token(token):
        return True

    device = await get_device_store().resolve(token)
    if device is not None:
        return device.has_scope("screenshare")

    api_key = os.getenv("PUPA_API_KEY")
    if api_key and hmac.compare_digest(token, api_key):
        return True

    return False


@router.websocket("/ws")
async def screenshare_ws(
    websocket: WebSocket,
    role: str | None = None,
    share_id: str | None = None,
) -> None:
    if not await _authorised(websocket):
        await websocket.close(code=4401)
        return

    if role not in ("publisher", "viewer"):
        await websocket.close(code=4400)
        return

    if role == "publisher":
        if not share_id:
            await websocket.close(code=4400)
            return
        await websocket.accept()
        await _serve_publisher(websocket, share_id)
    else:
        # Viewer: use the provided share_id or auto-discover the sole active publisher.
        resolved_id = share_id or await broker.sole_publisher_share_id()
        await websocket.accept()
        if resolved_id is None:
            await _send_error_and_close(
                websocket,
                code=4404,
                reason="no active publisher — start `pupa-backend screenshare` first",
            )
            return
        await _serve_viewer(websocket, resolved_id)


async def _serve_publisher(websocket: WebSocket, share_id: str) -> None:
    session = await broker.register_publisher(share_id, websocket)
    if session is None:
        await _send_error_and_close(
            websocket,
            code=4409,
            reason="another publisher is already registered for this share id",
        )
        return

    logger.info("screenshare publisher registered share_id=%s", share_id)
    try:
        while True:
            msg = await websocket.receive_json()
            viewer = session.viewer
            if viewer is not None:
                await _safe_send(viewer, msg)
    except WebSocketDisconnect:
        pass
    finally:
        peers = await broker.remove_publisher(share_id)
        if peers and peers.viewer is not None:
            await _safe_send(peers.viewer, {"type": "bye"})
            await _safe_close(peers.viewer, code=1000)


async def _serve_viewer(websocket: WebSocket, share_id: str) -> None:
    session, error = await broker.register_viewer(share_id, websocket)
    if session is None:
        assert error is not None
        reason = (
            "no publisher is running for this share id — start `pupa-backend screenshare` first"
            if error == 4404
            else "another viewer is already connected for this share id"
        )
        await _send_error_and_close(websocket, code=error, reason=reason)
        return

    publisher = session.publisher
    assert publisher is not None
    await _safe_send(publisher, {"type": "viewer_joined"})

    try:
        while True:
            msg = await websocket.receive_json()
            await _safe_send(publisher, msg)
    except WebSocketDisconnect:
        pass
    finally:
        peers = await broker.remove_viewer(share_id)
        if peers and peers.publisher is not None:
            await _safe_send(peers.publisher, {"type": "bye"})


async def _send_error_and_close(ws: WebSocket, *, code: int, reason: str) -> None:
    # Send the JSON before the close — URLSessionWebSocketTask on iOS hides
    # 4xxx close codes behind a generic abnormalClosure, so the message body
    # is the only reliable channel for "why was I rejected".
    await _safe_send(ws, {"type": "error", "code": code, "reason": reason})
    await _safe_close(ws, code=code)


async def _safe_send(ws: WebSocket, msg: dict) -> None:
    try:
        await ws.send_json(msg)
    except Exception:  # noqa: BLE001
        logger.debug("screenshare send failed, peer likely gone", exc_info=True)


async def _safe_close(ws: WebSocket, *, code: int) -> None:
    try:
        await ws.close(code=code)
    except Exception:  # noqa: BLE001
        pass
