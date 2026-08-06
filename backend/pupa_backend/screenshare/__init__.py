"""WebRTC screen-share signalling broker for Pupa.

The broker pairs one publisher (a macOS Swift sidecar capturing a window via
ScreenCaptureKit) with one viewer (the Pupa iOS / macOS app) on a
shared `share_id`. It relays opaque JSON signalling payloads (SDP offer /
answer / trickle ICE) between them and never inspects media. Video and audio
flow peer-to-peer over WebRTC — never through this process.

Mounted in `app.py` only when `PUPA_SCREENSHARE=1` so Linux backends
incur zero cost.
"""

from .routes import router

__all__ = ["router"]
