"""Tests for the screen-share signalling broker.

The broker pairs one publisher with one viewer on a shared `share_id` and
relays opaque JSON signalling payloads (offer / answer / ICE candidates)
between them. The broker never inspects the SDP — these tests use synthetic
strings and assert exact byte-for-byte relay.

The router is mounted on a minimal test app, not the production `app.py`,
because the latter loads the agent graph (Bedrock client at import time).
"""

import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pupa_backend.auth import api_key_middleware
from pupa_backend.auth.devices import DEFAULT_SCOPES, reset_for_testing
from pupa_backend.screenshare import router as screenshare_router
from pupa_backend.screenshare.config import is_enabled


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    # Most broker tests exercise the relay protocol, not auth — opt out of
    # the global default-required state. Individual tests that DO exercise
    # auth set the env vars themselves.
    monkeypatch.setenv("PUPA_AUTH_DISABLED", "1")
    monkeypatch.delenv("PUPA_API_KEY", raising=False)

    app = FastAPI()
    app.middleware("http")(api_key_middleware)
    app.include_router(screenshare_router, prefix="/screenshare")
    return app


def _share_id() -> str:
    return str(uuid.uuid4())


def test_publisher_then_viewer_pair_and_relay_offer_answer_and_ice(app: FastAPI) -> None:
    client = TestClient(app)
    share_id = _share_id()

    with client.websocket_connect(
        f"/screenshare/ws?role=publisher&share_id={share_id}"
    ) as pub:
        with client.websocket_connect(
            f"/screenshare/ws?role=viewer&share_id={share_id}"
        ) as viewer:
            # Publisher is notified a viewer has arrived.
            assert pub.receive_json() == {"type": "viewer_joined"}

            # Offer travels publisher -> viewer, verbatim.
            offer = {"type": "offer", "sdp": "v=0\r\nfake-offer"}
            pub.send_json(offer)
            assert viewer.receive_json() == offer

            # Answer travels viewer -> publisher, verbatim.
            answer = {"type": "answer", "sdp": "v=0\r\nfake-answer"}
            viewer.send_json(answer)
            assert pub.receive_json() == answer

            # ICE candidates flow both ways, verbatim.
            pub_ice = {
                "type": "ice",
                "candidate": {"candidate": "candidate:pub-1", "sdpMid": "0", "sdpMLineIndex": 0},
            }
            pub.send_json(pub_ice)
            assert viewer.receive_json() == pub_ice

            viewer_ice = {
                "type": "ice",
                "candidate": {"candidate": "candidate:viewer-1", "sdpMid": "0", "sdpMLineIndex": 0},
            }
            viewer.send_json(viewer_ice)
            assert pub.receive_json() == viewer_ice


def test_viewer_for_unknown_share_id_gets_error_message_then_4404_close(app: FastAPI) -> None:
    client = TestClient(app)
    with pytest.raises(Exception) as excinfo:
        with client.websocket_connect(
            f"/screenshare/ws?role=viewer&share_id={_share_id()}"
        ) as viewer:
            error = viewer.receive_json()
            assert error["type"] == "error"
            assert error["code"] == 4404
            assert "no publisher" in error["reason"].lower()
            # Next receive should pop the close frame.
            viewer.receive_json()
    assert getattr(excinfo.value, "code", None) == 4404


def test_second_concurrent_viewer_gets_error_message_then_4409_close(app: FastAPI) -> None:
    client = TestClient(app)
    share_id = _share_id()

    with client.websocket_connect(
        f"/screenshare/ws?role=publisher&share_id={share_id}"
    ) as _pub:
        with client.websocket_connect(
            f"/screenshare/ws?role=viewer&share_id={share_id}"
        ) as _viewer:
            with pytest.raises(Exception) as excinfo:
                with client.websocket_connect(
                    f"/screenshare/ws?role=viewer&share_id={share_id}"
                ) as second:
                    error = second.receive_json()
                    assert error["type"] == "error"
                    assert error["code"] == 4409
                    assert "another viewer" in error["reason"].lower()
                    second.receive_json()
            assert getattr(excinfo.value, "code", None) == 4409


def test_duplicate_publisher_gets_error_message_then_4409_close(app: FastAPI) -> None:
    client = TestClient(app)
    share_id = _share_id()

    with client.websocket_connect(
        f"/screenshare/ws?role=publisher&share_id={share_id}"
    ) as _pub:
        with pytest.raises(Exception) as excinfo:
            with client.websocket_connect(
                f"/screenshare/ws?role=publisher&share_id={share_id}"
            ) as second:
                error = second.receive_json()
                assert error["type"] == "error"
                assert error["code"] == 4409
                assert "publisher" in error["reason"].lower()
                second.receive_json()
        assert getattr(excinfo.value, "code", None) == 4409


def test_viewer_without_share_id_auto_joins_sole_publisher(app: FastAPI) -> None:
    client = TestClient(app)
    share_id = _share_id()
    with client.websocket_connect(
        f"/screenshare/ws?role=publisher&share_id={share_id}"
    ) as pub:
        with client.websocket_connect("/screenshare/ws?role=viewer") as viewer:
            assert pub.receive_json() == {"type": "viewer_joined"}
            offer = {"type": "offer", "sdp": "v=0\r\nauto-discover-offer"}
            pub.send_json(offer)
            assert viewer.receive_json() == offer


def test_viewer_without_share_id_gets_4404_when_no_publisher(app: FastAPI) -> None:
    client = TestClient(app)
    with pytest.raises(Exception) as excinfo:
        with client.websocket_connect("/screenshare/ws?role=viewer") as viewer:
            error = viewer.receive_json()
            assert error["type"] == "error"
            assert error["code"] == 4404
            viewer.receive_json()
    assert getattr(excinfo.value, "code", None) == 4404


def test_viewer_without_share_id_gets_4404_when_multiple_publishers(app: FastAPI) -> None:
    client = TestClient(app)
    sid1, sid2 = _share_id(), _share_id()
    with client.websocket_connect(f"/screenshare/ws?role=publisher&share_id={sid1}") as _p1:
        with client.websocket_connect(f"/screenshare/ws?role=publisher&share_id={sid2}") as _p2:
            with pytest.raises(Exception) as excinfo:
                with client.websocket_connect("/screenshare/ws?role=viewer") as viewer:
                    error = viewer.receive_json()
                    assert error["type"] == "error"
                    assert error["code"] == 4404
                    viewer.receive_json()
    assert getattr(excinfo.value, "code", None) == 4404


def test_missing_role_or_share_id_is_rejected_with_4400(app: FastAPI) -> None:
    client = TestClient(app)
    for url in (
        "/screenshare/ws",
        "/screenshare/ws?role=publisher",
        f"/screenshare/ws?share_id={_share_id()}",
        "/screenshare/ws?role=bogus&share_id=abc",
    ):
        with pytest.raises(Exception) as excinfo:
            with client.websocket_connect(url) as ws:
                ws.receive_json()
        assert getattr(excinfo.value, "code", None) == 4400, url


def test_publisher_disconnect_closes_viewer(app: FastAPI) -> None:
    client = TestClient(app)
    share_id = _share_id()

    with client.websocket_connect(
        f"/screenshare/ws?role=publisher&share_id={share_id}"
    ) as pub:
        with client.websocket_connect(
            f"/screenshare/ws?role=viewer&share_id={share_id}"
        ) as viewer:
            assert pub.receive_json() == {"type": "viewer_joined"}
            pub.close()
            with pytest.raises(Exception):
                # Viewer either gets a bye or a disconnect — either is fine.
                while True:
                    msg = viewer.receive_json()
                    if msg.get("type") == "bye":
                        viewer.receive_json()  # next call should raise


def test_ws_bypasses_http_middleware_so_broker_checks_auth_itself(
    monkeypatch: pytest.MonkeyPatch, app: FastAPI
) -> None:
    """HTTP middleware doesn't see WS upgrades. The broker must enforce auth."""
    # Undo the fixture's PUPA_AUTH_DISABLED so this test exercises the
    # real require-auth path against the broker.
    monkeypatch.delenv("PUPA_AUTH_DISABLED", raising=False)
    monkeypatch.setenv("PUPA_API_KEY", "s3cr3t")
    client = TestClient(app)
    with pytest.raises(Exception) as excinfo:
        with client.websocket_connect(
            f"/screenshare/ws?role=publisher&share_id={_share_id()}"
        ) as ws:
            ws.receive_json()
    assert getattr(excinfo.value, "code", None) == 4401


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, False),
        ("", False),
        ("0", False),
        ("false", False),
        ("False", False),
        ("1", True),
        ("true", True),
        ("yes", True),
    ],
)
def test_is_enabled_reads_env_var(
    monkeypatch: pytest.MonkeyPatch, value: str | None, expected: bool
) -> None:
    if value is None:
        monkeypatch.delenv("PUPA_SCREENSHARE", raising=False)
    else:
        monkeypatch.setenv("PUPA_SCREENSHARE", value)
    assert is_enabled() is expected


def test_ws_accepts_correct_bearer_when_auth_required(
    monkeypatch: pytest.MonkeyPatch, app: FastAPI
) -> None:
    monkeypatch.delenv("PUPA_AUTH_DISABLED", raising=False)
    monkeypatch.setenv("PUPA_API_KEY", "s3cr3t")
    client = TestClient(app)
    share_id = _share_id()
    with client.websocket_connect(
        f"/screenshare/ws?role=publisher&share_id={share_id}",
        headers={"Authorization": "Bearer s3cr3t"},
    ) as pub:
        with client.websocket_connect(
            f"/screenshare/ws?role=viewer&share_id={share_id}",
            headers={"Authorization": "Bearer s3cr3t"},
        ) as viewer:
            assert pub.receive_json() == {"type": "viewer_joined"}
            offer = {"type": "offer", "sdp": "v=0\r\nfake"}
            pub.send_json(offer)
            assert viewer.receive_json() == offer


async def _issue_token(tmp_path, scopes=DEFAULT_SCOPES) -> str:
    store = reset_for_testing(tmp_path / "auth.json")
    _, token = await store.issue(label="test-device", scopes=scopes)
    return token


async def test_paired_device_with_screenshare_scope_passes_auth(
    monkeypatch: pytest.MonkeyPatch, app: FastAPI, tmp_path
) -> None:
    monkeypatch.delenv("PUPA_AUTH_DISABLED", raising=False)
    monkeypatch.delenv("PUPA_API_KEY", raising=False)
    token = await _issue_token(tmp_path)
    client = TestClient(app)
    share_id = _share_id()
    with client.websocket_connect(
        f"/screenshare/ws?role=publisher&share_id={share_id}",
        headers={"Authorization": f"Bearer {token}"},
    ) as pub:
        with client.websocket_connect(
            f"/screenshare/ws?role=viewer&share_id={share_id}",
            headers={"Authorization": f"Bearer {token}"},
        ) as viewer:
            assert pub.receive_json() == {"type": "viewer_joined"}


async def test_paired_device_without_screenshare_scope_is_rejected(
    monkeypatch: pytest.MonkeyPatch, app: FastAPI, tmp_path
) -> None:
    monkeypatch.delenv("PUPA_AUTH_DISABLED", raising=False)
    monkeypatch.delenv("PUPA_API_KEY", raising=False)
    token = await _issue_token(tmp_path, scopes=["agent", "db"])
    client = TestClient(app)
    with pytest.raises(Exception) as excinfo:
        with client.websocket_connect(
            f"/screenshare/ws?role=publisher&share_id={_share_id()}",
            headers={"Authorization": f"Bearer {token}"},
        ) as ws:
            ws.receive_json()
    assert getattr(excinfo.value, "code", None) == 4401


async def test_sidecar_token_allows_publisher_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Per-process sidecar token authenticates the publisher but not the viewer."""
    import types
    import pupa_backend.screenshare.sidecar_token as st

    monkeypatch.delenv("PUPA_AUTH_DISABLED", raising=False)
    monkeypatch.delenv("PUPA_API_KEY", raising=False)
    # Inject a known in-process token without touching the filesystem.
    monkeypatch.setattr(st, "_token", "test-sidecar-secret")

    from pupa_backend.screenshare.routes import _authorised

    def ws(role: str, token: str) -> object:
        import types
        return types.SimpleNamespace(
            headers={"authorization": f"Bearer {token}"},
            query_params={"role": role},
            client=types.SimpleNamespace(host="127.0.0.1"),
        )

    # Correct token + publisher role → allowed.
    assert await _authorised(ws("publisher", "test-sidecar-secret"))

    # Wrong token → rejected.
    assert not await _authorised(ws("publisher", "wrong-token"))

    # Correct token but viewer role → rejected (sidecar token is publisher-only).
    assert not await _authorised(ws("viewer", "test-sidecar-secret"))

    # No token at all → rejected.
    no_token = types.SimpleNamespace(
        headers={"authorization": ""},
        query_params={"role": "publisher"},
        client=types.SimpleNamespace(host="127.0.0.1"),
    )
    assert not await _authorised(no_token)


async def test_a_refused_socket_is_accepted_before_it_is_closed() -> None:
    """Closing a socket that was never accepted has no frame channel to carry
    the close code — the ASGI server just answers the upgrade with a plain HTTP
    403, so 4403 (HTTPS required), 4401 (unauthorised) and 4400 (bad role) all
    reach the client as the same thing. TestClient is an in-process shim that
    surfaces the code either way, so only the call order can be asserted."""
    from pupa_backend.screenshare.routes import _refuse

    calls: list = []

    class _FakeWebSocket:
        async def accept(self) -> None:
            calls.append("accept")

        async def close(self, code: int) -> None:
            calls.append(("close", code))

    await _refuse(_FakeWebSocket(), 4403)
    assert calls == ["accept", ("close", 4403)]
