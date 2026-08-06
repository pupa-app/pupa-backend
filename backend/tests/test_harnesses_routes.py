"""Tests for `GET /harnesses` (the discovery document)."""

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pupa_backend.harnesses.routes import router as harnesses_router


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("PUPA_AUTH_DISABLED", "1")  # skip scope deps for the shape test
    app = FastAPI()
    app.include_router(harnesses_router, prefix="/harnesses")
    return TestClient(app)


def test_discovery_lists_langgraph_by_default(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PUPA_HARNESSES", raising=False)
    body = client.get("/harnesses").json()
    ids = {h["id"] for h in body}
    assert ids == {"deepagents"}
    lg = body[0]
    assert lg["isDefault"] is True
    assert lg["models"], "langgraph should advertise its MODEL_REGISTRY models"
    assert any(c["key"] == "shell_approval_disabled" for c in lg["permissions"])


def test_discovery_lists_both_harnesses(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "PUPA_HARNESSES",
        json.dumps(
            {
                "deepagents": {"enabled": True, "default": True},
                "claude_code": {"enabled": True},
            }
        ),
    )
    body = client.get("/harnesses").json()
    by_id = {h["id"]: h for h in body}
    assert set(by_id) == {"deepagents", "claude_code"}
    assert by_id["deepagents"]["isDefault"] is True
    assert by_id["claude_code"]["isDefault"] is False
    # Claude harness advertises its alias menu + native-scope control.
    assert {m["modelId"] for m in by_id["claude_code"]["models"]} >= {"opus", "sonnet"}
    assert any(
        c["key"] == "claude_loop_native" for c in by_id["claude_code"]["permissions"]
    )
