"""Codex newline JSON-RPC transport tests using a tiny fake child process."""

from __future__ import annotations

import os

from pupa_backend.harnesses.codex.transport import AppServerClient


async def test_transport_handshake_request_notification_and_server_request(tmp_path) -> None:
    binary = tmp_path / "fake-codex"
    binary.write_text(
        """#!/usr/bin/env python3
import json, sys
for line in sys.stdin:
    msg = json.loads(line)
    method = msg.get("method")
    if method == "initialized":
        print(json.dumps({"method": "notice", "params": {"ok": True}}), flush=True)
    elif method == "trigger":
        print(json.dumps({"method": "host/request", "id": "host-1", "params": {"x": 2}}), flush=True)
    elif msg.get("id") == "host-1":
        print(json.dumps({"id": 2, "result": {"host": msg.get("result")}}), flush=True)
    elif "id" in msg:
        print(json.dumps({"id": msg["id"], "result": {"method": method}}), flush=True)
"""
    )
    binary.chmod(0o755)
    notifications = []

    async def on_notification(method, params):
        notifications.append((method, params))

    async def on_request(method, params):
        assert method == "host/request"
        return {"answer": params["x"] + 1}

    client = AppServerClient(
        str(binary),
        env={"PATH": os.environ["PATH"]},
        cwd=str(tmp_path),
        on_notification=on_notification,
        on_request=on_request,
    )
    try:
        assert (await client.connect())["method"] == "initialize"
        result = await client.request("trigger", {})
        assert result == {"host": {"answer": 3}}
        for _ in range(20):
            if notifications:
                break
            await __import__("asyncio").sleep(0)
        assert notifications == [("notice", {"ok": True})]
    finally:
        await client.close()


async def test_transport_accepts_json_lines_larger_than_asyncio_default(tmp_path) -> None:
    binary = tmp_path / "fake-codex"
    binary.write_text(
        """#!/usr/bin/env python3
import json, sys
for line in sys.stdin:
    msg = json.loads(line)
    if "id" in msg:
        print(json.dumps({"id": msg["id"], "result": {"payload": "x" * 131072}}), flush=True)
"""
    )
    binary.chmod(0o755)

    client = AppServerClient(
        str(binary),
        env={"PATH": os.environ["PATH"]},
        cwd=str(tmp_path),
    )
    try:
        result = await client.connect()
        assert result["payload"] == "x" * 131072
    finally:
        await client.close()
