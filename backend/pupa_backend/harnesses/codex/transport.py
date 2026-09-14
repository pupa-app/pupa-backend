"""Async JSON-RPC transport for ``codex app-server`` over stdio."""

from __future__ import annotations

import asyncio
import json
from contextlib import suppress
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from pupa_backend.version import backend_version

logger = logging.getLogger("uvicorn.error")

NotificationHandler = Callable[[str, dict[str, Any]], Awaitable[None]]
RequestHandler = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]


class AppServerError(RuntimeError):
    """The Codex child or its JSON-RPC protocol failed."""


class AppServerClient:
    """One newline-delimited JSON-RPC connection to a Codex child process."""

    def __init__(
        self,
        binary: str,
        *,
        env: dict[str, str],
        cwd: str,
        on_notification: NotificationHandler | None = None,
        on_request: RequestHandler | None = None,
    ) -> None:
        self.binary = binary
        self.env = env
        self.cwd = cwd
        self.on_notification = on_notification
        self.on_request = on_request
        self.process: asyncio.subprocess.Process | None = None
        self._reader: asyncio.Task | None = None
        self._stderr: asyncio.Task | None = None
        self._request_tasks: set[asyncio.Task] = set()
        self._pending: dict[int, asyncio.Future] = {}
        self._next_id = 1
        self._write_lock = asyncio.Lock()
        self._closing = False

    async def connect(self) -> dict[str, Any]:
        try:
            self.process = await asyncio.create_subprocess_exec(
                self.binary,
                "app-server",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.cwd,
                env=self.env,
            )
        except FileNotFoundError as exc:
            raise AppServerError(
                f"Codex CLI not found at {self.binary!r}; install it and run `codex login`."
            ) from exc
        self._reader = asyncio.create_task(self._read_stdout(), name="codex-app-server-stdout")
        self._stderr = asyncio.create_task(self._read_stderr(), name="codex-app-server-stderr")
        result = await self.request(
            "initialize",
            {
                "clientInfo": {
                    "name": "pupa-backend",
                    "title": "Pupa Backend",
                    "version": backend_version(),
                },
                "capabilities": {"experimentalApi": True},
            },
        )
        await self.notify("initialized", {})
        return result

    async def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        if self.process is None or self.process.returncode is not None:
            raise AppServerError("Codex App Server is not running")
        request_id = self._next_id
        self._next_id += 1
        future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            await self._send({"method": method, "id": request_id, "params": params or {}})
            value = await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise AppServerError(f"Codex App Server timed out handling {method}") from exc
        finally:
            self._pending.pop(request_id, None)
        if not isinstance(value, dict):
            raise AppServerError(f"Codex App Server returned an invalid result for {method}")
        return value

    async def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        await self._send({"method": method, "params": params or {}})

    async def _send(self, message: dict[str, Any]) -> None:
        process = self.process
        if process is None or process.stdin is None or process.returncode is not None:
            raise AppServerError("Codex App Server stdin is closed")
        payload = json.dumps(message, separators=(",", ":"), default=str).encode() + b"\n"
        async with self._write_lock:
            process.stdin.write(payload)
            await process.stdin.drain()

    async def _read_stdout(self) -> None:
        process = self.process
        assert process is not None and process.stdout is not None
        try:
            while raw := await process.stdout.readline():
                try:
                    message = json.loads(raw)
                except (TypeError, ValueError):
                    logger.warning("codex harness: ignoring malformed App Server output")
                    continue
                if not isinstance(message, dict):
                    continue
                method = message.get("method")
                if method and "id" in message:
                    task = asyncio.create_task(self._handle_server_request(message))
                    self._request_tasks.add(task)
                    task.add_done_callback(self._request_tasks.discard)
                    continue
                if method:
                    if self.on_notification is not None:
                        task = asyncio.create_task(
                            self.on_notification(str(method), message.get("params") or {})
                        )
                        self._request_tasks.add(task)
                        task.add_done_callback(self._request_tasks.discard)
                    continue
                request_id = message.get("id")
                future = self._pending.get(request_id)
                if future is None or future.done():
                    continue
                if "error" in message:
                    error = message.get("error") or {}
                    future.set_exception(
                        AppServerError(
                            f"Codex App Server error {error.get('code')}: {error.get('message')}"
                        )
                    )
                else:
                    future.set_result(message.get("result") or {})
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - fail all callers, then let owner recover
            logger.exception("codex harness: App Server stdout reader failed")
        finally:
            if not self._closing:
                self._fail_pending(AppServerError("Codex App Server exited unexpectedly"))
                if self.on_notification is not None:
                    try:
                        await self.on_notification("process/exited", {})
                    except Exception:  # noqa: BLE001
                        logger.debug("codex harness: process-exit callback failed", exc_info=True)

    async def _handle_server_request(self, message: dict[str, Any]) -> None:
        request_id = message.get("id")
        try:
            if self.on_request is None:
                raise AppServerError(f"unsupported Codex request {message.get('method')!r}")
            result = await self.on_request(
                str(message.get("method")), message.get("params") or {}
            )
            await self._send({"id": request_id, "result": result})
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - reply with JSON-RPC error, don't kill reader
            try:
                await self._send(
                    {
                        "id": request_id,
                        "error": {"code": -32000, "message": str(exc)},
                    }
                )
            except Exception:  # noqa: BLE001
                logger.debug("codex harness: failed to answer server request", exc_info=True)

    async def _read_stderr(self) -> None:
        process = self.process
        assert process is not None and process.stderr is not None
        try:
            while raw := await process.stderr.readline():
                line = raw.decode(errors="replace").strip()
                if line:
                    logger.info("codex app-server: %s", line[:4000])
        except asyncio.CancelledError:
            raise

    def _fail_pending(self, error: Exception) -> None:
        for future in self._pending.values():
            if not future.done():
                future.set_exception(error)

    async def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        self._fail_pending(AppServerError("Codex App Server closed"))
        for task in list(self._request_tasks):
            task.cancel()
        process = self.process
        if process is not None and process.returncode is None:
            if process.stdin is not None:
                process.stdin.close()
            with suppress(ProcessLookupError):
                process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                with suppress(ProcessLookupError):
                    process.kill()
                await process.wait()
        for task in (self._reader, self._stderr):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(
            *(task for task in (self._reader, self._stderr) if task is not None),
            *self._request_tasks,
            return_exceptions=True,
        )
