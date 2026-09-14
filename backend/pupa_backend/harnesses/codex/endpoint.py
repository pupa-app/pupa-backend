"""AG-UI endpoint backed by a live Codex App Server session."""

from __future__ import annotations

import logging
from typing import Any

from ag_ui.core import RunAgentInput
from ag_ui.encoder import EventEncoder
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

from pupa_backend.agui import events
from pupa_backend.agui.approval import interpret_always, interpret_approval
from pupa_backend.agui.input import (
    image_inputs,
    latest_user_message,
    latest_user_text,
    message_content,
    render_context,
    render_transcript,
)
from pupa_backend.agui.tool_results import parse_tool_results
from pupa_backend.sse_replay import register_reattach_observer

from . import registry
from .models import ModelCatalog
from .tools import build_tool_surface

logger = logging.getLogger("uvicorn.error")


def _advertised_tools(run_input: RunAgentInput, resume_payload: Any) -> list[Any]:
    if isinstance(resume_payload, dict):
        after = resume_payload.get("tools_after_round")
        if isinstance(after, list):
            return after
    return list(run_input.tools or [])


def _turn_text(run_input: RunAgentInput, *, has_history: bool) -> str:
    text = latest_user_text(run_input.messages) if has_history else render_transcript(run_input.messages)
    context = render_context(run_input.context)
    if context:
        text = (
            f"{text}\n\nAmbient context supplied by Pupa for this turn:\n\n{context}"
            if text
            else f"Ambient context supplied by Pupa for this turn:\n\n{context}"
        )
    return text


def _stream(session: registry.LiveSession) -> StreamingResponse:
    encoder = EventEncoder()

    async def generate():
        async for event in registry.attach(session):
            yield encoder.encode(event)

    return StreamingResponse(generate(), media_type="text/event-stream")


def _error_stream(message: str) -> StreamingResponse:
    encoder = EventEncoder()

    async def generate():
        yield encoder.encode(events.run_error(message))

    return StreamingResponse(generate(), media_type="text/event-stream")


def register_codex_endpoint(
    app: FastAPI,
    *,
    path: str,
    mcp: Any,
    catalog: ModelCatalog,
) -> None:
    register_reattach_observer(registry.note_reattach)
    logger.info("codex agent harness active on POST %s", path)

    @app.post(path)
    async def codex_endpoint(request: Request):  # noqa: ANN202 - FastAPI route
        body = await request.json()
        run_input = RunAgentInput.model_validate(body)
        thread_id = run_input.thread_id
        run_id = run_input.run_id
        forwarded = run_input.forwarded_props or {}
        command = forwarded.get("command") if isinstance(forwarded, dict) else None
        resume_payload = command.get("resume") if isinstance(command, dict) else None
        keepalive = command.get("keepalive") if isinstance(command, dict) else None

        if keepalive is not None:
            session = registry.get(thread_id)
            if session is not None:
                session.touch()
            return Response(status_code=204)

        session = registry.get(thread_id)
        if resume_payload is not None:
            if session is None or session.disposed or not session.active_batch:
                return _error_stream(
                    "no parked Codex frontend-tool session for this thread; start a new message"
                )
            try:
                session.open_run(run_id)
                advertised = _advertised_tools(run_input, resume_payload)
                refreshed = build_tool_surface(advertised, run_input.state, mcp)
                surface_changed = (
                    session.surface is None
                    or refreshed.fingerprint != session.surface.fingerprint
                )
                if surface_changed:
                    session.arm_surface_continuation()
                await session.resolve_tool_results(parse_tool_results(resume_payload))
                if surface_changed:
                    model = catalog.resolve_model(run_input)
                    await session.continue_with_surface(
                        refreshed,
                        model=model,
                        effort=catalog.resolve_effort(run_input, model),
                        state=run_input.state,
                    )
                return _stream(session)
            except Exception as exc:  # noqa: BLE001
                logger.exception("codex harness: failed to resume frontend tools")
                await registry.remove(thread_id, session, notify=False)
                return _error_stream(f"Codex frontend-tool resume failed: {exc}")

        if (
            session is not None
            and not session.disposed
            and session.pending_approval is not None
            and not session.pending_approval.done()
        ):
            reply = latest_user_text(run_input.messages)
            session.open_run(run_id)
            await session.resolve_approval(
                interpret_approval(reply), always=interpret_always(reply)
            )
            return _stream(session)

        await registry.sweep_idle()
        session = registry.get(thread_id)
        if session is not None and session.turn_active:
            # A new user request supersedes an abandoned interrupted turn. Start
            # from the Pupa transcript instead of resuming ambiguous child state.
            await registry.remove(thread_id, session, notify=False)
            registry.forget_thread_id(thread_id)
            session = None

        try:
            surface = build_tool_surface(list(run_input.tools or []), run_input.state, mcp)
            remembered = registry.remembered_thread_id(thread_id)
            resume_id = (
                remembered
                if registry.remembered_surface(thread_id) == surface.fingerprint
                else None
            )
            has_history = bool(resume_id)
            if session is None or session.disposed:
                session = registry.create(thread_id, mcp)
                session.scope = registry.native_scope(run_input.state)
                has_history = await session.connect(surface, resume_id=resume_id)
            else:
                restarted = await session.reload_surface(surface)
                has_history = not restarted
            model = catalog.resolve_model(run_input)
            latest = latest_user_message(run_input.messages)
            await session.start_turn(
                text=_turn_text(run_input, has_history=has_history),
                images=image_inputs(message_content(latest)),
                model=model,
                effort=catalog.resolve_effort(run_input, model),
                state=run_input.state,
                transcript=render_transcript(run_input.messages),
                run_id=run_id,
            )
            return _stream(session)
        except Exception as exc:  # noqa: BLE001
            logger.exception("codex harness: failed to start turn")
            if session is not None:
                await registry.remove(thread_id, session, notify=False)
            return _error_stream(f"Codex harness failed to start: {exc}")
