"""
main.py — FastAPI Web Service for Adaptive Learning Companion
─────────────────────────────────────────────────────────────
Lab 8 · Tasks 1–3

Exposes the LangGraph multi-agent graph (v4) over HTTP:

  POST /chat              — Synchronous turn; returns ChatResponse.
  POST /stream            — Streaming turn; SSE stream of token/node chunks.
  POST /hitl/{thread_id}  — Human-in-the-loop decision (approve/cancel/edit).
  GET  /session/{thread_id} — Inspect the current checkpoint state.

Architecture notes
──────────────────
• The SqliteSaver checkpointer is created ONCE at startup (lifespan) and
  shared across all requests — satisfying Lab 8 Task 2 persistence requirement.
• /stream uses graph.astream_events() with SSE format — Task 3.
• All I/O is validated through the Pydantic models in schema.py — Task 1.
"""

from __future__ import annotations

import json
import logging
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator
import os

import uvicorn
from fastapi import FastAPI, HTTPException, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from feedback_routes import feedback_router, init_feedback_db

# ── local imports (from the v4 lab codebase) ──────────────────────────────
from multi_agent_graph import build_multi_agent_graph
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from schema import (
    ChatRequest,
    ChatResponse,
    AgentStatus,
    HitlDecision,
    HitlResponse,
)
from agents_config import HANDOFF_SIGNAL

# ─────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("api")

CHECKPOINT_DB = "/app/checkpoint_data/checkpoint_db.sqlite"

# Ensure directory exists (IMPORTANT for Docker volume)
os.makedirs("/app/checkpoint_data", exist_ok=True)


# ─────────────────────────────────────────────────────────
# APPLICATION STATE  (shared across requests)
# ─────────────────────────────────────────────────────────

class AppState:
    """Holds the single checkpointer + compiled graph for the process lifetime."""
    checkpointer: SqliteSaver | None = None
    app_graph   = None   # sync graph  — used by /chat and /hitl
    async_graph = None   # async graph — used by /stream


_app_state = AppState()


# ─────────────────────────────────────────────────────────
# LIFESPAN  (Task 2 — initialize checkpointer once at startup)
# ─────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler.
    Opens TWO checkpointers on startup:
      • SqliteSaver      — sync,  used by /chat and /hitl
      • AsyncSqliteSaver — async, used by /stream (astream_events)
    Closes both cleanly on shutdown.
    """
    logger.info("[STARTUP] Initialising SqliteSaver → %s", CHECKPOINT_DB)
    cm = SqliteSaver.from_conn_string(CHECKPOINT_DB)
    _app_state.checkpointer = cm.__enter__()
    _app_state.app_graph = build_multi_agent_graph(_app_state.checkpointer)

    logger.info("[STARTUP] Initialising AsyncSqliteSaver → %s", CHECKPOINT_DB)
    async_cm = AsyncSqliteSaver.from_conn_string(CHECKPOINT_DB)
    _app_state.async_checkpointer = await async_cm.__aenter__()
    _app_state.async_graph = build_multi_agent_graph(_app_state.async_checkpointer)

    logger.info("[STARTUP] Both graphs compiled. API ready.")
    init_feedback_db()
    yield

    logger.info("[SHUTDOWN] Closing checkpointers.")
    try:
        cm.__exit__(None, None, None)
    except Exception:
        pass
    try:
        await async_cm.__aexit__(None, None, None)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────
# FASTAPI APPLICATION
# ─────────────────────────────────────────────────────────

app = FastAPI(
    title="Adaptive Learning Companion API",
    description=(
        "REST interface for the LangGraph multi-agent tutoring system. "
        "Supports synchronous chat, streaming SSE, and Human-in-the-Loop review."
    ),
    version="4.0.0",
    lifespan=lifespan,
)

app.include_router(feedback_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────

def _make_config(thread_id: str) -> dict:
    """LangGraph config dict used as the persistence key."""
    return {"configurable": {"thread_id": thread_id}}


def _extract_answer(messages: list) -> str:
    """Return the last meaningful AIMessage content (handoff block stripped)."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            content = msg.content
            if HANDOFF_SIGNAL in content:
                content = content.split(HANDOFF_SIGNAL)[0].strip()
            if content:
                return content
    return ""


def _build_initial_state(student_id: str, message: str, thread_id: str) -> dict:
    """Build a fresh agent state dict for the first turn of a session."""
    return {
        "messages":        [HumanMessage(content=f"[Student ID: {student_id}] {message}")],
        "rolling_summary": "",
        "tools_called":    [],
        "tool_call_count": 0,
        "current_agent":   "researcher",
        "handoff_done":    False,
        "student_id":      student_id,
        "proposed_score":  "",
    }


def _append_message_to_state(existing_state: dict, student_id: str, message: str) -> dict:
    """Append a new HumanMessage to an existing checkpoint state."""
    existing_state["messages"] = existing_state["messages"] + [
        HumanMessage(content=f"[Student ID: {student_id}] {message}")
    ]
    existing_state["tools_called"]    = []
    existing_state["tool_call_count"] = 0
    existing_state["current_agent"]   = "researcher"
    existing_state["student_id"]      = student_id
    return existing_state


# ─────────────────────────────────────────────────────────
# ENDPOINT 1: POST /chat  (Task 1 + Task 2)
# ─────────────────────────────────────────────────────────

@app.post(
    "/chat",
    response_model=ChatResponse,
    summary="Synchronous chat turn",
    description=(
        "Send a message and receive the complete agent response. "
        "If the graph pauses at the HITL breakpoint, `hitl_pending` will be True "
        "and you must call POST /hitl/{thread_id} to continue."
    ),
)
async def chat(request: ChatRequest) -> ChatResponse:
    graph     = _app_state.app_graph
    config    = _make_config(request.thread_id)
    logger.info("[/chat] thread=%s student=%s", request.thread_id, request.student_id)

    # ── Load or create state ──────────────────────────────
    existing = _app_state.checkpointer.get(config)
    if existing:
        agent_state = existing["channel_values"]
        agent_state = _append_message_to_state(agent_state, request.student_id, request.message)
    else:
        agent_state = _build_initial_state(request.student_id, request.message, request.thread_id)

    # ── Run until breakpoint or END ───────────────────────
    try:
        result = graph.invoke(agent_state, config)
    except Exception as exc:
        logger.exception("[/chat] Graph invocation error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    # ── Detect HITL pause ─────────────────────────────────
    snapshot    = graph.get_state(config)
    next_nodes  = list(snapshot.next) if snapshot.next else []
    hitl_pending = "analyst" in next_nodes

    current_values = snapshot.values if hitl_pending else result

    answer = _extract_answer(current_values.get("messages", []))

    return ChatResponse(
        answer=answer or "(Agent is processing — HITL review required.)" if hitl_pending else answer,
        thread_id=request.thread_id,
        status=AgentStatus(
            current_agent=current_values.get("current_agent", "researcher"),
            tool_call_count=current_values.get("tool_call_count", 0),
            proposed_score=current_values.get("proposed_score") or None,
        ),
        hitl_pending=hitl_pending,
    )


# ─────────────────────────────────────────────────────────
# ENDPOINT 2: POST /stream  (Task 3 — SSE Streaming)
# ─────────────────────────────────────────────────────────

async def _event_generator(request: ChatRequest) -> AsyncGenerator[str, None]:
    """
    Async generator that streams LangGraph events as Server-Sent Events.

    Each SSE event has the form:
        data: <JSON payload>\n\n

    Event types emitted:
        node_start   — a graph node has been entered
        token        — an LLM token chunk (if model supports streaming)
        tool_use     — a tool has been called
        node_end     — a node has finished
        hitl_pause   — graph paused at the analyst breakpoint
        done         — final event carrying the full ChatResponse payload
        error        — unhandled exception
    """
    graph  = _app_state.async_graph
    config = _make_config(request.thread_id)

    existing = await _app_state.async_checkpointer.aget(config)
    if existing:
        agent_state = existing["channel_values"]
        agent_state = _append_message_to_state(agent_state, request.student_id, request.message)
    else:
        agent_state = _build_initial_state(request.student_id, request.message, request.thread_id)

    def _sse(event_type: str, payload: dict) -> str:
        data = json.dumps({"event": event_type, **payload})
        return f"data: {data}\n\n"

    try:
        # astream_events gives fine-grained token + node events
        async for event in graph.astream_events(agent_state, config, version="v2"):
            kind = event.get("event", "")
            name = event.get("name", "")

            if kind == "on_chain_start" and name in ("researcher", "analyst", "researcher_tools", "analyst_tools"):
                yield _sse("node_start", {"node": name})

            elif kind == "on_chain_end" and name in ("researcher", "analyst", "researcher_tools", "analyst_tools"):
                yield _sse("node_end", {"node": name})

            elif kind == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    yield _sse("token", {"token": chunk.content})

            elif kind == "on_tool_start":
                yield _sse("tool_use", {"tool": name, "input": str(event.get("data", {}).get("input", ""))[:200]})

        # After streaming, check for HITL pause
        snapshot   = await graph.aget_state(config)
        next_nodes = list(snapshot.next) if snapshot.next else []

        if "analyst" in next_nodes:
            current_values = snapshot.values
            yield _sse("hitl_pause", {
                "thread_id":      request.thread_id,
                "proposed_score": current_values.get("proposed_score", "unknown"),
                "message":        "POST /hitl/{thread_id} to approve, cancel, or edit.",
            })
        else:
            final_state = (await graph.aget_state(config)).values
            answer      = _extract_answer(final_state.get("messages", []))
            yield _sse("done", {
                "answer":    answer,
                "thread_id": request.thread_id,
                "status": {
                    "current_agent":  final_state.get("current_agent", "researcher"),
                    "tool_call_count": final_state.get("tool_call_count", 0),
                    "proposed_score": final_state.get("proposed_score") or None,
                },
            })

    except Exception as exc:
        logger.exception("[/stream] Error: %s", exc)
        yield _sse("error", {"detail": str(exc)})


@app.post(
    "/stream",
    summary="Streaming chat turn (SSE)",
    description=(
        "Stream the agent's response token-by-token and node-by-node using "
        "Server-Sent Events. Connect with an EventSource or curl --no-buffer."
    ),
)
async def stream(request: ChatRequest) -> StreamingResponse:
    logger.info("[/stream] thread=%s student=%s", request.thread_id, request.student_id)
    return StreamingResponse(
        _event_generator(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


# ─────────────────────────────────────────────────────────
# ENDPOINT 3: POST /hitl/{thread_id}  (Lab 5 HITL over HTTP)
# ─────────────────────────────────────────────────────────

@app.post(
    "/hitl/{thread_id}",
    response_model=HitlResponse,
    summary="Human-in-the-Loop decision",
    description=(
        "Submit an approve / cancel / edit decision for a paused analyst turn. "
        "Only valid when a previous /chat or /stream call returned hitl_pending=True."
    ),
)
async def hitl_decision(
    thread_id: str = Path(..., description="Thread ID returned from /chat"),
    decision: HitlDecision = ...,
) -> HitlResponse:
    graph  = _app_state.app_graph
    config = _make_config(thread_id)

    snapshot   = graph.get_state(config)
    next_nodes = list(snapshot.next) if snapshot.next else []

    if "analyst" not in next_nodes:
        raise HTTPException(
            status_code=409,
            detail=f"Thread '{thread_id}' is not paused at a HITL breakpoint.",
        )

    logger.info("[/hitl] thread=%s action=%s", thread_id, decision.action)

    if decision.action == "cancel":
        graph.update_state(config, {
            "current_agent":   "researcher",
            "handoff_done":    False,
            "tools_called":    [],
            "tool_call_count": 0,
        })
        return HitlResponse(
            thread_id=thread_id,
            action_taken="cancelled",
            answer=None,
        )

    if decision.action == "edit":
        if not decision.new_score:
            raise HTTPException(status_code=422, detail="new_score required when action is 'edit'.")
        graph.update_state(config, {"proposed_score": decision.new_score})
        logger.info("[/hitl] Score updated to %s", decision.new_score)

    # Approve (or post-edit) — resume from the breakpoint
    try:
        result = graph.invoke(None, config)
    except Exception as exc:
        logger.exception("[/hitl] Resume error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    answer = _extract_answer(result.get("messages", []))
    action_taken = "edited" if decision.action == "edit" else "approved"

    return HitlResponse(
        thread_id=thread_id,
        action_taken=action_taken,
        answer=answer,
    )


# ─────────────────────────────────────────────────────────
# UTILITY ENDPOINT: GET /session/{thread_id}
# ─────────────────────────────────────────────────────────

@app.get(
    "/session/{thread_id}",
    summary="Inspect session state",
    description="Return the current checkpoint values for a given thread (useful for debugging).",
)
async def get_session(thread_id: str = Path(...)):
    config   = _make_config(thread_id)
    snapshot = _app_state.app_graph.get_state(config)
    if not snapshot or not snapshot.values:
        raise HTTPException(status_code=404, detail=f"Thread '{thread_id}' not found.")

    vals = snapshot.values
    return {
        "thread_id":      thread_id,
        "current_agent":  vals.get("current_agent"),
        "tool_call_count": vals.get("tool_call_count"),
        "proposed_score": vals.get("proposed_score"),
        "message_count":  len(vals.get("messages", [])),
        "next_nodes":     list(snapshot.next) if snapshot.next else [],
    }


# ─────────────────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────────────────

@app.get("/health", summary="Health check")
async def health():
    return {
        "status": "ok",
        "graph_ready": _app_state.app_graph is not None,
        "checkpoint_db": CHECKPOINT_DB,
    }


# ─────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )