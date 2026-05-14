"""
feedback_routes.py — Feedback Collection & Drift Monitoring
────────────────────────────────────────────────────────────
Add to your existing main.py FastAPI app via:
    app.include_router(feedback_router)

Endpoints:
  POST /feedback          — Submit good/bad rating for an interaction
  GET  /feedback/stats    — Quick summary stats
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("feedback")

FEEDBACK_DB = Path("/app/checkpoint_data/feedback_log.db")
FEEDBACK_DB.parent.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────
# DB INIT
# ─────────────────────────────────────────────────────────

def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(FEEDBACK_DB))
    conn.row_factory = sqlite3.Row
    return conn


def init_feedback_db() -> None:
    """Create the feedback table if it doesn't exist. Call at startup."""
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback_log (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id      TEXT    NOT NULL,
                student_id     TEXT    NOT NULL DEFAULT 'unknown',
                user_input     TEXT    NOT NULL,
                agent_response TEXT    NOT NULL,
                agent_type     TEXT    NOT NULL DEFAULT 'researcher',
                feedback       TEXT    NOT NULL CHECK(feedback IN ('good','bad')),
                timestamp      TEXT    NOT NULL
            )
        """)
        conn.commit()
    logger.info("[FEEDBACK] DB ready at %s", FEEDBACK_DB)


# ─────────────────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    thread_id:      str
    student_id:     str = "unknown"
    user_input:     str = Field(..., min_length=1)
    agent_response: str = Field(..., min_length=1)
    agent_type:     str = "researcher"
    feedback:       Literal["good", "bad"]


class FeedbackResponse(BaseModel):
    success:    bool
    feedback_id: int
    message:    str


# ─────────────────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────────────────

feedback_router = APIRouter(prefix="/feedback", tags=["Feedback"])


@feedback_router.post("", response_model=FeedbackResponse)
async def submit_feedback(req: FeedbackRequest) -> FeedbackResponse:
    """Store a thumbs-up / thumbs-down rating for an agent interaction."""
    ts = datetime.now(timezone.utc).isoformat()
    try:
        with _get_conn() as conn:
            cur = conn.execute(
                """INSERT INTO feedback_log
                   (thread_id, student_id, user_input, agent_response, agent_type, feedback, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (req.thread_id, req.student_id, req.user_input,
                 req.agent_response, req.agent_type, req.feedback, ts),
            )
            conn.commit()
            fid = cur.lastrowid
    except Exception as exc:
        logger.exception("[FEEDBACK] DB write error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    logger.info("[FEEDBACK] id=%d thread=%s rating=%s", fid, req.thread_id, req.feedback)
    return FeedbackResponse(
        success=True,
        feedback_id=fid,
        message=f"Feedback recorded (id={fid})",
    )


@feedback_router.get("/stats")
async def feedback_stats():
    """Quick summary: totals, negative count, top failed queries."""
    with _get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) FROM feedback_log").fetchone()[0]
        neg   = conn.execute(
            "SELECT COUNT(*) FROM feedback_log WHERE feedback='bad'"
        ).fetchone()[0]
        top_bad = conn.execute(
            """SELECT user_input, COUNT(*) as cnt
               FROM feedback_log WHERE feedback='bad'
               GROUP BY user_input ORDER BY cnt DESC LIMIT 3"""
        ).fetchall()

    return {
        "total_responses":   total,
        "negative_feedback": neg,
        "positive_feedback": total - neg,
        "negative_rate":     round(neg / total * 100, 1) if total else 0,
        "top_failed_queries": [
            {"query": r["user_input"], "bad_count": r["cnt"]} for r in top_bad
        ],
    }