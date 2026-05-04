"""
schema.py — Pydantic Models for Adaptive Learning Companion API
───────────────────────────────────────────────────────────────
Lab 8 · Task 1: Endpoint Design & Schema Validation

Defines the request / response contracts between any HTTP client
and the LangGraph multi-agent backend.
"""

from __future__ import annotations

import uuid
from typing import Literal, Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────
# REQUEST MODELS
# ─────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """
    Payload sent by the client to POST /chat or POST /stream.

    Fields
    ------
    message   : The student's free-text input (question, answer, command).
    thread_id : Identifies the conversation session.  A new UUID is generated
                automatically when omitted, creating a fresh session.
    student_id: Identifies the learner; defaults to 'student_001' so existing
                progress-DB entries remain compatible with earlier lab data.
    """

    message: str = Field(
        ...,
        min_length=1,
        max_length=4_000,
        description="The student's message to the learning agent.",
        examples=["Explain recursion with an example."],
    )

    thread_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description=(
            "Conversation thread identifier used as the LangGraph checkpoint key. "
            "Supply the same value across requests to resume a session."
        ),
        examples=["a1b2c3d4"],
    )

    student_id: str = Field(
        default="student_001",
        description="Learner identifier; matched against the progress database.",
        examples=["student_042"],
    )

    class Config:
        json_schema_extra = {
            "example": {
                "message": "I think the answer to the recursion problem is n * factorial(n-1).",
                "thread_id": "a1b2c3d4",
                "student_id": "student_042",
            }
        }


# ─────────────────────────────────────────────────────────
# RESPONSE MODELS
# ─────────────────────────────────────────────────────────

class AgentStatus(BaseModel):
    """Snapshot of which agent is active and how many tool calls were made."""

    current_agent: Literal["researcher", "analyst"] = Field(
        description="The agent that produced the final message in this turn."
    )
    tool_call_count: int = Field(
        ge=0,
        description="Cumulative tool invocations in this turn.",
    )
    proposed_score: Optional[str] = Field(
        default=None,
        description=(
            "Score extracted from the Researcher's handoff package. "
            "Present only when a HITL review is pending."
        ),
    )


class ChatResponse(BaseModel):
    """
    Returned by POST /chat after the agent completes a full turn.

    Fields
    ------
    answer     : The last meaningful AI message (handoff block stripped).
    thread_id  : Echo the thread so clients can persist it for future requests.
    status     : Lightweight agent telemetry.
    hitl_pending: True when the graph paused at the analyst breakpoint and
                  is awaiting human approval via POST /hitl/{thread_id}.
    """

    answer: str = Field(
        description="The agent's response to the student's message."
    )
    thread_id: str = Field(
        description="Session identifier — store this to continue the conversation."
    )
    status: AgentStatus = Field(
        description="Operational metadata about the agent turn."
    )
    hitl_pending: bool = Field(
        default=False,
        description=(
            "When True the Analyst update is paused. "
            "Call POST /hitl/{thread_id} to approve, cancel, or edit the score."
        ),
    )

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Great job! Your recursive definition is correct. Score: 90.",
                "thread_id": "a1b2c3d4",
                "status": {
                    "current_agent": "analyst",
                    "tool_call_count": 2,
                    "proposed_score": "90",
                },
                "hitl_pending": False,
            }
        }


# ─────────────────────────────────────────────────────────
# HITL (Human-in-the-Loop) MODELS
# ─────────────────────────────────────────────────────────

class HitlDecision(BaseModel):
    """
    Payload for POST /hitl/{thread_id}.

    action     : 'approve' resumes as-is; 'cancel' aborts the analyst update;
                 'edit' overrides the proposed score.
    new_score  : Required when action == 'edit'.
    """

    action: Literal["approve", "cancel", "edit"] = Field(
        description="Human decision on the pending analyst action.",
        examples=["approve"],
    )
    new_score: Optional[str] = Field(
        default=None,
        description="Replacement score; only used when action is 'edit'.",
        examples=["85"],
    )

    class Config:
        json_schema_extra = {
            "example": {
                "action": "edit",
                "new_score": "85",
            }
        }


class HitlResponse(BaseModel):
    """Confirmation returned after a HITL decision is processed."""

    thread_id: str
    action_taken: Literal["approved", "cancelled", "edited"]
    answer: Optional[str] = Field(
        default=None,
        description="Analyst response after resumption (absent when cancelled).",
    )