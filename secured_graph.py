"""
secured_graph.py  (Security Layer)
Adaptive Learning Companion — Secured Multi-Agent LangGraph
────────────────────────────────────────────────────────────
Lab: Defensive Guardrails:

  ① guardrail_node — Runs BEFORE researcher_node on every turn.
                      Stage 1: Pydantic deterministic keyword / topic check.
                      Stage 2: LLM-as-Judge semantic classification (gpt-4o-mini).
                      Verdict is stored in state["guardrail_verdict"].

  ② alert_node     — Activated when verdict == "UNSAFE".
                      Returns a standardised, context-aware refusal and logs
                      the attack vector to security_events.log.

  ③ Output sanitiser — Applied to every AIMessage and ToolMessage before
                        it reaches the user.  Redacts file paths, metadata
                        keys, API credentials, emails, and raw SQL.

  ④ Routing patch  — graph_router inspects guardrail_verdict first and
                      short-circuits to alert_node, bypassing the entire
                      Researcher → Analyst pipeline for unsafe inputs.

All previous features (SqliteSaver persistence, HITL breakpoint, sliding-window
memory, role-restricted tools) are fully preserved.

GRAPH TOPOLOGY:
  START
    └─► guardrail_node ──(SAFE)──► researcher_node ──► … (v4 pipeline)
                        │
                      (UNSAFE)
                        └──────► alert_node ──► END
"""

from __future__ import annotations

import os
import re
import json
import uuid
import logging
from typing import Annotated, Literal
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    SystemMessage, HumanMessage, AIMessage, ToolMessage, BaseMessage
)
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver
from typing_extensions import TypedDict

from tools import retrieve_content, get_student_progress, update_student_progress
from agents_config import (
    RESEARCHER_CONFIG,
    ANALYST_CONFIG,
    HANDOFF_SIGNAL,
    SESSION_COMPLETE_SIGNAL,
    WINDOW_SIZE,
)
from guardrails_config import (
    run_input_guardrails,
    sanitise_output,
    get_refusal_message,
)

load_dotenv()

MAX_TOOL_CALLS = 6
CHECKPOINT_DB  = "checkpoint_db.sqlite"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ─────────────────────────────────────────────────────────
# LOGGING  (two handlers: console + security audit file)
# ─────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("secured_agent")

# Collaboration trace (same as v4)
fh_trace = logging.FileHandler("collaboration_trace.log", mode="a")
fh_trace.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s"))
logger.addHandler(fh_trace)

# Security-specific audit log
security_logger = logging.getLogger("security_audit")
fh_security = logging.FileHandler("security_events.log", mode="a")
fh_security.setFormatter(
    logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
)
security_logger.addHandler(fh_security)
security_logger.setLevel(logging.WARNING)


# ─────────────────────────────────────────────────────────
# GRAPH STATE  (extends v4 with security fields)
# ─────────────────────────────────────────────────────────

class SecuredAgentState(TypedDict):
    # ── v4 fields (unchanged) ──────────────────────────────
    messages:        Annotated[list, add_messages]
    rolling_summary: str
    tools_called:    list
    tool_call_count: int
    current_agent:   str
    handoff_done:    bool
    student_id:      str
    proposed_score:  str
    # ── v5 security fields ────────────────────────────────
    guardrail_verdict: str   # "SAFE" | "UNSAFE" | ""
    guardrail_reason:  str   # Why UNSAFE was triggered
    blocked_count:     int   # Running total of blocked attempts this session


# ─────────────────────────────────────────────────────────
# TOOL SETS & LLM INSTANCES  (identical to v4)
# ─────────────────────────────────────────────────────────

RESEARCHER_TOOL_LIST = [retrieve_content, get_student_progress]
ANALYST_TOOL_LIST    = [update_student_progress]

researcher_llm = ChatOpenAI(
    model="gpt-4o", temperature=0.3, api_key=OPENAI_API_KEY,
).bind_tools(RESEARCHER_TOOL_LIST)

analyst_llm = ChatOpenAI(
    model="gpt-4o", temperature=0.2, api_key=OPENAI_API_KEY,
).bind_tools(ANALYST_TOOL_LIST)

researcher_tool_node = ToolNode(tools=RESEARCHER_TOOL_LIST)
analyst_tool_node    = ToolNode(tools=ANALYST_TOOL_LIST)


# ─────────────────────────────────────────────────────────
# MEMORY HELPERS  (unchanged from v4)
# ─────────────────────────────────────────────────────────

def trim_messages(state: SecuredAgentState) -> tuple[list[BaseMessage], str]:
    msgs    = state["messages"]
    summary = state.get("rolling_summary", "")
    if len(msgs) <= WINDOW_SIZE:
        return msgs, summary
    evicted = msgs[:-WINDOW_SIZE]
    kept    = msgs[-WINDOW_SIZE:]
    new_lines = []
    for m in evicted:
        role    = type(m).__name__.replace("Message", "")
        content = (getattr(m, "content", "") or "")[:80].replace("\n", " ")
        new_lines.append(f"[{role}]: {content}...")
    if new_lines:
        summary = (summary + "\n" + "\n".join(new_lines)).strip()
    return kept, summary


def build_context_header(summary: str) -> str | None:
    if not summary.strip():
        return None
    return (
        "=== EARLIER CONVERSATION (summarised) ===\n"
        + summary.strip()
        + "\n=== END OF SUMMARY ==="
    )


def extract_handoff_package(messages: list[BaseMessage]) -> str | None:
    for msg in reversed(messages):
        content = getattr(msg, "content", "") or ""
        if "---HANDOFF_TO_ANALYST---" in content and "---END_HANDOFF---" in content:
            match = re.search(
                r"---HANDOFF_TO_ANALYST---.*?---END_HANDOFF---", content, re.DOTALL
            )
            if match:
                return match.group(0)
    return None


def extract_proposed_score(handoff_text: str) -> str:
    match = re.search(r"score[:\s]+([0-9.]+)", handoff_text, re.IGNORECASE)
    return match.group(1) if match else "unknown"


# ─────────────────────────────────────────────────────────
# NODE 0: GUARDRAIL NODE  ← NEW IN v5
# ─────────────────────────────────────────────────────────

def guardrail_node(state: SecuredAgentState) -> dict:
    """
    Two-stage security gate that runs before every researcher invocation.

    Stage 1 — Pydantic deterministic check (keyword blocklist + topic allowlist)
    Stage 2 — LLM-as-Judge semantic classification via gpt-4o-mini

    If UNSAFE → sets guardrail_verdict="UNSAFE" so the router can divert
    directly to alert_node, bypassing the entire agent pipeline.
    """
    logger.info("━━━ GUARDRAIL NODE ACTIVATED ━━━")

    # Extract the latest human message
    last_human = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_human = msg.content or ""
            break

    if not last_human.strip():
        logger.info("[GUARDRAIL] Empty input — marking SAFE")
        return {"guardrail_verdict": "SAFE", "guardrail_reason": ""}

    verdict, reason = run_input_guardrails(
        prompt=last_human,
        openai_api_key=OPENAI_API_KEY,
        use_llm_judge=True,
    )

    if verdict == "UNSAFE":
        blocked_count = state.get("blocked_count", 0) + 1
        security_logger.warning(
            "BLOCKED | student=%s | count=%d | reason=%s | prompt=%.120s",
            state.get("student_id", "?"),
            blocked_count,
            reason,
            last_human,
        )
        logger.warning("[GUARDRAIL] ✗ UNSAFE — %s", reason)
        return {
            "guardrail_verdict": "UNSAFE",
            "guardrail_reason":  reason,
            "blocked_count":     blocked_count,
        }

    logger.info("[GUARDRAIL] ✓ SAFE — proceeding to Researcher")
    return {"guardrail_verdict": "SAFE", "guardrail_reason": ""}


# ─────────────────────────────────────────────────────────
# NODE 0b: ALERT NODE  ← NEW IN v5
# ─────────────────────────────────────────────────────────

def alert_node(state: SecuredAgentState) -> dict:
    """
    Activated when guardrail_verdict == "UNSAFE".

    Generates a standardised, context-aware refusal message and appends
    it as an AIMessage so it is displayed to the user in the normal flow.
    """
    logger.info("━━━ ALERT NODE ACTIVATED ━━━")
    reason  = state.get("guardrail_reason", "")
    refusal = get_refusal_message(reason)

    security_logger.warning(
        "ALERT_RESPONSE | student=%s | refusal_key=%s",
        state.get("student_id", "?"),
        reason[:60],
    )

    refusal_msg = AIMessage(content=refusal)
    return {
        "messages":          [refusal_msg],
        "current_agent":     "guardrail",
        "guardrail_verdict": "",   # reset for next turn
        "guardrail_reason":  "",
        "tools_called":      [],
        "tool_call_count":   0,
    }


# ─────────────────────────────────────────────────────────
# NODE 1: RESEARCHER  (unchanged logic from v4)
# ─────────────────────────────────────────────────────────

def researcher_node(state: SecuredAgentState) -> dict:
    logger.info("━━━ RESEARCHER NODE ACTIVATED ━━━")
    window, summary = trim_messages(state)
    logger.info("[RESEARCHER] Window=%d | ToolCalls=%d/%d",
                len(window), state.get("tool_call_count", 0), MAX_TOOL_CALLS)

    called_set: list = state.get("tools_called", [])
    dedup_reminder = ""
    if called_set:
        dedup_reminder = (
            "\n\nIMPORTANT — already called this turn (do NOT repeat):\n"
            + "\n".join(f"  - {c}" for c in called_set)
            + "\nPresent the practice problem NOW if content is ready."
        )

    prompt: list[BaseMessage] = [
        SystemMessage(content=RESEARCHER_CONFIG["system_prompt"] + dedup_reminder)
    ]
    if header := build_context_header(summary):
        prompt.append(HumanMessage(content=header))
    prompt.extend(window)

    response = researcher_llm.invoke(prompt)

    new_called = list(called_set)
    new_count  = state.get("tool_call_count", 0)

    if hasattr(response, "tool_calls") and response.tool_calls:
        for tc in response.tool_calls:
            key = f"{tc['name']}({tc['args']})"
            logger.info("[RESEARCHER] → Tool: %s", key)
            if key not in new_called:
                new_called.append(key)
        new_count += len(response.tool_calls)
    else:
        preview = (response.content or "")[:120].replace("\n", " ")
        logger.info("[RESEARCHER] → %s...", preview)

    proposed_score = state.get("proposed_score", "")
    if HANDOFF_SIGNAL in (response.content or ""):
        proposed_score = extract_proposed_score(response.content or "")
        logger.info("[RESEARCHER] ✓ Handoff signal detected")

    # ── OUTPUT SANITISATION: apply to researcher response ──
    if response.content:
        sanitised, triggered = sanitise_output(response.content)
        if triggered:
            logger.warning("[SANITISER] Researcher output redacted: %s", triggered)
            security_logger.warning(
                "OUTPUT_SANITISED | agent=researcher | rules=%s | student=%s",
                triggered, state.get("student_id", "?"),
            )
            response = AIMessage(
                content=sanitised,
                tool_calls=getattr(response, "tool_calls", []),
            )

    return {
        "messages":        window + [response],
        "rolling_summary": summary,
        "tools_called":    new_called,
        "tool_call_count": new_count,
        "current_agent":   "researcher",
        "proposed_score":  proposed_score,
    }


# ─────────────────────────────────────────────────────────
# NODE 2: ANALYST  (with output sanitisation added)
# ─────────────────────────────────────────────────────────

def analyst_node(state: SecuredAgentState) -> dict:
    logger.info("━━━ ANALYST NODE ACTIVATED ━━━")
    all_msgs = state["messages"]
    handoff  = extract_handoff_package(all_msgs)

    if handoff:
        human_score = state.get("proposed_score", "")
        score_note  = (
            f"\n\n[HUMAN OVERRIDE] Use score = {human_score}."
            if human_score and human_score != "unknown" else ""
        )
        activation_content = (
            "You have received a handoff from the Researcher. "
            "Evaluate the student's answer and update their progress.\n\n"
            + handoff + score_note
            + "\n\nExecute: PARSE → EVALUATE → call update_student_progress "
              "→ FEEDBACK → SESSION_COMPLETE."
        )
    else:
        last_user = next(
            (m.content for m in reversed(all_msgs) if isinstance(m, HumanMessage)),
            "No student answer available.",
        )
        activation_content = (
            f"Student's answer: {last_user}\n\n"
            "Evaluate, call update_student_progress, give feedback, end with SESSION_COMPLETE."
        )

    prompt: list[BaseMessage] = [
        SystemMessage(content=ANALYST_CONFIG["system_prompt"]),
        HumanMessage(content=activation_content),
    ]
    response = analyst_llm.invoke(prompt)

    # ── OUTPUT SANITISATION: apply to analyst response ──
    if response.content:
        sanitised, triggered = sanitise_output(response.content)
        if triggered:
            logger.warning("[SANITISER] Analyst output redacted: %s", triggered)
            security_logger.warning(
                "OUTPUT_SANITISED | agent=analyst | rules=%s | student=%s",
                triggered, state.get("student_id", "?"),
            )
            response = AIMessage(content=sanitised)

    if hasattr(response, "tool_calls") and response.tool_calls:
        for tc in response.tool_calls:
            logger.info("[ANALYST] → Tool: %s(%s)", tc["name"], tc["args"])
    else:
        preview = (response.content or "")[:120].replace("\n", " ")
        logger.info("[ANALYST] → %s...", preview)
        if SESSION_COMPLETE_SIGNAL in (response.content or ""):
            logger.info("[ANALYST] ✓ SESSION_COMPLETE → END")

    return {"messages": [response], "current_agent": "analyst"}


# ─────────────────────────────────────────────────────────
# OUTPUT SANITISATION: wrap ToolNodes to intercept messages
# ─────────────────────────────────────────────────────────

def sanitise_tool_messages(messages: list[BaseMessage], agent_label: str, student_id: str) -> list[BaseMessage]:
    """Redact sensitive data from ToolMessage content before it enters state."""
    cleaned = []
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.content:
            sanitised, triggered = sanitise_output(msg.content)
            if triggered:
                logger.warning("[SANITISER] Tool output redacted (%s): %s", agent_label, triggered)
                security_logger.warning(
                    "OUTPUT_SANITISED | agent=%s | rules=%s | student=%s",
                    agent_label, triggered, student_id,
                )
                msg = ToolMessage(
                    content=sanitised,
                    tool_call_id=msg.tool_call_id,
                    name=getattr(msg, "name", None),
                )
        cleaned.append(msg)
    return cleaned


def researcher_tools_sanitised(state: SecuredAgentState) -> dict:
    """Wrapper: run researcher tools then sanitise tool messages."""
    result = researcher_tool_node.invoke(state)
    msgs   = result.get("messages", [])
    cleaned = sanitise_tool_messages(msgs, "researcher_tools", state.get("student_id", "?"))
    return {"messages": cleaned}


def analyst_tools_sanitised(state: SecuredAgentState) -> dict:
    """Wrapper: run analyst tools then sanitise tool messages."""
    result = analyst_tool_node.invoke(state)
    msgs   = result.get("messages", [])
    cleaned = sanitise_tool_messages(msgs, "analyst_tools", state.get("student_id", "?"))
    return {"messages": cleaned}


# ─────────────────────────────────────────────────────────
# ROUTERS
# ─────────────────────────────────────────────────────────

def guardrail_router(state: SecuredAgentState) -> Literal["researcher", "alert"]:
    """Route after guardrail_node: SAFE → researcher, UNSAFE → alert."""
    if state.get("guardrail_verdict") == "UNSAFE":
        logger.info("[ROUTER] Guardrail → alert_node")
        return "alert"
    logger.info("[ROUTER] Guardrail → researcher_node")
    return "researcher"


def researcher_router(state: SecuredAgentState) -> Literal["researcher_tools", "analyst", "__end__"]:
    last = state["messages"][-1]
    if state.get("tool_call_count", 0) >= MAX_TOOL_CALLS:
        logger.warning("[ROUTER] MAX_TOOL_CALLS reached — forcing handoff")
        return "analyst"
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "researcher_tools"
    if HANDOFF_SIGNAL in (last.content or ""):
        logger.info("[ROUTER] Handoff signal → analyst")
        return "analyst"
    return END


def analyst_router(state: SecuredAgentState) -> Literal["analyst_tools", "__end__"]:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "analyst_tools"
    return END


# ─────────────────────────────────────────────────────────
# BUILD SECURED GRAPH
# ─────────────────────────────────────────────────────────

def build_secured_graph(checkpointer):
    """
    Compile the secured graph:
      guardrail_node → researcher_node / alert_node
      … rest of v4 pipeline with HITL breakpoint preserved …
    """
    graph = StateGraph(SecuredAgentState)

    # Register nodes
    graph.add_node("guardrail",          guardrail_node)
    graph.add_node("alert",              alert_node)
    graph.add_node("researcher",         researcher_node)
    graph.add_node("researcher_tools",   researcher_tools_sanitised)
    graph.add_node("analyst",            analyst_node)
    graph.add_node("analyst_tools",      analyst_tools_sanitised)

    # Entry point is now the guardrail
    graph.set_entry_point("guardrail")

    # Guardrail routing: SAFE → researcher, UNSAFE → alert
    graph.add_conditional_edges(
        "guardrail", guardrail_router,
        {"researcher": "researcher", "alert": "alert"},
    )

    # Alert terminates the turn
    graph.add_edge("alert", END)

    # Researcher routing (unchanged from v4)
    graph.add_conditional_edges(
        "researcher", researcher_router,
        {"researcher_tools": "researcher_tools", "analyst": "analyst", END: END},
    )
    graph.add_edge("researcher_tools", "researcher")

    # Analyst routing (unchanged from v4)
    graph.add_conditional_edges(
        "analyst", analyst_router,
        {"analyst_tools": "analyst_tools", END: END},
    )
    graph.add_edge("analyst_tools", "analyst")

    return graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["analyst"],   # HITL breakpoint preserved from v4
    )


# ─────────────────────────────────────────────────────────
# HITL REVIEW  (unchanged from v4)
# ─────────────────────────────────────────────────────────

def hitl_review(app, config: dict, state: dict) -> bool:
    print("\n" + "=" * 60)
    print("  ⚠  SAFETY PAUSE — Human-in-the-Loop Review")
    print("=" * 60)
    print(f"  Proposed score : {state.get('proposed_score', 'unknown')}")
    print(f"  Student ID     : {state.get('student_id', '?')}")

    handoff = extract_handoff_package(state.get("messages", []))
    if handoff:
        print("\n  --- Handoff Package ---")
        lines = handoff.splitlines()
        for line in lines[:20]:
            print("  " + line)
        if len(lines) > 20:
            print(f"  ... ({len(lines)-20} more lines)")
        print("  -----------------------")

    print("\n  Commands:")
    print("    approve          — proceed with the update")
    print("    cancel           — skip this update")
    print("    edit:<score>     — change the score (e.g. edit:85)")
    print("=" * 60)

    while True:
        cmd = input("  Your decision: ").strip().lower()
        if cmd == "approve":
            logger.info("[HITL] ✓ Human approved analyst action")
            return True
        elif cmd == "cancel":
            logger.info("[HITL] ✗ Human cancelled analyst action")
            print("  → Update cancelled.\n")
            return False
        elif cmd.startswith("edit:"):
            new_score = cmd.split(":", 1)[1].strip()
            app.update_state(config, {"proposed_score": new_score})
            logger.info("[HITL] ✎ Human edited score to: %s", new_score)
            print(f"  → Score updated to {new_score}. Resuming...\n")
            return True
        else:
            print("  Unrecognised command.")


# ─────────────────────────────────────────────────────────
# INTERACTIVE CLI
# ─────────────────────────────────────────────────────────

def run_secured_agent():
    with SqliteSaver.from_conn_string(CHECKPOINT_DB) as checkpointer:
        app = build_secured_graph(checkpointer)

        print("\n" + "=" * 60)
        print("  ADAPTIVE LEARNING COMPANION  [Secured v5]")
        print("  Guardrail: Pydantic + LLM-Judge  |  HITL: analyst")
        print("  Memory: Sliding window + SQLite checkpoint")
        print("=" * 60)
        print("  Type 'quit' to exit.\n")

        security_logger.warning("=" * 55)
        security_logger.warning("  SECURED SESSION STARTED (v5)")
        security_logger.warning("=" * 55)

        student_id = input("Enter your student ID (or Enter for 'student_001'): ").strip() or "student_001"
        thread_id  = input("Enter thread ID to resume (or Enter for new): ").strip()
        if not thread_id:
            thread_id = str(uuid.uuid4())[:8]
            print(f"  → New session. Thread ID: {thread_id}\n")
        else:
            print(f"  → Resuming thread: {thread_id}\n")

        config = {"configurable": {"thread_id": thread_id}}

        existing = checkpointer.get(config)
        if existing:
            print("  ✓ Previous session restored.\n")
            agent_state = existing["channel_values"]
            agent_state["student_id"] = student_id
        else:
            agent_state = {
                "messages":          [],
                "rolling_summary":   "",
                "tools_called":      [],
                "tool_call_count":   0,
                "current_agent":     "researcher",
                "handoff_done":      False,
                "student_id":        student_id,
                "proposed_score":    "",
                "guardrail_verdict": "",
                "guardrail_reason":  "",
                "blocked_count":     0,
            }

        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ("quit", "exit", "q"):
                print(f"\nGoodbye! Thread '{thread_id}' saved. 📚")
                break
            if not user_input:
                continue

            contextual_input = f"[Student ID: {student_id}] {user_input}"
            agent_state["messages"] = agent_state["messages"] + [
                HumanMessage(content=contextual_input)
            ]
            # Reset guardrail fields for this new turn
            agent_state["guardrail_verdict"] = ""
            agent_state["guardrail_reason"]  = ""

            result = app.invoke(agent_state, config)

            # Check for HITL breakpoint
            snapshot   = app.get_state(config)
            next_nodes = list(snapshot.next) if snapshot.next else []

            if "analyst" in next_nodes:
                current_state  = snapshot.values
                should_proceed = hitl_review(app, config, current_state)
                if should_proceed:
                    result = app.invoke(None, config)
                else:
                    app.update_state(config, {
                        "current_agent":   "researcher",
                        "handoff_done":    False,
                        "tools_called":    [],
                        "tool_call_count": 0,
                    })
                    result = app.get_state(config).values

            # Display last meaningful AI response
            for msg in reversed(result["messages"]):
                if isinstance(msg, AIMessage) and msg.content:
                    display = msg.content
                    if "---HANDOFF_TO_ANALYST---" in display:
                        display = display.split("---HANDOFF_TO_ANALYST---")[0].strip()
                    if display:
                        label = result.get("current_agent", "agent").upper()
                        print(f"\n[{label}]: {display}")
                        break

            # Show blocked count if non-zero
            bc = result.get("blocked_count", 0)
            if bc:
                print(f"\n  ⚠  [{bc} blocked attempt(s) this session]")

            # Carry state forward
            agent_state = {
                "messages":          result["messages"],
                "rolling_summary":   result.get("rolling_summary", ""),
                "tools_called":      result.get("tools_called", []),
                "tool_call_count":   result.get("tool_call_count", 0),
                "current_agent":     result.get("current_agent", "researcher"),
                "handoff_done":      result.get("handoff_done", False),
                "student_id":        student_id,
                "proposed_score":    result.get("proposed_score", ""),
                "guardrail_verdict": "",
                "guardrail_reason":  "",
                "blocked_count":     bc,
            }

            if result.get("current_agent") == "analyst":
                agent_state.update({
                    "current_agent":   "researcher",
                    "handoff_done":    False,
                    "tools_called":    [],
                    "tool_call_count": 0,
                    "proposed_score":  "",
                })


if __name__ == "__main__":
    run_secured_agent()