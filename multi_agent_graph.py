"""
multi_agent_graph.py  (v4)
Adaptive Learning Companion — Multi-Agent LangGraph System
──────────────────────────────────────────────────────────
WHAT'S NEW IN v4 (Lab: Persistence + HITL):
  ① SqliteSaver checkpointer  — full state persisted to checkpoint_db.sqlite.
                                 Resume any session by re-supplying its thread_id.
  ② Safety Breakpoint (HITL)  — graph is compiled with interrupt_before=["analyst"]
                                 so execution pauses before the Analyst writes to
                                 the progress DB.  Human sees the handoff package,
                                 can APPROVE, CANCEL, or EDIT the score before
                                 the agent continues.
  ③ State Editing              — human can type  edit:<new_score>  at the pause
                                 to overwrite the proposed score inside the state
                                 before resuming.

MULTI-AGENT ARCHITECTURE (unchanged — kept for viva):
  Agent A (Researcher) → [tools: retrieve_content, get_student_progress]
        │  HANDOFF_TO_ANALYST signal
        ▼
  Agent B (Analyst)    → [tools: update_student_progress]
        │  SESSION_COMPLETE signal
        ▼
       END

Two specialized agent personas, role-restricted tools, proper state handover,
routing logic, and collaborative task execution are all preserved from v3.
"""

import os
import re
import json
import logging
from typing import Annotated, Literal
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    SystemMessage, HumanMessage, AIMessage, BaseMessage
)
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver          # ← persistence
from typing_extensions import TypedDict

from tools import retrieve_content, get_student_progress, update_student_progress
from agents_config import (
    RESEARCHER_CONFIG,
    ANALYST_CONFIG,
    HANDOFF_SIGNAL,
    SESSION_COMPLETE_SIGNAL,
    WINDOW_SIZE,
)

load_dotenv()

MAX_TOOL_CALLS   = 6
CHECKPOINT_DB    = "checkpoint_db.sqlite"   # ← deliverable ③

# ─────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("multi_agent")
fh = logging.FileHandler("collaboration_trace.log", mode="a")
fh.setLevel(logging.INFO)
fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(fh)


# ─────────────────────────────────────────────────────────
# GRAPH STATE
# ─────────────────────────────────────────────────────────

class MultiAgentState(TypedDict):
    """
    Shared state for the multi-agent graph.

      messages         — Bounded sliding window (max WINDOW_SIZE).
      rolling_summary  — Plain-text summary of evicted messages.
      tools_called     — Tools already invoked this turn (loop prevention).
      tool_call_count  — Hard-cap counter; forced handoff at MAX_TOOL_CALLS.
      current_agent    — "researcher" | "analyst"
      handoff_done     — True after HANDOFF_TO_ANALYST detected.
      student_id       — Injected at session start.
      proposed_score   — [HITL] Score extracted from handoff package so a human
                         can inspect / edit it before the Analyst commits it.
    """
    messages:        Annotated[list, add_messages]
    rolling_summary: str
    tools_called:    list
    tool_call_count: int
    current_agent:   str
    handoff_done:    bool
    student_id:      str
    proposed_score:  str   


# ─────────────────────────────────────────────────────────
# TOOL SETS  (role-restricted — viva requirement)
# ─────────────────────────────────────────────────────────

RESEARCHER_TOOL_LIST = [retrieve_content, get_student_progress]  # read-only
ANALYST_TOOL_LIST    = [update_student_progress]                 # write (high-risk)

# ─────────────────────────────────────────────────────────
# LLM INSTANCES  (each bound only to its permitted tools)
# ─────────────────────────────────────────────────────────

researcher_llm = ChatOpenAI(
    model="gpt-4o", temperature=0.3, api_key=os.getenv("OPENAI_API_KEY"),
).bind_tools(RESEARCHER_TOOL_LIST)

analyst_llm = ChatOpenAI(
    model="gpt-4o", temperature=0.2, api_key=os.getenv("OPENAI_API_KEY"),
).bind_tools(ANALYST_TOOL_LIST)

# ─────────────────────────────────────────────────────────
# TOOL NODES
# ─────────────────────────────────────────────────────────

researcher_tool_node = ToolNode(tools=RESEARCHER_TOOL_LIST)
analyst_tool_node    = ToolNode(tools=ANALYST_TOOL_LIST)


# ─────────────────────────────────────────────────────────
# MEMORY HELPERS  
# ─────────────────────────────────────────────────────────

def trim_messages(state: MultiAgentState) -> tuple[list[BaseMessage], str]:
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
        logger.info("[MEMORY] Evicted %d msgs → window=%d", len(evicted), len(kept))
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
    """Pull the score value from the handoff block for the HITL review screen."""
    match = re.search(r"score[:\s]+([0-9.]+)", handoff_text, re.IGNORECASE)
    return match.group(1) if match else "unknown"


# ─────────────────────────────────────────────────────────
# NODE 1: RESEARCHER  (Agent A — persona: curious tutor)
# ─────────────────────────────────────────────────────────

def researcher_node(state: MultiAgentState) -> dict:
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
        if HANDOFF_SIGNAL in (response.content or ""):
            logger.info("[RESEARCHER] ✓ Handoff signal detected")

    # Extract proposed score for HITL state review
    proposed_score = state.get("proposed_score", "")
    if HANDOFF_SIGNAL in (response.content or ""):
        handoff_text   = response.content or ""
        proposed_score = extract_proposed_score(handoff_text)

    return {
        "messages":        window + [response],
        "rolling_summary": summary,
        "tools_called":    new_called,
        "tool_call_count": new_count,
        "current_agent":   "researcher",
        "proposed_score":  proposed_score,
    }


# ─────────────────────────────────────────────────────────
# NODE 2: ANALYST  (Agent B — persona: strict evaluator)
# HIGH-RISK node: graph interrupts BEFORE this node (HITL)
# ─────────────────────────────────────────────────────────

def analyst_node(state: MultiAgentState) -> dict:
    logger.info("━━━ ANALYST NODE ACTIVATED ━━━")
    all_msgs = state["messages"]
    handoff  = extract_handoff_package(all_msgs)

    if handoff:
        # Allow human-edited score to override the proposed score in the handoff
        human_score = state.get("proposed_score", "")
        if human_score and human_score != "unknown":
            # Inject the (potentially human-edited) score into the activation msg
            score_note = f"\n\n[HUMAN OVERRIDE] Use score = {human_score} if it differs from the handoff."
        else:
            score_note = ""

        logger.info("[ANALYST] ✓ Handoff package found")
        activation_content = (
            "You have received a handoff from the Researcher. "
            "Evaluate the student's answer and update their progress.\n\n"
            + handoff
            + score_note
            + "\n\nExecute: PARSE → EVALUATE → call update_student_progress → FEEDBACK → SESSION_COMPLETE."
        )
    else:
        logger.warning("[ANALYST] ⚠ No handoff block — using last user message")
        last_user = next(
            (m.content for m in reversed(all_msgs) if isinstance(m, HumanMessage)),
            "No student answer available."
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

    if hasattr(response, "tool_calls") and response.tool_calls:
        for tc in response.tool_calls:
            logger.info("[ANALYST] → Tool: %s(%s)", tc['name'], tc['args'])
    else:
        preview = (response.content or "")[:120].replace("\n", " ")
        logger.info("[ANALYST] → %s...", preview)
        if SESSION_COMPLETE_SIGNAL in (response.content or ""):
            logger.info("[ANALYST] ✓ SESSION_COMPLETE → END")

    return {"messages": [response], "current_agent": "analyst"}


# ─────────────────────────────────────────────────────────
# ROUTERS  (unchanged logic from v3)
# ─────────────────────────────────────────────────────────

def researcher_router(state: MultiAgentState) -> Literal["researcher_tools", "analyst", "__end__"]:
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


def analyst_router(state: MultiAgentState) -> Literal["analyst_tools", "__end__"]:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "analyst_tools"
    return END


# ─────────────────────────────────────────────────────────
# BUILD GRAPH  — compiled with interrupt_before analyst (HITL)
# ─────────────────────────────────────────────────────────

def build_multi_agent_graph(checkpointer):
    """
    Compile the graph with:
      • checkpointer  → SqliteSaver for persistent memory (Task 1)
      • interrupt_before=["analyst"]  → HITL safety pause (Task 2 & 3)
    """
    graph = StateGraph(MultiAgentState)

    graph.add_node("researcher",       researcher_node)
    graph.add_node("researcher_tools", researcher_tool_node)
    graph.add_node("analyst",          analyst_node)
    graph.add_node("analyst_tools",    analyst_tool_node)

    graph.set_entry_point("researcher")

    graph.add_conditional_edges(
        "researcher", researcher_router,
        {"researcher_tools": "researcher_tools", "analyst": "analyst", END: END},
    )
    graph.add_edge("researcher_tools", "researcher")

    graph.add_conditional_edges(
        "analyst", analyst_router,
        {"analyst_tools": "analyst_tools", END: END},
    )
    graph.add_edge("analyst_tools", "analyst")

    return graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["analyst"],   # ← HITL breakpoint (Task 2)
    )


# ─────────────────────────────────────────────────────────
# HITL REVIEW  (Task 2 + Task 3)
# ─────────────────────────────────────────────────────────

def hitl_review(app, config: dict, state: dict) -> bool:
    """
    Pause execution, show the analyst's proposed action, and wait for:
      • 'approve'         → resume as-is
      • 'cancel'          → abort this turn
      • 'edit:<score>'    → overwrite proposed_score, then resume (Task 3)

    Returns True if execution should continue, False if cancelled.
    """
    print("\n" + "="*60)
    print("  ⚠  SAFETY PAUSE — Human-in-the-Loop Review")
    print("="*60)
    print(f"  Proposed score : {state.get('proposed_score', 'unknown')}")
    print(f"  Student ID     : {state.get('student_id', '?')}")

    # Show the handoff block so the human can read the full context
    handoff = extract_handoff_package(state.get("messages", []))
    if handoff:
        print("\n  --- Handoff Package ---")
        # Print a trimmed version for readability
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
    print("="*60)

    while True:
        cmd = input("  Your decision: ").strip().lower()

        if cmd == "approve":
            logger.info("[HITL] ✓ Human approved analyst action")
            return True

        elif cmd == "cancel":
            logger.info("[HITL] ✗ Human cancelled analyst action")
            print("  → Update cancelled. Returning to Researcher.\n")
            return False

        elif cmd.startswith("edit:"):
            new_score = cmd.split(":", 1)[1].strip()
            # Update the state on the checkpoint so analyst sees the edited score
            app.update_state(config, {"proposed_score": new_score})
            logger.info("[HITL] ✎ Human edited score to: %s", new_score)
            print(f"  → Score updated to {new_score}. Resuming analyst...\n")
            return True

        else:
            print("  Unrecognised command. Type 'approve', 'cancel', or 'edit:<score>'.")


# ─────────────────────────────────────────────────────────
# INTERACTIVE CLI  (with persistence + HITL)
# ─────────────────────────────────────────────────────────

def run_multi_agent():
    # SqliteSaver opens / creates checkpoint_db.sqlite automatically
    with SqliteSaver.from_conn_string(CHECKPOINT_DB) as checkpointer:
        app = build_multi_agent_graph(checkpointer)

        print("\n" + "="*60)
        print("  ADAPTIVE LEARNING COMPANION  [Multi-Agent v4]")
        print("  Agent A: Researcher  |  Agent B: Analyst (HITL)")
        print("  Memory : Sliding window (%d msgs) + SQLite checkpoint" % WINDOW_SIZE)
        print("="*60)
        print("  Type 'quit' to exit.\n")

        logger.info("=" * 55)
        logger.info("  MULTI-AGENT SESSION STARTED  (v4 — Persistence + HITL)")
        logger.info("=" * 55)

        student_id = input("Enter your student ID (or Enter for 'student_001'): ").strip() or "student_001"

        # Thread ID determines which checkpoint to load/save
        thread_id = input("Enter thread ID to resume (or Enter for new session): ").strip()
        if not thread_id:
            import uuid
            thread_id = str(uuid.uuid4())[:8]
            print(f"  → New session created. Thread ID: {thread_id}")
            print(f"     (save this ID to resume later)\n")
        else:
            print(f"  → Resuming thread: {thread_id}\n")

        logger.info("Student: %s | Thread: %s", student_id, thread_id)

        # LangGraph config — thread_id is the persistence key
        config = {"configurable": {"thread_id": thread_id}}

        # Load existing checkpoint state if available
        existing = checkpointer.get(config)
        if existing:
            print("  ✓ Previous session restored from checkpoint.\n")
            logger.info("[CHECKPOINT] Restored thread %s", thread_id)
            agent_state = existing["channel_values"]
            # Ensure student_id is consistent
            agent_state["student_id"] = student_id
        else:
            agent_state = {
                "messages":        [],
                "rolling_summary": "",
                "tools_called":    [],
                "tool_call_count": 0,
                "current_agent":   "researcher",
                "handoff_done":    False,
                "student_id":      student_id,
                "proposed_score":  "",
            }

        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ("quit", "exit", "q"):
                print(f"\nGoodbye! Session saved as thread '{thread_id}'. 📚")
                logger.info("Session ended by user. Thread: %s", thread_id)
                break
            if not user_input:
                continue

            contextual_input = f"[Student ID: {student_id}] {user_input}"
            logger.info("[USER] %s", contextual_input)

            agent_state["messages"] = agent_state["messages"] + [
                HumanMessage(content=contextual_input)
            ]

            # ── Run until the HITL breakpoint (interrupt_before analyst) ──
            result = app.invoke(agent_state, config)

            # ── Check if we are paused at the analyst breakpoint ──
            snapshot = app.get_state(config)
            next_nodes = list(snapshot.next) if snapshot.next else []

            if "analyst" in next_nodes:
                # We're paused before analyst — present HITL review
                current_state = snapshot.values
                should_proceed = hitl_review(app, config, current_state)

                if should_proceed:
                    # Resume from breakpoint — pass None to continue from checkpoint
                    result = app.invoke(None, config)
                else:
                    # Human cancelled — reset to researcher for next question
                    app.update_state(config, {
                        "current_agent":   "researcher",
                        "handoff_done":    False,
                        "tools_called":    [],
                        "tool_call_count": 0,
                    })
                    result = app.get_state(config).values

            # ── Display last meaningful AI response ──
            for msg in reversed(result["messages"]):
                if isinstance(msg, AIMessage) and msg.content:
                    display = msg.content
                    if "---HANDOFF_TO_ANALYST---" in display:
                        display = display.split("---HANDOFF_TO_ANALYST---")[0].strip()
                    if display:
                        label = result.get("current_agent", "agent").upper()
                        print(f"\n[{label}]: {display}")
                        break

            n = len(result["messages"])
            s = len((result.get("rolling_summary") or "").splitlines())
            logger.info("[MEMORY] Live=%d | Summary lines=%d | Thread=%s", n, s, thread_id)

            # ── Carry state forward ──
            agent_state = {
                "messages":        result["messages"],
                "rolling_summary": result.get("rolling_summary", ""),
                "tools_called":    result.get("tools_called", []),
                "tool_call_count": result.get("tool_call_count", 0),
                "current_agent":   result.get("current_agent", "researcher"),
                "handoff_done":    result.get("handoff_done", False),
                "student_id":      student_id,
                "proposed_score":  result.get("proposed_score", ""),
            }

            # Reset per-turn tracking after analyst completes
            if result.get("current_agent") == "analyst":
                agent_state.update({
                    "current_agent":   "researcher",
                    "handoff_done":    False,
                    "tools_called":    [],
                    "tool_call_count": 0,
                    "proposed_score":  "",
                })


if __name__ == "__main__":
    run_multi_agent()