"""
multi_agent_graph.py  (v3)
Adaptive Learning Companion — Multi-Agent LangGraph System
──────────────────────────────────────────────────────────
Two-agent ReAct pipeline:

  Agent A (Researcher) → [tools: retrieve_content, get_student_progress]
        │  HANDOFF_TO_ANALYST signal
        ▼
  Agent B (Analyst)    → [tools: update_student_progress]
        │  SESSION_COMPLETE signal
        ▼
       END

CHANGES v3 (fixes two regressions found in v2):
  ① Tool-loop prevention  — State tracks which tool/args combos the Researcher
                             has already called this turn (tools_called set).
                             researcher_node injects a "DO NOT REPEAT" reminder
                             listing already-called tools, so the LLM stops
                             re-issuing identical retrieve_content calls.
                             researcher_router enforces a hard MAX_TOOL_CALLS cap
                             as a safety net, forcing handoff if the loop runs away.
  ② Real memory bounding  — trim_messages() now ALSO returns the trimmed list so
                             researcher_node writes it back into state["messages"].
                             Previously trim only filtered what the LLM *saw*;
                             state kept growing unboundedly via add_messages.
                             Now state["messages"] is capped at WINDOW_SIZE after
                             every node execution.
"""

import os
import re
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

# Hard cap on Researcher tool calls per user turn (safety net against infinite loops)
MAX_TOOL_CALLS = 6

# ─────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("multi_agent")

file_handler = logging.FileHandler("collaboration_trace.log", mode="a")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
)
logger.addHandler(file_handler)


# ─────────────────────────────────────────────────────────
# GRAPH STATE
# ─────────────────────────────────────────────────────────

class MultiAgentState(TypedDict):
    """
    Shared state for the multi-agent graph.

      messages        — Bounded sliding window (max WINDOW_SIZE). Written back
                        to state after every node so it never exceeds the cap.
      rolling_summary — Plain-text summary of evicted messages, injected as a
                        context header so learning history is never silently lost.
      tools_called    — Set of "tool_name:arg_hash" strings the Researcher has
                        already invoked this user turn. Reset to empty after each
                        full turn. Prevents identical tool calls from looping.
      tool_call_count — Integer count of tool calls this turn; hard-capped at
                        MAX_TOOL_CALLS by the router as a safety net.
      current_agent   — "researcher" | "analyst"
      handoff_done    — True after HANDOFF_TO_ANALYST is detected
      student_id      — Injected at session start
    """
    messages:        Annotated[list, add_messages]
    rolling_summary: str
    tools_called:    list   # serialised as list (sets aren't TypedDict-safe)
    tool_call_count: int
    current_agent:   str
    handoff_done:    bool
    student_id:      str


# ─────────────────────────────────────────────────────────
# TOOL SETS
# ─────────────────────────────────────────────────────────

RESEARCHER_TOOL_LIST = [retrieve_content, get_student_progress]
ANALYST_TOOL_LIST    = [update_student_progress]

# ─────────────────────────────────────────────────────────
# LLM INSTANCES  (each bound only to its permitted tools)
# ─────────────────────────────────────────────────────────

researcher_llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.3,
    api_key=os.getenv("OPENAI_API_KEY"),
).bind_tools(RESEARCHER_TOOL_LIST)

analyst_llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.2,
    api_key=os.getenv("OPENAI_API_KEY"),
).bind_tools(ANALYST_TOOL_LIST)

# ─────────────────────────────────────────────────────────
# TOOL NODES  (isolated per agent)
# ─────────────────────────────────────────────────────────

researcher_tool_node = ToolNode(tools=RESEARCHER_TOOL_LIST)
analyst_tool_node    = ToolNode(tools=ANALYST_TOOL_LIST)


# ─────────────────────────────────────────────────────────
# MEMORY HELPERS
# ─────────────────────────────────────────────────────────

def trim_messages(state: MultiAgentState) -> tuple[list[BaseMessage], str]:
    """
    Sliding-window memory management.

    Returns (trimmed_window, updated_summary).

    The caller (researcher_node) must write `trimmed_window` back
    into state["messages"] so the state list itself is bounded, not just the
    prompt sent to the LLM.  Previously only the prompt was trimmed while
    state["messages"] kept growing without limit via add_messages.
    """
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
        logger.info(
            "[MEMORY] Evicted %d msgs → window=%d, summary_lines=%d",
            len(evicted), len(kept), len(summary.splitlines()),
        )

    return kept, summary


def build_context_header(summary: str) -> str | None:
    """
    If there is a non-empty rolling summary, format it as a header message
    to inject at the front of the prompt so the LLM knows what happened earlier.
    """
    if not summary.strip():
        return None
    return (
        "=== EARLIER CONVERSATION (summarised, oldest first) ===\n"
        + summary.strip()
        + "\n=== END OF SUMMARY — continue from live messages below ==="
    )


def extract_handoff_package(messages: list[BaseMessage]) -> str | None:
    """
    Scan messages (newest-first) for a HANDOFF block and return it.
    Returns None if no block is found (safety guard).
    """
    for msg in reversed(messages):
        content = getattr(msg, "content", "") or ""
        if "---HANDOFF_TO_ANALYST---" in content and "---END_HANDOFF---" in content:
            # Extract just the block
            match = re.search(
                r"---HANDOFF_TO_ANALYST---.*?---END_HANDOFF---",
                content,
                re.DOTALL,
            )
            if match:
                return match.group(0)
    return None


# ─────────────────────────────────────────────────────────
# NODE 1: RESEARCHER
# ─────────────────────────────────────────────────────────

def researcher_node(state: MultiAgentState) -> dict:
    """
    Agent A — Researcher.

      • Writes trimmed window back into state["messages"] so the list is
        genuinely bounded (not just the LLM prompt).
      • Injects a "DO NOT REPEAT" reminder listing already-called tools so
        the LLM stops re-issuing identical retrieve_content calls.
      • Updates tools_called and tool_call_count after each invocation.
    """
    logger.info("━━━ RESEARCHER NODE ACTIVATED ━━━")

    # ── 1. Trim & prepare write-back ───────────────────────
    window, summary = trim_messages(state)
    logger.info("[RESEARCHER] Window=%d msgs | Summary=%s | ToolCalls=%d/%d",
                len(window), bool(summary),
                state.get("tool_call_count", 0), MAX_TOOL_CALLS)

    # ── 2. Build already-called reminder ───────────────────
    called_set: list = state.get("tools_called", [])
    dedup_reminder = ""
    if called_set:
        dedup_reminder = (
            "\n\nIMPORTANT — you have ALREADY called these tools this turn "
            "(do NOT call them again with the same arguments):\n"
            + "\n".join(f"  - {c}" for c in called_set)
            + "\nIf you have retrieved prerequisites, explanation, AND practice content, "
            "present the practice problem to the student NOW and wait for their answer."
        )

    # ── 3. Build prompt ─────────────────────────────────────
    system_content = RESEARCHER_CONFIG["system_prompt"] + dedup_reminder
    prompt: list[BaseMessage] = [SystemMessage(content=system_content)]
    if header := build_context_header(summary):
        prompt.append(HumanMessage(content=header))
    prompt.extend(window)

    response = researcher_llm.invoke(prompt)

    # ── 4. Track tool calls ─────────────────────────────────
    new_called = list(called_set)
    new_count  = state.get("tool_call_count", 0)

    if hasattr(response, "tool_calls") and response.tool_calls:
        for tc in response.tool_calls:
            key = f"{tc['name']}({tc['args']})"
            logger.info("[RESEARCHER] → Tool call: %s", key)
            if key not in new_called:
                new_called.append(key)
        new_count += len(response.tool_calls)
    else:
        preview = (response.content or "")[:120].replace("\n", " ")
        logger.info("[RESEARCHER] → Response: %s...", preview)
        if HANDOFF_SIGNAL in (response.content or ""):
            logger.info("[RESEARCHER] ✓ Handoff signal — transitioning to Analyst")

    # ── 5. Write trimmed window + new response back to state ─
    # This is the key fix: state["messages"] is replaced with the bounded
    # window rather than letting add_messages append indefinitely.
    return {
        "messages":        window + [response],
        "rolling_summary": summary,
        "tools_called":    new_called,
        "tool_call_count": new_count,
        "current_agent":   "researcher",
    }


# ─────────────────────────────────────────────────────────
# NODE 2: ANALYST
# ─────────────────────────────────────────────────────────

def analyst_node(state: MultiAgentState) -> dict:
    """
    Agent B — Analyst.

    Instead of passing the full noisy conversation:
      1. Extract the handoff package from wherever it sits in history.
      2. Build a clean, isolated activation message containing only the
         handoff block + a clear instruction to evaluate.
      3. Send ONLY [system_prompt, activation_message] to the LLM.
    """
    logger.info("━━━ ANALYST NODE ACTIVATED ━━━")

    # Pull handoff package from message history
    all_msgs = state["messages"]
    handoff  = extract_handoff_package(all_msgs)

    if handoff:
        logger.info("[ANALYST] ✓ Handoff package extracted successfully")
        activation_content = (
            "You have received a handoff from the Researcher. "
            "A student has just answered a question and is waiting for your evaluation.\n\n"
            "Here is the complete handoff package:\n\n"
            + handoff
            + "\n\nPlease execute your mandatory 5-step workflow now: "
            "PARSE → EVALUATE → call update_student_progress → FEEDBACK → SESSION_COMPLETE."
        )
    else:
        # Safety fallback: no handoff found — scan raw messages for student answer
        logger.warning("[ANALYST] ⚠ No handoff block found — using last user message as fallback")
        last_user = next(
            (m.content for m in reversed(all_msgs) if isinstance(m, HumanMessage)),
            "No student answer available."
        )
        activation_content = (
            "A student has answered a question. No formal handoff package was detected, "
            "but their most recent message was:\n\n"
            f"{last_user}\n\n"
            "Please evaluate their response as best you can, call update_student_progress "
            "with an appropriate score, give feedback, and end with SESSION_COMPLETE."
        )

    # Clean isolated prompt — ONLY system prompt + activation message
    prompt: list[BaseMessage] = [
        SystemMessage(content=ANALYST_CONFIG["system_prompt"]),
        HumanMessage(content=activation_content),
    ]

    response = analyst_llm.invoke(prompt)

    # Log
    if hasattr(response, "tool_calls") and response.tool_calls:
        for tc in response.tool_calls:
            logger.info(f"[ANALYST] → Tool call: {tc['name']}({tc['args']})")
    else:
        preview = (response.content or "")[:120].replace("\n", " ")
        logger.info(f"[ANALYST] → Response: {preview}...")
        if SESSION_COMPLETE_SIGNAL in (response.content or ""):
            logger.info("[ANALYST] ✓ SESSION_COMPLETE signal → END")

    return {
        "messages":      [response],
        "current_agent": "analyst",
    }


# ─────────────────────────────────────────────────────────
# ROUTERS
# ─────────────────────────────────────────────────────────

def researcher_router(
    state: MultiAgentState,
) -> Literal["researcher_tools", "analyst", "__end__"]:
    last = state["messages"][-1]

    # Hard cap — if tool calls have exceeded MAX_TOOL_CALLS, force handoff
    # This is the safety net that stops runaway loops even if dedup fails
    if state.get("tool_call_count", 0) >= MAX_TOOL_CALLS:
        logger.warning(
            "[ROUTER] MAX_TOOL_CALLS (%d) reached — forcing handoff to Analyst", MAX_TOOL_CALLS
        )
        return "analyst"

    if hasattr(last, "tool_calls") and last.tool_calls:
        logger.info("[ROUTER] Researcher tool calls → researcher_tools")
        return "researcher_tools"

    if HANDOFF_SIGNAL in (last.content or ""):
        logger.info("[ROUTER] Handoff signal → ANALYST")
        return "analyst"

    logger.info("[ROUTER] No tool calls / no handoff → END")
    return END


def analyst_router(
    state: MultiAgentState,
) -> Literal["analyst_tools", "__end__"]:
    last = state["messages"][-1]

    if hasattr(last, "tool_calls") and last.tool_calls:
        logger.info("[ROUTER] Analyst tool calls → analyst_tools")
        return "analyst_tools"

    if SESSION_COMPLETE_SIGNAL in (last.content or ""):
        logger.info("[ROUTER] SESSION_COMPLETE → END")

    return END


# ─────────────────────────────────────────────────────────
# BUILD THE GRAPH
# ─────────────────────────────────────────────────────────

def build_multi_agent_graph() -> StateGraph:
    """
    Graph topology:

        START
          │
          ▼
        researcher ──(tool calls)──► researcher_tools
          ▲                               │
          └───────────────────────────────┘
          │
        (HANDOFF_TO_ANALYST)
          │
          ▼
        analyst ──(tool calls)──► analyst_tools
          ▲                            │
          └────────────────────────────┘
          │
        (SESSION_COMPLETE)
          │
          ▼
         END
    """
    graph = StateGraph(MultiAgentState)

    graph.add_node("researcher",       researcher_node)
    graph.add_node("researcher_tools", researcher_tool_node)
    graph.add_node("analyst",          analyst_node)
    graph.add_node("analyst_tools",    analyst_tool_node)

    graph.set_entry_point("researcher")

    graph.add_conditional_edges(
        "researcher",
        researcher_router,
        {"researcher_tools": "researcher_tools", "analyst": "analyst", END: END},
    )
    graph.add_edge("researcher_tools", "researcher")

    graph.add_conditional_edges(
        "analyst",
        analyst_router,
        {"analyst_tools": "analyst_tools", END: END},
    )
    graph.add_edge("analyst_tools", "analyst")

    return graph.compile()


# ─────────────────────────────────────────────────────────
# INTERACTIVE CLI
# ─────────────────────────────────────────────────────────

def run_multi_agent():
    app = build_multi_agent_graph()

    print("\n" + "="*60)
    print("  ADAPTIVE LEARNING COMPANION  [Multi-Agent v3]")
    print("  Agent A: Researcher  |  Agent B: Analyst")
    print("  Memory : Sliding window (%d msgs) + rolling summary" % WINDOW_SIZE)
    print("="*60)
    print("  Type 'quit' to exit.\n")

    logger.info("=" * 55)
    logger.info("  MULTI-AGENT SESSION STARTED  (v3)")
    logger.info("  Researcher : retrieve_content, get_student_progress")
    logger.info("  Analyst    : update_student_progress")
    logger.info("  Memory     : WINDOW_SIZE=%d + rolling summary", WINDOW_SIZE)
    logger.info("=" * 55)

    student_id = input("Enter your student ID (or press Enter for 'student_001'): ").strip()
    if not student_id:
        student_id = "student_001"

    logger.info("Student ID: %s", student_id)

    agent_state: MultiAgentState = {
        "messages":        [],
        "rolling_summary": "",
        "tools_called":    [],
        "tool_call_count": 0,
        "current_agent":   "researcher",
        "handoff_done":    False,
        "student_id":      student_id,
    }

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("\nGoodbye! Keep studying! 📚")
            logger.info("Session ended by user.")
            break
        if not user_input:
            continue

        contextual_input = f"[Student ID: {student_id}] {user_input}"
        logger.info("[USER] %s", contextual_input)

        agent_state["messages"] = agent_state["messages"] + [
            HumanMessage(content=contextual_input)
        ]

        result = app.invoke(agent_state)

        # Show last meaningful AI text (strip internal handoff block from display)
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                display = msg.content
                if "---HANDOFF_TO_ANALYST---" in display:
                    display = display.split("---HANDOFF_TO_ANALYST---")[0].strip()
                if display:
                    label = result.get("current_agent", "agent").upper()
                    print(f"\n[{label}]: {display}")
                    break

        # Log memory stats
        n = len(result["messages"])
        s = len((result.get("rolling_summary") or "").splitlines())
        logger.info("[MEMORY] Live messages: %d | Summary lines: %d", n, s)

        # Carry state forward
        agent_state = {
            "messages":        result["messages"],
            "rolling_summary": result.get("rolling_summary", ""),
            "tools_called":    result.get("tools_called", []),
            "tool_call_count": result.get("tool_call_count", 0),
            "current_agent":   result.get("current_agent", "researcher"),
            "handoff_done":    result.get("handoff_done", False),
            "student_id":      student_id,
        }

        # Reset per-turn tracking when analyst completes a round
        if result.get("current_agent") == "analyst":
            agent_state["current_agent"]   = "researcher"
            agent_state["handoff_done"]    = False
            agent_state["tools_called"]    = []    # ← reset for next question
            agent_state["tool_call_count"] = 0     # ← reset for next question


if __name__ == "__main__":
    run_multi_agent()