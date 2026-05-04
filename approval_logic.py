"""
approval_logic.py
──────────────────────────────────────────────────────────
Lab deliverable: LangGraph HITL configuration.

Shows exactly how interrupt_before is wired to the "analyst" node
and how a human can APPROVE, CANCEL, or EDIT the agent's state
before execution resumes.

This file is self-contained and runnable as a demo:
  $ python approval_logic.py

It uses a minimal stub graph so you can see the HITL mechanics
without needing API keys or the full tool stack. 

NOTE: THE ACTUAL LOGIC IS IMPLEMENTED IN multi_agent_graph.py
"""

from __future__ import annotations

import logging
from typing import Annotated, Literal
from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

CHECKPOINT_DB = "checkpoint_db.sqlite"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("approval_logic")


# ─────────────────────────────────────────────────────────
# MINIMAL STATE
# ─────────────────────────────────────────────────────────

class DemoState(TypedDict):
    messages:       Annotated[list, add_messages]
    current_agent:  str
    proposed_score: str   # ← the value a human can edit before analyst commits it
    student_id:     str


# ─────────────────────────────────────────────────────────
# STUB NODES  (replace with real LLM calls in production)
# ─────────────────────────────────────────────────────────

def researcher_node(state: DemoState) -> dict:
    """Agent A: produces a handoff with a proposed score."""
    logger.info("[RESEARCHER] Generating handoff...")
    handoff = (
        "---HANDOFF_TO_ANALYST---\n"
        "student_id: student_001\n"
        "topic: Python Lists\n"
        "student_answer: 'append() adds an item to the end'\n"
        "proposed_score: 90\n"
        "---END_HANDOFF---"
    )
    return {
        "messages":       [AIMessage(content=f"Great answer! {handoff}")],
        "current_agent":  "researcher",
        "proposed_score": "90",
    }


def analyst_node(state: DemoState) -> dict:
    """
    Agent B: commits the (possibly human-edited) score.
    This node is preceded by a HITL pause — it only runs after human approval.
    """
    score = state.get("proposed_score", "unknown")
    logger.info("[ANALYST] Committing score: %s", score)
    # In production: calls update_student_progress(student_id, score)
    return {
        "messages":      [AIMessage(content=f"Score {score} recorded. SESSION_COMPLETE")],
        "current_agent": "analyst",
    }


# ─────────────────────────────────────────────────────────
# ROUTERS
# ─────────────────────────────────────────────────────────

def researcher_router(state: DemoState) -> Literal["analyst", "__end__"]:
    last = state["messages"][-1]
    if "---HANDOFF_TO_ANALYST---" in (getattr(last, "content", "") or ""):
        return "analyst"
    return END


# ─────────────────────────────────────────────────────────
# GRAPH COMPILATION
# Key: interrupt_before=["analyst"] — this is the HITL safety breakpoint.
# Execution pauses here; human must explicitly resume via app.invoke(None, config).
# ─────────────────────────────────────────────────────────

def build_demo_graph(checkpointer):
    graph = StateGraph(DemoState)

    graph.add_node("researcher", researcher_node)
    graph.add_node("analyst",    analyst_node)      # ← HIGH-RISK node

    graph.set_entry_point("researcher")

    graph.add_conditional_edges(
        "researcher", researcher_router,
        {"analyst": "analyst", END: END},
    )
    graph.add_edge("analyst", END)

    return graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["analyst"],   # ◄── HITL BREAKPOINT (Task 2)
    )


# ─────────────────────────────────────────────────────────
# HITL APPROVAL LOOP  (Tasks 2 + 3)
# ─────────────────────────────────────────────────────────

def hitl_approval_loop(app, config: dict) -> None:
    """
    After the graph pauses at the analyst breakpoint:
      • Display the current state to the human
      • Wait for APPROVE / CANCEL / EDIT
      • Optionally mutate state (Task 3: state editing)
      • Resume or abort

    approve       → app.invoke(None, config)          resumes from checkpoint
    cancel        → state reset, analyst skipped
    edit:<score>  → app.update_state() patches score, then resumes
    """
    snapshot     = app.get_state(config)
    current_vals = snapshot.values

    print("\n" + "="*60)
    print("  ⚠  SAFETY PAUSE — Human Approval Required")
    print("="*60)
    print(f"  Node about to execute : analyst  (writes to DB)")
    print(f"  Student ID            : {current_vals.get('student_id', '?')}")
    print(f"  Proposed score        : {current_vals.get('proposed_score', '?')}")
    print()
    print("  Commands:")
    print("    approve          → execute analyst as-is")
    print("    cancel           → skip analyst, end session")
    print("    edit:<score>     → change score then execute  [Task 3]")
    print("="*60)

    while True:
        cmd = input("  Decision: ").strip().lower()

        # ── APPROVE ────────────────────────────────────────
        if cmd == "approve":
            logger.info("[HITL] ✓ Approved — resuming analyst")
            result = app.invoke(None, config)   # None = resume from checkpoint
            for msg in reversed(result["messages"]):
                if isinstance(msg, AIMessage) and msg.content:
                    print(f"\n  [ANALYST]: {msg.content}\n")
                    break
            return

        # ── CANCEL ─────────────────────────────────────────
        elif cmd == "cancel":
            logger.info("[HITL] ✗ Cancelled — analyst skipped")
            print("  → Analyst action cancelled. No score written.\n")
            return

        # ── EDIT (Task 3: state editing) ───────────────────
        elif cmd.startswith("edit:"):
            new_score = cmd.split(":", 1)[1].strip()
            if not new_score.isdigit():
                print("  Score must be a number. Try again.")
                continue

            logger.info("[HITL] ✎ Human edited score: %s → %s",
                        current_vals.get("proposed_score"), new_score)

            # ── THIS IS THE STATE EDIT ──
            # update_state patches the checkpoint in-place before resuming.
            # The analyst node will see proposed_score = new_score.
            app.update_state(config, {"proposed_score": new_score})
            print(f"  → Score updated to {new_score}. Resuming analyst...\n")

            result = app.invoke(None, config)   # resume with edited state
            for msg in reversed(result["messages"]):
                if isinstance(msg, AIMessage) and msg.content:
                    print(f"  [ANALYST]: {msg.content}\n")
                    break
            return

        else:
            print("  Unknown command. Type 'approve', 'cancel', or 'edit:<score>'.")


# ─────────────────────────────────────────────────────────
# DEMO RUNNER
# ─────────────────────────────────────────────────────────

def run_demo():
    print("\n" + "="*60)
    print("  HITL Approval Logic Demo")
    print("  Checkpoint DB:", CHECKPOINT_DB)
    print("="*60 + "\n")

    import uuid
    thread_id = str(uuid.uuid4())[:8]
    config    = {"configurable": {"thread_id": thread_id}}

    with SqliteSaver.from_conn_string(CHECKPOINT_DB) as checkpointer:
        app = build_demo_graph(checkpointer)

        initial_state: DemoState = {
            "messages":       [HumanMessage(content="append() adds items to the end of a list")],
            "current_agent":  "researcher",
            "proposed_score": "",
            "student_id":     "student_001",
        }

        print("  → Running graph until HITL breakpoint...\n")
        app.invoke(initial_state, config)

        # Graph paused before analyst — now enter the approval loop
        hitl_approval_loop(app, config)

    print("  Demo complete. State saved to", CHECKPOINT_DB)


if __name__ == "__main__":
    run_demo()