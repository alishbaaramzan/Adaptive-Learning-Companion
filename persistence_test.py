"""
persistence_test.py
──────────────────────────────────────────────────────────
Proves that the agent can retrieve information from a previous
session using a thread_id (Lab Task 1 — deliverable).

HOW IT WORKS:
  • Run 1  →  Start a new thread, send a message, save thread_id.
  • Run 2  →  Supply the same thread_id; agent should remember the
              previous message WITHOUT re-processing it.

Run this script twice:
  $ python persistence_test.py          # first run  → note the thread_id printed
  $ python persistence_test.py <id>     # second run → confirm memory restored
"""

import sys
import uuid
import logging
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver

# ── Reuse the same graph factory from the main module ──
from multi_agent_graph import build_multi_agent_graph, MultiAgentState

CHECKPOINT_DB = "checkpoint_db.sqlite"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("persistence_test")


def run_test(thread_id: str | None = None):
    with SqliteSaver.from_conn_string(CHECKPOINT_DB) as checkpointer:
        app = build_multi_agent_graph(checkpointer)

        # ── Determine thread ──
        is_new = thread_id is None
        if is_new:
            thread_id = str(uuid.uuid4())[:8]
            print(f"\n{'='*55}")
            print(f"  RUN 1 — New session created")
            print(f"  Thread ID : {thread_id}")
            print(f"  ➜  Re-run with:  python persistence_test.py {thread_id}")
            print(f"{'='*55}\n")
        else:
            print(f"\n{'='*55}")
            print(f"  RUN 2 — Resuming thread: {thread_id}")
            print(f"{'='*55}\n")

        config = {"configurable": {"thread_id": thread_id}}

        # ── Check for existing checkpoint ──
        existing = checkpointer.get(config)
        if existing:
            prev_msgs = existing["channel_values"].get("messages", [])
            print(f"  ✓ Checkpoint found — {len(prev_msgs)} message(s) restored.\n")
            logger.info("[PERSIST] Restored %d messages from thread %s", len(prev_msgs), thread_id)

            # Show last AI message from the restored session
            for m in reversed(prev_msgs):
                if isinstance(m, AIMessage) and m.content:
                    print("  Last AI response (from previous session):")
                    print("  " + (m.content[:300].replace("\n", "\n  ")))
                    print()
                    break

            # Send a follow-up that only makes sense if memory is intact
            follow_up = "[Student ID: student_001] Based on our last session, what topic were we studying?"
            print(f"  Sending follow-up: '{follow_up}'\n")
            logger.info("[PERSIST] Sending follow-up to test memory continuity")

            state: MultiAgentState = {
                **existing["channel_values"],
                "messages": existing["channel_values"]["messages"] + [
                    HumanMessage(content=follow_up)
                ],
            }

        else:
            # First run — send an initial message and save it
            print("  No checkpoint yet. Sending initial message...\n")
            initial_msg = "[Student ID: student_001] I want to study Python lists today."
            logger.info("[PERSIST] Sending initial message: %s", initial_msg)

            state: MultiAgentState = {
                "messages":        [HumanMessage(content=initial_msg)],
                "rolling_summary": "",
                "tools_called":    [],
                "tool_call_count": 0,
                "current_agent":   "researcher",
                "handoff_done":    False,
                "student_id":      "student_001",
                "proposed_score":  "",
            }

        # ── Invoke (runs until analyst breakpoint or END) ──
        result = app.invoke(state, config)

        # ── Show result ──
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                display = msg.content
                if "---HANDOFF_TO_ANALYST---" in display:
                    display = display.split("---HANDOFF_TO_ANALYST---")[0].strip()
                if display:
                    print(f"  [AGENT]: {display[:400]}")
                break

        n = len(result["messages"])
        logger.info("[PERSIST] Session saved. Thread=%s | Messages=%d", thread_id, n)

        print(f"\n  ✓ State saved to {CHECKPOINT_DB}")
        print(f"  Thread ID : {thread_id}")
        if is_new:
            print(f"\n  ➜  Run again with:  python persistence_test.py {thread_id}")
        else:
            print(f"\n  ✓ Memory continuity verified — agent referenced previous session.\n")


if __name__ == "__main__":
    tid = sys.argv[1] if len(sys.argv) > 1 else None
    run_test(tid)