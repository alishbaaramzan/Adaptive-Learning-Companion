"""
Lab 3: graph.py
Adaptive Learning Companion — LangGraph ReAct Agent
─────────────────────────────────────────────────────
Implements a ReAct (Reason + Act) loop using LangGraph:
  1. Agent Node  — LLM reasons and decides which tool (if any) to call
  2. Tool Node   — Executes the chosen tool and returns the result
  3. Router      — If tool calls exist → loop to Tool Node; else → END

State flows:  START → agent → [tools → agent]* → END

"""

import os
from typing import Annotated
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from tools import retrieve_content, get_student_progress, update_student_progress

load_dotenv()

# ─────────────────────────────────────────────────────────
# GRAPH STATE
# ─────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """
    Shared state passed between every node in the graph.
    
    `messages` uses the `add_messages` reducer so each node
    appends to the history rather than overwriting it.
    This gives the LLM full conversation + tool-call context.
    """
    messages: Annotated[list, add_messages]


# ─────────────────────────────────────────────────────────
# LLM SETUP
# ─────────────────────────────────────────────────────────

TOOLS = [retrieve_content, get_student_progress, update_student_progress]

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.3,
    api_key=os.getenv("OPENAI_API_KEY")
).bind_tools(TOOLS)

SYSTEM_PROMPT = """You are an Adaptive Learning Companion — a patient, encouraging AI tutor.

Your job is to help students master concepts step-by-step using this workflow:

1. ASSESS   – Find out what the student already knows (ask a diagnostic question).
2. CHECK    – Call get_student_progress to see their mastery score for this topic.
3. PREREQS  – If mastery < 0.7, call retrieve_content(..., "prerequisites", ...) first.
4. EXPLAIN  – Call retrieve_content(..., "explanation", ...) to ground your explanation.
5. PRACTICE – Call retrieve_content(..., "practice", ...) to give them a problem.
6. EVALUATE – Ask the student to answer, then judge their response (score 0.0–1.0).
7. UPDATE   – Call update_student_progress with the score.
8. DECIDE   – If mastery ≥ 0.7, move to next concept. Else repeat with harder focus.

Rules:
- Always ground explanations in retrieved content — never rely solely on your memory.
- Use analogies and examples appropriate to the student's difficulty level.
- Be encouraging. Normalise mistakes as part of learning.
- Only call update_student_progress AFTER the student has answered a question.
"""


# ─────────────────────────────────────────────────────────
# NODE 1: AGENT
# ─────────────────────────────────────────────────────────

def agent_node(state: AgentState) -> AgentState:
    """
    The brain of the agent.
    Takes the current message history, calls the LLM (with tools bound),
    and returns the LLM's response (which may contain tool call requests).
    """
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


# ─────────────────────────────────────────────────────────
# NODE 2: TOOL NODE
# ─────────────────────────────────────────────────────────

# LangGraph's built-in ToolNode automatically:
#   - Reads tool_calls from the last AIMessage
#   - Executes the matching tool function
#   - Appends ToolMessage results back to state
tool_node = ToolNode(tools=TOOLS)


# ─────────────────────────────────────────────────────────
# CONDITIONAL ROUTER (the "logic gate")
# ─────────────────────────────────────────────────────────

def router(state: AgentState) -> str:
    """
    Inspect the last message from the agent.
    - If it contains tool_calls → route to 'tools' node (keep looping)
    - If no tool_calls         → route to END (final answer ready)
    """
    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    return END


# ─────────────────────────────────────────────────────────
# BUILD THE GRAPH
# ─────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Assemble and compile the LangGraph StateGraph.

    Graph structure:
        START
          │
          ▼
        agent ──(has tool calls?)──► tools
          ▲                            │
          └────────────────────────────┘
          │
        (no tool calls)
          │
          ▼
         END
    """
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    # Entry point
    graph.set_entry_point("agent")

    # Conditional edge from agent: call router to decide next step
    graph.add_conditional_edges(
        source="agent",
        path=router,
        path_map={
            "tools": "tools",
            END: END
        }
    )

    # After tools run → always return to agent for next reasoning step
    graph.add_edge("tools", "agent")

    return graph.compile()


# ─────────────────────────────────────────────────────────
# RUN (interactive CLI loop)
# ─────────────────────────────────────────────────────────

def run_agent():
    """Interactive CLI to chat with the learning agent."""
    app = build_graph()

    print("\n" + "="*60)
    print("  ADAPTIVE LEARNING COMPANION")
    print("  Powered by LangGraph ReAct Agent")
    print("="*60)
    print("  Type 'quit' to exit.\n")

    student_id = input("Enter your student ID (or press Enter for 'student_001'): ").strip()
    if not student_id:
        student_id = "student_001"

    conversation_history = []

    while True:
        user_input = input(f"\nYou: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("\nGoodbye! Keep studying! 📚")
            break
        if not user_input:
            continue

        # Inject student context into message
        contextual_input = f"[Student ID: {student_id}] {user_input}"
        conversation_history.append(HumanMessage(content=contextual_input))

        # Run one full ReAct cycle
        result = app.invoke({"messages": conversation_history})

        # Extract final assistant message
        final_message = result["messages"][-1]
        print(f"\nAgent: {final_message.content}")

        # Update history for multi-turn memory
        conversation_history = result["messages"]


if __name__ == "__main__":
    run_agent()