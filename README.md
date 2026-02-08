# Adaptive-Learning-Companion

## Project Overview
An AI agent that helps students understand complex learning materials by adapting explanations to their knowledge level, generating practice problems, and tracking progress.

**Why it's agentic**: Requires 5-7 sequential steps with conditional logic, tool orchestration, and multi-source data integration.

## Deliverables

1. PRD.md
Complete product requirements document containing:
- Problem statement
- User personas
- Success metrics
- Tool & data inventory
- Agent workflow example

2. Architecture_Diagram.png
System architecture showing:
- LangGraph agent states (7 steps)
- 4 Python tools
- 4 data sources
- Data flow connections


**Why This is Agentic (Not a Chatbot)**:

- Perceive: Extracts from 4 different data sources
- Reason: LangGraph manages conditional multi-step logic
- Execute: Calls Python functions to query DBs and update records
- Cannot be solved with a single prompt - requires coordinated workflow.
