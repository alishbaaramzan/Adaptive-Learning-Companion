# Product Requirements Document

## Adaptive Learning Companion

## Problem Statement

**What bottleneck are we solving?**

Students struggle to understand complex learning materials because textbook explanations don't adapt to their current knowledge level. When a concept is confusing, students waste time searching multiple resources, don't know if they truly understand it, and can't identify their specific knowledge gaps.

**Why this needs an agent (not just a chatbot):**
- Must **assess** what student already knows
- Must **check** if prerequisites are mastered
- Must **retrieve** appropriate course materials
- Must **generate** personalized explanations
- Must **create** practice problems
- Must **track** progress over time

This requires **5-7 sequential steps with decision logic** - impossible with a single prompt.

---

## User Personas

### Primary: "Struggling High School/College Student

**Profile:**
- Age: 16-22, studying STEM subjects
- Studies 2-4 hours/day, often gets stuck on specific concepts
- Comfortable with chat interfaces and mobile apps

**Pain Points:**
- Textbook explanations are too abstract
- Doesn't know if she's "getting it" until exam day
- Wastes 40% of study time on ineffective methods

**Goals:**
- Understand *why* she's confused (identify the gap)
- Get explanations that build on what she knows
- Practice until confident

---

## Success Metrics

| Metric | Target | How Measured |
|--------|--------|--------------|
| **Understanding Improvement** | +35% | Pre/post quiz scores |
| **Time to Mastery** | -25% | Time tracking per topic |
| **Engagement Rate** | 60% | Complete multi-turn sessions |

---

## Tool & Data Inventory

### Tools (3 Python functions the agent will call)

**1. `retrieve_content(topic, content_type, difficulty)`**
- Purpose: Fetch course materials, prerequisites, or practice problems
- Example: `retrieve_content("photosynthesis", "explanation", "beginner")`
- Returns: Relevant content from vector database

**2. `get_student_progress(student_id, topic)`**
- Purpose: Check student's mastery level and history
- Example: `get_student_progress("student_123", "photosynthesis")`
- Returns: Mastery score, attempts, last studied date

**3. `update_student_progress(student_id, topic, score)`**
- Purpose: Log performance and track learning
- Example: `update_student_progress("student_123", "photosynthesis", 0.85)`
- Returns: Success confirmation

### Data Sources (what grounds the agent)

**1. Course Knowledge Base (Vector Database)**
- **Format:** ChromaDB with embeddings
- **Contains:**
  - Course materials (textbook chapters, lecture notes)
  - Prerequisite mappings (topic dependencies)
  - Practice problems with solutions
- **Structure:** Each document has metadata (topic, type, difficulty, prerequisites)
- **Access:** Semantic search with filtering
- **Example:** 
  ```
  Topic: "Photosynthesis"
  Type: "explanation" | "prerequisites" | "practice"
  Difficulty: "beginner" | "intermediate" | "advanced"
  ```

**2. Student Progress Database (SQLite)**
- **Format:** Relational database
- **Tables:**
  - `student_progress`: student_id, topic, mastery_score, attempts
  - `study_sessions`: session_id, timestamp, duration
- **Tracks:** Quiz scores, time spent, mastery level per topic
- **Purpose:** Personalize learning path and measure improvement

---

## Agent Workflow Example

**Scenario:** Student asks: *"I don't understand how ATP is produced"*

```
Step 1: [ASSESS] Ask diagnostic question: "Do you know what mitochondria are?"
        Student: "Kind of?"

Step 2: [CHECK_UNDERSTANDING]
        Tool call: get_student_progress("student_123", "mitochondria")
        Result: Mastery = 0.5 (needs review)

Step 3: [RETRIEVE_PREREQUISITES]
        Tool call: retrieve_content("ATP_synthesis", "prerequisites", "beginner")
        Returns: "Requires: mitochondria structure, energy basics"

Step 4: [RETRIEVE_EXPLANATION]
        Tool call: retrieve_content("mitochondria", "explanation", "beginner")
        Returns: Simplified explanation with diagrams

Step 5: [GENERATE_EXPLANATION]
        LLM adapts content: "Think of mitochondria like a power plant..."
        Uses analogy appropriate for beginner level

Step 6: [RETRIEVE_PRACTICE]
        Tool call: retrieve_content("mitochondria", "practice", "beginner")
        Returns: "Label the parts where ATP is produced..."

Step 7: [EVALUATE]
        Student completes problem → Score: 0.8

Step 8: [UPDATE_PROGRESS]
        Tool call: update_student_progress("student_123", "mitochondria", 0.8)
        
Decision: Student ready for ATP synthesis? Yes → proceed
```

**Total: 8 steps with 5 tool calls (using 3 tools)**

---

## Why This is "Agentic"

**Perceive:** Extracts data from 2 sources
- Course Knowledge Base (materials, prerequisites, practice problems)
- Student Progress Database (mastery levels, history)

**Reason:** LangGraph manages conditional logic:
- If prerequisites missing → explain those first
- If concept understood → generate harder problems
- If score < 0.7 → review and retry

**Execute:** Calls 3 Python tools to:
- Query vector database with filters
- Retrieve student history
- Update progress records
- Make decisions based on data

**Not a chatbot:** Requires coordinated multi-step process with tools, not just text generation.
