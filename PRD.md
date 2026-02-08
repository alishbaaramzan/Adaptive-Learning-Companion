## Problem Statement
What bottleneck are we solving?

Students struggle to understand complex learning materials because textbook explanations don't adapt to their current knowledge level. When a concept is confusing, students waste time searching multiple resources, don't know if they truly understand it, and can't identify their specific knowledge gaps.
Why this needs an agent (not just a chatbot):

1. Must assess what student already knows
2. Must check if prerequisites are mastered
3. Must retrieve appropriate course materials
4. Must generate personalized explanations
5. Must create practice problems
6. Must track progress over time

This requires 5-7 sequential steps with decision logic - impossible with a single prompt.

## User Personas

Primary: "Struggling High School/College Student"

### Profile:
- Age: 16-22, studying STEM subjects
- Studies 2-4 hours/day, often gets stuck on specific concepts
- Comfortable with chat interfaces and mobile apps

### Pain Points:

- Textbook explanations are too abstract
- Doesn't know if she's "getting it" until exam day
- Wastes 40% of study time on ineffective methods

### Goals:

- Understand why she's confused (identify the gap)
- Get explanations that build on what she knows
- Practice until confident

## Success Metrics

| Metric                  | Target | How Measured                 |
|-------------------------|----------|------------------------------|
| Understanding Improvement | +35%     | Pre/post quiz scores        |
| Time to Mastery          | -25%     | Time tracking per topic     |
| Engagement Rate          | 60%      | Complete multi-turn sessions |

## Tool & Data Inventory

### Tools (4 Python functions the agent will call)

**1. `retrieve_learning_material(topic, difficulty)`**
- Purpose: Fetch relevant course content
- Example: `retrieve_learning_material("photosynthesis", "beginner")`
- Returns: Top 3 text chunks from vector database

**2. `check_prerequisites(topic, student_id)`**
- Purpose: Verify prerequisite knowledge mastery
- Example: `check_prerequisites("derivatives", "student_123")`
- Returns: List of required topics + student's mastery status

**3. `generate_practice_problem(topic, difficulty)`**
- Purpose: Create practice questions with solutions
- Example: `generate_practice_problem("quadratic_equations", difficulty=2)`
- Returns: Problem + worked solution + explanation

**4. `update_student_progress(student_id, topic, score)`**
- Purpose: Log performance and track learning
- Example: `update_student_progress("student_123", "enzymes", 0.85)`
- Returns: Success confirmation

### Data Sources (what grounds the agent)

**1. Course Material Database (Vector Store)**
- Format: PDF textbooks, lecture notes
- Structure: Organized by Subject → Topic → Subtopic
- Access: ChromaDB with semantic search
- Example content: "Biology → Cell Biology → Mitochondria"

**2. Prerequisite Knowledge Graph**
- Format: JSON mapping topic dependencies
- Example: `{"derivatives": ["limits", "algebra", "functions"]}`
- Purpose: Identify what student must know first

**3. Practice Problem Bank**
- Format: JSON with problems, solutions, explanations
- Structure: Categorized by topic and difficulty (1-5)
- Contains: 10 problems per topic for testing

**4. Student Progress Database (SQL)**
- Format: SQLite database
- Tracks: Quiz scores, time spent, mastery level per topic
- Purpose: Personalize learning path and measure improvement

---

## Agent Workflow Example

**Scenario:** Student asks: *"I don't understand how ATP is produced"*

```
Step 1: [ASSESS] Ask diagnostic question: "Do you know what mitochondria are?"
        Student: "Kind of?"

Step 2: [CHECK_PREREQUISITES] 
        Tool call: check_prerequisites("ATP_synthesis", "student_123")
        Result: ✓ cell_structure (0.9), ✗ energy_basics (0.4)

Step 3: [RETRIEVE_MATERIAL]
        Tool call: retrieve_learning_material("energy_basics", "beginner")
        Returns: Simple explanation of chemical energy

Step 4: [EXPLAIN]
        LLM generates: "Think of ATP like a rechargeable battery..."
        Uses analogy appropriate for beginner level

Step 5: [PRACTICE]
        Tool call: generate_practice_problem("energy_basics", 2)
        Returns: "Label the parts where energy is stored..."

Step 6: [EVALUATE]
        Student completes problem → Score: 0.8

Step 7: [UPDATE_PROGRESS]
        Tool call: update_student_progress("student_123", "energy_basics", 0.8)
        
Decision: Student ready for ATP synthesis? Yes → proceed
```

**Total: 7 steps with 4 tool calls**

