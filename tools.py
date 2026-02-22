"""
Lab 3: tools.py
Adaptive Learning Companion — Agent Tool Definitions
─────────────────────────────────────────────────────
Three LangChain tools with Pydantic input validation and descriptive docstrings.
The LLM reads the docstrings to decide WHEN and HOW to call each tool.

Tools:
    1. retrieve_content        — queries ChromaDB vector store (the "Grounding" tool)
    2. get_student_progress    — reads student mastery from SQLite
    3. update_student_progress — writes quiz score back to SQLite

Install:
    pip install langchain langchain-core langchain-openai chromadb openai pydantic
"""

import os
import sqlite3
from datetime import datetime
from typing import Literal

import chromadb
import openai
from chromadb.config import Settings
from dotenv import load_dotenv
from langchain_core.tools import tool
from pydantic import BaseModel, Field

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ─────────────────────────────────────────────────────────
# SHARED: ChromaDB client (re-used across calls)
# ─────────────────────────────────────────────────────────

def _get_collection(collection_name: str = "learning_companion_kb"):
    client = chromadb.PersistentClient(
        path="./chroma_db",
        settings=Settings(anonymized_telemetry=False)
    )
    return client.get_or_create_collection(name=collection_name)


def _embed(text: str) -> list[float]:
    """Generate a single embedding via OpenAI for querying ChromaDB."""
    response = openai.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    )
    return response.data[0].embedding


# ─────────────────────────────────────────────────────────
# SHARED: SQLite setup
# ─────────────────────────────────────────────────────────

DB_PATH = "./student_progress.db"

def _init_db():
    """Create SQLite tables if they don't exist yet."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS student_progress (
            student_id   TEXT NOT NULL,
            topic        TEXT NOT NULL,
            mastery_score REAL DEFAULT 0.0,
            attempts     INTEGER DEFAULT 0,
            last_studied TEXT,
            PRIMARY KEY (student_id, topic)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS study_sessions (
            session_id   TEXT PRIMARY KEY,
            student_id   TEXT,
            topic        TEXT,
            score        REAL,
            timestamp    TEXT
        )
    """)
    conn.commit()
    conn.close()

_init_db()


# ═════════════════════════════════════════════════════════
# TOOL 1: retrieve_content
# ═════════════════════════════════════════════════════════

class RetrieveContentInput(BaseModel):
    topic: str = Field(
        description="The subject topic to retrieve content about. E.g. 'neural_networks', 'photosynthesis'."
    )
    content_type: Literal["explanation", "prerequisites", "practice"] = Field(
        description=(
            "Type of content to fetch: "
            "'explanation' for concept explanations, "
            "'prerequisites' for required background knowledge, "
            "'practice' for exercise problems."
        )
    )
    difficulty: Literal["beginner", "intermediate", "advanced"] = Field(
        description="The student's current difficulty level so we retrieve appropriately pitched content."
    )
    n_results: int = Field(
        default=3,
        description="Number of relevant chunks to return. Default is 3."
    )


@tool(args_schema=RetrieveContentInput)
def retrieve_content(topic: str,
                     content_type: str,
                     difficulty: str,
                     n_results: int = 3) -> str:
    """
    Retrieve relevant course material from the vector knowledge base.

    Use this tool whenever you need:
    - An explanation of a concept the student is struggling with
    - The prerequisite knowledge required before teaching a topic
    - Practice problems to test the student's understanding

    The tool performs a semantic search filtered by topic, content_type, and difficulty,
    so results are always appropriate for the student's current level.

    Returns the top matching chunks of course content as a single string.
    """
    collection = _get_collection()
    query_text = f"{topic} {content_type} {difficulty}"
    query_embedding = _embed(query_text)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where={
            "$and": [
                {"topic":        {"$eq": topic.lower().replace(" ", "_")}},
                {"content_type": {"$eq": content_type}},
                {"difficulty":   {"$eq": difficulty}},
            ]
        },
        include=["documents", "metadatas", "distances"]
    )

    if not results["documents"] or not results["documents"][0]:
        # Fallback: semantic search without strict metadata filter
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas"]
        )

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    if not docs:
        return f"No content found for topic='{topic}', type='{content_type}', difficulty='{difficulty}'."

    output_parts = []
    for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
        source = meta.get("source_file", "unknown")
        page   = meta.get("start_page", "?")
        output_parts.append(
            f"[Result {i} | Source: {source}, Page {page}]\n{doc}"
        )

    return "\n\n---\n\n".join(output_parts)


# ═════════════════════════════════════════════════════════
# TOOL 2: get_student_progress
# ═════════════════════════════════════════════════════════

class GetStudentProgressInput(BaseModel):
    student_id: str = Field(
        description="Unique identifier for the student. E.g. 'student_123'."
    )
    topic: str = Field(
        description="The subject topic to check progress on. E.g. 'neural_networks'."
    )


@tool(args_schema=GetStudentProgressInput)
def get_student_progress(student_id: str, topic: str) -> str:
    """
    Check a student's current mastery level and learning history for a given topic.

    Use this tool to:
    - Determine if the student already knows a concept before teaching it
    - Decide whether prerequisites need review (mastery_score < 0.7 = needs review)
    - Personalise difficulty: if mastery is high, use harder explanations/problems

    Returns mastery score (0.0–1.0), number of attempts, and last study date.
    A mastery score of 0.0 means the student has not studied this topic yet.
    """
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT mastery_score, attempts, last_studied FROM student_progress "
        "WHERE student_id = ? AND topic = ?",
        (student_id, topic.lower().replace(" ", "_"))
    ).fetchone()
    conn.close()

    if row is None:
        return (
            f"No progress record found for student '{student_id}' on topic '{topic}'. "
            f"This is likely a new topic for them. Mastery: 0.0, Attempts: 0."
        )

    mastery, attempts, last_studied = row
    status = "needs review" if mastery < 0.7 else "proficient"

    return (
        f"Student '{student_id}' | Topic: '{topic}'\n"
        f"  Mastery Score : {mastery:.2f} ({status})\n"
        f"  Attempts      : {attempts}\n"
        f"  Last Studied  : {last_studied or 'never'}"
    )


# ═════════════════════════════════════════════════════════
# TOOL 3: update_student_progress
# ═════════════════════════════════════════════════════════

class UpdateStudentProgressInput(BaseModel):
    student_id: str = Field(
        description="Unique identifier for the student."
    )
    topic: str = Field(
        description="The subject topic the student just practiced."
    )
    score: float = Field(
        ge=0.0, le=1.0,
        description=(
            "The student's quiz/practice score as a decimal between 0.0 and 1.0. "
            "E.g. 0.8 means 80% correct."
        )
    )


@tool(args_schema=UpdateStudentProgressInput)
def update_student_progress(student_id: str, topic: str, score: float) -> str:
    """
    Record a student's latest quiz or practice score and update their mastery level.

    Use this tool AFTER the student has completed a practice problem or quiz.
    The new mastery score is a running average of all attempts, giving a fair
    picture of long-term understanding rather than just the last attempt.

    Call this tool to close the learning loop so progress is tracked over sessions.
    Returns a confirmation with the updated mastery score.
    """
    topic_key = topic.lower().replace(" ", "_")
    now = datetime.now().isoformat()
    session_id = f"{student_id}_{topic_key}_{now}"

    conn = sqlite3.connect(DB_PATH)

    # Upsert: running average of mastery
    existing = conn.execute(
        "SELECT mastery_score, attempts FROM student_progress WHERE student_id=? AND topic=?",
        (student_id, topic_key)
    ).fetchone()

    if existing:
        old_mastery, attempts = existing
        new_attempts = attempts + 1
        new_mastery = ((old_mastery * attempts) + score) / new_attempts
        conn.execute(
            "UPDATE student_progress SET mastery_score=?, attempts=?, last_studied=? "
            "WHERE student_id=? AND topic=?",
            (new_mastery, new_attempts, now, student_id, topic_key)
        )
    else:
        new_mastery = score
        new_attempts = 1
        conn.execute(
            "INSERT INTO student_progress (student_id, topic, mastery_score, attempts, last_studied) "
            "VALUES (?, ?, ?, ?, ?)",
            (student_id, topic_key, score, 1, now)
        )

    # Log session
    conn.execute(
        "INSERT INTO study_sessions (session_id, student_id, topic, score, timestamp) VALUES (?,?,?,?,?)",
        (session_id, student_id, topic_key, score, now)
    )
    conn.commit()
    conn.close()

    status = "✓ Mastery achieved!" if new_mastery >= 0.7 else "⟳ Needs more practice."
    return (
        f"Progress updated for '{student_id}' on '{topic}'.\n"
        f"  Latest Score  : {score:.2f}\n"
        f"  New Mastery   : {new_mastery:.2f} (after {new_attempts} attempt(s))\n"
        f"  Status        : {status}"
    )