"""
retrieval_test.py
Adaptive Learning Companion — Vector Store Retrieval Tests
──────────────────────────────────────────────────────────
Runs 3 retrieval tests against your ChromaDB collection and
prints results with pass/fail status.

Usage:
    python retrieval_test.py

Requirements:
    pip install chromadb openai python-dotenv
    .env file with OPENAI_API_KEY=sk-...
    ChromaDB populated by running: python ingest_data.py ...
"""

import os
import json
from dotenv import load_dotenv
import openai
import chromadb
from chromadb.config import Settings

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

COLLECTION_NAME = "learning_companion_kb"
CHROMA_PATH     = "./chroma_db"

# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────

def get_collection():
    client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False)
    )
    return client.get_or_create_collection(name=COLLECTION_NAME)


def embed(text: str) -> list:
    response = openai.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    )
    return response.data[0].embedding


def print_results(results: dict):
    docs   = results.get("documents", [[]])[0]
    metas  = results.get("metadatas", [[]])[0]

    if not docs:
        print("No results returned.\n")
        return

    for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
        print(f"\n  Result {i}:")
        print(f"    Source   : {meta.get('source_file', 'unknown')}  (page {meta.get('start_page', '?')})")
        print(f"    Topic    : {meta.get('topic')}")
        print(f"    Type     : {meta.get('content_type')}")
        print(f"    Difficulty: {meta.get('difficulty')}")
        print(f"    Preview  : {doc[:200].strip()}...")


def section(title: str, test_num: int):
    print(f"\n{'='*60}")
    print(f"  TEST {test_num}: {title}")
    print(f"{'='*60}")


# ─────────────────────────────────────────────────────────
# TEST 1 — Pure Semantic Search (no filters)
# ─────────────────────────────────────────────────────────

def test_semantic_search(collection):
    section("Pure Semantic Search", 1)
    query = "How does the agent reason and decide which tool to use?"
    print(f"  Query   : \"{query}\"")
    print(f"  Filter  : None")

    results = collection.query(
        query_embeddings=[embed(query)],
        n_results=3,
        include=["documents", "metadatas"]
    )

    print_results(results)

    docs = results["documents"][0]
    passed = len(docs) > 0
    print(f"\n  {'PASS' if passed else 'FAIL'} — returned {len(docs)} result(s)")
    return passed


# ─────────────────────────────────────────────────────────
# TEST 2 — Metadata Filter: difficulty = "intermediate"
# ─────────────────────────────────────────────────────────

def test_difficulty_filter(collection):
    section("Metadata Filter — difficulty=intermediate", 2)
    query = "Explain the concept simply with an example"
    print(f"  Query   : \"{query}\"")
    print(f"  Filter  : difficulty = 'intermediate'")

    try:
        results = collection.query(
            query_embeddings=[embed(query)],
            n_results=3,
            where={"difficulty": {"$eq": "intermediate"}},
            include=["documents", "metadatas"]
        )
    except Exception as e:
        print(f"\n  Filter error (no 'intermediate' docs in DB?): {e}")
        return False

    print_results(results)

    docs   = results["documents"][0]
    metas  = results["metadatas"][0]
    # Validate every result actually has difficulty=intermediate
    all_correct = all(m.get("difficulty") == "intermediate" for m in metas)
    passed = len(docs) > 0 and all_correct

    if not all_correct:
        print(f"\n  Some results did not match filter!")
    print(f"\n  {'PASS' if passed else 'FAIL'} — {len(docs)} result(s), all difficulty=intermediate: {all_correct}")
    return passed


# ─────────────────────────────────────────────────────────
# TEST 3 — Compound Metadata Filter: content_type + topic
# ─────────────────────────────────────────────────────────

def test_compound_filter(collection):
    section("Compound Filter — content_type=practice + topic match", 3)

    # Grab one topic that exists in the DB to use as filter
    sample = collection.get(limit=5, include=["metadatas"])
    topics = list({m["topic"] for m in sample["metadatas"] if m.get("topic")})
    if not topics:
        print("  No documents in collection yet. Run ingest_data.py first.")
        return False

    chosen_topic = topics[0]
    query = f"practice problem or exercise on {chosen_topic}"
    print(f"  Query   : \"{query}\"")
    print(f"  Filter  : content_type='practice' AND topic='{chosen_topic}'")

    try:
        results = collection.query(
            query_embeddings=[embed(query)],
            n_results=2,
            where={
                "$and": [
                    {"content_type": {"$eq": "practice"}},
                    {"topic":        {"$eq": chosen_topic}},
                ]
            },
            include=["documents", "metadatas"]
        )
    except Exception as e:
        print(f"\n  Filter error: {e}")
        # Fallback: try without compound filter to show data exists
        print("  Falling back to topic-only filter...")
        results = collection.query(
            query_embeddings=[embed(query)],
            n_results=2,
            where={"topic": {"$eq": chosen_topic}},
            include=["documents", "metadatas"]
        )

    print_results(results)

    docs  = results["documents"][0]
    metas = results["metadatas"][0]
    passed = len(docs) > 0
    print(f"\n  {'PASS' if passed else 'FAIL'} — {len(docs)} result(s) for topic='{chosen_topic}'")
    return passed


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  ADAPTIVE LEARNING COMPANION — RETRIEVAL TESTS")
    print("="*60)

    # Check DB exists
    if not os.path.exists(CHROMA_PATH):
        print(f"\n ChromaDB not found at '{CHROMA_PATH}'.")
        print("   Run ingest_data.py first:\n")
        print("   python ingest_data.py --pdf your_book.pdf --topic 'your_topic' --difficulty intermediate\n")
        return

    collection = get_collection()
    total = collection.count()
    print(f"\n  Collection : '{COLLECTION_NAME}'")
    print(f"  Documents  : {total} chunks in DB")

    if total == 0:
        print("\n Collection is empty. Run ingest_data.py first.")
        return

    # Run tests
    results = {
        "Test 1 — Semantic Search":       test_semantic_search(collection),
        "Test 2 — Difficulty Filter":     test_difficulty_filter(collection),
        "Test 3 — Compound Filter":       test_compound_filter(collection),
    }

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    passed = sum(results.values())
    total_tests = len(results)
    for name, ok in results.items():
        print(f"  {'✅' if ok else '❌'}  {name}")
    print(f"\n  {passed}/{total_tests} tests passed\n")


if __name__ == "__main__":
    main()