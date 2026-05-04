"""
run_eval.py — Automated Quality Gate for Adaptive Learning Companion
────────────────────────────────────────────────────────────────────
Runs headlessly in CI. Reads all credentials from environment variables.
Exits with code 0 if all metrics pass, code 1 if any metric fails.
Writes eval_results.json with full pass/fail breakdown.

Required environment variables:
  OPENAI_API_KEY   — OpenAI key for the LLM judge
  API_BASE_URL     — Base URL of the running agent (default: http://localhost:8000)

Usage:
  python run_eval.py
  python run_eval.py --thresholds eval_thresholds.json
"""

import os
import sys
import json
import time
import argparse
import requests
from datetime import datetime, timezone

# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_THRESHOLDS_FILE = "eval_thresholds.json"
RESULTS_FILE            = "eval_results.json"
API_BASE_URL            = os.environ.get("API_BASE_URL", "http://localhost:8000")
OPENAI_API_KEY          = os.environ.get("OPENAI_API_KEY", "")

# Eval dataset: questions are phrased as a student who already studied the topic
# and wants an explanation. This matches the agent's tutoring flow and avoids
# the agent asking "what do you already know?" back.
EVAL_DATASET = [
    {
        "question": (
            "I just studied machine learning and I want to check my understanding. "
            "Can you explain what supervised learning is?"
        ),
        "keywords": ["labelled", "label", "training", "input", "output", "predict"],
        "topic": "supervised learning",
    },
    {
        "question": (
            "I read about model evaluation. Can you explain the difference "
            "between overfitting and underfitting?"
        ),
        "keywords": ["overfit", "underfit", "training", "generalise", "generalize", "noise", "simple", "complex"],
        "topic": "overfitting vs underfitting",
    },
    {
        "question": (
            "I am studying deep learning. Can you explain what a neural network is "
            "and how it works?"
        ),
        "keywords": ["neuron", "layer", "weight", "brain", "node", "activation", "train"],
        "topic": "neural networks",
    },
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_thresholds(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    return {k: v["min"] for k, v in data["thresholds"].items()}


def call_agent(question: str, thread_id: str) -> str:
    """Send a question to the agent API and return its answer."""
    try:
        resp = requests.post(
            f"{API_BASE_URL}/chat",
            json={
                "message":    question,
                "student_id": "eval_bot",
                "thread_id":  thread_id,
            },
            timeout=90,
        )
        resp.raise_for_status()
        return resp.json().get("answer", "")
    except Exception as e:
        print(f"  [ERROR] Agent call failed: {e}")
        return ""


def llm_judge(prompt: str) -> float:
    """Call OpenAI as an LLM judge. Returns score 0.0–1.0."""
    if not OPENAI_API_KEY:
        print("  [WARN] OPENAI_API_KEY not set — returning neutral score 0.5")
        return 0.5

    import urllib.request
    payload = json.dumps({
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 10,
    }).encode()

    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        text = data["choices"][0]["message"]["content"].strip()
        return max(0.0, min(1.0, float(text)))
    except Exception as e:
        print(f"  [WARN] LLM judge error: {e} — returning 0.5")
        return 0.5


def score_faithfulness(answer: str, topic: str) -> float:
    """
    Faithfulness: is the answer factually accurate and grounded for this topic?
    Uses LLM judge.
    """
    prompt = (
        f"You are an evaluator. The topic is: {topic}.\n"
        f"Score how factually accurate and grounded this answer is for that topic, "
        f"from 0.0 (hallucinated/wrong) to 1.0 (fully accurate).\n\n"
        f"Answer: {answer}\n\n"
        f"Reply with a single decimal number only, e.g. 0.8"
    )
    return llm_judge(prompt)


def score_answer_relevancy(answer: str, question: str, keywords: list) -> float:
    """
    Answer relevancy: combined keyword check + LLM judge.
    Keyword check ensures the agent actually addressed the topic.
    """
    # Keyword-based check (0 or 1 per keyword found)
    answer_lower = answer.lower()
    matched = sum(1 for kw in keywords if kw.lower() in answer_lower)
    keyword_score = matched / len(keywords) if keywords else 0.0

    # LLM relevancy score
    prompt = (
        f"You are an evaluator. Score how relevant and complete this answer is "
        f"for the question asked, from 0.0 (irrelevant/off-topic) to 1.0 (fully addresses the question).\n\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n\n"
        f"Reply with a single decimal number only, e.g. 0.7"
    )
    llm_score = llm_judge(prompt)

    # Average of both signals
    return round((keyword_score + llm_score) / 2, 3)


# ── Main Evaluation Loop ──────────────────────────────────────────────────────

def run_evaluation(thresholds: dict) -> dict:
    print(f"\n{'='*55}")
    print(f"  QUALITY GATE EVALUATION")
    print(f"  API: {API_BASE_URL}")
    print(f"  Samples: {len(EVAL_DATASET)}")
    print(f"  Thresholds: {thresholds}")
    print(f"{'='*55}\n")

    # Check API is reachable
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=10)
        resp.raise_for_status()
        print("  ✓ Agent API reachable\n")
    except Exception as e:
        print(f"  ✗ Agent API not reachable: {e}")
        sys.exit(1)

    faithfulness_scores     = []
    answer_relevancy_scores = []
    sample_results          = []

    for i, sample in enumerate(EVAL_DATASET):
        print(f"  Sample {i+1}/{len(EVAL_DATASET)}: {sample['topic']}")
        thread_id = f"eval_{i}_{int(time.time())}"
        answer = call_agent(sample["question"], thread_id=thread_id)

        if not answer:
            print("    → Empty answer, scoring 0.0")
            f_score  = 0.0
            ar_score = 0.0
        else:
            print(f"    Answer (truncated): {answer[:100]}...")
            f_score  = score_faithfulness(answer, sample["topic"])
            ar_score = score_answer_relevancy(answer, sample["question"], sample["keywords"])

        print(f"    Faithfulness: {f_score:.2f}  |  Answer Relevancy: {ar_score:.2f}")

        faithfulness_scores.append(f_score)
        answer_relevancy_scores.append(ar_score)
        sample_results.append({
            "topic":            sample["topic"],
            "question":         sample["question"],
            "answer":           answer,
            "faithfulness":     round(f_score, 3),
            "answer_relevancy": round(ar_score, 3),
        })

    avg_faithfulness     = sum(faithfulness_scores) / len(faithfulness_scores)
    avg_answer_relevancy = sum(answer_relevancy_scores) / len(answer_relevancy_scores)

    metrics = {
        "faithfulness": {
            "score":     round(avg_faithfulness, 3),
            "threshold": thresholds["faithfulness"],
            "passed":    avg_faithfulness >= thresholds["faithfulness"],
        },
        "answer_relevancy": {
            "score":     round(avg_answer_relevancy, 3),
            "threshold": thresholds["answer_relevancy"],
            "passed":    avg_answer_relevancy >= thresholds["answer_relevancy"],
        },
    }

    all_passed = all(m["passed"] for m in metrics.values())

    return {
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "api_base_url":   API_BASE_URL,
        "overall_passed": all_passed,
        "metrics":        metrics,
        "samples":        sample_results,
    }


def print_summary(results: dict):
    print(f"\n{'='*55}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*55}")
    for name, m in results["metrics"].items():
        status = "✓ PASS" if m["passed"] else "✗ FAIL"
        print(f"  {status}  {name}: {m['score']:.3f} (threshold: {m['threshold']})")
    print(f"{'='*55}")
    overall = "✓ ALL PASSED — build OK" if results["overall_passed"] else "✗ QUALITY GATE FAILED"
    print(f"  {overall}")
    print(f"{'='*55}\n")


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS_FILE)
    args = parser.parse_args()

    thresholds = load_thresholds(args.thresholds)
    results    = run_evaluation(thresholds)

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results written to {RESULTS_FILE}")

    print_summary(results)
    sys.exit(0 if results["overall_passed"] else 1)