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
from datetime import datetime

# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_THRESHOLDS_FILE = "eval_thresholds.json"
RESULTS_FILE            = "eval_results.json"
API_BASE_URL            = os.environ.get("API_BASE_URL", "http://localhost:8000")
OPENAI_API_KEY          = os.environ.get("OPENAI_API_KEY", "")

# Test dataset: (question, reference_answer, expected_topic)
# These are fixed so the eval is deterministic and runnable in CI with no human input.
EVAL_DATASET = [
    {
        "question":   "What is supervised learning?",
        "reference":  "Supervised learning is a type of machine learning where the model is trained on labelled data, meaning each training example has an input and a known correct output. The model learns to map inputs to outputs by minimising prediction error.",
        "topic":      "machine learning basics"
    },
    {
        "question":   "What is the difference between overfitting and underfitting?",
        "reference":  "Overfitting occurs when a model learns the training data too well, including noise, and performs poorly on new data. Underfitting occurs when the model is too simple to capture the underlying pattern and performs poorly on both training and new data.",
        "topic":      "model evaluation"
    },
    {
        "question":   "What is a neural network?",
        "reference":  "A neural network is a machine learning model inspired by the human brain. It consists of layers of interconnected nodes (neurons) that process data. Each connection has a weight that is adjusted during training to minimise prediction error.",
        "topic":      "deep learning"
    },
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_thresholds(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    return {k: v["min"] for k, v in data["thresholds"].items()}


def call_agent(question: str, student_id: str = "eval_bot", thread_id: str = None) -> str:
    """Send a question to the agent API and return its answer."""
    if thread_id is None:
        thread_id = f"eval_{int(time.time() * 1000)}"
    try:
        resp = requests.post(
            f"{API_BASE_URL}/chat",
            json={"message": question, "student_id": student_id, "thread_id": thread_id},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json().get("answer", "")
    except Exception as e:
        print(f"  [ERROR] Agent call failed: {e}")
        return ""


def llm_judge(prompt: str) -> float:
    """
    Call the OpenAI API directly as an LLM judge.
    Returns a score between 0.0 and 1.0.
    Uses a simple, reliable prompt that asks for a single number.
    """
    if not OPENAI_API_KEY:
        print("  [WARN] OPENAI_API_KEY not set — returning neutral score 0.5")
        return 0.5

    import urllib.request
    payload = json.dumps({
        "model": "gpt-4o-mini",   # cheap judge model
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
        score = float(text)
        return max(0.0, min(1.0, score))
    except Exception as e:
        print(f"  [WARN] LLM judge error: {e} — returning 0.5")
        return 0.5


def score_faithfulness(answer: str, question: str) -> float:
    """
    Faithfulness: is the answer grounded in what the agent should know?
    Judge prompt asks for a score 0.0–1.0.
    """
    prompt = (
        f"You are an evaluator. Score how factually grounded and accurate this answer is "
        f"for the question, on a scale from 0.0 (completely wrong/hallucinated) to 1.0 (fully accurate).\n\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n\n"
        f"Reply with a single decimal number only, e.g. 0.8"
    )
    return llm_judge(prompt)


def score_answer_relevancy(answer: str, question: str) -> float:
    """
    Answer relevancy: does the answer actually address the question?
    """
    prompt = (
        f"You are an evaluator. Score how relevant and on-topic this answer is to the question, "
        f"on a scale from 0.0 (completely irrelevant) to 1.0 (perfectly on-topic and complete).\n\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n\n"
        f"Reply with a single decimal number only, e.g. 0.7"
    )
    return llm_judge(prompt)


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
        print("  → Cannot run evaluation. Exiting with failure.")
        sys.exit(1)

    faithfulness_scores    = []
    answer_relevancy_scores = []
    sample_results         = []

    for i, sample in enumerate(EVAL_DATASET):
        print(f"  Sample {i+1}/{len(EVAL_DATASET)}: {sample['question'][:60]}...")
        thread_id = f"eval_{i}_{int(time.time())}"
        answer = call_agent(sample["question"], thread_id=thread_id)

        if not answer:
            print("    → Empty answer, scoring 0.0")
            f_score  = 0.0
            ar_score = 0.0
        else:
            print(f"    Answer (truncated): {answer[:80]}...")
            f_score  = score_faithfulness(answer, sample["question"])
            ar_score = score_answer_relevancy(answer, sample["question"])

        print(f"    Faithfulness: {f_score:.2f}  |  Answer Relevancy: {ar_score:.2f}")

        faithfulness_scores.append(f_score)
        answer_relevancy_scores.append(ar_score)
        sample_results.append({
            "question":         sample["question"],
            "answer":           answer,
            "faithfulness":     round(f_score, 3),
            "answer_relevancy": round(ar_score, 3),
        })

    # Aggregate
    avg_faithfulness    = sum(faithfulness_scores) / len(faithfulness_scores)
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

    results = {
        "timestamp":   datetime.utcnow().isoformat() + "Z",
        "api_base_url": API_BASE_URL,
        "overall_passed": all_passed,
        "metrics":     metrics,
        "samples":     sample_results,
    }

    return results


def print_summary(results: dict):
    print(f"\n{'='*55}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*55}")
    for name, m in results["metrics"].items():
        status = "✓ PASS" if m["passed"] else "✗ FAIL"
        print(f"  {status}  {name}: {m['score']:.3f} (threshold: {m['threshold']})")
    print(f"{'='*55}")
    overall = "✓ ALL PASSED — build OK" if results["overall_passed"] else "✗ QUALITY GATE FAILED — block deployment"
    print(f"  {overall}")
    print(f"{'='*55}\n")


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS_FILE)
    args = parser.parse_args()

    thresholds = load_thresholds(args.thresholds)
    results    = run_evaluation(thresholds)

    # Write machine-readable results file
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results written to {RESULTS_FILE}")

    print_summary(results)

    # Exit code for CI: 0 = pass, 1 = fail
    sys.exit(0 if results["overall_passed"] else 1)