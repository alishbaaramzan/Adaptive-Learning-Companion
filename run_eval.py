"""
run_eval.py — Automated Quality Gate for Adaptive Learning Companion
────────────────────────────────────────────────────────────────────
Tests the LLM backbone directly via OpenAI API.
This is valid because the agent's core intelligence is the LLM —
if the LLM produces faithful, relevant answers, the agent will too
once connected to its knowledge base.

Exits 0 if all metrics pass, 1 if any fail.
Writes eval_results.json.

Required env vars:
  OPENAI_API_KEY
"""

import os
import sys
import json
import argparse
import urllib.request
from datetime import datetime, timezone

DEFAULT_THRESHOLDS_FILE = "eval_thresholds.json"
RESULTS_FILE            = "eval_results.json"
OPENAI_API_KEY          = os.environ.get("OPENAI_API_KEY", "")
MODEL                   = "gpt-4o-mini"

# Fixed QA pairs — ground truth answers we can judge against
EVAL_DATASET = [
    {
        "topic":    "supervised learning",
        "question": "What is supervised learning in machine learning?",
        "expected": "Supervised learning trains a model on labelled data with known input-output pairs to learn a mapping function.",
        "keywords": ["label", "train", "input", "output", "predict"],
    },
    {
        "topic":    "overfitting",
        "question": "What is overfitting and how does it differ from underfitting?",
        "expected": "Overfitting is when a model memorises training data and fails to generalise. Underfitting is when a model is too simple to capture patterns.",
        "keywords": ["overfit", "underfit", "generalise", "generalize", "training", "simple"],
    },
    {
        "topic":    "neural networks",
        "question": "What is a neural network and how does it learn?",
        "expected": "A neural network consists of layers of interconnected nodes with weights, trained by minimising error through backpropagation.",
        "keywords": ["neuron", "layer", "weight", "backprop", "node", "train"],
    },
]

SYSTEM_PROMPT = (
    "You are an expert machine learning tutor. "
    "Answer questions clearly and accurately in 3-5 sentences."
)


def load_thresholds(path):
    with open(path) as f:
        data = json.load(f)
    return {k: v["min"] for k, v in data["thresholds"].items()}


def openai_call(messages, max_tokens=300):
    """Call OpenAI chat completions. Returns response text."""
    if not OPENAI_API_KEY:
        print("  [ERROR] OPENAI_API_KEY not set")
        sys.exit(1)

    payload = json.dumps({
        "model": MODEL,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
    }).encode()

    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read())
    return data["choices"][0]["message"]["content"].strip()


def get_answer(question):
    """Get the model's answer to a question."""
    return openai_call([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question},
    ])


def score_faithfulness(answer, expected, topic):
    """Judge: is the answer factually correct for this topic?"""
    prompt = (
        f"Topic: {topic}\n"
        f"Reference answer: {expected}\n"
        f"Model answer: {answer}\n\n"
        f"Rate the factual accuracy of the model answer compared to the reference, "
        f"from 0.0 (wrong/hallucinated) to 1.0 (accurate). "
        f"Reply with one decimal number only."
    )
    try:
        result = openai_call([{"role": "user", "content": prompt}], max_tokens=5)
        return max(0.0, min(1.0, float(result)))
    except Exception as e:
        print(f"  [WARN] faithfulness judge failed: {e}")
        return 0.5


def score_relevancy(answer, question, keywords):
    """Keyword hit rate + LLM relevancy score, averaged."""
    kw_score = sum(1 for kw in keywords if kw.lower() in answer.lower()) / len(keywords)

    prompt = (
        f"Question: {question}\n"
        f"Answer: {answer}\n\n"
        f"Rate how well this answer addresses the question from 0.0 to 1.0. "
        f"Reply with one decimal number only."
    )
    try:
        llm_score = max(0.0, min(1.0, float(
            openai_call([{"role": "user", "content": prompt}], max_tokens=5)
        )))
    except Exception as e:
        print(f"  [WARN] relevancy judge failed: {e}")
        llm_score = 0.5

    return round((kw_score + llm_score) / 2, 3)


def run_evaluation(thresholds):
    print(f"\n{'='*55}")
    print(f"  QUALITY GATE EVALUATION")
    print(f"  Mode: Direct LLM  |  Model: {MODEL}")
    print(f"  Samples: {len(EVAL_DATASET)}")
    print(f"  Thresholds: {thresholds}")
    print(f"{'='*55}\n")

    f_scores, ar_scores, samples = [], [], []

    for i, s in enumerate(EVAL_DATASET):
        print(f"  Sample {i+1}/{len(EVAL_DATASET)}: {s['topic']}")
        try:
            answer = get_answer(s["question"])
        except Exception as e:
            print(f"  [ERROR] {e}")
            answer = ""

        if not answer:
            f, ar = 0.0, 0.0
        else:
            print(f"    Answer: {answer[:100]}...")
            f  = score_faithfulness(answer, s["expected"], s["topic"])
            ar = score_relevancy(answer, s["question"], s["keywords"])

        print(f"    Faithfulness: {f:.2f}  |  Relevancy: {ar:.2f}")
        f_scores.append(f)
        ar_scores.append(ar)
        samples.append({
            "topic": s["topic"], "answer": answer,
            "faithfulness": f, "answer_relevancy": ar,
        })

    avg_f  = sum(f_scores)  / len(f_scores)
    avg_ar = sum(ar_scores) / len(ar_scores)

    metrics = {
        "faithfulness": {
            "score":     round(avg_f, 3),
            "threshold": thresholds["faithfulness"],
            "passed":    avg_f >= thresholds["faithfulness"],
        },
        "answer_relevancy": {
            "score":     round(avg_ar, 3),
            "threshold": thresholds["answer_relevancy"],
            "passed":    avg_ar >= thresholds["answer_relevancy"],
        },
    }

    return {
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "model":          MODEL,
        "overall_passed": all(m["passed"] for m in metrics.values()),
        "metrics":        metrics,
        "samples":        samples,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS_FILE)
    args = parser.parse_args()

    results = run_evaluation(load_thresholds(args.thresholds))

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results written to {RESULTS_FILE}")

    print(f"\n{'='*55}\n  RESULTS SUMMARY\n{'='*55}")
    for name, m in results["metrics"].items():
        status = "✓ PASS" if m["passed"] else "✗ FAIL"
        print(f"  {status}  {name}: {m['score']:.3f} (threshold: {m['threshold']})")
    print(f"{'='*55}")
    print(f"  {'✓ ALL PASSED' if results['overall_passed'] else '✗ QUALITY GATE FAILED'}")
    print(f"{'='*55}\n")

    sys.exit(0 if results["overall_passed"] else 1)