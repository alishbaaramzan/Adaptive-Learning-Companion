"""
run_eval.py — Automated Quality Gate for Adaptive Learning Companion
Two-turn eval: first message prompts the agent, second message answers
its follow-up so it proceeds to explain the topic.
"""

import os
import sys
import json
import time
import argparse
import requests
from datetime import datetime, timezone

DEFAULT_THRESHOLDS_FILE = "eval_thresholds.json"
RESULTS_FILE            = "eval_results.json"
API_BASE_URL            = os.environ.get("API_BASE_URL", "http://localhost:8000")
OPENAI_API_KEY          = os.environ.get("OPENAI_API_KEY", "")

# Two-turn dataset:
# turn1 — initial question
# turn2 — student says they don't know, forcing the agent to explain
EVAL_DATASET = [
    {
        "topic":  "supervised learning",
        "turn1":  "Can you teach me about supervised learning?",
        "turn2":  "I don't know anything about it yet, please explain it to me from scratch.",
        "keywords": ["label", "train", "input", "output", "predict", "data"],
    },
    {
        "topic":  "overfitting vs underfitting",
        "turn1":  "Can you explain overfitting and underfitting to me?",
        "turn2":  "I have no prior knowledge, please just explain both concepts to me.",
        "keywords": ["overfit", "underfit", "training", "generalise", "generalize", "noise"],
    },
    {
        "topic":  "neural networks",
        "turn1":  "Can you teach me what a neural network is?",
        "turn2":  "I am a complete beginner, please explain it from the beginning.",
        "keywords": ["neuron", "layer", "weight", "node", "network", "brain"],
    },
]


def load_thresholds(path):
    with open(path) as f:
        data = json.load(f)
    return {k: v["min"] for k, v in data["thresholds"].items()}


def call_agent(message, thread_id):
    try:
        resp = requests.post(
            f"{API_BASE_URL}/chat",
            json={"message": message, "student_id": "eval_bot", "thread_id": thread_id},
            timeout=90,
        )
        resp.raise_for_status()
        return resp.json().get("answer", "")
    except Exception as e:
        print(f"  [ERROR] {e}")
        return ""


def llm_judge(prompt):
    if not OPENAI_API_KEY:
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
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {OPENAI_API_KEY}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read())
        return max(0.0, min(1.0, float(data["choices"][0]["message"]["content"].strip())))
    except Exception as e:
        print(f"  [WARN] judge error: {e}")
        return 0.5


def score_faithfulness(answer, topic):
    return llm_judge(
        f"Topic: {topic}\nRate factual accuracy of this answer from 0.0 to 1.0.\n"
        f"Answer: {answer}\nReply with one decimal number only."
    )


def score_relevancy(answer, topic, keywords):
    # keyword hit rate
    kw_score = sum(1 for kw in keywords if kw.lower() in answer.lower()) / len(keywords)
    # llm score
    llm_score = llm_judge(
        f"Topic: {topic}\nRate how well this answer explains the topic from 0.0 to 1.0.\n"
        f"Answer: {answer}\nReply with one decimal number only."
    )
    return round((kw_score + llm_score) / 2, 3)


def run_evaluation(thresholds):
    print(f"\n{'='*55}")
    print(f"  QUALITY GATE EVALUATION  (two-turn mode)")
    print(f"  API: {API_BASE_URL}  |  Samples: {len(EVAL_DATASET)}")
    print(f"  Thresholds: {thresholds}")
    print(f"{'='*55}\n")

    try:
        requests.get(f"{API_BASE_URL}/health", timeout=10).raise_for_status()
        print("  ✓ Agent API reachable\n")
    except Exception as e:
        print(f"  ✗ API not reachable: {e}")
        sys.exit(1)

    f_scores, ar_scores, samples = [], [], []

    for i, s in enumerate(EVAL_DATASET):
        print(f"  Sample {i+1}/{len(EVAL_DATASET)}: {s['topic']}")
        tid = f"eval_{i}_{int(time.time())}"

        # Turn 1 — ask the question
        call_agent(s["turn1"], tid)
        time.sleep(2)

        # Turn 2 — say we don't know, so agent explains
        answer = call_agent(s["turn2"], tid)

        if not answer:
            f, ar = 0.0, 0.0
            print("    → Empty answer")
        else:
            print(f"    Answer: {answer[:120]}...")
            f  = score_faithfulness(answer, s["topic"])
            ar = score_relevancy(answer, s["topic"], s["keywords"])

        print(f"    Faithfulness: {f:.2f}  |  Relevancy: {ar:.2f}")
        f_scores.append(f)
        ar_scores.append(ar)
        samples.append({"topic": s["topic"], "answer": answer,
                        "faithfulness": f, "answer_relevancy": ar})

    avg_f  = sum(f_scores)  / len(f_scores)
    avg_ar = sum(ar_scores) / len(ar_scores)

    metrics = {
        "faithfulness":     {"score": round(avg_f, 3),  "threshold": thresholds["faithfulness"],
                             "passed": avg_f  >= thresholds["faithfulness"]},
        "answer_relevancy": {"score": round(avg_ar, 3), "threshold": thresholds["answer_relevancy"],
                             "passed": avg_ar >= thresholds["answer_relevancy"]},
    }

    return {
        "timestamp":      datetime.now(timezone.utc).isoformat(),
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