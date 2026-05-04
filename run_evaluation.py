from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

os.environ.setdefault("LANGCHAIN_PROJECT", "adaptive-learning-companion-eval")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")

sys.path.insert(0, str(Path(__file__).parent))

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from openai import OpenAI

# ============================
# RAGAS (FIXED IMPORTS)
# ============================
from ragas import evaluate as ragas_evaluate
from ragas import EvaluationDataset, SingleTurnSample
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextRecall,
    ContextPrecision,
    AnswerCorrectness,
)

from langchain_openai import OpenAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper

# ============================
# DEEPEVAL (FIXED USAGE)
# ============================
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from multi_agent_graph import build_multi_agent_graph

# ============================
# CONFIG
# ============================
BASE_DIR = Path(__file__).parent

DATASET_PATH = BASE_DIR / "test_dataset.json"
RAW_OUTPUT = BASE_DIR / "eval_results_raw.json"
EVAL_CHECKPOINT = BASE_DIR / "eval_checkpoint.sqlite"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ============================
# EVAL PREFIX
# ============================
EVAL_MODE_PREFIX = "[EVAL MODE] Answer directly and use tools immediately. "

# ============================
# TOOL MESSAGE PARSER
# ============================
def extract_tool_message_text(msg: ToolMessage) -> str:
    content = msg.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "\n".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )

    if isinstance(content, dict):
        return content.get("text", "")

    return str(content)

# ============================
# RAGAS METRICS BUILDER
# ============================
def build_ragas_metrics(api_key: str):
    emb = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
    )

    return {
        "faithfulness": Faithfulness(),
        "answer_relevancy": AnswerRelevancy(),
        "context_recall": ContextRecall(),
        "context_precision": ContextPrecision(),
        "answer_correctness": AnswerCorrectness(),
        "embeddings": emb,
    }

# ============================
# RAGAS RUNNER
# ============================
def run_ragas_single(tc, response, context, metrics):
    nulls = {k: None for k in metrics.keys() if k != "embeddings"}

    if not response.strip():
        return nulls

    sample = SingleTurnSample(
        user_input=tc["user_query"],
        response=response,
        reference=tc["expected_ground_truth"],
        retrieved_contexts=[context] if context else []
    )

    dataset = EvaluationDataset(samples=[sample])

    metric_list = [
        metrics["faithfulness"],
        metrics["answer_relevancy"],
        metrics["context_recall"],
        metrics["context_precision"],
        metrics["answer_correctness"],
    ]

    try:
        result = ragas_evaluate(dataset=dataset, metrics=metric_list)
        row = result.to_pandas().iloc[0]

        return {
            k: round(float(row[k]), 3) if k in row else None
            for k in nulls.keys()
        }

    except Exception as e:
        print(f"[RAGAS ERROR] {e}")
        return nulls

# ============================
# DEEPEVAL RUNNER (FIXED)
# ============================
def run_deepeval(tc, response, context):
    if not response.strip():
        return {"relevancy": None, "faithfulness": None}

    try:
        case = LLMTestCase(
            input=tc["user_query"],
            actual_output=response,
            expected_output=tc["expected_ground_truth"],
            retrieval_context=[context] if context else []
        )

        relevancy = AnswerRelevancyMetric(threshold=0.5)
        faithfulness = FaithfulnessMetric(threshold=0.5)

        relevancy.measure(case)
        faithfulness.measure(case)

        return {
            "relevancy": round(relevancy.score, 3),
            "faithfulness": round(faithfulness.score, 3),
        }

    except Exception as e:
        print(f"[DEEPEVAL ERROR] {e}")
        return {"relevancy": None, "faithfulness": None}

# ============================
# AGENT RUNNER
# ============================
def invoke_agent(query: str):
    thread_id = f"eval_{uuid.uuid4().hex[:6]}"
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "messages": [HumanMessage(content=EVAL_MODE_PREFIX + query)],
        "current_agent": "researcher",
    }

    tool_calls = []
    context_parts = []

    with SqliteSaver.from_conn_string(str(EVAL_CHECKPOINT)) as cp:
        app = build_multi_agent_graph(cp)
        result = app.invoke(initial_state, config)

        for msg in result.get("messages", []):
            if isinstance(msg, AIMessage):
                for tc in getattr(msg, "tool_calls", []) or []:
                    tool_calls.append(tc.get("name"))

            if isinstance(msg, ToolMessage):
                text = extract_tool_message_text(msg)
                if text:
                    context_parts.append(text)

    response = ""
    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.content:
            response = msg.content
            break

    return {
        "response": response,
        "tool_calls": tool_calls,
        "context": "\n".join(context_parts),
    }

# ============================
# MAIN
# ============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="")
    args = parser.parse_args()

    with open(DATASET_PATH) as f:
        dataset = json.load(f)["test_cases"]

    if args.cases:
        ids = set(args.cases.split(","))
        dataset = [tc for tc in dataset if tc["id"] in ids]

    metrics = build_ragas_metrics(OPENAI_API_KEY)

    results = []

    for tc in dataset:
        print(f"\nRunning {tc['id']}...")

        out = invoke_agent(tc["user_query"])

        ragas_scores = run_ragas_single(
            tc,
            out["response"],
            out["context"],
            metrics
        )

        deepeval_scores = run_deepeval(
            tc,
            out["response"],
            out["context"]
        )

        print("RAGAS:", ragas_scores)
        print("DeepEval:", deepeval_scores)

        results.append({
            "id": tc["id"],
            "ragas": ragas_scores,
            "deepeval": deepeval_scores
        })

    with open(RAW_OUTPUT, "w") as f:
        json.dump(results, f, indent=2)

    print("\nDone.")

if __name__ == "__main__":
    main()