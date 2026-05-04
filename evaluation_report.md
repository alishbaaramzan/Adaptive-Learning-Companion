# Evaluation Report — Adaptive Learning Companion (v4)

**Project:** Adaptive Learning Companion — Multi-Agent LangGraph  
**Evaluation Frameworks:** RAGAS v0.4.x · Tool scoring  
**Model Under Test:** GPT-4o (Researcher & Analyst)  
**Judge Model:** GPT-4o-mini  
**Dataset:** test_dataset.json — 3 test cases  
**Generated:** 2026-04-13 09:23  
**Eval Fixes Applied:** EVAL_MODE_PREFIX · robust ToolMessage extraction · RAGAS 0.4.x API · partial RAGAS fallback  

---

## 1. RAGAS Scores — Retrieval-Augmented Queries

*2 cases in categories: retrieval, guardrail_safe, memory, handoff*

| Metric | Average Score | Threshold | Status |
|---|---|---|---|
| **Faithfulness** | n/a | ≥ 0.80 | — |
| **Answer Relevancy** | n/a | ≥ 0.75 | — |
| **Context Recall** | n/a | ≥ 0.70 | — |
| **Context Precision** | n/a | ≥ 0.70 | — |
| **Answer Correctness** | n/a | ≥ 0.75 | — |

---

## 2. Tool Call Accuracy — All Cases

*3 cases scored*

| Metric | Average Score | Threshold | Status |
|---|---|---|---|
| **Tool Selection Accuracy** | 1.000 | ≥ 0.85 | ✅ PASS |
| **Argument Binding Accuracy** | 1.000 | ≥ 0.80 | ✅ PASS |
| **No-Duplicate-Call Rate** | 1.000 | ≥ 0.90 | ✅ PASS |

---

## 3. Guardrail Accuracy

| Metric | Score | Cases |
|---|---|---|
| **True Positive Rate** (block when unsafe) | n/a | 0 adversarial cases |
| **False Positive Rate** (block when safe) | n/a | 0 legitimate cases |

---

## 4. Overall Summary

| Dimension | Score |
|---|---|
| RAGAS Faithfulness | n/a |
| RAGAS Answer Relevancy | n/a |
| RAGAS Answer Correctness | n/a |
| Tool Selection | 1.000 |
| Tool Binding | 1.000 |
| Guardrail TP Rate | n/a |
| **Weighted Average** | **1.000** |

---

## 5. Per-Case Results

| ID | Category | Faithfulness | Relevancy | Correctness | Tool Sel. | Tool Bind. | Latency | Status |
|---|---|---|---|---|---|---|---|---|
| TC-001 | retrieval | n/a | n/a | n/a | 1.000 | 1.000 | 4.89s | ✅ |
| TC-002 | tool_call | — | — | — | 1.000 | 1.000 | 2.54s | ✅ |
| TC-003 | retrieval | n/a | n/a | n/a | 1.000 | 1.000 | 3.82s | ✅ |

---

## 6. Latency

| Stat | Value |
|---|---|
| Fastest | TC-002 (2.54s) |
| Slowest | TC-001 (4.89s) |
| Average | 3.75s |

---

## 7. Errors

_No errors recorded._

---

## 8. Diagnostic Notes

- **Faithfulness / context metrics show `n/a`** when `retrieve_content` returned
  empty content (vector store gap). Check console ⚠️ lines to see which topics
  need indexing.
- **Answer Relevancy / Correctness** are always scored, even without retrieval
  context, using the agent response as a proxy.
- **No-Duplicate-Call FAIL** means the same tool was called more than once in
  a single turn (-0.2 penalty per duplicate).
