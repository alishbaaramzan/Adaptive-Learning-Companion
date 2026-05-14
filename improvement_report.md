# Improvement Demo

## Identified Issue

Feedback analysis showed that users frequently gave negative feedback for beginner programming explanations, especially recursion-related concepts. Responses lacked examples and beginner-friendly explanations.

## Fix Applied

The system prompt was improved to:

* explain coding concepts step-by-step
* include short examples
* avoid overly technical wording
* provide beginner-friendly explanations

## Before Improvement

### User Query

“first explain what recursion is”

### Agent Response

“Recursion is when a function calls itself until a base case is reached.”

### Feedback

👎 Bad

---

## After Improvement

### User Query

“first explain what recursion is”

### Agent Response

“Recursion is a programming technique where a function repeatedly calls itself to solve smaller parts of a problem.

For example, a countdown function can call itself like this:

5 → 4 → 3 → 2 → 1

The condition that stops the repeated calls is called the base case.”

### Feedback

👍 Good

## Result

After updating the prompt, the agent produced clearer and more beginner-friendly explanations for coding-related questions, improving overall response quality.
