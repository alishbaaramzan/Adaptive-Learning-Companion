# Analysis Report

## Overview

The deployed agent system collected user feedback using thumbs up/down interactions stored in SQLite. Feedback logs were analyzed using `analyze.py`.

### Statistics

* Total Responses: 5
* Positive Feedback: 2
* Negative Feedback: 3
* Negative Feedback Rate: 60%

## Top Failed Queries

1. “traditional is related to regression, classification etc while agentic one gives memory, tool and reasong”
2. “i think it means running a loop till a base case it met”
3. “first explain what recursion is”

## Observations

Analysis showed that most negative feedback came from beginner-level programming and conceptual explanation tasks. Responses were often too short or lacked step-by-step explanations and examples.


## Conclusion

The feedback monitoring system successfully identified recurring failure patterns. The collected feedback was then used to improve prompt instructions for educational and coding-related responses.
