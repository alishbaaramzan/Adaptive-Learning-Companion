# export_feedback.py

import sqlite3
import json
from pathlib import Path

DB_PATH = "/app/checkpoint_data/feedback_log.db"
OUTPUT_FILE = "feedback_readable.json"

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row

rows = conn.execute("""
SELECT user_input, agent_response, feedback, timestamp
FROM feedback_log
ORDER BY timestamp DESC
""").fetchall()

data = [dict(r) for r in rows]

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"Exported {len(data)} entries to {OUTPUT_FILE}")