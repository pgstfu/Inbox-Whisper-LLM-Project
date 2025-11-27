# utils/priority.py
from datetime import datetime, timezone
import math

def score_task(task):
    """
    Simple scoring:
      - Base 0
      - deadline closeness: if due_date present, score higher as date approaches
      - type weight: assignment/exam get higher base
      - keywords: urgent/submit/ASAP raises score
    Output: priority float (higher = more urgent)
    """
    score = 0.0
    typ = (task.get("type") or "").lower()
    if typ == "assignment":
        score += 2.0
    elif typ == "exam":
        score += 3.0
    elif typ == "announcement":
        score += 0.5

    s = (task.get("summary") or "").lower() + " " + (task.get("action_item") or "").lower()
    for kw, w in [("urgent",3),("asap",3),("submit",2),("due",1.5),("deadline",1.5),("required",1)]:
        if kw in s:
            score += w

    # deadline effect (closer => bigger)
    due = task.get("due_date")
    if due:
        try:
            dt = datetime.fromisoformat(due)
            now = datetime.now(dt.tzinfo or timezone.utc)
            diff = (dt - now).total_seconds()
            # if overdue => very high priority
            if diff < 0:
                score += 10.0
            else:
                # scaled score: near-term deadlines get higher
                days = diff / 86400.0
                score += max(0, 5.0 - math.log1p(days+0.1))  # aggressive near-term bump
        except Exception:
            pass

    return round(score,3)
