# utils/priority.py
from datetime import datetime, timezone
import math
import re

YOUR_NAME = "pranav gupta"
YOUR_ROLL = "2410110241"

def classify_sender(email):
    e = (email or "").lower()

    # Professor pattern: firstname.lastname@snu.edu.in
    if re.match(r"^[a-z]+\.[a-z]+@snu\.edu\.in$", e):
        return "professor"

    # Student pattern: fn888@snu.edu.in
    if re.match(r"^[a-z]{2}\d{3}@snu\.edu\.in$", e):
        return "student"

    # Clubs
    if "club" in e or "society" in e or "noreply" in e:
        return "club"

    return "other"

def score_task(task):
    score = 0.0
    raw = task.get("raw", {})

    # Sender classification
    sender = raw.get("from","").lower()
    sender_type = classify_sender(sender)

    if sender_type == "professor":
        score += 4
    elif sender_type == "student":
        score += 0.5
    elif sender_type == "club":
        return 0.0

    # Course-related tasks
    if task.get("course"):
        score += 3

    # Type
    typ = (task.get("type") or "").lower()
    if typ == "assignment":
        score += 3
    elif typ == "exam":
        score += 5

    # Attachment-based logic
    attach_text = ""
    for att in raw.get("attachments", []):
        attach_text += (att.get("text","") or "").lower()

    # Name/Roll requirements
    if "name:" in attach_text or "roll" in attach_text:
        score += 3

    if YOUR_NAME in attach_text:
        score += 2

    if YOUR_ROLL in attach_text:
        score += 2

    # Deadlines
    due = task.get("due_date")
    if due:
        try:
            dt = datetime.fromisoformat(due)
            now = datetime.now(dt.tzinfo or timezone.utc)
            diff = (dt - now).total_seconds()

            if diff < 0:
                score += 8
            else:
                days = diff / 86400
                score += max(0, 5 - math.log1p(days + 0.1))
        except:
            pass

    return round(score,2)
