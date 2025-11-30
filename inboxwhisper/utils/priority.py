# utils/priority.py
from datetime import datetime, timezone
import math
import re

from .student_profile import normalized_name, normalized_roll


def classify_sender(email):
    e = (email or "").lower()

    # Professor pattern: firstname.lastname@snu.edu.in
    if re.match(r"^[a-z]+\.[a-z]+@snu\.edu\.in$", e):
        return "professor"

    # Student pattern: fn888@snu.edu.in
    if re.match(r"^[a-z]{2}\d{3}@snu\.edu\.in$", e):
        return "student"

    # Clubs / broadcast
    if "club" in e or "society" in e or "noreply" in e:
        return "club"

    return "other"


def score_task(task):
    """
    Richer prioritization tuned for academic workflows:
      - Boost professor emails & known courses
      - Highlight exams/vivas and seating-plan matches for the student
      - Penalize club spam
      - Keep deadline pressure with aggressive near-term bump
    """
    score = 0.0
    raw = task.get("raw", {}) or {}

    # Sender classification
    sender_type = classify_sender(raw.get("from"))

    if sender_type == "professor":
        score += 4
    elif sender_type == "student":
        score += 0.5
    elif sender_type == "club":
        # deprioritize generic club blasts
        return 0.0

    # Course-related tasks
    if task.get("course"):
        score += 3

    # Type
    typ = (task.get("type") or "").lower()
    if typ == "assignment":
        score += 3
    elif typ in {"exam", "viva"}:
        score += 5

    # Attachment-based logic
    attach_text = ""
    for att in raw.get("attachments", []):
        attach_text += (att.get("text", "") or "").lower()

    # Name/Roll requirements
    if "name:" in attach_text or "roll" in attach_text:
        score += 3

    name = normalized_name()
    roll = normalized_roll()

    if name in attach_text:
        score += 2

    if roll in attach_text:
        score += 2

    if task.get("student_slot_details"):
        # Direct match for the student's slot (exam/viva seating plans)
        score += 5

    # Deadline pressure
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
        except Exception:
            pass

    # Keyword urgency (for reordering in list view)
    combined_text = (
        ((task.get("summary") or "") + " " + (task.get("action_item") or "")).lower()
    )
    for kw, weight in [
        ("urgent", 3),
        ("asap", 3),
        ("submit", 2),
        ("due", 1.5),
        ("deadline", 1.5),
        ("seating plan", 2),
        ("slot", 1.5),
    ]:
        if kw in combined_text:
            score += weight

    return round(score, 2)
