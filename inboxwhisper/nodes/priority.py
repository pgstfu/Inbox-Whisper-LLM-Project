# utils/priority.py
from datetime import datetime, timezone
import math
import re

def classify_sender(email):
    """
    Returns: "professor", "student", "club", or "other"
    """
    e = email.lower()

    # Professor pattern: firstname.lastname@snu.edu.in
    if re.match(r"^[a-z]+\.[a-z]+@snu\.edu\.in$", e):
        return "professor"

    # Student pattern: fn888@snu.edu.in
    if re.match(r"^[a-z]{2}\d{3}@snu\.edu\.in$", e):
        return "student"

    # Club pattern: clubname@snu.edu.in
    if "club" in e or "society" in e:
        return "club"

    return "other"

# Your authentic academic courses
YOUR_COURSES = [
    "OOPS", "COA", "DISCRETE", "DIGITAL", "LLM",     # course-code detection
    "CSD211", "CSD213", "CSD205", "ECE103", "MAT496"  # add more as needed
]

# Senders that should have low or zero priority (clubs, events, spam)
LOW_PRIORITY_SENDERS = [
    "noreply@clubs.snu.edu.in",
    "events@snu.edu.in",
    "studentaffairs@snu.edu.in"
]

def is_course_related(task):
    subj = (task.get("title") or "").lower()
    body = (task.get("summary") or "").lower()
    for c in YOUR_COURSES:
        if c in subj or c in body:
            return True
    return False

def is_low_priority_sender(email_obj):
    sender = email_obj.get("from", "").lower()
    for low in LOW_PRIORITY_SENDERS:
        if low in sender:
            return True
    return False


def score_task(task):
    score = 0.0

    # If club/spam sender → priority = 0
    if is_low_priority_sender(task.get("raw", {})):
        return 0.0

    # Course-related emails get a base boost
    if is_course_related(task):
        score += 3.0

    # Type-based scoring
    typ = (task.get("type") or "other").lower()
    if typ == "assignment":
        score += 2.0
    elif typ == "exam":
        score += 3.0

    # Attachments boost (assignment PDFs)
    attachments = task.get("raw", {}).get("attachments", [])
    if attachments:
        score += 2.0  # assignment with attachment is more urgent

    # Keyword urgency
    text = (task.get("summary") or "").lower()
    for kw, val in [("urgent", 4), ("submit", 3), ("deadline", 2), ("due", 1.5)]:
        if kw in text:
            score += val

    # Deadline proximity
    due = task.get("due_date")
    if due:
        try:
            dt = datetime.fromisoformat(due)
            now = datetime.now(dt.tzinfo or timezone.utc)

            diff = (dt - now).total_seconds()
            if diff < 0:
                score += 8.0  # overdue = major problem
            else:
                days = diff / 86400
                score += max(0, 5 - math.log1p(days + 0.1))
        except:
            pass
        
    # Attachment requires your details → major bump
    text = (task.get("summary") or "").lower()
    attach_text = ""
    raw = task.get("raw", {})
    for att in raw.get("attachments", []):
        attach_text += att.get("text", "").lower()

    need_name = "name:" in attach_text or "write your name" in attach_text
    need_roll = "roll" in attach_text or "roll number" in attach_text

    if need_name or need_roll:
        score += 4.0  # templates requiring your info get extra boost
    
    # If it literally mentions YOUR NAME / ROLL in the PDF
    if "pranav gupta" in attach_text:
        score += 2.0

    if "2410110241" in attach_text:
        score += 2.0

    
    sender = task.get("raw", {}).get("from", "").lower()
    sender_type = classify_sender(sender)

    # Professors → strong boost
    if sender_type == "professor":
        score += 4.0

    # Students → neutral (could be group members)
    if sender_type == "student":
        score += 0.5

    # Clubs → extremely low priority (almost zero)
    if sender_type == "club":
        return 0.0

    attach_text = ""
    for att in raw.get("attachments", []):
        attach_text += (att.get("text", "") or "").lower()
        
    # detect name and roll requirements
    if "name:" in attach_text or "write your name" in attach_text:
        score += 3

    if "roll" in attach_text or "roll number" in attach_text:
        score += 3

    # If your name/roll already mentioned → high urgency
    if "pranav gupta" in attach_text:
        score += 2

    if "2410110241" in attach_text:
        score += 2

    return round(score, 3)
