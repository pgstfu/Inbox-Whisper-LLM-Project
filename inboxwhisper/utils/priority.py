# utils/priority.py
from datetime import datetime, timezone
import math
import re

from .student_profile import normalized_name, normalized_roll


def classify_sender(email):
    """
    Classifies the email sender based on email address patterns.
    
    This function uses regex patterns to identify:
    - Professors: firstname.lastname@snu.edu.in (e.g., john.smith@snu.edu.in)
    - Students: fn888@snu.edu.in (e.g., ab123@snu.edu.in)
    - Clubs/Societies: Contains "club", "society", or "noreply" in the address
    
    Args:
        email: Email address string (e.g., "john.smith@snu.edu.in")
    
    Returns:
        str: One of "professor", "student", "club", or "other"
    """
    e = (email or "").lower()

    # Professor pattern: firstname.lastname@snu.edu.in
    if re.match(r"^[a-z]+\.[a-z]+@snu\.edu\.in$", e):
        return "professor"

    # Student pattern: fn888@snu.edu.in (2 letters + 3 digits)
    if re.match(r"^[a-z]{2}\d{3}@snu\.edu\.in$", e):
        return "student"

    # Clubs / broadcast emails
    if "club" in e or "society" in e or "noreply" in e:
        return "club"

    return "other"


def score_task(task):
    """
    Computes a priority score for a task based on multiple factors.
    
    This function implements a weighted scoring system that prioritizes:
    - Professor emails (higher authority)
    - Course-related tasks (academic relevance)
    - Exams/vivas (high-stakes events)
    - Tasks with student-specific details (personal relevance)
    - Urgent deadlines (time pressure)
    - Urgency keywords in the content
    
    The scoring formula:
    - Base score starts at 0.0
    - Sender type: professor (+4), student (+0.5), club (returns 0.0 immediately)
    - Course present: +3
    - Task type: assignment (+3), exam/viva (+5)
    - Attachment mentions name/roll: +3 (if "name:" or "roll" keywords found)
    - Student name/roll in attachment text: +2 each
    - Student slot details found: +5 (direct match in seating plan)
    - Deadline proximity: +8 if overdue, else +max(0, 5 - log(days+0.1))
    - Urgency keywords: "urgent"/"asap" (+3), "submit" (+2), "due"/"deadline" (+1.5), etc.
    
    Args:
        task: Dict containing task fields (type, course, due_date, raw, student_slot_details, etc.)
    
    Returns:
        float: Priority score (higher = more urgent). Rounded to 2 decimal places.
    """
    score = 0.0
    raw = task.get("raw", {}) or {}

    # Step 1: Sender classification
    # Professor emails are most important, student emails less so, club emails are filtered out
    sender_type = classify_sender(raw.get("from"))

    if sender_type == "professor":
        score += 4  # Professors send important academic communications
    elif sender_type == "student":
        score += 0.5  # Student emails are usually less critical
    elif sender_type == "club":
        # Club/society emails are often spam - deprioritize completely
        return 0.0

    # Step 2: Course relevance
    # Tasks related to enrolled courses are more important than general announcements
    if task.get("course"):
        score += 3

    # Step 3: Task type importance
    # Exams and vivas are high-stakes, assignments require work, announcements are informational
    typ = (task.get("type") or "").lower()
    if typ == "assignment":
        score += 3
    elif typ in {"exam", "viva"}:
        score += 5  # Exams/vivas are critical events

    # Step 4: Attachment analysis
    # Search attachment text for student-specific mentions
    attach_text = ""
    for att in raw.get("attachments", []):
        attach_text += (att.get("text", "") or "").lower()

    # If attachments mention name/roll fields, it's likely a seating plan or roster
    if "name:" in attach_text or "roll" in attach_text:
        score += 3

    # Check if the student's name or roll number appears in attachments
    # This indicates the task is directly relevant to the student
    name = normalized_name()
    roll = normalized_roll()

    if name in attach_text:
        score += 2  # Student's name found in attachment
    if roll in attach_text:
        score += 2  # Student's roll number found in attachment

    # If student_slot_details exists, it means we found a direct match in seating plan
    if task.get("student_slot_details"):
        score += 5  # High priority: student's specific slot was found

    # Step 5: Deadline pressure
    # Tasks with closer deadlines get higher scores
    # Overdue tasks get a large boost to ensure they're addressed immediately
    due = task.get("due_date")
    if due:
        try:
            dt = datetime.fromisoformat(due)
            now = datetime.now(dt.tzinfo or timezone.utc)
            diff = (dt - now).total_seconds()

            if diff < 0:
                # Overdue: very high priority
                score += 8
            else:
                # Not overdue: score decreases logarithmically with days until deadline
                # Formula: max(0, 5 - log(days + 0.1))
                # This gives high scores for near-term deadlines, low scores for far-future deadlines
                days = diff / 86400  # Convert seconds to days
                score += max(0, 5 - math.log1p(days + 0.1))
        except Exception:
            # If date parsing fails, skip deadline scoring
            pass

    # Step 6: Urgency keywords
    # Scan task summary and action_item for urgency indicators
    combined_text = (
        ((task.get("summary") or "") + " " + (task.get("action_item") or "")).lower()
    )
    for kw, weight in [
        ("urgent", 3),      # Explicit urgency
        ("asap", 3),       # As soon as possible
        ("submit", 2),     # Submission required
        ("due", 1.5),      # Deadline mentioned
        ("deadline", 1.5), # Deadline keyword
        ("seating plan", 2), # Exam/viva scheduling
        ("slot", 1.5),     # Time slot mentioned
    ]:
        if kw in combined_text:
            score += weight

    return round(score, 2)
