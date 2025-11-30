# nodes/email_to_task.py

from nodes.email_parser import parse_email
from utils.course_detection import detect_course
from utils.deadline_extractor import extract_deadline
from utils.student_slot import extract_student_slots
from utils.location_extractor import extract_location
import dateutil.parser
from pathlib import Path

def email_to_task_struct(email_obj):
    """
    Convert raw email + attachments into a Task dict.
    Uses LLM parsing + deadline extraction + course detection.
    """

    # Combine email body + ALL attachment text
    attachment_text = ""
    for att in email_obj.get("attachments", []):
        attachment_text += "\n\n[ATTACHMENT TEXT]\n" + (att.get("text","") or "")

    subject = email_obj.get("subject") or ""
    full_text = subject + "\n" + email_obj.get("body","") + attachment_text

    # LLM parsing with full context
    parsed = parse_email({
        "subject": subject,
        "body": full_text
    })

    # -------- 1. COURSE DETECTION --------
    course = detect_course(full_text) or parsed.get("course")

    # -------- 2. DEADLINE DETECTION --------
    deadline = parsed.get("deadline") or extract_deadline(full_text)

    if deadline:
        try:
            dt = dateutil.parser.parse(deadline)
            deadline = dt.isoformat()
        except:
            pass

    # -------- 3. LOCATION DETECTION --------
    location = parsed.get("location") or extract_location(full_text)

    # -------- 4. Student slot extraction --------
    student_slots = extract_student_slots(email_obj.get("attachments", []))
    slot_summary = ""
    if student_slots:
        pretty = []
        for slot in student_slots:
            attachment_label = ""
            if slot.get("attachment"):
                attachment_label = f" (source: {Path(slot['attachment']).name})"
            pretty.append(f"{slot.get('snippet')}{attachment_label}".strip())
        slot_summary = "\n".join(pretty).strip()

    # Append slot/location info to action item if found
    action_item = parsed.get("action_item")
    details = []
    if location:
        details.append(f"Venue: {location}")
    if slot_summary:
        details.append(f"Your slot:\n{slot_summary}")

    if details:
        detail_block = "\n".join(details)
        if action_item:
            action_item = action_item.strip() + "\n\n" + detail_block
        else:
            action_item = detail_block

    # -------- 5. Build task --------
    task = {
        "source_id": email_obj.get("id"),
        "raw": email_obj,
        "title": parsed.get("summary") or email_obj.get("subject"),
        "type": parsed.get("type") or "other",
        "course": course,
        "due_date": deadline,
        "summary": parsed.get("summary"),
        "action_item": action_item,
        "location": location,
        "status": "pending",
        "priority": 0.0,
        "student_slot_details": student_slots,
    }

    return task
