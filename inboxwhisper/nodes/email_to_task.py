# nodes/email_to_task.py
"""
This module converts raw email objects (with attachments) into structured task dictionaries.
It combines LLM-based parsing with heuristic extraction utilities to produce a canonical task schema.
"""

from nodes.email_parser import parse_email
from utils.course_detection import detect_course
from utils.deadline_extractor import extract_deadline
from utils.student_slot import extract_student_slots
from utils.location_extractor import extract_location
import dateutil.parser
from pathlib import Path

def email_to_task_struct(email_obj):
    """
    Converts a raw email object (with subject, body, attachments) into a structured task dictionary.
    
    This function implements a hybrid extraction approach:
    1. LLM-based parsing: Uses GPT-4o-mini to extract structured fields from the email text
    2. Heuristic fallbacks: Uses regex/pattern matching for course codes, deadlines, locations
    3. Attachment processing: Extracts text from PDFs/Excel files and searches for student-specific slots
    
    The function combines all extracted information into a single canonical task schema that can be
    stored in the database and displayed in the UI.
    
    Args:
        email_obj: Dict containing:
            - 'id': Email message ID (from Microsoft Graph API)
            - 'subject': Email subject line
            - 'body': Cleaned email body text (HTML stripped)
            - 'attachments': List of dicts, each with 'path' and 'text' keys
    
    Returns:
        dict: Task object with keys:
            - source_id: Original email ID
            - raw: Full email object (for reference)
            - title: Task title (from summary or subject)
            - type: Task type (assignment/exam/viva/announcement/other)
            - course: Course code (e.g., "CSO203")
            - due_date: ISO-formatted deadline string
            - summary: One-sentence summary
            - action_item: Detailed instructions (may include location/slot info)
            - location: Venue/room if mentioned
            - status: "pending" (default)
            - priority: 0.0 (will be computed later by score_task)
            - student_slot_details: List of slot snippets if student's name/roll found in attachments
    """

    # Step 1: Combine email body with all attachment text
    # Attachments are pre-extracted (PDF/Excel text) and stored in attachment['text']
    # We prefix with [ATTACHMENT TEXT] so the LLM knows this content is from attachments
    attachment_text = ""
    for att in email_obj.get("attachments", []):
        attachment_text += "\n\n[ATTACHMENT TEXT]\n" + (att.get("text","") or "")

    subject = email_obj.get("subject") or ""
    full_text = subject + "\n" + email_obj.get("body","") + attachment_text

    # Step 2: LLM-based parsing with full context (subject + body + attachments)
    # The parse_email function uses GPT-4o-mini to extract structured JSON
    parsed = parse_email({
        "subject": subject,
        "body": full_text
    })

    # Step 3: Course detection using hybrid approach
    # First try heuristic pattern matching (faster, more reliable for known codes)
    # Fall back to LLM-extracted course if heuristics fail
    course = detect_course(full_text) or parsed.get("course")

    # Step 4: Deadline extraction with normalization
    # LLM may extract deadline in various formats; we normalize to ISO format
    deadline = parsed.get("deadline") or extract_deadline(full_text)

    if deadline:
        try:
            # Parse any date format into datetime, then convert to ISO string
            dt = dateutil.parser.parse(deadline)
            deadline = dt.isoformat()
        except Exception:
            # If parsing fails, keep original string (will be handled by UI)
            pass

    # Step 5: Location extraction
    # LLM should capture explicit venues, but we also run heuristic extraction as fallback
    location = parsed.get("location") or extract_location(full_text)

    # Step 6: Student-specific slot extraction from attachments
    # This searches attachment text for the student's name or roll number
    # Returns list of snippets (context windows) where matches are found
    student_slots = extract_student_slots(email_obj.get("attachments", []))
    slot_summary = ""
    if student_slots:
        # Format slot snippets for display, including source attachment filename
        pretty = []
        for slot in student_slots:
            attachment_label = ""
            if slot.get("attachment"):
                attachment_label = f" (source: {Path(slot['attachment']).name})"
            pretty.append(f"{slot.get('snippet')}{attachment_label}".strip())
        slot_summary = "\n".join(pretty).strip()

    # Step 7: Enhance action_item with location and slot details
    # If location or slot info was found, append it to the action_item for clarity
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

    # Step 8: Build final task dictionary
    # This is the canonical schema that will be stored in SQLite and displayed in UI
    task = {
        "source_id": email_obj.get("id"),  # For deduplication (upsert by source_id)
        "raw": email_obj,  # Store full email for reference/debugging
        "title": parsed.get("summary") or email_obj.get("subject"),  # Fallback to subject if no summary
        "type": parsed.get("type") or "other",
        "course": course,
        "due_date": deadline,  # ISO format string or None
        "summary": parsed.get("summary"),
        "action_item": action_item,  # May include location/slot details appended above
        "location": location,
        "status": "pending",  # Can be updated to "completed" via UI checkbox
        "priority": 0.0,  # Will be computed by score_task() before DB insertion
        "student_slot_details": student_slots,  # List of slot snippets for UI display
    }

    return task
