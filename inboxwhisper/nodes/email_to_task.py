# nodes/email_to_task.py
from nodes.email_parser import parse_email
from datetime import datetime
from dateutil import parser as dateparser

def email_to_task_struct(email_obj):
    """
    Run your LLM parser and normalize returned fields into a Task dict.
    """
    # include attachment text inside parsing prompt
    attachment_text = ""
    for att in email_obj.get("attachments", []):
        attachment_text += "\n\n[ATTACHMENT TEXT]\n" + att.get("text", "")
    parsed = parse_email(
        {
            "subject": email_obj.get("subject"),
            "body": email_obj.get("body") + attachment_text
        }
    )

    # If parse failed, create fallback
    if isinstance(parsed, dict) and parsed.get("error"):
        return {
            "source_id": email_obj.get("id"),
            "title": email_obj.get("subject"),
            "raw": email_obj,
            "type": "other",
            "course": None,
            "due_date": None,
            "summary": parsed.get("raw_output") if parsed.get("raw_output") else email_obj.get("body"),
            "action_item": parsed.get("action_item") if isinstance(parsed, dict) else None,
            "priority": 0.0
        }
    # compute due_date normalization
    due = parsed.get("deadline") or parsed.get("date")
    due_dt = None
    if due:
        try:
            due_dt = dateparser.parse(due)
        except Exception:
            due_dt = None

    return {
        "source_id": email_obj.get("id"),
        "title": parsed.get("summary") or email_obj.get("subject"),
        "raw": email_obj,
        "type": parsed.get("type"),
        "course": parsed.get("course"),
        "due_date": due_dt.isoformat() if due_dt else None,
        "summary": parsed.get("summary"),
        "action_item": parsed.get("action_item"),
        "priority": 0.0
    }
