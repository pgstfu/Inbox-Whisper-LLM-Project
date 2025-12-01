# inboxwhisper/nodes/email_parser.py

from openai import OpenAI
from utils.config import Config
import json
import re

client = OpenAI(api_key=Config.OPENAI_API_KEY)

def normalize_email(email_dict):
    """
    Normalize various email dict formats into:
    {
        "subject": "...",
        "body": "..."
    }
    """

    # Gmail-like (headers nested)
    if "payload" in email_dict and "headers" in email_dict["payload"]:
        headers = email_dict["payload"]["headers"]
        subject = ""
        for h in headers:
            if h["name"].lower() == "subject":
                subject = h["value"]
        body = email_dict.get("body") or email_dict.get("snippet") or ""
        return {"subject": subject, "body": body}

    # Common formats
    subject = (
        email_dict.get("subject")
        or email_dict.get("Subject")
        or email_dict.get("SUBJECT")
        or ""
    )

    body = (
        email_dict.get("body")
        or email_dict.get("Body")
        or email_dict.get("BODY")
        or email_dict.get("text")
        or email_dict.get("snippet")
        or ""
    )

    return {"subject": subject, "body": body}


def parse_email(email_dict):
    """
    Main parser: Takes any email dict, normalizes it,
    and sends it to OpenAI for structured extraction.
    """

    # --- Normalize Keys -------------------------------------
    email = normalize_email(email_dict)
    subject = email["subject"]
    body = email["body"]

    # --- Build prompt ---------------------------------------
    prompt = f"""
You are InboxWhisper+, an academic email assistant for a university student.

Extract structured information from this email and return ONLY valid JSON (no markdown, no backticks).

JSON schema:
{{
  "type": "assignment" | "exam" | "viva" | "announcement" | "task" | "other",
  "course": string | null,
  "date": string | null,
  "time": string | null,
  "deadline": string | null,
  "location": string | null,
  "summary": string,
  "action_item": string,
  "raw_subject": string,
  "raw_body": string
}}

Email:
SUBJECT: {subject}
BODY: {body}

Guidelines:
- Return STRICT JSON.
- "course": extract codes like CSO203, CSE201 if present.
- "location": extract rooms, labs, buildings.
- "summary": 1–2 sentence summary.
- "action_item": clear steps for the student.
"""

    # --- Call OpenAI ----------------------------------------
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
             "content": "Return ONLY raw JSON. No markdown. No ```."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
    )

    content = response.choices[0].message.content

    # --- Clean accidental fences -----------------------------
    content = re.sub(r"```json", "", content, flags=re.IGNORECASE)
    content = re.sub(r"```", "", content).strip()

    # --- Parse JSON safely -----------------------------------
    try:
        parsed = json.loads(content)
        parsed["raw_subject"] = subject
        parsed["raw_body"] = body
        return parsed

    except json.JSONDecodeError as e:
        # Return structured error object
        return {
            "error": "Failed to parse JSON output",
            "exception": str(e),
            "raw_output": content,
            "raw_subject": subject,
            "raw_body": body
        }
