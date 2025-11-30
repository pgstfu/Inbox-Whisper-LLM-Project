# utils/deadline_extractor.py

import re
from dateutil import parser
from datetime import datetime, timedelta

DATE_PATTERNS = [
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",           # 05/03/2025
    r"\b\d{1,2}\s+[A-Za-z]{3,9}\b",                 # 4 Feb
    r"\b[A-Za-z]{3,9}\s+\d{1,2}\b",                 # Feb 4
    r"\b\d{1,2}[a-z]{2}\s+[A-Za-z]{3,9}\b",         # 4th Feb
    r"\b(?:tomorrow|today|tonight|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", # natural
    r"\b\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?\b",        # 11:59 PM
]

def extract_deadline(text):
    t = text.lower()

    # Look for keyword scopes
    if "due" in t or "deadline" in t or "submit" in t:
        for pattern in DATE_PATTERNS:
            match = re.search(pattern, t, re.IGNORECASE)
            if match:
                try:
                    dt = parser.parse(match.group())
                    return dt.isoformat()
                except:
                    continue

    return None
    