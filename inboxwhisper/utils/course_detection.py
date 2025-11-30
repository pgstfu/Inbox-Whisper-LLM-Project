# utils/course_detection.py
import re

# Known mappings of aliases -> canonical codes for stronger detection
COURSE_MAP = {
    "CSO203": ["operating systems", "cso203", "cso 203", "os assignment"],
    "CSE201": ["data structures", "ds", "cse201", "cse 201"],
    "CSE202": ["algorithms", "algo", "cse202", "cse 202"],
    "CSH205": ["computer networks", "networks", "csh205", "csh 205"],
    "CSD204": ["discrete math", "discrete mathematics", "csd204", "csd 204"],
    "MAT101": ["calculus", "mat101", "mat 101", "math 101"],
    "MAT201": ["linear algebra", "mat201", "mat 201"],
    "PHY101": ["physics", "phy101", "phy 101"],
    "ECO101": ["microeconomics", "eco101", "eco 101"],
    "ENG101": ["english", "english composition", "eng101", "eng 101"],
}

GENERIC_PATTERN = re.compile(r"\b([A-Za-z]{2,4})\s*-?(\d{3})\b")


def detect_course(text):
    if not text:
        return None

    lower_text = text.lower()

    # First pass: check friendly aliases
    for code, aliases in COURSE_MAP.items():
        for alias in aliases:
            if alias in lower_text:
                return code

    # Second pass: generic COURSECODE (e.g., CSO203 or CSO-203)
    match = GENERIC_PATTERN.search(text)
    if match:
        prefix = match.group(1).upper()
        digits = match.group(2)
        return f"{prefix}{digits}"

    return None
