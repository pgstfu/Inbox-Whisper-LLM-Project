"""
Utilities to extract student-specific slots/timings from attachment text.
Looks for either the configured student name or roll number and returns the
relevant line-level context so the UI can surface exact timings.
"""
from typing import List, Dict

from .student_profile import normalized_name, normalized_roll


def _context_window(lines, idx, window=1):
    start = max(0, idx - window)
    end = min(len(lines), idx + window + 1)
    snippet = "\n".join(lines[start:end]).strip()
    return snippet


def extract_student_slots(attachments: List[Dict], context_window: int = 1):
    """
    Returns a list of snippets that mention the student's name or roll number.
    Each entry contains the attachment path and the matching text slice.
    """
    name = normalized_name()
    roll = normalized_roll()
    matches = []

    for att in attachments or []:
        text = (att.get("text") or "").strip()
        if not text:
            continue

        lines = text.splitlines()
        lines_lower = [line.lower() for line in lines]
        seen_indices = set()

        for idx, line_lower in enumerate(lines_lower):
            if name in line_lower or roll in line_lower:
                if idx in seen_indices:
                    continue
                snippet = _context_window(lines, idx, window=context_window)
                matches.append(
                    {
                        "attachment": att.get("path"),
                        "snippet": snippet,
                    }
                )
                # Avoid duplicating when name & roll appear on same line
                seen_indices.add(idx)

    return matches

