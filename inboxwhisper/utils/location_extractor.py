import re

LOCATION_PATTERNS = [
    r"(?:venue|location|room|hall|block)\s*[:\-]\s*(.+)",
    r"(?:exam|viva|assessment)\s+venue\s+is\s+(.+)",
    r"report\s+to\s+(.+?)(?:\n|$)",
]


def extract_location(text: str | None) -> str | None:
    if not text:
        return None

    cleaned = text.replace("\r", "")
    lines = cleaned.split("\n")

    for line in lines:
        lower = line.lower()
        for pattern in LOCATION_PATTERNS:
            match = re.search(pattern, lower)
            if match:
                captured = match.group(1).strip(" .")
                # Return the original line segment to keep capitalization
                start = lower.find(match.group(1))
                if start != -1:
                    return line[start : start + len(match.group(1))].strip()
                return captured

    # fallback: look for keywords on same line as building names
    for line in lines:
        if any(keyword in line.lower() for keyword in ["hf", "audi", "lab", "block", "room"]):
            return line.strip()

    return None



