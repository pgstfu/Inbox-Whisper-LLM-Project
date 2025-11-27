# utils/html_cleaner.py

from bs4 import BeautifulSoup
import re

def clean_html_to_text(html):
    if not html:
        return ""

    soup = BeautifulSoup(html, "html.parser")

    # Remove script and style elements
    for tag in soup(["script", "style"]):
        tag.decompose()

    text = soup.get_text(separator="\n")

    # Remove tracking pixels, very short lines, disclaimers
    cleaned_lines = []
    for line in text.split("\n"):
        l = line.strip()

        # skip blank lines
        if not l:
            continue

        # skip signatures
        if any(x in l.lower() for x in [
            "sent from my iphone", 
            "best regards", 
            "confidentiality notice",
            "please do not print",
            "unsubscribe"
        ]):
            continue
        
        # skip lines that are junk / 1-2 chars
        if len(l) < 3:
            continue

        cleaned_lines.append(l)

    text = "\n".join(cleaned_lines)

    # remove multiple blank lines
    text = re.sub(r"\n\s*\n", "\n\n", text)

    return text.strip()
