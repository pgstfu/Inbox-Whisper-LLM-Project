# utils/attachments.py
import base64
import os
from pathlib import Path
from utils.graph_api import graph_get
from utils.token_manager import get_token_silent
from utils.attachment_text import extract_attachment_text


ATTACH_DIR = Path("attachments")
ATTACH_DIR.mkdir(exist_ok=True)

def fetch_attachments_for_message(message_id):
    """
    Fetches and saves attachments for a specific Outlook message.
    Returns a list of saved file paths.
    """
    token = get_token_silent()
    url = f"https://graph.microsoft.com/v1.0/me/messages/{message_id}/attachments"
    data = graph_get(url, token)

    saved = []
    for att in data.get("value", []):
        # File attachments only
        if att.get("@odata.type") != "#microsoft.graph.fileAttachment":
            continue
        
        name = att.get("name")
        content_bytes = base64.b64decode(att.get("contentBytes"))

        msg_dir = ATTACH_DIR / message_id
        msg_dir.mkdir(parents=True, exist_ok=True)

        file_path = msg_dir / name
        with open(file_path, "wb") as f:
            f.write(content_bytes)

        text_content = extract_attachment_text(file_path)
        saved.append(
            {
                "path": str(file_path),
                "text": text_content
            }
        )

    return saved
