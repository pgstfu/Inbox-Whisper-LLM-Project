# scripts/rescan_existing_db_tasks.py

import sys, os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(CURRENT_DIR, "..")))

from utils.task_db import list_tasks, insert_or_update
from utils.token_manager import get_token_silent
from utils.attachments import fetch_attachments_for_message
from utils.priority import score_task
from nodes.email_to_task import email_to_task_struct
from utils.graph_api import graph_get

def get_email_by_id(message_id):
    access = get_token_silent()
    url = f"https://graph.microsoft.com/v1.0/me/messages/{message_id}?$select=subject,body,bodyPreview,from,receivedDateTime,hasAttachments"
    return graph_get(url, access)

def run():
    print("🔍 Rescanning existing tasks in DB...")
    tasks = list_tasks(limit=1000)

    updated = 0
    for t in tasks:
        msg_id = t.get("source_id")
        if not msg_id:
            continue

        # 1. Fetch the original email body (but NOT all emails)
        mail = get_email_by_id(msg_id)

        if "error" in mail:
            continue

        # 2. Fetch attachments for this email only
        attachments = fetch_attachments_for_message(msg_id)

        # 3. Build a normalized email object
        email_obj = {
            "id": msg_id,
            "from": mail.get("from", {}).get("emailAddress", {}).get("address"),
            "subject": mail.get("subject", ""),
            "body": mail.get("body", {}).get("content", "") or mail.get("bodyPreview", ""),
            "received": mail.get("receivedDateTime"),
            "attachments": attachments
        }

        # 4. Convert to task structure (LLM parse + attachment text)
        new_task = email_to_task_struct(email_obj)

        # 5. Recompute priority
        new_task["priority"] = score_task(new_task)

        # 6. Update in DB
        insert_or_update(new_task)
        updated += 1

    print(f"✔ Updated {updated} tasks with new attachment/text/priority info.")

if __name__ == "__main__":
    run()
