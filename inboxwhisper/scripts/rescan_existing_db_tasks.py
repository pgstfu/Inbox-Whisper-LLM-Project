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
from utils.html_cleaner import clean_html_to_text

def get_email_by_id(message_id):
    access = get_token_silent()
    url = f"https://graph.microsoft.com/v1.0/me/messages/{message_id}?$select=subject,body,bodyPreview,from,receivedDateTime,hasAttachments"
    return graph_get(url, access)

def log(msg):
    print(msg, flush=True)


def run():
    log("🔍 Rescanning existing tasks in DB...")
    tasks = list_tasks(limit=1000)
    total = len(tasks)

    if not total:
        log("No tasks found in DB. Nothing to rescan.")
        return

    updated = 0
    for idx, t in enumerate(tasks, start=1):
        msg_id = t.get("source_id")
        if not msg_id:
            log(f"[{idx}/{total}] Skipping task without source_id (id={t.get('id')})")
            continue

        log(f"[{idx}/{total}] Fetching email {msg_id} ...")

        # 1. Fetch the original email body (but NOT all emails)
        mail = get_email_by_id(msg_id)

        if "error" in mail:
            log(f"   ✖ Graph API error for {msg_id}: {mail.get('error')}")
            continue

        # 2. Fetch attachments for this email only
        attachments = fetch_attachments_for_message(msg_id)
        log(f"   📎 Retrieved {len(attachments)} attachments")

        # 3. Build a normalized email object
        email_obj = {
            "id": msg_id,
            "from": mail.get("from", {}).get("emailAddress", {}).get("address"),
            "subject": mail.get("subject", ""),
            "body": clean_html_to_text(
                mail.get("body", {}).get("content", "") or mail.get("bodyPreview", "")
            ),
            "received": mail.get("receivedDateTime"),
            "attachments": attachments
        }

        log("   🧠 Parsing email + attachments via LLM ...")
        try:
            # 4. Convert to task structure (LLM parse + attachment text)
            new_task = email_to_task_struct(email_obj)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            log(f"   ✖ Failed to parse {msg_id}: {exc}")
            continue

        # 5. Recompute priority
        new_task["priority"] = score_task(new_task)

        # 6. Update in DB
        insert_or_update(new_task)
        updated += 1
        log(f"   ✅ Updated task for {msg_id} (priority={new_task['priority']})")

    log(f"✔ Updated {updated} tasks with new attachment/text/priority info.")

if __name__ == "__main__":
    run()
