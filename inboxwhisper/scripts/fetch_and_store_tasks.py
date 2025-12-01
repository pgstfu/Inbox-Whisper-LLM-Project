# scripts/fetch_and_store_tasks.py

import sys
import os

# Add parent directory to Python path so imports work
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(CURRENT_DIR, "..")))

from nodes.email_ingest_batch import ingest_emails_batch
from nodes.email_to_task import email_to_task_struct
from utils.priority import score_task
from utils.task_db import insert_or_update, init_db

def run(limit=200):
    """
    Fetch emails from Microsoft 365, parse them into tasks, and store in SQLite database.
    
    Args:
        limit: Maximum number of emails to process (default 200)
    """
    init_db()
    emails = ingest_emails_batch(limit=limit)
    print(f"📥 Fetched {len(emails)} emails")

    for e in emails:
        task = email_to_task_struct(e)
        task["priority"] = score_task(task)
        insert_or_update(task)

    print("✔ Done updating DB with attachments, courses, deadlines & priority.")

if __name__ == "__main__":
    run()
