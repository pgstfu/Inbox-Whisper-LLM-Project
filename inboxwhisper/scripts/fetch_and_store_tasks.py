# scripts/fetch_and_store_tasks.py
import sys, os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(CURRENT_DIR,"..")))

from nodes.email_ingest_batch import ingest_emails_batch
from nodes.email_to_task import email_to_task_struct
from utils.task_db import init_db, insert_or_update, list_tasks
from utils.priority import score_task

def run(limit=200):
    init_db()
    emails = ingest_emails_batch(limit=limit)
    print(f"Fetched {len(emails)} emails")
    count = 0
    for e in emails:
        t = email_to_task_struct(e)
        t["priority"] = score_task(t)
        insert_or_update(t)
        count += 1
    print(f"Stored/updated {count} tasks")

if __name__ == "__main__":
    run()
