from nodes.email_ingest_batch import ingest_emails_batch
from nodes.email_to_task import email_to_task_struct
from utils.priority import score_task
from utils.task_db import insert_or_update, init_db

def run(limit=200):
    init_db()
    emails = ingest_emails_batch(limit=limit)
    print(f"📥 Fetched {len(emails)} emails")

    for e in emails:
        task = email_to_task_struct(e)
        task["priority"] = score_task(task)
        insert_or_update(task)

    print("✔ Done updating DB with attachments, courses, deadlines & priority.")
