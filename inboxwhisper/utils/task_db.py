# utils/task_db.py
import sqlite3
from pathlib import Path
import json

DB_PATH = Path(__file__).resolve().parent.parent / "inboxwhisper_tasks.db"

def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS tasks(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_id TEXT,
        title TEXT,
        type TEXT,
        course TEXT,
        due_date TEXT,
        priority REAL,
        summary TEXT,
        action_item TEXT,
        raw_json TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """)
    con.commit()
    con.close()

def insert_or_update(task):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    # update if source_id exists
    cur.execute("SELECT id FROM tasks WHERE source_id = ?", (task.get("source_id"),))
    row = cur.fetchone()
    if row:
        cur.execute("""
           UPDATE tasks SET title=?, type=?, course=?, due_date=?, priority=?, summary=?, action_item=?, raw_json=?, updated_at=CURRENT_TIMESTAMP
           WHERE id=?
        """, (task.get("title"), task.get("type"), task.get("course"), task.get("due_date"), task.get("priority"), task.get("summary"), task.get("action_item"), json.dumps(task.get("raw")), row[0]))
    else:
        cur.execute("""
           INSERT INTO tasks(source_id,title,type,course,due_date,priority,summary,action_item,raw_json)
           VALUES(?,?,?,?,?,?,?,?,?)
        """, (task.get("source_id"),task.get("title"),task.get("type"),task.get("course"),task.get("due_date"),task.get("priority"),task.get("summary"),task.get("action_item"),json.dumps(task.get("raw"))))
    con.commit()
    con.close()

def list_tasks(limit=200, order_by="priority DESC, due_date ASC"):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    q = f"SELECT id,source_id,title,type,course,due_date,priority,summary,action_item,created_at FROM tasks ORDER BY {order_by} LIMIT ?"
    cur.execute(q,(limit,))
    rows = cur.fetchall()
    con.close()
    keys = ["id","source_id","title","type","course","due_date","priority","summary","action_item","created_at"]
    return [dict(zip(keys,r)) for r in rows]
