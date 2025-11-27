# utils/ics_export.py
from icalendar import Calendar, Event
from datetime import datetime, timezone
from pathlib import Path

def tasks_to_ics(tasks, out_path="inboxwhisper_tasks.ics"):
    cal = Calendar()
    cal.add('prodid', '-//InboxWhisper+//mx//')
    cal.add('version', '2.0')
    for t in tasks:
        ev = Event()
        title = t.get("title") or (t.get("summary") or "Task")
        ev.add('summary', title)
        due = t.get("due_date")
        if due:
            try:
                dt = datetime.fromisoformat(due)
                ev.add('dtstart', dt)
                ev.add('dtend', dt)
            except Exception:
                pass
        ev.add('description', t.get("summary") or "")
        ev['uid'] = f"inboxwhisper-{t.get('id')}"
        cal.add_component(ev)
    out = Path(out_path)
    out.write_bytes(cal.to_ical())
    return str(out.resolve())
