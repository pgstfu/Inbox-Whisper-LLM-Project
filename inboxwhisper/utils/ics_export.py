# utils/ics_export.py
from icalendar import Calendar, Event
from datetime import datetime
from pathlib import Path


def build_calendar(tasks):
    cal = Calendar()
    cal.add("prodid", "-//InboxWhisper+//mx//")
    cal.add("version", "2.0")
    for t in tasks:
        ev = Event()
        title = t.get("title") or (t.get("summary") or "Task")
        ev.add("summary", title)
        due = t.get("due_date")
        if due:
            try:
                dt = datetime.fromisoformat(due)
                ev.add("dtstart", dt)
                ev.add("dtend", dt)
            except Exception:
                pass
        description_lines = [
            t.get("summary") or "",
            t.get("action_item") or "",
            f"Course: {t.get('course') or 'General'}",
        ]
        ev.add("description", "\n".join(filter(None, description_lines)))
        ev["uid"] = f"inboxwhisper-{t.get('id')}-{t.get('source_id')}"
        cal.add_component(ev)
    return cal


def tasks_to_ics(tasks, out_path="inboxwhisper_tasks.ics"):
    cal = build_calendar(tasks)
    out = Path(out_path)
    out.write_bytes(cal.to_ical())
    return str(out.resolve())


def tasks_ics_bytes(tasks):
    return build_calendar(tasks).to_ical()
