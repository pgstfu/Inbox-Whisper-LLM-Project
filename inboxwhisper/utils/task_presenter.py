from datetime import datetime


def format_due_date(due_date: str | None) -> str:
    if not due_date:
        return "No due date"
    try:
        dt = datetime.fromisoformat(due_date)
        return dt.strftime("%a, %d %b %Y %I:%M %p")
    except Exception:
        return due_date


def concise_task_lines(task: dict) -> list[str]:
    title = task.get("title") or "Untitled Task"
    due = format_due_date(task.get("due_date"))
    course = task.get("course") or "General"
    priority = task.get("priority")
    slot = ""
    if task.get("student_slot_details"):
        first = task["student_slot_details"][0]
        slot = first.get("snippet") or ""

    lines = [
        f"• {title}",
        f"  Course: {course} | Due: {due}",
    ]
    if priority is not None:
        lines.append(f"  Priority: {priority}")
    if task.get("location"):
        lines.append(f"  Location: {task['location']}")
    if slot:
        lines.append(f"  Slot: {slot.strip()}")
    if task.get("action_item"):
        lines.append(f"  Action: {task['action_item'].strip()}")
    return lines

