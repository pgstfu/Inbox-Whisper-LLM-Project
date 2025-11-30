import streamlit as st
from datetime import datetime, date
import calendar as pycal
from graph.main_graph import build_inbox_graph
from utils.task_db import init_db, list_tasks, update_task_status
from utils.task_presenter import concise_task_lines, format_due_date
from utils.task_chatbot import answer_task_question
from utils.ics_export import tasks_ics_bytes

st.set_page_config(page_title="InboxWhisper+", layout="wide", initial_sidebar_state="expanded")
st.title("📚 InboxWhisper+ — Email Academic Assistant")


@st.cache_resource
def get_graph():
    return build_inbox_graph()


init_db()
graph = get_graph()


@st.cache_data(ttl=30)
def fetch_tasks():
    return list_tasks(limit=500)


def is_completed(task):
    return (task.get("status") or "").lower() == "completed"


def refresh_tasks():
    st.cache_data.clear()


tasks_snapshot = fetch_tasks()
priority_tasks = [t for t in tasks_snapshot if (t.get("priority") or 0) > 0]

tab_analyze, tab_tasks, tab_chat, tab_calendar = st.tabs(
    ["Analyze Email", "Task Board", "Chatbot", "Calendar"]
)


with tab_analyze:
    st.write("Click the button below to fetch your latest email and analyze it.")

    if st.button("🔄 Analyze Latest Email"):
        with st.spinner("Fetching email and analyzing..."):
            result = graph.invoke({}, config={"reset": True})

        refresh_tasks()

        st.success("Analysis complete!")

        st.subheader("📥 Raw Email")
        st.json(result.get("email_raw"))

        parsed = result.get("parsed") or {}

        st.subheader("🧠 Parsed Academic Data")
        st.json(parsed)

        st.subheader("🧾 Final Instructions")
        lines = concise_task_lines(parsed)
        st.code("\n".join(lines), language="markdown")

        if parsed.get("student_slot_details"):
            st.subheader("🪑 Your Exam/Viva Slot")
            st.json(parsed["student_slot_details"])

        st.subheader("📝 Summary")
        st.write(result.get("summary"))


with tab_tasks:
    st.subheader("🎯 Task Board & Checklist")
    if not priority_tasks:
        st.info("No high-priority tasks yet. Run a fetch to populate the database.")
    else:
        include_completed = st.checkbox("Show completed tasks", value=False)
        show_low_priority = st.checkbox("Show low/zero priority tasks", value=False)

        board_candidates = tasks_snapshot if show_low_priority else priority_tasks
        if not include_completed:
            board_candidates = [t for t in board_candidates if not is_completed(t)]

        courses = sorted({t.get("course") for t in board_candidates if t.get("course")})
        types = sorted({(t.get("type") or "other").title() for t in board_candidates})

        col1, col2, col3 = st.columns([1.2, 1, 1.2])
        with col1:
            selected_courses = st.multiselect(
                "Filter by course",
                courses,
                default=courses,
                placeholder="All courses" if courses else "",
            )
        with col2:
            selected_type = st.selectbox("Type", ["All"] + types if types else ["All"])
        with col3:
            keyword = st.text_input("Search keywords").strip().lower()

        def matches_filters(task):
            if selected_courses and task.get("course") and task.get("course") not in selected_courses:
                return False
            if selected_courses and not task.get("course") and courses:
                return False
            if selected_type != "All":
                if (task.get("type") or "other").title() != selected_type:
                    return False
            if keyword:
                blob = " ".join(
                    filter(
                        None,
                        [
                            task.get("title"),
                            task.get("summary"),
                            task.get("action_item"),
                            task.get("course"),
                            task.get("location"),
                        ],
                    )
                ).lower()
                if keyword not in blob:
                    return False
            return True

        filtered = [t for t in board_candidates if matches_filters(t)]
        if not filtered:
            st.info("No tasks match the current filters.")
        else:
            st.caption(f"Showing {len(filtered)} tasks")
            for task in filtered:
                completed = is_completed(task)
                cols = st.columns([0.9, 0.1])
                with cols[0]:
                    st.markdown(f"### {task.get('title') or 'Untitled Task'}")
                    st.write(
                        f"Course: {task.get('course') or 'General'} · "
                        f"Type: {(task.get('type') or 'other').title()} · "
                        f"Priority: {task.get('priority') or 0}"
                    )
                    st.markdown(f"**Due:** {format_due_date(task.get('due_date'))}")
                    if task.get("location"):
                        st.markdown(f"**Venue:** {task.get('location')}")
                    st.markdown(f"**Instructions:** {task.get('action_item') or 'No action provided.'}")
                    if task.get("student_slot_details"):
                        st.markdown("**Slot Details:**")
                        st.json(task["student_slot_details"])
                    with st.expander("More details"):
                        st.write(task.get("summary") or "No summary.")
                        st.caption(
                            f"Source ID: {task.get('source_id')} • Created {task.get('created_at')} • Status: {task.get('status') or 'pending'}"
                        )
                    st.divider()
                with cols[1]:
                    checked = st.checkbox(
                        "Done",
                        value=completed,
                        key=f"task-done-{task['id']}",
                    )
                    if checked != completed:
                        update_task_status(task["id"], "completed" if checked else "pending")
                        refresh_tasks()
                        st.rerun()


with tab_chat:
    st.subheader("💬 Task Chatbot")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        st.chat_message(message["role"]).write(message["content"])

    question = st.chat_input("Ask about your reminders and tasks")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        st.chat_message("user").write(question)
        with st.spinner("Thinking..."):
            answer = answer_task_question(question, tasks_snapshot)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)


with tab_calendar:
    st.subheader("📆 Task Calendar")

    # Collect tasks that have a parsable due date
    dated_tasks: list[dict] = []
    for t in tasks_snapshot:
        due = t.get("due_date")
        if not due:
            continue
        try:
            dt = datetime.fromisoformat(due)
        except Exception:
            continue
        t_copy = dict(t)
        t_copy["_due_dt"] = dt
        dated_tasks.append(t_copy)

    if not dated_tasks:
        st.info("No tasks with valid due dates to display on the calendar.")
    else:
        today = datetime.today().date()
        min_day = min(t["_due_dt"].date() for t in dated_tasks)
        max_day = max(t["_due_dt"].date() for t in dated_tasks)

        col_year, col_month = st.columns(2)
        with col_year:
            year = st.number_input(
                "Year",
                min_value=min_day.year,
                max_value=max_day.year,
                value=today.year,
                step=1,
            )
        with col_month:
            month = st.selectbox(
                "Month",
                options=list(range(1, 13)),
                index=today.month - 1,
                format_func=lambda m: datetime(2000, m, 1).strftime("%B"),
            )

        # Group tasks by day for the selected month
        tasks_by_day: dict[int, list[dict]] = {}
        for t in dated_tasks:
            d = t["_due_dt"].date()
            if d.year == int(year) and d.month == int(month):
                tasks_by_day.setdefault(d.day, []).append(t)

        st.markdown("#### Month view")

        # Weekday header
        weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        header_cols = st.columns(7)
        for i, name in enumerate(weekday_names):
            with header_cols[i]:
                st.markdown(f"**{name}**")

        # Calendar grid
        cal = pycal.Calendar(firstweekday=0)  # Monday
        for week in cal.monthdayscalendar(int(year), int(month)):
            cols = st.columns(7)
            for i, day_num in enumerate(week):
                with cols[i]:
                    if day_num == 0:
                        st.write(" ")
                    else:
                        st.markdown(f"**{day_num}**")
                        day_tasks = tasks_by_day.get(day_num, [])
                        for task in day_tasks:
                            title = task.get("title") or "Untitled Task"
                            st.caption(f"- {title}")

        st.markdown("---")

        # Detailed view for a specific date
        selected_date = st.date_input(
            "View tasks for a specific day",
            value=today if min_day <= today <= max_day else min_day,
            min_value=min_day,
            max_value=max_day,
        )
        detailed_tasks = [
            t for t in dated_tasks if t["_due_dt"].date() == selected_date
        ]

        if not detailed_tasks:
            st.info("No tasks scheduled for this date.")
        else:
            st.markdown(f"### Tasks on {selected_date.strftime('%a, %d %b %Y')}")
            for task in detailed_tasks:
                st.markdown(f"#### {task.get('title') or 'Untitled Task'}")
                st.markdown(f"**Due:** {format_due_date(task.get('due_date'))}")
                st.markdown(
                    f"**Course:** {task.get('course') or 'General'} · "
                    f"Type: {(task.get('type') or 'other').title()} · "
                    f"Priority: {task.get('priority') or 0}"
                )
                if task.get("location"):
                    st.markdown(f"**Venue:** {task.get('location')}")
                if task.get("action_item"):
                    st.markdown(f"**Instructions:** {task.get('action_item')}")
                st.markdown("---")

        st.caption("Need an .ics file instead? Download and import it into any calendar app.")
        ics_bytes = tasks_ics_bytes([t for t in tasks_snapshot if t.get("due_date")])
        st.download_button(
            "⬇️ Download ICS",
            data=ics_bytes,
            file_name="inboxwhisper_tasks.ics",
            mime="text/calendar",
        )
