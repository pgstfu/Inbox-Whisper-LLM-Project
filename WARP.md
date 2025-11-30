# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Development environment

- Primary code lives under `inboxwhisper/`.
- Python dependencies are listed in `inboxwhisper/requirements.txt`.
- Configuration is loaded from `.env` in the `inboxwhisper/` directory via `utils.config.Config`.

Typical setup (from repo root):

```bash
cd inboxwhisper
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

### Required external services

- **OpenAI**: `Config.OPENAI_API_KEY` (from `.env`) is required by `nodes.email_parser`, `nodes.summary_node`, and `utils.task_chatbot`.
- **Microsoft 365 / Graph API** (for real email ingestion):
  - `Config.MS_CLIENT_ID`, `Config.MS_CLIENT_SECRET`, `Config.MS_TENANT_ID`, `Config.MS_REDIRECT_URI`, `Config.MS_SCOPES` are used by `utils.azure_auth` and `utils.token_manager`.
  - OAuth callback handling is implemented in `utils.oauth_callback_server.run_callback_server`, which exchanges the authorization code for tokens via `azure_auth.exchange_code_for_token` and stores them via `token_manager.store_initial_token`.
- **LangSmith** (optional but assumed when running the graph):
  - `utils.langsmith_setup.setup_langsmith()` is called from `graph/main_graph.py` and expects a LangSmith/LangChain API key in `.env` (`LANGSMITH_API_KEY` or `LANGCHAIN_API_KEY`).

## Common commands

All commands below assume you start in the repo root and use a virtualenv as shown above.

### Run the Streamlit app (main UI)

```bash
cd inboxwhisper
streamlit run app.py
```

This starts the InboxWhisper+ dashboard with four tabs:
- **Analyze Email**: One-off run of the LangGraph pipeline on the latest email.
- **Task Board**: Task list UI backed by the SQLite DB.
- **Chatbot**: LLM Q&A over existing tasks.
- **Calendar**: Due-date calendar plus `.ics` export.

### Batch-ingest emails into the task database

Fetch a batch of recent emails via Microsoft Graph, parse them into tasks, score priority, and upsert into the SQLite DB:

```bash
cd inboxwhisper
python scripts/fetch_and_store_tasks.py
```

`EMAIL_SOURCE_MODE` is not used here; this always talks to Microsoft Graph via `nodes.email_ingest_batch`.

### Rescan existing DB tasks with updated parsing/attachments

Re-fetch the original emails + attachments for tasks already in the DB, re-run parsing, and recompute priority:

```bash
cd inboxwhisper
python scripts/rescan_existing_db_tasks.py
```

This script uses `utils.token_manager.get_token_silent`, `utils.attachments.fetch_attachments_for_message`, and `nodes.email_to_task.email_to_task_struct` to refresh stored tasks.

### One-off graph invocation (without Streamlit)

To experiment with the LangGraph pipeline directly from the command line:

```bash
cd inboxwhisper
python - << 'PY'
from graph.main_graph import build_inbox_graph

graph = build_inbox_graph()
result = graph.invoke({}, config={"reset": True})
print(result.keys())
PY
```

This runs the full `ingest -> parse -> summary` pipeline once and prints the top-level keys of the resulting state.

### Tests

As of this snapshot there is no `tests/` directory or test runner configuration in the repo.
If you add tests using `pytest`, the usual patterns apply, for example:

- Run all tests: `pytest`
- Run a single test file: `pytest path/to/test_file.py`
- Run a single test: `pytest path/to/test_file.py::test_name`

## High-level architecture

### Overview

InboxWhisper+ turns academic emails into structured tasks with priorities, stores them in a local SQLite database, and exposes them via a Streamlit UI plus LLM-powered helper features.

Core layers:
- **Ingestion layer**: Pulls emails (and attachments) from Microsoft Graph or IMAP/mock.
- **LLM + rules layer**: Uses OpenAI for structured parsing and summarization, plus deterministic heuristics for course detection, deadlines, locations, and student slots.
- **Task model & persistence**: Normalizes everything into a task dict and stores it in SQLite via `utils.task_db`.
- **Presentation layer**: Streamlit app for task management, calendar view, and a small task-aware chatbot.
- **Orchestration**: A LangGraph state machine in `graph/main_graph.py` wires ingestion, parsing, and summarization into a reproducible pipeline with LangSmith tracing.

### Ingestion and preprocessing

- `nodes/email_ingest_azure.ingest_email_azure(access_token)`
  - Uses `utils.graph_api.graph_get` to call `https://graph.microsoft.com/v1.0/me/messages`.
  - Cleans HTML bodies into plain text with `utils.html_cleaner.clean_html_to_text`.
  - Optionally filters by sender domain (default `@snu.edu.in`).
  - Fetches and attaches file attachments via `utils.attachments.fetch_attachments_for_message`, which saves attachments under `attachments/<message_id>/` and extracts text using `utils.attachment_text.extract_attachment_text`.
- `nodes/email_ingest_batch.ingest_emails_batch` performs the same normalization in bulk, using `utils.graph_pagination.graph_get_with_paging` to follow `@odata.nextLink` pages and `utils.token_manager.get_token_silent()` to keep access tokens fresh.
- `nodes/email_ingest.ingest_email` is an older, simpler entry point that switches between `imap` and `mock` based on `Config.EMAIL_SOURCE_MODE` and uses `utils.imap_reader.fetch_latest_email` or `utils.sample_inputs.SAMPLE_EMAIL`.

### LLM parsing and task normalization

- `nodes/email_parser.parse_email(email_dict)`
  - Calls OpenAI Chat Completions (`gpt-4o-mini`) with a strict JSON-only system prompt.
  - Returns a JSON structure describing type, course, deadlines, location, summary, action_item, and raw fields.
  - Includes post-processing to strip accidental ```json code fences and attempts to `json.loads` the final string.
- `nodes/email_to_task.email_to_task_struct(email_obj)` builds the canonical **task dict**:
  - Concatenates the email body and all attachment text into a single `[ATTACHMENT TEXT]`-delimited string.
  - Calls `parse_email` for LLM-based extraction.
  - Refines/overrides fields using deterministic utilities:
    - `utils.course_detection.detect_course` for course codes.
    - `utils.deadline_extractor.extract_deadline` for natural-language due dates.
    - `utils.location_extractor.extract_location` for venues/rooms.
    - `utils.student_slot.extract_student_slots` to find the current student's specific slot from attachment tables (using `utils.student_profile` for normalized name/roll).
  - Normalizes deadlines using `dateutil.parser.parse` to ISO 8601 strings when possible.
  - Augments the `action_item` text with venue and student-slot details if they were found.
  - Returns a dict with keys such as `source_id`, `raw`, `title`, `type`, `course`, `due_date`, `summary`, `action_item`, `location`, `status`, `priority`, and `student_slot_details`.

### Priority scoring

- `utils.priority.score_task(task)` implements heuristic scoring based on:
  - **Sender type** (`classify_sender`): treats `firstname.lastname@snu.edu.in` as professors, short id addresses as students, and filters out clubs/"noreply".
  - **Task type** (assignment vs exam vs other).
  - **Course association** (course-tagged tasks receive extra weight).
  - **Attachment content**: boosts tasks whose attachments mention "name:", "roll", the configured student name, or roll number.
  - **Deadlines**: boosts near-term and overdue tasks relative to current time.
- This score is:
  - Assigned in the LangGraph `parse` node (`graph/main_graph.py`).
  - Recomputed in the batch scripts after re-parsing tasks.

### Orchestration via LangGraph + LangSmith

- `graph/main_graph.build_inbox_graph()` defines a `StateGraph` with an `InboxState` dict containing `email_raw`, `parsed`, and `summary`.
- Nodes:
  1. **`ingest`**: calls `get_token_silent()` and `ingest_email_azure()` to produce `email_raw`.
  2. **`parse`**: converts `email_raw` into a scored task (`parsed`) using `email_to_task_struct` and `score_task`.
  3. **`summary`**: calls `nodes.summary_node.summarize_parsed` to generate a short, human-readable `summary`.
- Each node runs inside `tracing_v2_enabled` spans so that LangSmith captures per-node traces.
- `build_inbox_graph()` returns `graph.compile()`, which is what the Streamlit app and CLI snippet invoke.

### Persistence layer (SQLite)

- `utils.task_db` manages a single SQLite file `inboxwhisper_tasks.db` under `inboxwhisper/`:
  - `init_db()` creates or migrates the `tasks` table and ensures `location` and `status` columns exist.
  - `insert_or_update(task)` upserts tasks keyed by `source_id` and stores the original email object as JSON in `raw_json`.
  - `list_tasks(limit, order_by)` returns task dicts consumed by the UI and scripts.
  - `update_task_status(task_id, status)` toggles completion state.
- Batch scripts interact with this layer:
  - `scripts/fetch_and_store_tasks.run(limit=200)` populates the DB from batches of emails.
  - `scripts/rescan_existing_db_tasks.run()` re-derives tasks from the latest Graph/attachment content for existing DB rows.

### Presentation layer (Streamlit UI)

The main UI is implemented in `app.py` and draws heavily on the lower-level utilities:

- Caches the compiled graph with `@st.cache_resource` and task snapshots with `@st.cache_data`.
- Uses `utils.task_db` for initialization and read/write:
  - `init_db()` at startup.
  - `list_tasks()` to populate views.
  - `update_task_status()` when toggling completion.
- Delegates formatting and LLM calls to utilities:
  - `utils.task_presenter.concise_task_lines` and `format_due_date` for readable task summaries and due dates.
  - `utils.task_chatbot.answer_task_question` for the chatbot tab (re-ranking tasks and querying OpenAI again).
  - `utils.ics_export.tasks_ics_bytes` to generate a downloadable `.ics` file containing dated tasks.
- The calendar tab uses pure Python (`datetime`, `calendar`) and the existing task snapshot to render a month grid and per-day task details.

### Configuration and personalization

- `utils.config.Config` centralizes all environment-driven configuration; `.env` is loaded from the `inboxwhisper/` directory at import time.
- The student identity (name and roll number) used for slot detection and priority boosts is defined in `utils.student_profile` and referenced from `utils.student_slot` and `utils.priority`.
- `Config.EMAIL_SOURCE_MODE` controls whether legacy ingestion (`nodes.email_ingest`) reads from IMAP or mock input; the newer Graph-based flows ignore this flag and rely solely on Microsoft Graph credentials.
