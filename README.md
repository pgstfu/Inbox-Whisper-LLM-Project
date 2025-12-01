# InboxWhisper+: LangGraph-based Academic Email Task Manager

## Overview

This project builds an AI assistant that turns raw academic emails into a structured task board using LangGraph and OpenAI.  
It ingests emails (via Microsoft 365 / Graph), cleans and normalizes the content (including attachments), and then uses an LLM with carefully designed prompts to extract:

- **Task type** (assignment, exam, viva, announcement, etc.)
- **Course code** (e.g., CSO203, CSE201) via pattern matching + LLM
- **Deadlines and locations** (explicit venues/rooms for exams/vivas)
- **Actionable next steps** (detailed instructions for the student)
- **Student-specific details** (slot/roll-based info extracted from PDF/Excel attachments)

The extracted information is normalized into a canonical task schema, stored in a local SQLite database, and exposed through a Streamlit UI with:

- **Analyze Email**: One-shot pipeline run that fetches the latest email and displays parsed results
- **Task Board**: Filterable list of all tasks with priorities, course filters, and completion checkboxes
- **Chatbot**: RAG-style Q&A that searches the task database and answers questions using GPT-4o-mini
- **Calendar**: Month view of tasks with due dates, plus `.ics` export for Google Calendar integration

### Architecture

The core pipeline is a **LangGraph StateGraph** with three sequential nodes:

1. **`ingest` node**: Fetches the latest email from Microsoft 365 via Graph API
   - Gets OAuth token (auto-refreshes if expired)
   - Downloads email body (HTML cleaned to plain text)
   - Downloads and extracts text from attachments (PDF, Excel, Word)
   - Returns email object with `id`, `subject`, `body`, `attachments`, `from`, `receivedDateTime`

2. **`parse` node**: Converts raw email into structured task
   - Combines email body + attachment text into full context
   - Calls GPT-4o-mini with structured output prompt to extract: type, course, deadline, location, summary, action_item
   - Applies heuristic fallbacks: `detect_course()`, `extract_deadline()`, `extract_location()`
   - Extracts student-specific slots from attachments using `extract_student_slots()` (searches for name/roll number)
   - Computes priority score using `score_task()` (based on sender type, course, deadline proximity, urgency keywords)
   - Returns task dict with all fields normalized

3. **`summary` node**: Generates human-readable summary
   - Calls GPT-4o-mini to convert structured task into 2-sentence natural language summary
   - Includes course, due date, location, and key action in the summary

The pipeline is instrumented with **LangSmith tracing** for debugging prompt quality, token usage, and response consistency.

## Reason for picking up this project

This project is tightly aligned with the topics from MAT496:

### 1. **Prompt Engineering & Structured Output**
- **Custom prompts**: The `parse_email()` function uses a detailed prompt that specifies an exact JSON schema (type, course, deadline, location, summary, action_item). The system message explicitly forbids markdown/code fences to ensure pure JSON output.
- **Post-processing**: The code strips any accidental markdown wrappers (```json ... ```) and validates JSON parsing, returning error dicts if parsing fails. This embodies the "structured output" pattern from class.
- **Prompt refinement**: The prompts were iteratively refined based on test emails to better capture locations (venues/rooms) and student-specific details.

### 2. **Retrieval Augmented Generation (RAG)**
- **Task retrieval**: The chatbot (`answer_task_question()`) implements a RAG workflow:
  1. **Retrieval**: Searches the SQLite task database using keyword matching (scores tasks by token overlap with the question)
  2. **Augmentation**: Selects top 5 most relevant tasks and formats them as context
  3. **Generation**: Passes the context + question to GPT-4o-mini with instructions to answer only from the provided tasks
- This is a classic RAG loop over a small, local corpus (the task database).

### 3. **Hybrid LLM + Heuristic Approach**
- **LLM for complex extraction**: GPT-4o-mini handles ambiguous cases (e.g., "viva will be held in Room 205" → extracts location)
- **Heuristics for reliability**: Pattern matching for course codes (`detect_course()`), deadline extraction (`extract_deadline()`), and location extraction (`extract_location()`) provide fast, deterministic fallbacks
- **Best of both worlds**: LLM captures nuanced information, heuristics ensure consistency for known patterns

### 4. **LangGraph (State, Nodes, Graph)**
- **StateGraph architecture**: The main pipeline is a LangGraph `StateGraph` with:
  - **State schema**: `InboxState` dict with `email_raw`, `parsed`, `summary` fields
  - **Nodes**: Three sequential nodes (`ingest` → `parse` → `summary`)
  - **Edges**: Linear flow with `graph.add_edge()` connecting nodes
  - **Compilation**: `graph.compile()` validates structure and returns executable pipeline
- **State management**: Each node reads from and writes to the shared state dict, allowing data to flow through the pipeline

### 5. **LangSmith Tracing & Observability**
- **Tracing enabled**: Each node wraps its LLM calls with `tracing_v2_enabled("node_name")` to log to LangSmith
- **Debugging workflow**: LangSmith dashboard shows prompt inputs, LLM responses, token usage, and latency for each node
- **Iterative improvement**: Tracing data was used to refine prompts and debug structured output inconsistencies

### 6. **Tool-calling Style LLM Usage**
- **LLM as one tool in a pipeline**: The LLM is invoked as part of a broader toolchain:
  - Ingestion utilities (Graph API, HTML cleaning, attachment extraction)
  - Parsing functions (LLM + heuristics)
  - Priority scoring (`score_task()`)
  - Database utilities (SQLite insert/update/query)
  - Calendar export (`ics_export.py`)
- **Orchestration**: LangGraph orchestrates these tools, with the LLM being just one component in the workflow

Overall, this project takes the techniques from MAT496 and applies them to a real, high-friction problem for students: managing academic email overload. The hybrid approach (LLM + heuristics) ensures both accuracy and reliability, while the RAG chatbot and structured output patterns demonstrate practical applications of the course concepts.

## Setup & Installation

### Prerequisites
- Python 3.10+ with virtual environment support
- Microsoft 365 account (for email access via Graph API)
- OpenAI API key (for GPT-4o-mini)
- (Optional) Google Calendar API credentials (for calendar sync)

### Installation Steps

1. **Clone the repository and navigate to the project directory:**
   ```bash
   cd inboxwhisper
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   Create a `.env` file in the `inboxwhisper/` directory with:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   LANGCHAIN_API_KEY=your_langsmith_api_key_here  # Optional, for tracing
   ```

5. **Authenticate with Microsoft 365:**
   - Run the app once: `streamlit run app.py`
   - Click "Analyze Latest Email" - this will trigger OAuth flow
   - Complete Microsoft 365 authentication in the browser
   - Token will be stored automatically for future use

6. **(Optional) Set up Google Calendar integration:**
   - Download `client_secret.json` from Google Cloud Console
   - Place it in the `inboxwhisper/` directory
   - Click "Connect Google Calendar" in the Calendar tab to authenticate

### Running the Application

**Start the Streamlit UI:**
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` with four tabs:
- **Analyze Email**: Fetch and parse the latest email
- **Task Board**: View, filter, and manage all tasks
- **Chatbot**: Ask questions about your tasks
- **Calendar**: View tasks on a calendar and export to `.ics`

**Batch processing (fetch multiple emails):**
```bash
python -m scripts.fetch_and_store_tasks
```

**Rescan existing tasks (re-parse with new code):**
```bash
python -m scripts.rescan_existing_db_tasks
```

## Video Summary Link:

[ADD YOUR 3–5 MIN VIDEO LINK HERE AFTER RECORDING]  
(e.g., unlisted YouTube / Google Drive link)

## Plan

I plan to execute these steps to complete my project.

- [DONE] Step 1: Set up the Python project environment, install LangGraph / LangChain / Streamlit / OpenAI, and verify a minimal “hello world” LLM call.
- [DONE] Step 2: Design the LangGraph state schema and implement the `ingest` node to fetch emails (via Microsoft Graph or mock input) and normalize bodies + attachments into a clean text representation.
- [DONE] Step 3: Implement the `parse` node that calls the OpenAI model with a carefully engineered prompt to produce strictly JSON structured output (task type, course, deadlines, location, summary, action item, etc.), plus post-processing / validation of the JSON.
- [DONE] Step 4: Add heuristic post-processing utilities (course detection, deadline extraction, location extraction, student-slot detection) and a priority scoring function to turn parsed emails into a canonical “task” object.
- [DONE] Step 5: Create the persistence layer using SQLite (task table, upsert logic, listing and status update functions) so that parsed tasks are stored and can be reloaded across sessions.
- [DONE] Step 6: Build the Streamlit UI with multiple tabs: Analyze Email (one-off graph run), Task Board (list + prioritize tasks), Chatbot (ask questions about tasks using a RAG-style workflow), and Calendar (due-date calendar + `.ics` export).
- [DONE] Step 7: Integrate LangSmith tracing for the LangGraph pipeline and run several example emails to debug prompt quality, structured output consistency, and node-level behavior.
- [TODO] Step 8: Record a 3–5 minute summary video (showing my face, explaining inputs/outputs, briefly walking through how the agent works, and demonstrating an example run) and add the link above.
- [TODO] Step 9: Final polishing: refine prompts based on a few more test emails, clean up any rough edges in the UI, and ensure the README and code comments are clear enough that I can explain every line in a viva.

## Key Design Decisions

### Why Hybrid LLM + Heuristics?
- **LLM strengths**: Handles ambiguous language, extracts nuanced information (e.g., "viva will be held in Room 205")
- **Heuristic strengths**: Fast, deterministic, reliable for known patterns (course codes like CSO203, date formats)
- **Combination**: Best of both worlds - LLM for complex cases, heuristics for consistency

### Why Structured Output with Post-processing?
- **Problem**: LLMs sometimes wrap JSON in markdown code fences despite instructions
- **Solution**: Explicit system message + regex post-processing to strip markdown
- **Result**: Reliable JSON parsing even when LLM doesn't follow instructions perfectly

### Why RAG for Chatbot Instead of Fine-tuning?
- **Fine-tuning limitations**: Would require labeled Q&A pairs, doesn't adapt to new tasks
- **RAG advantages**: Can answer questions about any task in the database, no training needed
- **Implementation**: Simple keyword matching for retrieval, LLM for generation - works well for small corpus

### Why LangGraph Instead of Simple Function Calls?
- **State management**: LangGraph handles state passing between nodes automatically
- **Observability**: Built-in LangSmith integration for tracing
- **Extensibility**: Easy to add new nodes (e.g., notification node, calendar sync node)
- **Error handling**: Can add conditional edges based on state (e.g., skip summary if parsing fails)

## Conclusion

I had planned to build an end-to-end LangGraph-based agent that can take unstructured academic emails as input and output a structured, prioritized task board with a supporting chatbot and calendar view.  
I believe I have largely achieved this goal: the core pipeline (ingest → parse → summarize), the heuristics for deadlines/locations/course codes, the SQLite-backed task DB, and the multi-tab Streamlit interface are all working together as intended. The prompts have been refined based on test emails to better capture locations and student-specific details. The code is well-commented for viva explanation, and the README provides clear architecture documentation. The remaining work is mostly polish (recording the final demo video), so I am reasonably satisfied with the technical outcome and with how well it reflects the main topics from MAT496.
