# InboxWhisper+: LangGraph-based Academic Email Task Manager

## Overview

This project builds an AI assistant that turns raw academic emails into a structured task board using LangGraph and OpenAI.  
It ingests emails (via Microsoft 365 / Graph), cleans and normalizes the content (including attachments), and then uses an LLM with carefully designed prompts to extract:

- Task type (assignment, exam, announcement, etc.)
- Course code
- Deadlines and locations
- Actionable next steps
- Student-specific details (e.g., slot/roll-based info from attachments)

The extracted information is normalized into a task schema, stored in a local SQLite database, and exposed through a Streamlit UI with:

- An “Analyze Email” one-shot pipeline run
- A “Task Board” of all tasks with priorities
- A “Chatbot” that answers questions about your tasks (RAG-style over the task DB)
- A “Calendar” view with `.ics` export for due-date tasks

Internally, the core pipeline is orchestrated using LangGraph (nodes for ingest → parse → summarize) and instrumented with LangSmith for debugging and tracing.

## Reason for picking up this project

This project is tightly aligned with the topics from MAT496:

- **Prompting**:  
  Custom system and user prompts are used to (1) force strictly JSON-only structured output from the LLM for parsing emails, and (2) generate concise human-readable summaries.

- **Structured Output**:  
  The parsing node expects a rigid JSON schema (type, course, deadlines, location, summary, action_item, etc.), and the code post-processes / validates this JSON, embodying the “structured output” ideas from class.

- **Semantic Search**:  
  For the task-aware chatbot, the system ranks/selects the most relevant tasks based on semantic similarity between the user’s question and the stored task titles/descriptions before passing them to the LLM.

- **Retrieval Augmented Generation (RAG)**:  
  The chatbot uses the retrieved subset of tasks (from the SQLite DB) as context and then asks the LLM to answer questions grounded in these retrieved tasks — a classic RAG loop over a small, local corpus.

- **Tool-calling style LLM usage**:  
  The LLM is invoked as part of a broader toolchain: ingestion utilities, parsing functions, priority scoring, DB utilities, and calendar export functions all act like “tools” that the graph orchestrates around the LLM calls.

- **LangGraph (State, Nodes, Graph)**:  
  The main pipeline is a LangGraph `StateGraph` with nodes for:
  - `ingest` (fetch and normalize an email),
  - `parse` (LLM + heuristics → structured task),
  - `summary` (short natural language summary).  
  The compiled graph is used both from the command line and inside the Streamlit app.

- **LangSmith**:  
  LangSmith tracing is enabled for the LangGraph pipeline and run on example emails to inspect prompt quality, structured output consistency, and node-level behavior.

Overall, this project takes the techniques from MAT496 and applies them to a real, high-friction problem for students: managing academic email overload.

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

## Conclusion

I had planned to build an end-to-end LangGraph-based agent that can take unstructured academic emails as input and output a structured, prioritized task board with a supporting chatbot and calendar view.  
I believe I have largely achieved this goal: the core pipeline (ingest → parse → summarize), the heuristics for deadlines/locations/course codes, the SQLite-backed task DB, and the multi-tab Streamlit interface are all working together as intended. The remaining work is mostly polish (prompt tuning, UX improvements, and recording the final demo video), so I am reasonably satisfied with the technical outcome and with how well it reflects the main topics from MAT496.
