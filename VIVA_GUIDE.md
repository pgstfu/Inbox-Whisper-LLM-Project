# InboxWhisper+ Viva Guide: Core AI/ML Concepts

This document maps the key AI/ML concepts in your project to exact code locations and explains how they work. Use this as a reference for your viva presentation.

---

## Table of Contents

1. [Prompting](#1-prompting)
2. [Structured Output](#2-structured-output)
3. [Semantic Search](#3-semantic-search)
4. [Retrieval Augmented Generation (RAG)](#4-retrieval-augmented-generation-rag)
5. [Tool Calling LLMs & MCP](#5-tool-calling-llms--mcp)
6. [LangGraph: State, Nodes, Graph](#6-langgraph-state-nodes-graph)

---

## 1. Prompting

Prompting is the core technique for instructing LLMs to perform specific tasks. Your project uses three distinct prompt patterns.

### 1.1 Email → Structured JSON (Schema-Based Extraction)

**Location**: `nodes/email_parser.py` → `parse_email` function

**What it does**: Takes a raw email (subject/body) and asks GPT-4o-mini to extract structured information as JSON.

**Code snippet** (lines 55-101):

```python
# --- Normalize Keys -------------------------------------
email = normalize_email(email_dict)
subject = email["subject"]
body = email["body"]

# --- Build prompt ---------------------------------------
prompt = f"""
You are InboxWhisper+, an academic email assistant for a university student.

Extract structured information from this email and return ONLY valid JSON (no markdown, no backticks).

JSON schema:
{{
  "type": "assignment" | "exam" | "viva" | "announcement" | "task" | "other",
  "course": string | null,
  "date": string | null,
  "time": string | null,
  "deadline": string | null,
  "location": string | null,
  "summary": string,
  "action_item": string,
  "raw_subject": string,
  "raw_body": string
}}

Email:
SUBJECT: {subject}
BODY: {body}

Guidelines:
- Return STRICT JSON.
- "course": extract codes like CSO203, CSE201 if present.
- "location": extract rooms, labs, buildings.
- "summary": 1–2 sentence summary.
- "action_item": clear steps for the student.
"""

# --- Call OpenAI ----------------------------------------
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system",
         "content": "Return ONLY raw JSON. No markdown. No ```."},
        {"role": "user", "content": prompt}
    ],
    temperature=0,
)
```

**Key prompt engineering techniques**:
- **Role-based messaging**: System message enforces strict JSON output format
- **Schema definition**: Explicit JSON schema with all possible fields and types
- **Few-shot-like guidelines**: Examples of what to extract (course codes, locations)
- **Constraints**: "STRICT JSON", "no markdown", "no backticks"
- **Temperature=0**: Deterministic output for structured extraction

---

### 1.2 Task → Natural Language Summary (Generation)

**Location**: `nodes/summary_node.py` → `summarize_parsed` function

**What it does**: Converts structured task data into a human-readable 2-sentence summary.

**Code snippet** (lines 23-44):

```python
prompt = f"""
Create a concise, human-readable summary of this academic task for a student.

Task details:
{parsed}

Summary requirements:
- First sentence: State the task type (assignment/exam/viva/announcement), course code if available, and due date/time.
- Second sentence: Mention the key action required (what the student needs to do).
- If "location" is present, include the venue/room in the first sentence.
- If "student_slot_details" exists, explicitly state the student's specific slot/timing in the second sentence.
- If "priority" is high (>7), mention urgency.

Keep it to exactly 2 sentences. Be specific and actionable.
"""

# Use temperature=0.3 for slight variation while maintaining consistency
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.3
)

return response.choices[0].message.content.strip()
```

**Key prompt engineering techniques**:
- **Task description**: Clear explanation of what the LLM should produce
- **Structured requirements**: Bullet-point list of what each sentence should contain
- **Conditional logic**: "If X exists, include Y" - teaching the model context-aware generation
- **Length constraint**: "exactly 2 sentences"
- **Temperature=0.3**: Allows slight variation for natural language while maintaining consistency

---

### 1.3 Chatbot QA (RAG-Style Prompting)

**Location**: `utils/task_chatbot.py` → `answer_task_question` function

**What it does**: Answers user questions based only on retrieved task data.

**Code snippet** (lines 48-70):

```python
prompt = f"""
You are InboxWhisper+, a personal academic assistant. You have access to the following tasks:

{context_text}

Answer this question using ONLY the provided tasks. If the answer is not in the tasks, say you cannot find it. Keep responses short, actionable, and cite the task number when helpful.

Question: {question}
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant that answers only from provided task data.",
        },
        {"role": "user", "content": prompt},
    ],
    temperature=0.2,
)

return response.choices[0].message.content.strip()
```

**Key prompt engineering techniques**:
- **System-level guardrails**: "answers only from provided task data" prevents hallucination
- **Context injection**: `{context_text}` contains retrieved tasks (RAG pattern)
- **Explicit constraints**: "ONLY the provided tasks", "say you cannot find it" if no answer
- **Citation instruction**: "cite the task number when helpful"
- **Temperature=0.2**: Low temperature for factual, grounded responses

---

## 2. Structured Output

Structured output ensures the LLM returns parseable, consistent data instead of free-form text. Your project implements a two-layer structured output strategy.

### 2.1 Layer 1: LLM → JSON (Primary Extraction)

**Location**: `nodes/email_parser.py` → `parse_email` function

**Process**:

1. **Prompt for strict JSON** (shown in Prompting section above)
2. **Parse and validate the response**:

```python
content = response.choices[0].message.content

# --- Clean accidental fences -----------------------------
content = re.sub(r"```json", "", content, flags=re.IGNORECASE)
content = re.sub(r"```", "", content).strip()

# --- Parse JSON safely -----------------------------------
try:
    parsed = json.loads(content)
    parsed["raw_subject"] = subject
    parsed["raw_body"] = body
    return parsed

except json.JSONDecodeError as e:
    # Return structured error object
    return {
        "error": "Failed to parse JSON output",
        "exception": str(e),
        "raw_output": content,
        "raw_subject": subject,
        "raw_body": body
    }
```

**Why this matters**:
- **Post-processing**: Strips markdown code fences that LLMs sometimes add
- **Validation**: Uses `json.loads()` to ensure valid JSON
- **Error handling**: Returns a structured error object (not a crash) if parsing fails
- **Augmentation**: Adds `raw_subject` and `raw_body` to the parsed result for debugging

---

### 2.2 Layer 2: Canonical Task Schema (Hybrid LLM + Heuristics)

**Location**: `nodes/email_to_task.py` → `email_to_task_struct` function

**What it does**: Combines LLM-extracted JSON with heuristic extraction to produce a consistent internal task format.

**Code snippet** (lines 50-137):

```python
# Step 1: Combine email body with all attachment text
# Attachments are pre-extracted (PDF/Excel text) and stored in attachment['text']
# We prefix with [ATTACHMENT TEXT] so the LLM knows this content is from attachments
attachment_text = ""
for att in email_obj.get("attachments", []):
    attachment_text += "\n\n[ATTACHMENT TEXT]\n" + (att.get("text","") or "")

subject = email_obj.get("subject") or ""
full_text = subject + "\n" + email_obj.get("body","") + attachment_text

# Step 2: LLM-based parsing with full context (subject + body + attachments)
# The parse_email function uses GPT-4o-mini to extract structured JSON
parsed = parse_email({
    "subject": subject,
    "body": full_text
})

# Step 3: Course detection using hybrid approach
# First try heuristic pattern matching (faster, more reliable for known codes)
# Fall back to LLM-extracted course if heuristics fail
course = detect_course(full_text) or parsed.get("course")

# Step 4: Deadline extraction with normalization
# LLM may extract deadline in various formats; we normalize to ISO format
deadline = parsed.get("deadline") or extract_deadline(full_text)

if deadline:
    try:
        # Parse any date format into datetime, then convert to ISO string
        dt = dateutil.parser.parse(deadline)
        deadline = dt.isoformat()
    except Exception:
        # If parsing fails, keep original string (will be handled by UI)
        pass

# Step 5: Location extraction
# LLM should capture explicit venues, but we also run heuristic extraction as fallback
location = parsed.get("location") or extract_location(full_text)

# Step 6: Student-specific slot extraction from attachments
# This searches attachment text for the student's name or roll number
# Returns list of snippets (context windows) where matches are found
student_slots = extract_student_slots(email_obj.get("attachments", []))
slot_summary = ""
if student_slots:
    # Format slot snippets for display, including source attachment filename
    pretty = []
    for slot in student_slots:
        attachment_label = ""
        if slot.get("attachment"):
            attachment_label = f" (source: {Path(slot['attachment']).name})"
        pretty.append(f"{slot.get('snippet')}{attachment_label}".strip())
    slot_summary = "\n".join(pretty).strip()

# Step 7: Enhance action_item with location and slot details
# If location or slot info was found, append it to the action_item for clarity
action_item = parsed.get("action_item")
details = []
if location:
    details.append(f"Venue: {location}")
if slot_summary:
    details.append(f"Your slot:\n{slot_summary}")

if details:
    detail_block = "\n".join(details)
    if action_item:
        action_item = action_item.strip() + "\n\n" + detail_block
    else:
        action_item = detail_block

# Step 8: Build final task dictionary
# This is the canonical schema that will be stored in SQLite and displayed in UI
task = {
    "source_id": email_obj.get("id"),  # For deduplication (upsert by source_id)
    "raw": email_obj,  # Store full email for reference/debugging
    "title": parsed.get("summary") or email_obj.get("subject"),  # Fallback to subject if no summary
    "type": parsed.get("type") or "other",
    "course": course,
    "due_date": deadline,  # ISO format string or None
    "summary": parsed.get("summary"),
    "action_item": action_item,  # May include location/slot details appended above
    "location": location,
    "status": "pending",  # Can be updated to "completed" via UI checkbox
    "priority": 0.0,  # Will be computed by score_task() before DB insertion
    "student_slot_details": student_slots,  # List of slot snippets for UI display
}

return task
```

**Hybrid extraction strategy**:

1. **LLM extraction**: `parse_email()` gets initial structured fields
2. **Heuristic fallbacks**:
   - `detect_course(full_text)`: Regex patterns for course codes (e.g., CSO203, CSE201)
   - `extract_deadline(full_text)`: Date pattern matching
   - `extract_location(full_text)`: Room/building pattern matching
3. **Priority**: Heuristics override LLM if both exist (more reliable for known patterns)
4. **Normalization**: Deadlines → ISO format, locations → standardized strings
5. **Enrichment**: Attachment parsing adds student-specific slot details
6. **Final schema**: Consistent 12-field task object for DB storage

**Why this approach**:
- **LLM is not trusted blindly**: Heuristics catch mistakes and fill gaps
- **Best of both worlds**: LLM handles unstructured variety, heuristics ensure precision
- **Canonical schema**: Downstream code (DB, UI, calendar) always gets the same structure

---

## 3. Semantic Search

Semantic search finds information by meaning rather than exact keyword matches. Your project currently uses keyword search but has designed semantic search upgrades.

### 3.1 Current Implementation: Keyword-Based Retrieval

**Location**: `utils/task_chatbot.py`

**How it works**:

```python
def _task_score(task, question_lower):
    haystack = " ".join(
        filter(
            None,
            [
                task.get("title"),
                task.get("summary"),
                task.get("action_item"),
                task.get("course"),
                task.get("due_date"),
            ],
        )
    ).lower()
    score = 0
    for token in question_lower.split():
        if token in haystack:
            score += 1
    return score


def select_relevant_tasks(question, tasks, limit=5):
    question_lower = question.lower()
    scored = []
    for t in tasks:
        score = _task_score(t, question_lower)
        if score > 0:
            scored.append((score, t))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [t for _, t in scored[:limit]] or tasks[:1]
```

**Algorithm**:
1. Combine all searchable task fields (title, summary, action_item, course, due_date)
2. Tokenize the user's question by spaces
3. Count how many question tokens appear in the task text
4. Sort tasks by token overlap count (descending)
5. Return top 5 tasks

**Limitations**:
- No understanding of synonyms ("exam" vs "test", "deadline" vs "due date")
- No semantic similarity ("machine learning assignment" won't match "AI homework")
- Simple lexical matching

---

### 3.2 Planned Semantic Search (Embeddings-Based)

**Location**: `RAG_IMPLEMENTATIONS.md` (design document)

**Option A: OpenAI Embeddings**:

```python
# utils/task_chatbot_semantic.py
from openai import OpenAI
import numpy as np
from typing import List, Dict

client = OpenAI(api_key=Config.OPENAI_API_KEY)

def get_embedding(text: str) -> List[float]:
    """Get embedding vector for text using OpenAI."""
    response = client.embeddings.create(
        model="text-embedding-3-small",  # or text-embedding-ada-002
        input=text
    )
    return response.data[0].embedding

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def select_relevant_tasks_semantic(question: str, tasks: List[Dict], limit: int = 5) -> List[Dict]:
    """Select tasks using semantic similarity."""
    # Get question embedding
    question_embedding = get_embedding(question)
    
    # Score each task
    scored = []
    for task in tasks:
        # Create searchable text from task fields
        task_text = " ".join([
            task.get("title", ""),
            task.get("summary", ""),
            task.get("action_item", ""),
            task.get("course", ""),
            task.get("location", "")
        ])
        
        # Get task embedding
        task_embedding = get_embedding(task_text)
        
        # Compute similarity
        similarity = cosine_similarity(question_embedding, task_embedding)
        scored.append((similarity, task))
    
    # Sort by similarity and return top N
    scored.sort(key=lambda x: x[0], reverse=True)
    return [task for _, task in scored[:limit]]
```

**How semantic search works**:
1. **Embedding**: Convert text into high-dimensional vectors (e.g., 1536 dimensions for OpenAI)
2. **Vector space**: Semantically similar texts have similar vector representations
3. **Cosine similarity**: Measure angle between question vector and task vectors
4. **Ranking**: Tasks with highest similarity scores are most relevant

**Benefits over keyword search**:
- Handles synonyms: "test" and "exam" have similar embeddings
- Understands context: "AI assignment" matches "machine learning homework"
- Language-agnostic: Works across paraphrasing

---

### 3.3 Hybrid Search (Keyword + Semantic)

**Location**: `RAG_IMPLEMENTATIONS.md`

```python
def select_relevant_tasks_hybrid(question: str, tasks: List[Dict], limit: int = 5, alpha: float = 0.5) -> List[Dict]:
    """
    Hybrid retrieval: combines BM25 (keyword) and semantic search.
    
    Args:
        alpha: Weight for semantic (0-1). 0.5 = equal weight.
    """
    query_tokens = question.lower().split()
    
    # Prepare task texts
    task_texts = []
    for task in tasks:
        task_text = " ".join([
            task.get("title", ""),
            task.get("summary", ""),
            task.get("action_item", ""),
            task.get("course", "")
        ])
        task_texts.append(task_text)
    
    # Semantic scores
    question_embedding = model.encode(question)
    task_embeddings = model.encode(task_texts)
    semantic_scores = np.dot(task_embeddings, question_embedding) / (
        np.linalg.norm(task_embeddings, axis=1) * np.linalg.norm(question_embedding)
    )
    
    # Normalize scores to 0-1
    semantic_scores = (semantic_scores - semantic_scores.min()) / (semantic_scores.max() - semantic_scores.min() + 1e-8)
    
    # BM25 scores
    bm25_scores = [bm25_score(query_tokens, text) for text in task_texts]
    bm25_scores = np.array(bm25_scores)
    if bm25_scores.max() > 0:
        bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)
    
    # Combine scores
    combined_scores = alpha * semantic_scores + (1 - alpha) * bm25_scores
    
    # Get top N
    top_indices = np.argsort(combined_scores)[::-1][:limit]
    return [tasks[i] for i in top_indices]
```

**Why hybrid is best**:
- **Keyword (BM25)**: Good for exact matches (course codes, specific terms)
- **Semantic**: Good for conceptual matches, synonyms, paraphrasing
- **Combined**: Gets precision from keywords + recall from semantics

---

## 4. Retrieval Augmented Generation (RAG)

RAG combines information retrieval with LLM generation to answer questions grounded in specific data. Your chatbot is a working RAG system.

### 4.1 Current RAG Implementation (Keyword Retrieval)

**Location**: `utils/task_chatbot.py` → `answer_task_question`

**Complete RAG pipeline** (lines 39-70):

```python
def answer_task_question(question, tasks):
    # STEP 1: RETRIEVAL
    # Select top 5 relevant tasks using keyword scoring
    relevant = select_relevant_tasks(question, tasks)
    
    # STEP 2: AUGMENTATION (Context Construction)
    # Format retrieved tasks into a structured context
    context_blocks = []
    for idx, task in enumerate(relevant, start=1):
        lines = concise_task_lines(task)
        context_blocks.append(f"Task {idx}:\n" + "\n".join(lines))

    context_text = "\n\n".join(context_blocks) if context_blocks else "No tasks available."

    # STEP 3: GENERATION
    # Build prompt with retrieved context and user question
    prompt = f"""
You are InboxWhisper+, a personal academic assistant. You have access to the following tasks:

{context_text}

Answer this question using ONLY the provided tasks. If the answer is not in the tasks, say you cannot find it. Keep responses short, actionable, and cite the task number when helpful.

Question: {question}
"""

    # Call LLM with augmented context
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that answers only from provided task data.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()
```

**Three-stage RAG pipeline**:

1. **Retrieval**: 
   - Input: User question + full task database
   - Process: `select_relevant_tasks()` scores each task by keyword overlap
   - Output: Top 5 most relevant tasks

2. **Augmentation**:
   - Input: Retrieved tasks
   - Process: Format tasks into readable text blocks with numbering
   - Output: Structured context string injected into prompt

3. **Generation**:
   - Input: Augmented prompt (context + question) + system guardrails
   - Process: LLM generates answer conditioned on provided context
   - Output: Grounded answer (or "cannot find it" if no match)

**Why RAG matters**:
- **Grounding**: LLM answers are based on actual user data, not training data
- **No hallucination**: System message + prompt constraints prevent making up information
- **Scalability**: Can work with databases too large to fit in a single prompt
- **Freshness**: Always uses latest task data without retraining

---

### 4.2 Advanced RAG Variants (Planned)

**Location**: `RAG_IMPLEMENTATIONS.md` documents 8 RAG strategies:

1. **Semantic Search RAG**: Replace keyword retrieval with embeddings
2. **Hybrid Search RAG**: Combine BM25 + embeddings (shown in section 3.3)
3. **Multi-Vector RAG**: Split each task into chunks (title, summary, action_item) and search across all chunks
4. **Query Expansion RAG**: Use LLM to expand user question with synonyms before retrieval
5. **Re-Ranking RAG**: Two-stage retrieval (fast initial + accurate cross-encoder re-ranking)
6. **Contextual RAG**: Use conversation history to improve retrieval
7. **Temporal RAG**: Boost recent tasks in retrieval
8. **Task-Type Specific RAG**: Different retrieval strategies for different question types (deadline vs course vs location)

**Example: Query Expansion RAG**:

```python
def expand_query(question: str) -> str:
    """Use LLM to expand query with synonyms and related terms."""
    prompt = f"""
    Expand this question with synonyms and related academic terms.
    Return the original question plus 2-3 related terms, separated by spaces.
    
    Question: {question}
    
    Expanded query (just the terms, no explanation):
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    
    expanded = response.choices[0].message.content.strip()
    # Combine original and expanded
    return f"{question} {expanded}"

def select_relevant_tasks_expanded(question: str, tasks: List[Dict], limit: int = 5) -> List[Dict]:
    """Retrieval with query expansion."""
    # Expand query
    expanded_query = expand_query(question)
    
    # Use existing semantic or keyword search with expanded query
    return select_relevant_tasks_semantic_local(expanded_query, tasks, limit)
```

This improves recall by including related terms (e.g., "exam" → "test evaluation assessment").

---

## 5. Tool Calling LLMs & MCP

Tool calling allows LLMs to invoke external functions/APIs. Your project uses tools at the Python orchestration layer, not via OpenAI's tool-calling API.

### 5.1 Current Approach: Python-Level Tool Orchestration

**Your project does NOT use**:
- OpenAI's `tools` parameter in `chat.completions.create()`
- MCP (Model Context Protocol) servers
- LLM-driven tool selection

**Instead, you use**:
- LangGraph nodes that call tools as regular Python functions
- Manual orchestration: code explicitly decides which tools to call and when

---

### 5.2 Tools in Your System

Even without LLM tool-calling, your project integrates multiple "tools":

#### Tool 1: Microsoft Graph API

**Location**: `utils/graph_api.py` → `graph_get`

```python
import requests

def graph_get(url, access_token, consistency=False):
    """
    Helper to call Microsoft Graph GET endpoints.
    Set consistency=True when using $search queries.
    """
    headers = {
        "Authorization": f"Bearer {access_token}"
    }

    if consistency:
        headers["ConsistencyLevel"] = "eventual"

    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return resp.json()
```

**Used in**: `nodes/email_ingest_azure.py` → `ingest_email_azure`

```python
# Fetch more emails to check if domain filter is enabled
top_count = max_check if domain_only else 1
url = (
    f"https://graph.microsoft.com/v1.0/me/messages?"
    f"$top={top_count}&"
    "$orderby=receivedDateTime DESC&"
    "$select=id,subject,from,body,bodyPreview,receivedDateTime,hasAttachments"
)

data = graph_get(url, access_token)
```

**Purpose**: Fetches latest email from Microsoft 365 inbox

---

#### Tool 2: Attachment Parser

**Location**: `utils/attachments.py` → `fetch_attachments_for_message`

**Purpose**: Downloads and extracts text from PDF/Excel attachments

Used alongside `utils/attachment_text.py` which contains PDF and Excel text extraction logic.

---

#### Tool 3: Calendar Export (ICS)

**Location**: `utils/ics_export.py` → `tasks_ics_bytes`

**Purpose**: Converts tasks to iCalendar format for importing into Google Calendar, Outlook, etc.

---

#### Tool 4: Task Database (SQLite)

**Location**: `utils/task_db.py`

Functions:
- `init_db()`: Create tasks table
- `upsert_task()`: Insert or update task
- `list_tasks()`: Retrieve all tasks
- `update_task_status()`: Mark task as completed

---

### 5.3 How Tools Are Orchestrated

**Location**: `graph/main_graph.py` → LangGraph nodes

Each node calls tools as needed:

```python
def node_ingest(state):
    """
    Ingests the latest email from Microsoft 365.
    """
    with tracing_v2_enabled("email_ingest_azure"):
        # Tool 1: OAuth token manager
        access_token = get_token_silent()

        # Tool 2: Microsoft Graph API
        email = ingest_email_azure(access_token)
        
        # Store in state for next node
        state["email_raw"] = email
        return state

def node_parse(state):
    """
    Parses email into structured task.
    """
    with tracing_v2_enabled("email_parse_llm"):
        # Tool 3: LLM + heuristic extractors + attachment parser
        task = email_to_task_struct(state["email_raw"])
        
        # Tool 4: Priority scorer
        task["priority"] = score_task(task)
        
        state["parsed"] = task
        return state
```

---

### 5.4 Why Not LLM Tool-Calling?

**Current architecture**:
- Tools are deterministic and always called in the same order
- No need for LLM to decide "should I fetch email now?"
- LangGraph provides explicit control flow

**Future enhancement**:
You could add OpenAI tool-calling for dynamic scenarios like:
- User asks "What's my next deadline?" → LLM calls `get_next_deadline_tool()`
- User asks "Add a reminder for tomorrow" → LLM calls `create_task_tool(date="2024-12-17")`

**Example of how tool-calling would work** (not implemented):

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_tasks_by_course",
            "description": "Retrieve all tasks for a specific course code",
            "parameters": {
                "type": "object",
                "properties": {
                    "course_code": {
                        "type": "string",
                        "description": "Course code like CSO203 or CSE201"
                    }
                },
                "required": ["course_code"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Show me all CSO203 tasks"}],
    tools=tools,
    tool_choice="auto"
)

# LLM would return: tool_calls=[{"function": {"name": "get_tasks_by_course", "arguments": '{"course_code": "CSO203"}'}}]
```

**For viva**: Explain that your tools exist as Python functions, orchestrated by LangGraph. This is a valid architecture; LLM tool-calling would be useful for more dynamic, conversational scenarios.

---

## 6. LangGraph: State, Nodes, Graph

LangGraph is a framework for building stateful, multi-step LLM workflows. Your project uses it to orchestrate the email→task→summary pipeline.

### 6.1 State Definition

**Location**: `graph/main_graph.py` → `build_inbox_graph`

**Code** (lines 48-61):

```python
# Define the state schema as a TypedDict-like class
# LangGraph uses this to validate state structure across nodes
class InboxState(dict):
    """
    State schema for the InboxWhisper+ pipeline.
    
    Fields:
        email_raw: Dict containing fetched email (subject, body, attachments, from, etc.)
        parsed: Dict containing structured task (type, course, due_date, action_item, etc.)
        summary: String containing human-readable summary of the task
    """
    email_raw: dict
    parsed: dict
    summary: str
```

**What is State?**
- **Shared memory** that persists across all nodes in the graph
- Each node can **read from** and **write to** the state
- LangGraph automatically passes state between nodes
- State schema defines what fields are expected

**Your state fields**:
1. `email_raw`: Output of the ingest node (raw email from Graph API)
2. `parsed`: Output of the parse node (structured task with course, deadline, etc.)
3. `summary`: Output of the summary node (2-sentence natural language summary)

---

### 6.2 Graph Initialization

**Code** (lines 63-64):

```python
# Create a new StateGraph with the defined state schema
graph = StateGraph(InboxState)
```

**What this does**:
- Creates a new graph instance
- Tells LangGraph to validate state against `InboxState` schema
- Prepares an empty graph (no nodes or edges yet)

---

### 6.3 Node Definitions

Your pipeline has **three nodes**, each wrapped as a function that receives and returns state.

#### Node 1: Email Ingestion (`node_ingest`)

**Code** (lines 71-94):

```python
def node_ingest(state):
    """
    Ingests the latest email from Microsoft 365.
    
    This node:
    1. Gets a fresh OAuth access token (handles token refresh automatically)
    2. Calls ingest_email_azure() to fetch the latest email via Graph API
    3. Stores the raw email in state["email_raw"]
    
    The email object includes: id, subject, from, body (cleaned HTML), attachments, receivedDateTime
    """
    with tracing_v2_enabled("email_ingest_azure"):
        # Get OAuth token (automatically refreshes if expired)
        access_token = get_token_silent()

        # Fetch latest email from Microsoft 365
        email = ingest_email_azure(access_token)
        
        # Store in state for next node
        state["email_raw"] = email
        return state

graph.add_node("ingest", node_ingest)
graph.set_entry_point("ingest")  # Pipeline starts here
```

**Responsibilities**:
- OAuth authentication via `get_token_silent()`
- Microsoft Graph API call via `ingest_email_azure()`
- HTML cleaning and attachment downloading
- Writes `state["email_raw"]`

---

#### Node 2: LLM Parsing & Task Structuring (`node_parse`)

**Code** (lines 101-126):

```python
def node_parse(state):
    """
    Parses email into structured task and computes priority score.
    
    This node:
    1. Calls email_to_task_struct() which:
       - Combines email body with attachment text
       - Uses GPT-4o-mini to extract structured fields (type, course, deadline, location, etc.)
       - Applies heuristic fallbacks (course detection, deadline extraction, location extraction)
       - Extracts student-specific slots from attachments
    2. Computes priority score using score_task() (based on sender, course, deadline, urgency keywords)
    3. Stores the structured task in state["parsed"]
    """
    with tracing_v2_enabled("email_parse_llm"):
        # Convert email to structured task (LLM + heuristics)
        task = email_to_task_struct(state["email_raw"])
        
        # Compute priority score (higher = more urgent)
        task["priority"] = score_task(task)
        
        # Store in state for next node
        state["parsed"] = task
        return state

graph.add_node("parse", node_parse)
graph.add_edge("ingest", "parse")  # Linear flow: ingest → parse
```

**Responsibilities**:
- Reads `state["email_raw"]`
- LLM-based structured extraction (see Structured Output section)
- Heuristic course/deadline/location extraction
- Attachment text parsing and student slot detection
- Priority scoring
- Writes `state["parsed"]`

---

#### Node 3: LLM Summary Generation (`node_summary`)

**Code** (lines 133-152):

```python
def node_summary(state):
    """
    Generates a concise human-readable summary from the parsed task.
    
    This node:
    1. Calls summarize_parsed() which uses GPT-4o-mini to generate a 2-sentence summary
    2. The summary includes: task type, course, due date, location, and key action
    3. Stores the summary string in state["summary"]
    """
    with tracing_v2_enabled("summary_llm"):
        # Generate human-readable summary from parsed task
        summary = summarize_parsed(state["parsed"])
        
        # Store in state
        state["summary"] = summary
        return state

graph.add_node("summary", node_summary)
graph.add_edge("parse", "summary")  # Linear flow: parse → summary
graph.add_edge("summary", END)  # Pipeline ends here
```

**Responsibilities**:
- Reads `state["parsed"]`
- LLM-based natural language generation (see Prompting section 1.2)
- Writes `state["summary"]`
- Marks end of pipeline

---

### 6.4 Graph Compilation

**Code** (lines 154-156):

```python
# Compile the graph into an executable pipeline
# This validates the graph structure and prepares it for execution
return graph.compile()
```

**What compilation does**:
- Validates that all edges connect valid nodes
- Checks that entry point is set
- Checks that all execution paths lead to `END`
- Returns a `CompiledStateGraph` ready for execution

---

### 6.5 Graph Execution

**Location**: `app.py` (Streamlit UI)

**Code** (lines 14-53):

```python
@st.cache_resource
def get_graph():
    return build_inbox_graph()

graph = get_graph()

# ... later in the UI ...

with tab_analyze:
    st.write("Click the button below to fetch your latest email and analyze it.")

    if st.button("🔄 Analyze Latest Email"):
        with st.spinner("Fetching email and analyzing..."):
            # Execute the graph with empty initial state
            result = graph.invoke({}, config={"reset": True})

        refresh_tasks()

        st.success("Analysis complete!")
```

**Execution flow**:

1. **User clicks button** in Streamlit UI
2. **`graph.invoke({}, config={"reset": True})`** is called:
   - `{}`: Empty initial state (LangGraph will populate it)
   - `config={"reset": True}`: Start fresh (don't reuse previous state)
3. **LangGraph executes nodes in order**:
   - `ingest` → `parse` → `summary`
   - Each node updates the state
4. **Final state is returned**:
   ```python
   {
       "email_raw": {...},  # Raw email from Graph API
       "parsed": {...},     # Structured task
       "summary": "..."     # 2-sentence summary
   }
   ```
5. **Task is saved to database** (happens inside `email_to_task_struct` via DB upsert)
6. **UI refreshes** to show the new task

---

### 6.6 LangSmith Tracing

**Location**: `graph/main_graph.py` → `build_inbox_graph`

**Code** (lines 43-46):

```python
# Initialize LangSmith tracing
# This enables automatic logging of all LLM calls to LangSmith dashboard
# Useful for debugging prompt quality, token usage, and response consistency
setup_langsmith()
```

And wrapped around each node:

```python
with tracing_v2_enabled("email_ingest_azure"):
    # node code...
```

**What LangSmith tracing provides**:
- **LLM call logging**: Every `client.chat.completions.create()` is logged
- **Prompt inspection**: See exact prompts sent to GPT-4o-mini
- **Response analysis**: Token counts, latency, model outputs
- **State transitions**: Track how state changes across nodes
- **Debugging**: Identify which node/LLM call failed or produced bad output

**Example trace view**:
```
Graph Execution
├── email_ingest_azure (0.8s)
│   └── graph_get (0.5s)
├── email_parse_llm (2.3s)
│   ├── OpenAI chat.completions.create (1.9s)
│   │   Prompt: "Extract structured information from this email..."
│   │   Response: {"type": "assignment", "course": "CSO203", ...}
│   └── Heuristics (0.4s)
└── summary_llm (1.2s)
    └── OpenAI chat.completions.create (1.0s)
        Prompt: "Create a concise, human-readable summary..."
        Response: "CSO203 assignment due on Dec 20th. Submit via email."
```

---

### 6.7 Visualization of the Graph

**Your graph structure**:

```
┌─────────┐
│ START   │
└────┬────┘
     │
     ▼
┌──────────────┐
│   ingest     │ ← Entry point (set_entry_point)
│ (node_ingest)│
└──────┬───────┘
       │
       ▼
┌──────────────┐
│    parse     │
│ (node_parse) │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   summary    │
│(node_summary)│
└──────┬───────┘
       │
       ▼
   ┌──────┐
   │ END  │
   └──────┘
```

**Code that creates this structure**:

```python
graph.add_node("ingest", node_ingest)
graph.add_node("parse", node_parse)
graph.add_node("summary", node_summary)

graph.set_entry_point("ingest")
graph.add_edge("ingest", "parse")
graph.add_edge("parse", "summary")
graph.add_edge("summary", END)
```

---

## Summary: How to Present in Viva

### Quick Talking Points

**1. Prompting**:
- "We use three distinct prompt patterns: schema-based extraction for emails, constrained generation for summaries, and RAG-style QA for the chatbot."
- "Each prompt is carefully engineered with role-based messaging, explicit constraints, and schema definitions."
- Files: `nodes/email_parser.py`, `nodes/summary_node.py`, `utils/task_chatbot.py`

**2. Structured Output**:
- "We implement two-layer structured output: LLM produces JSON, then we validate and enhance it with heuristics."
- "This hybrid approach combines LLM flexibility with rule-based reliability for course codes, deadlines, and locations."
- Files: `nodes/email_parser.py`, `nodes/email_to_task.py`

**3. Semantic Search**:
- "Currently using keyword-based retrieval, but we've designed and documented semantic search upgrades using embeddings."
- "The design doc includes hybrid BM25+semantic, multi-vector, query expansion, and re-ranking strategies."
- Files: `utils/task_chatbot.py` (current), `RAG_IMPLEMENTATIONS.md` (planned)

**4. RAG**:
- "Our chatbot is a working RAG system: retrieve relevant tasks, inject into prompt, generate grounded answers."
- "Three-stage pipeline: retrieval → augmentation → generation, with strict constraints to prevent hallucination."
- File: `utils/task_chatbot.py`

**5. Tool Calling**:
- "We use Python-level tool orchestration via LangGraph nodes, not OpenAI's tool-calling API."
- "Tools include Microsoft Graph API, attachment parsers, calendar export, and SQLite database."
- Files: `utils/graph_api.py`, `utils/attachments.py`, `utils/ics_export.py`, `utils/task_db.py`

**6. LangGraph**:
- "Three-node stateful pipeline: ingest → parse → summary, with state carrying email_raw, parsed task, and summary."
- "LangGraph orchestrates the workflow, LangSmith traces all LLM calls for debugging."
- File: `graph/main_graph.py`

---

## Appendix: Full File References

| Concept | Primary Files |
|---------|---------------|
| **Prompting** | `nodes/email_parser.py`, `nodes/summary_node.py`, `utils/task_chatbot.py` |
| **Structured Output** | `nodes/email_parser.py`, `nodes/email_to_task.py` |
| **Semantic Search** | `utils/task_chatbot.py`, `RAG_IMPLEMENTATIONS.md` |
| **RAG** | `utils/task_chatbot.py`, `RAG_IMPLEMENTATIONS.md` |
| **Tool Calling** | `utils/graph_api.py`, `utils/attachments.py`, `utils/ics_export.py`, `utils/task_db.py` |
| **LangGraph** | `graph/main_graph.py`, `app.py` |
| **Heuristics** | `utils/course_detection.py`, `utils/deadline_extractor.py`, `utils/location_extractor.py`, `utils/student_slot.py` |
| **UI** | `app.py` |
| **Database** | `utils/task_db.py` |
| **OAuth/Auth** | `utils/azure_auth.py`, `utils/token_manager.py`, `utils/oauth_callback_server.py` |

---

**Good luck with your viva! 🚀**
