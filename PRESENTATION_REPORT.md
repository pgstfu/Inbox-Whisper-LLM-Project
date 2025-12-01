# InboxWhisper+ Presentation Report
## 3-5 Minute Video Demo Guide

---

## 🎯 Project Overview

**InboxWhisper+** is an AI-powered academic email assistant that automatically transforms unstructured emails into a structured, prioritized task board. Built with LangGraph, OpenAI GPT-4o-mini, and Streamlit, it solves the problem of email overload for students by:

- **Automatically extracting** task details (assignments, exams, deadlines, locations)
- **Prioritizing** tasks based on urgency, course importance, and deadlines
- **Organizing** everything into a searchable, filterable task board
- **Answering questions** about tasks using RAG-powered chatbot
- **Syncing** with calendar via ICS export

---

## 🏗️ Architecture Highlights

### Core Pipeline: LangGraph StateGraph

The system uses a **3-node LangGraph pipeline**:

```
Email Ingestion → LLM Parsing → Summary Generation
     ↓                ↓                ↓
  Microsoft      GPT-4o-mini      Human-readable
  Graph API      + Heuristics      Summary
```

**Key Design Pattern**: Hybrid LLM + Heuristics
- **LLM** handles ambiguous language and complex extraction
- **Heuristics** provide fast, reliable fallbacks for known patterns

---

## 📋 Video Script Outline (3-5 minutes)

### **Segment 1: Introduction & Problem (30 seconds)**
- Show a cluttered inbox with academic emails
- Explain the pain point: "Students receive dozens of emails daily with assignments, exams, and deadlines buried in unstructured text"
- **Demo**: Open email inbox showing mixed academic emails

### **Segment 2: Live Demo - Email Analysis (1.5 minutes)**
- **Open Streamlit app**: `streamlit run app.py`
- **Navigate to "Analyze Email" tab**
- **Click "Analyze Latest Email"**
- **Show results**:
  - Raw email JSON
  - Parsed structured data (type, course, deadline, location)
  - Final summary
  - Student slot details (if applicable)

**Key Points to Highlight**:
- Automatic extraction of course codes (e.g., CSO203)
- Deadline normalization
- Location extraction (rooms, venues)
- Student-specific slot detection from attachments

### **Segment 3: Code Walkthrough - LangGraph Pipeline (1 minute)**
- **Open**: `inboxwhisper/graph/main_graph.py`
- **Show**: StateGraph definition (lines 64-156)
- **Highlight**:
  - Three nodes: `ingest`, `parse`, `summary`
  - State schema: `InboxState` with `email_raw`, `parsed`, `summary`
  - LangSmith tracing integration
- **Open**: `inboxwhisper/nodes/email_to_task.py`
- **Show**: Hybrid extraction approach (lines 60-87)
  - LLM parsing with `parse_email()`
  - Heuristic fallbacks: `detect_course()`, `extract_deadline()`, `extract_location()`

### **Segment 4: Task Board & Features (1 minute)**
- **Navigate to "Task Board" tab**
- **Show**:
  - Filterable task list (by course, type, keywords)
  - Priority scores
  - Completion checkboxes
  - Task details (deadlines, locations, instructions)
- **Navigate to "Chatbot" tab**
- **Ask a question**: "What assignments are due this week?"
- **Show**: RAG-powered response using task database

### **Segment 5: Calendar & Export (30 seconds)**
- **Navigate to "Calendar" tab**
- **Show**: Month view with tasks on due dates
- **Click**: "Download ICS" button
- **Explain**: Can import into Google Calendar, Outlook, etc.

### **Segment 6: Technical Highlights (30 seconds)**
- **Show**: LangSmith dashboard (if available) showing traces
- **Mention**: 
  - Structured output with JSON parsing
  - RAG implementation for chatbot
  - SQLite persistence
  - Microsoft Graph API integration

---

## 💻 Code Snippets to Show

### 1. LangGraph Pipeline Definition
**File**: `inboxwhisper/graph/main_graph.py`

```python
# Show lines 64-93: Graph creation and ingest node
graph = StateGraph(InboxState)

def node_ingest(state):
    with tracing_v2_enabled("email_ingest_azure"):
        access_token = get_token_silent()
        email = ingest_email_azure(access_token)
        state["email_raw"] = email
        return state

graph.add_node("ingest", node_ingest)
graph.set_entry_point("ingest")
```

### 2. Hybrid LLM + Heuristics Parsing
**File**: `inboxwhisper/nodes/email_to_task.py`

```python
# Show lines 60-87: Hybrid extraction
# LLM-based parsing
parsed = parse_email({
    "subject": subject,
    "body": full_text
})

# Heuristic fallbacks
course = detect_course(full_text) or parsed.get("course")
deadline = parsed.get("deadline") or extract_deadline(full_text)
location = parsed.get("location") or extract_location(full_text)
```

### 3. Structured Output with JSON Parsing
**File**: `inboxwhisper/nodes/email_parser.py` (mention, don't show full file)

**Key Point**: System prompt explicitly forbids markdown, post-processing strips accidental code fences

### 4. RAG Chatbot Implementation
**File**: `inboxwhisper/utils/task_chatbot.py` or `task_chatbot_advanced.py`

**Key Point**: 
- Retrieval: Keyword/semantic search over task database
- Augmentation: Top 5 relevant tasks as context
- Generation: GPT-4o-mini answers from provided context

---

## 🎬 Demo Checklist

### Pre-Recording Setup
- [ ] Ensure `.env` file has valid API keys (OpenAI, LangSmith)
- [ ] Microsoft 365 OAuth token is cached (run app once to authenticate)
- [ ] Database has some sample tasks (run `python scripts/fetch_and_store_tasks.py` if needed)
- [ ] Streamlit app runs without errors
- [ ] Have a sample email ready in inbox for live analysis

### During Recording
- [ ] **Show face** (if required) at beginning/end
- [ ] **Clear audio** - explain what you're doing
- [ ] **Screen recording** of code and UI
- [ ] **Highlight cursor** on important code sections
- [ ] **Smooth transitions** between segments
- [ ] **Keep pace** - aim for 3-5 minutes total

### Key Moments to Capture
1. ✅ **Email analysis in action** - show the button click and results
2. ✅ **Code structure** - show LangGraph pipeline definition
3. ✅ **Task board filtering** - demonstrate interactivity
4. ✅ **Chatbot Q&A** - show RAG in action
5. ✅ **Calendar view** - show visual organization

---

## 🔑 Key Features to Emphasize

### 1. **Intelligent Email Parsing**
- Extracts: task type, course code, deadline, location, instructions
- Handles attachments (PDFs, Excel files)
- Finds student-specific slots from attachment tables

### 2. **Hybrid Extraction Approach**
- **LLM** for complex, ambiguous cases
- **Heuristics** for reliable pattern matching
- **Best of both worlds**: accuracy + consistency

### 3. **Priority Scoring**
- Considers: sender type, task type, course, deadline proximity, urgency keywords
- Automatically surfaces high-priority tasks

### 4. **RAG-Powered Chatbot**
- Searches task database semantically
- Answers questions about assignments, deadlines, courses
- Multiple retrieval strategies (hybrid, semantic, rerank, temporal)

### 5. **Calendar Integration**
- Month view of all tasks
- ICS export for external calendar apps
- Date-based filtering and navigation

---

## 📊 Technical Stack Summary

| Component | Technology |
|-----------|-----------|
| **LLM** | OpenAI GPT-4o-mini |
| **Orchestration** | LangGraph (StateGraph) |
| **Tracing** | LangSmith |
| **Email API** | Microsoft Graph API |
| **Database** | SQLite |
| **UI** | Streamlit |
| **Calendar Export** | iCalendar (.ics) |

---

## 🎯 Key Takeaways for Audience

1. **Real-world application** of LangGraph and LLMs
2. **Hybrid approach** (LLM + heuristics) for reliability
3. **Structured output** with robust JSON parsing
4. **RAG implementation** for task-aware chatbot
5. **End-to-end pipeline** from email to actionable tasks

---

## 📝 Presentation Tips

### Do's ✅
- Start with the problem (email overload)
- Show live demo first (most engaging)
- Explain code architecture clearly
- Highlight unique features (hybrid extraction, RAG chatbot)
- Keep it concise (3-5 minutes)

### Don'ts ❌
- Don't get lost in code details
- Don't skip the demo
- Don't rush through key features
- Don't forget to show the UI working

---

## 🔗 Additional Resources

- **README.md**: Full project documentation
- **WARP.md**: Development environment guide
- **RAG_QUICKSTART.md**: RAG implementation details
- **LangSmith Dashboard**: Live tracing and debugging

---

## 📹 Video Structure Summary

```
[0:00-0:30]   Introduction & Problem Statement
[0:30-2:00]   Live Demo: Email Analysis
[2:00-3:00]   Code Walkthrough: LangGraph Pipeline
[3:00-4:00]   Task Board & Chatbot Demo
[4:00-4:30]   Calendar & Export
[4:30-5:00]   Technical Highlights & Conclusion
```

---

## 🎬 Final Notes

This report serves as both a **presentation guide** and **documentation** for your video demo. Use it to:
- Plan your recording session
- Ensure you cover all key features
- Reference code locations during walkthrough
- Maintain consistent messaging

**Good luck with your presentation!** 🚀

