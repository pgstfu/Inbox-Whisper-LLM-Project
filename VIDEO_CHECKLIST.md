# Video Recording Quick Checklist

## Pre-Recording Setup (5 minutes)

- [ ] **Environment Setup**
  - [ ] `.env` file has `OPENAI_API_KEY` and `LANGCHAIN_API_KEY`
  - [ ] Virtual environment activated
  - [ ] Dependencies installed (`pip install -r requirements.txt`)

- [ ] **Authentication**
  - [ ] Microsoft 365 OAuth token cached (run app once, authenticate)
  - [ ] Test "Analyze Latest Email" button works

- [ ] **Data Preparation**
  - [ ] Database has sample tasks (run `python scripts/fetch_and_store_tasks.py` if empty)
  - [ ] At least 3-5 tasks in database for demo

- [ ] **Application Test**
  - [ ] Streamlit app starts: `streamlit run app.py`
  - [ ] All 4 tabs load without errors
  - [ ] Chatbot responds to test questions

## Recording Script (3-5 minutes)

### [0:00-0:30] Introduction
**Say**: "Hi, I'm [Name]. Today I'm presenting InboxWhisper+, an AI-powered assistant that transforms academic emails into structured tasks."

**Show**: 
- Cluttered email inbox
- Problem: "Students receive dozens of emails daily with assignments, exams, and deadlines buried in unstructured text"

### [0:30-2:00] Live Demo - Email Analysis
**Do**:
1. Open Streamlit app (already running)
2. Click "Analyze Email" tab
3. Click "🔄 Analyze Latest Email" button
4. Wait for results (show spinner)
5. Scroll through results:
   - Raw email JSON (show Physics Seminar email)
   - Parsed structured data (type: announcement, location: R-002, date: Dec 3)
   - Final summary
   - Action item: "Attend the seminar on December 3rd at 12:00 PM in R-002"

**Say**: 
- "The system automatically extracts task types, dates, locations, and creates actionable instructions"
- "For this Physics Seminar announcement, it identified the venue R-002, the date and time, and generated clear instructions"
- "Notice it correctly classified this as an 'announcement' type, not an assignment"

### [2:00-3:00] Code Walkthrough
**Do**:
1. Open `inboxwhisper/graph/main_graph.py`
2. Scroll to line 64: `graph = StateGraph(InboxState)`
3. Show node definitions (lines 71-152)
4. Open `inboxwhisper/nodes/email_to_task.py`
5. Show hybrid extraction (lines 60-87)

**Say**:
- "The pipeline uses LangGraph with three sequential nodes: ingest, parse, and summary"
- "We use a hybrid approach: LLM for complex extraction, heuristics for reliable pattern matching"

### [3:00-4:00] Task Board & Chatbot
**Do**:
1. Switch to "Task Board" tab
2. Show filters (course, type, keywords)
3. Filter by type: "Announcement" (or show the Physics seminar task)
4. Show task details (location: R-002, priority: 3.0, action item)
5. Switch to "Chatbot" tab
6. Type: "What seminars or events are coming up?" or "Tell me about the Physics seminar"
7. Show response

**Say**:
- "Tasks are automatically prioritized and can be filtered by course, type, or keywords"
- "This seminar has a priority of 3.0 - the system scores tasks based on importance"
- "The chatbot uses RAG to answer questions about your tasks - it found the Physics seminar and provided details"

### [4:00-4:30] Calendar View
**Do**:
1. Switch to "Calendar" tab
2. Show month view with tasks
3. Navigate to December 2025
4. Click on December 3rd (Physics seminar date)
5. Show detailed task list for that date
6. Click "Download ICS" button

**Say**: 
- "Tasks can be viewed on a calendar - here's December 3rd with the Physics seminar"
- "I can export this to Google Calendar or Outlook, and the seminar will appear with all details: time, location, and description"

### [4:30-5:00] Conclusion
**Say**:
- "InboxWhisper+ demonstrates real-world application of LangGraph, LLMs, and RAG"
- "Key features: hybrid extraction, structured output, priority scoring, and task-aware chatbot"
- "Thank you for watching!"

## Key Files to Have Open

1. `inboxwhisper/graph/main_graph.py` - LangGraph pipeline
2. `inboxwhisper/nodes/email_to_task.py` - Hybrid extraction
3. `inboxwhisper/app.py` - Streamlit UI (optional, for reference)

## Quick Talking Points

- **Hybrid Approach**: "We combine LLM intelligence with heuristic reliability"
- **Structured Output**: "GPT-4o-mini extracts JSON, with post-processing for robustness"
- **RAG Chatbot**: "Searches task database and answers questions using retrieved context"
- **Priority Scoring**: "Automatically surfaces urgent tasks based on deadlines and course importance"
- **End-to-End**: "From email inbox to actionable task board in one click"

## Post-Recording

- [ ] Review video for clarity
- [ ] Check audio quality
- [ ] Verify all demos work in recording
- [ ] Add video link to README.md (line 159)

## Troubleshooting

**If "Analyze Latest Email" fails**:
- Check OAuth token is cached
- Verify Microsoft Graph API credentials
- Check internet connection

**If chatbot doesn't respond**:
- Verify OpenAI API key in `.env`
- Check database has tasks
- Try simpler question

**If Streamlit errors**:
- Check all dependencies installed
- Verify `.env` file exists
- Check Python version (3.10+)

---

**Remember**: Keep it natural, show enthusiasm, and focus on the value proposition!

