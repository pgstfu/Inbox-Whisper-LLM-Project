# InboxWhisper+ Video Presentation Speech
## 3-5 Minute Script

---

## 📧 Your Actual Email Example

**Subject**: "Physics Seminar: Experimental Nuclear Science: Probing Matter's Core | 3 Dec | 12:00 PM | R002"

**Key Details to Highlight**:
- **Type**: Announcement
- **Location**: R-002
- **Date**: December 3, 2025
- **Time**: 12:00-01:00 PM
- **Priority**: 3.0
- **Action Item**: "Attend the seminar on December 3rd at 12:00 PM in R-002"

**What the system extracted**:
- Correctly identified as "announcement" type
- Extracted venue (R-002) from the email
- Parsed date and time information
- Created actionable instructions
- Generated a concise summary

---

## [0:00-0:30] Opening & Problem Statement

Hi, I'm [Your Name], and today I'm excited to show you **InboxWhisper+**, an AI-powered assistant that solves a real problem every student faces: email overload.

As a student, I receive dozens of academic emails every day—assignments, exam schedules, viva notifications, deadline reminders. The information I need is buried in unstructured text, scattered across multiple emails, and often hidden in PDF attachments. It's overwhelming, and I was constantly missing deadlines or forgetting important details.

That's why I built InboxWhisper+. It automatically transforms your cluttered inbox into a clean, organized task board with priorities, deadlines, and actionable instructions.

---

## [0:30-2:00] Live Demo - Email Analysis

Let me show you how it works. Here's the Streamlit interface. I'll click "Analyze Latest Email" to fetch and process my most recent academic email.

[Click button, wait for results]

Perfect! In just a few seconds, the system has processed this Physics Seminar announcement. Let me show you what it extracted:

- **Identified the task type**—correctly classified as an "announcement"
- **Extracted the location**—R-002, the seminar room
- **Parsed the date and time**—December 3rd, 2025, from 12:00 to 1:00 PM
- **Created a clear action item**—"Attend the seminar on December 3rd at 12:00 PM in R-002"
- **Generated a summary**—condensed the entire email into key information

Notice how the system understood the context—this is a seminar invitation, not an assignment. It extracted the venue, time, and created actionable instructions automatically. The raw email was just unstructured text, but now it's a structured task I can track and manage.

This is all happening automatically. No manual data entry, no copying and pasting. The AI understands the context and extracts structured information from unstructured emails.

---

## [2:00-3:00] Code Walkthrough - Architecture

Let me show you how this works under the hood. The core is a **LangGraph pipeline** with three sequential nodes.

[Open graph/main_graph.py]

Here's the state graph definition. We have three nodes: **ingest**, **parse**, and **summary**. The ingest node fetches emails from Microsoft 365 using the Graph API. The parse node is where the magic happens—it uses GPT-4o-mini to extract structured information.

[Open nodes/email_to_task.py]

But here's the key design decision: we use a **hybrid approach**. The LLM handles complex, ambiguous language—like "the seminar is scheduled for 3 December 2025 from 12:00-01:00 PM in R-002." But for known patterns—like course codes or date formats—we use fast, reliable heuristics. This gives us the best of both worlds: the intelligence of an LLM with the reliability of deterministic pattern matching.

Notice the LangSmith tracing integration—every LLM call is logged so we can debug and refine prompts. This is crucial for production systems.

---

## [3:00-4:00] Task Board & Chatbot

Now let's see how these parsed tasks are organized. Here's the Task Board.

[Switch to Task Board tab]

All my tasks are here, automatically prioritized. I can filter by course, by type—assignments, exams, vivas, announcements—or search by keywords. Each task shows the location, priority score, and detailed instructions.

[Filter by type: "Announcement" or show the Physics seminar task]

See how easy it is to filter? The priority scores help me know what's urgent. This seminar has a priority of 3.0—tasks from official communication channels, with specific dates and locations, get appropriate priority scores. The system helps me stay on top of important events and deadlines.

[Switch to Chatbot tab]

But here's my favorite feature—the chatbot. Let me ask it a question.

[Type: "What seminars or events are coming up?" or "Tell me about the Physics seminar"]

This uses **RAG—Retrieval Augmented Generation**. The system searches my task database, finds the most relevant tasks—like this Physics seminar—and then uses GPT-4o-mini to generate a natural language answer based on that context. It's not just a keyword search—it understands the question and provides a helpful response with the date, time, and location.

---

## [4:00-4:30] Calendar & Export

Finally, let's look at the calendar view.

[Switch to Calendar tab]

All my tasks are organized by date. I can see at a glance what's coming up this month. For example, here's December 3rd—that's when the Physics seminar is scheduled. Click on any date to see detailed task information.

And here's the export feature—I can download an ICS file and import it directly into Google Calendar, Outlook, or any calendar app. The seminar will appear on my calendar with all the details: time, location, and description. My tasks are now synced across all my devices.

---

## [4:30-5:00] Conclusion & Key Takeaways

So to summarize, InboxWhisper+ demonstrates several important concepts:

**First**, it's a real-world application of LangGraph and LLMs. We're not just calling an API—we're orchestrating a multi-step pipeline with state management, error handling, and observability.

**Second**, the hybrid LLM-plus-heuristics approach ensures both accuracy and reliability. The LLM handles ambiguity, while heuristics provide fast, deterministic fallbacks.

**Third**, we use structured output with robust JSON parsing. Even when the LLM doesn't follow instructions perfectly, our post-processing ensures reliable extraction.

**Fourth**, the RAG chatbot shows how retrieval-augmented generation works in practice—searching a local database and generating context-aware answers.

And **finally**, this is an end-to-end system—from email ingestion to task management to calendar integration. It's not just a demo; it's a working tool that solves a real problem.

The code is well-documented, uses LangSmith for tracing, and follows best practices for production LLM applications. I'm proud of how it demonstrates the techniques we learned in class applied to a practical, high-impact use case.

Thank you for watching! I'm happy to answer any questions.

---

## Alternative Shorter Version (if running long)

### [0:00-0:20] Opening

Hi, I'm [Your Name]. Today I'm showing you **InboxWhisper+**—an AI assistant that turns academic emails into organized tasks automatically.

### [0:20-1:30] Demo

Let me show you. [Click Analyze Email] In seconds, it extracts task types, dates, locations, and creates actionable instructions. For this Physics seminar, it identified the venue, time, and generated a clear action item automatically.

### [1:30-2:30] Architecture

The core is a LangGraph pipeline with three nodes. We use a hybrid approach—LLM for complex extraction, heuristics for reliability. [Show code briefly]

### [2:30-3:30] Features

The Task Board organizes everything with priorities and filters. The chatbot uses RAG to answer questions about my tasks. [Show both]

### [3:30-4:00] Conclusion

This demonstrates LangGraph, structured output, RAG, and hybrid extraction in a real-world application. Thank you!

---

## Tips for Delivery

1. **Be conversational**—speak naturally, not robotically
2. **Show enthusiasm**—you built something cool!
3. **Pause for effect**—let the demo results load before explaining
4. **Point with cursor**—highlight what you're talking about
5. **Smile**—it comes through in your voice
6. **Practice once**—run through the demo before recording

---

## Key Phrases to Remember

- "Hybrid approach: LLM intelligence with heuristic reliability"
- "Structured output with robust JSON parsing"
- "RAG-powered chatbot searches and generates context-aware answers"
- "End-to-end pipeline from email to actionable tasks"
- "Real-world application of LangGraph and LLMs"

---

Good luck with your presentation! 🚀

