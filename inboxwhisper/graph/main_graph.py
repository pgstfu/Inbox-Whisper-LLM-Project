"""
Main LangGraph pipeline definition for InboxWhisper+.

This module defines a StateGraph with three nodes:
1. ingest: Fetches the latest email from Microsoft 365 via Graph API
2. parse: Converts email + attachments into a structured task (LLM + heuristics)
3. summary: Generates a human-readable summary from the parsed task

The graph uses LangSmith tracing for debugging and monitoring LLM calls.
"""

from langgraph.graph import StateGraph, END
from utils.langsmith_setup import setup_langsmith
from nodes.email_ingest_azure import ingest_email_azure
from nodes.email_to_task import email_to_task_struct
from nodes.summary_node import summarize_parsed
from utils.oauth_callback_server import last_token
from utils.token_manager import get_token_silent
from utils.priority import score_task

# Import LangSmith tracing context manager
from langchain_core.tracers.context import tracing_v2_enabled


def build_inbox_graph():
    """
    Builds and compiles the LangGraph pipeline for email-to-task conversion.
    
    This function:
    1. Sets up LangSmith tracing (for debugging LLM calls in LangSmith dashboard)
    2. Defines the state schema (InboxState) as a dict with email_raw, parsed, summary
    3. Creates three nodes: ingest → parse → summary
    4. Connects nodes with edges to form a linear pipeline
    5. Compiles and returns the graph
    
    The graph can be invoked with: graph.invoke({}, config={"reset": True})
    This starts with an empty state dict, and each node populates/modifies the state.
    
    Returns:
        CompiledStateGraph: A compiled LangGraph ready for execution
    """

    # Initialize LangSmith tracing
    # This enables automatic logging of all LLM calls to LangSmith dashboard
    # Useful for debugging prompt quality, token usage, and response consistency
    setup_langsmith()

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

    # Create a new StateGraph with the defined state schema
    graph = StateGraph(InboxState)

    # -------------------------------
    # Node 1: Email Ingestion
    # -------------------------------
    # This node fetches the latest email from Microsoft 365 using Graph API

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

    # -------------------------------
    # Node 2: LLM Parsing & Task Structuring
    # -------------------------------
    # This node converts raw email into a structured task using LLM + heuristics

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

    # -------------------------------
    # Node 3: LLM Summary Generation
    # -------------------------------
    # This node generates a human-readable summary from the parsed task

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

    # Compile the graph into an executable pipeline
    # This validates the graph structure and prepares it for execution
    return graph.compile()
