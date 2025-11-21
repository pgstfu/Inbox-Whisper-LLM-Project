from langgraph.graph import StateGraph, END
from utils.langsmith_setup import setup_langsmith
from nodes.email_ingest_azure import ingest_email_azure
from nodes.email_parser import parse_email
from nodes.summary_node import summarize_parsed
from utils.oauth_callback_server import last_token
from utils.token_manager import get_token_silent

# NEW: import tracer context
from langchain_core.tracers.context import tracing_v2_enabled


def build_inbox_graph():

    # Enable LangSmith
    setup_langsmith()

    # Define state structure
    class InboxState(dict):
        email_raw: dict
        parsed: dict
        summary: str

    graph = StateGraph(InboxState)

    # -------------------------------
    # Node 1: Email ingestion
    # -------------------------------

    def node_ingest(state):
        with tracing_v2_enabled("email_ingest_azure"):
            # NEW — get fresh token automatically (no kernel restart needed)
            access_token = get_token_silent()

            email = ingest_email_azure(access_token)
            state["email_raw"] = email
            return state


    graph.add_node("ingest", node_ingest)
    graph.set_entry_point("ingest")

    # -------------------------------
    # Node 2: LLM Parsing
    # -------------------------------
    def node_parse(state):
        with tracing_v2_enabled("email_parse_llm"):
            parsed = parse_email(state["email_raw"])
            state["parsed"] = parsed
            return state

    graph.add_node("parse", node_parse)
    graph.add_edge("ingest", "parse")

    # -------------------------------
    # Node 3: LLM Summary
    # -------------------------------
    def node_summary(state):
        with tracing_v2_enabled("summary_llm"):
            summary = summarize_parsed(state["parsed"])
            state["summary"] = summary
            return state

    graph.add_node("summary", node_summary)
    graph.add_edge("parse", "summary")
    graph.add_edge("summary", END)

    return graph.compile()
