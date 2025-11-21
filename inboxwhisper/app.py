import streamlit as st
from graph.main_graph import build_inbox_graph
from utils.token_manager import get_token_silent

st.set_page_config(page_title="InboxWhisper+", layout="centered")

st.title("📚 InboxWhisper+ — Email Academic Assistant")

graph = build_inbox_graph()

st.write("Click the button below to fetch your latest email and analyze it.")

if st.button("🔄 Analyze Latest Email"):
    with st.spinner("Fetching email and analyzing..."):
        
        # Run LangGraph pipeline
        result = graph.invoke({}, config={"reset": True})

    st.success("Analysis complete!")

    st.subheader("📥 Raw Email")
    st.json(result.get("email_raw"))

    st.subheader("🧠 Parsed Academic Data")
    st.json(result.get("parsed"))

    st.subheader("📝 Summary")
    st.write(result.get("summary"))
