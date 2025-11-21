import os
from utils.config import Config

def setup_langsmith():
    """
    Sets LangSmith environment variables for tracing.
    """
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    
    # Use whichever key exists
    if Config.LANGSMITH_API_KEY:
        os.environ["LANGCHAIN_API_KEY"] = Config.LANGSMITH_API_KEY
    elif os.getenv("LANGCHAIN_API_KEY"):
        pass  # already set
    else:
        raise ValueError("LangSmith API key missing. Add it to .env")

    os.environ["LANGCHAIN_PROJECT"] = "InboxWhisper+"
    print("LangSmith tracing enabled.")
