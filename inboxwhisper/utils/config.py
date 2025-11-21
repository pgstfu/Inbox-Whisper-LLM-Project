import os
from dotenv import load_dotenv

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(ROOT_DIR, ".env")

print("Loading .env from:", ENV_PATH)
load_dotenv(ENV_PATH, override=True)

class Config:
    # Mode selector (mock / imap / azure)
    EMAIL_SOURCE_MODE = os.getenv("EMAIL_SOURCE_MODE", "mock")

    # IMAP
    IMAP_SERVER = os.getenv("IMAP_SERVER")
    IMAP_PORT = os.getenv("IMAP_PORT")
    OUTLOOK_EMAIL = os.getenv("OUTLOOK_EMAIL")
    OUTLOOK_PASSWORD = os.getenv("OUTLOOK_PASSWORD")

    # Azure SSO
    MS_CLIENT_ID = os.getenv("MS_CLIENT_ID")
    MS_TENANT_ID = os.getenv("MS_TENANT_ID", "common")
    MS_CLIENT_SECRET = os.getenv("MS_CLIENT_SECRET")
    MS_REDIRECT_URI = os.getenv("MS_REDIRECT_URI")
    MS_SCOPES = os.getenv("MS_SCOPES", "")

    # OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # LangSmith
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")

    # Blackboard
    BLACKBOARD_BASE_URL = os.getenv("BLACKBOARD_BASE_URL", "https://blackboard.snu.edu.in")
    BLACKBOARD_USERNAME = os.getenv("BLACKBOARD_USERNAME")
    BLACKBOARD_PASSWORD = os.getenv("BLACKBOARD_PASSWORD")

    # Playwright
    PLAYWRIGHT_HEADLESS = os.getenv("PLAYWRIGHT_HEADLESS", "1")
    DOWNLOAD_DIR = os.getenv("DOWNLOAD_DIR", "./files")
