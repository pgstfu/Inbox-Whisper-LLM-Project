import os
from dotenv import load_dotenv

# Force load .env from project root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(ROOT_DIR, ".env")

print("Loading .env from:", ENV_PATH)
load_dotenv(ENV_PATH, override=True)

class Config:
    # Mode selector
    EMAIL_SOURCE_MODE = os.getenv("EMAIL_SOURCE_MODE", "mock")

    # IMAP settings (not used now, but keep them)
    IMAP_SERVER = os.getenv("IMAP_SERVER")
    IMAP_PORT = os.getenv("IMAP_PORT")
    OUTLOOK_EMAIL = os.getenv("OUTLOOK_EMAIL")
    OUTLOOK_PASSWORD = os.getenv("OUTLOOK_PASSWORD")

    # Azure OAuth settings
    MS_CLIENT_ID = os.getenv("MS_CLIENT_ID")
    MS_TENANT_ID = os.getenv("MS_TENANT_ID", "common")
    MS_CLIENT_SECRET = os.getenv("MS_CLIENT_SECRET")
    MS_REDIRECT_URI = os.getenv("MS_REDIRECT_URI")
    MS_SCOPES = os.getenv("MS_SCOPES", "")

    # OpenAI key
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
