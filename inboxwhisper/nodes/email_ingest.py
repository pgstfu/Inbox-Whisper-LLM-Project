from utils.config import Config
from utils.imap_reader import fetch_latest_email
from utils.sample_inputs import SAMPLE_EMAIL

def ingest_email():
    mode = Config.EMAIL_SOURCE_MODE

    if mode == "imap":
        print("Reading email from IMAP...")
        return fetch_latest_email()

    elif mode == "mock":
        print("Using mock email...")
        return SAMPLE_EMAIL

    else:
        raise ValueError("Unknown EMAIL_SOURCE_MODE. Use 'imap' or 'mock'.")
