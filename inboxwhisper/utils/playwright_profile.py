from pathlib import Path

PROFILE_DIR = Path(".playwright_user_data")

def ensure_profile():
    """
    Makes sure the persistent Playwright profile directory exists.
    """
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    return str(PROFILE_DIR)
