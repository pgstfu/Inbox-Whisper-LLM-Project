# utils/playwright_auth.py
import os
from pathlib import Path
from playwright.sync_api import sync_playwright
from utils.config import Config

PROFILE_DIR = Path(".playwright_user_data")  # persistent profile to keep logged-in session

def ensure_profile():
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    return str(PROFILE_DIR)

def interactive_sso_login(start_url=None, headless=False):
    """
    Open a browser window for manual Microsoft SSO login and store session in PROFILE_DIR.
    Use this once; subsequent runs will reuse stored cookies/profile.
    """
    profile_dir = ensure_profile()
    with sync_playwright() as p:
        browser = p.chromium.launch_persistent_context(user_data_dir=profile_dir, headless=headless,
                                                      args=["--start-maximized", "--disable-dev-shm-usage"])
        page = browser.new_page()
        target = start_url or Config.BLACKBOARD_BASE_URL + "/ultra/stream"
        print("Opening:", target)
        page.goto(target)
        print("If not logged in, complete the Microsoft SSO login in the opened browser window.")
        print("Wait until you see your Blackboard stream.")
        # Block until user confirms login complete by pressing Enter in terminal.
        input("When you finished login and can see the stream page in the browser, press Enter here...")
        # Optionally check for an element in the stream page to confirm
        try:
            page.wait_for_selector("div.stream", timeout=5000)
            print("Stream page element found — login looks successful.")
        except Exception:
            print("Warning: stream selector not found — but session may still be active.")
        browser.close()
        return profile_dir
