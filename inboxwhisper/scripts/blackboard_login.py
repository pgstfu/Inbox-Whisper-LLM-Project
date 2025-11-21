# scripts/blackboard_login.py

import os
import sys
from pathlib import Path
from time import sleep

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from utils.config import Config
from playwright.sync_api import sync_playwright

PROFILE_DIR = Path(".playwright_user_data")

def ensure_profile():
    PROFILE_DIR.mkdir(exist_ok=True)
    return str(PROFILE_DIR)

def interactive_sso_login():
    profile = ensure_profile()
    with sync_playwright() as p:
        browser = p.chromium.launch_persistent_context(
            user_data_dir=profile,
            headless=False,
            args=["--start-maximized"]
        )
        page = browser.new_page()

        print("Opening login page…")
        page.goto(Config.BLACKBOARD_BASE_URL + "/ultra/stream")

        # 🔥 If login button exists — CLICK IT
        try:
            page.wait_for_selector("button.login-button", timeout=8000)
            print("Found login button → clicking it…")
            page.click("button.login-button")
        except:
            print("Login button not found — maybe already logged in.")

        print("Please COMPLETE Microsoft login in the browser…")
        print("DO NOT press Enter until you reach /ultra/stream")

        # 🔥 WAIT for Blackboard dashboard after SSO
        page.wait_for_url("**/ultra/stream**", timeout=0)

        print("🎉 SUCCESS — Logged in and cookies saved.")
        input("Press ENTER to close browser…")

        browser.close()

if __name__ == "__main__":
    interactive_sso_login()
