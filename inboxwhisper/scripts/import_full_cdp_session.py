import json
import sys
import os
from pathlib import Path
from playwright.sync_api import sync_playwright
from utils.playwright_profile import ensure_profile

profile = ensure_profile()

cookies = json.load(open(Path(profile) / "cloned_cookies.json"))
localstorage = json.load(open(Path(profile) / "cloned_localstorage.json"))

with sync_playwright() as p:
    browser = p.chromium.launch_persistent_context(
        user_data_dir=profile,
        headless=False
    )
    page = browser.new_page()

    # Apply cookies
    browser.clear_cookies()
    browser.add_cookies(cookies)

    # Apply localStorage
    for key, value in localstorage:
        page.add_init_script(f"""
            localStorage.setItem("{key}", "{value}");
        """)

    print("✔ Session applied. Opening dashboard...")
    page.goto("https://blackboard.snu.edu.in/ultra/stream")
    input("Press ENTER once dashboard loads...")
    browser.close()
