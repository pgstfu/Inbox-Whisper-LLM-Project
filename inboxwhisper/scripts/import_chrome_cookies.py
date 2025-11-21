import os
import sys
import json
from pathlib import Path

# Add project root
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

import browser_cookie3
from utils.playwright_profile import ensure_profile

BLACKBOARD_DOMAIN = "blackboard.snu.edu.in"

profile_dir = Path(ensure_profile())
cookies_path = profile_dir / "cookies.json"

print("Extracting cookies from Chrome...")

cj = browser_cookie3.chrome(domain_name=BLACKBOARD_DOMAIN)

cookies_list = []

for c in cj:
    cookies_list.append({
        "name": c.name,
        "value": c.value,
        "domain": c.domain,
        "path": c.path,
        "secure": c.secure,
        "httpOnly": c.has_nonstandard_attr('HttpOnly'),
        "sameSite": c._rest.get("SameSite", "None")
    })

cookies_json_path = profile_dir / "imported_cookies.json"
with open(cookies_json_path, "w") as f:
    json.dump(cookies_list, f, indent=2)

print(f"✔ Extracted {len(cookies_list)} cookies")
print(f"✔ Saved to {cookies_json_path}")

print("\nNow injecting into Playwright...")

from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch_persistent_context(
        user_data_dir=str(profile_dir),
        headless=False
    )
    context = browser

    context.clear_cookies()
    context.add_cookies(cookies_list)

    print("✔ Cookies injected!")
    print("Opening Blackboard to verify...")

    page = context.new_page()
    page.goto("https://blackboard.snu.edu.in/ultra/stream")

    input("Press ENTER once the page loads in Chrome-authenticated login.")
    browser.close()

print("Done.")
