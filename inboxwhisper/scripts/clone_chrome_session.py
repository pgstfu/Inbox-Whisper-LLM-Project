import os
import sys
import json
from pathlib import Path
import pychrome

# --- FIX: Add project root to sys.path ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))         # /inboxwhisper/scripts
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))  # /inboxwhisper
sys.path.insert(0, PROJECT_ROOT)

from utils.playwright_profile import ensure_profile   # <-- now it will work

# Connect to Chrome remote debugging
browser = pychrome.Browser(url="http://127.0.0.1:9222")

# Get active tab
tabs = browser.list_tab()
tab = tabs[0]
tab.start()

# Enable storage
tab.call_method("Storage.enable")

profile_dir = Path(ensure_profile())

# === Clone Cookies ===
print("Extracting cookies from Chrome...")
cookies_data = tab.call_method("Network.getAllCookies")["cookies"]

cookie_file = profile_dir / "cloned_cookies.json"
with open(cookie_file, "w") as f:
    json.dump(cookies_data, f, indent=2)

print(f"✔ Extracted {len(cookies_data)} cookies")
print(f"✔ Saved to {cookie_file}")

# === Clone LocalStorage ===
print("\nExtracting LocalStorage...")
try:
    storage_items = tab.call_method(
        "DOMStorage.getDOMStorageItems",
        storageId={"isLocalStorage": True, "securityOrigin": "https://blackboard.snu.edu.in"}
    )["entries"]
except Exception as e:
    storage_items = []
    print(f"⚠ Could not extract LocalStorage: {e}")

local_file = profile_dir / "cloned_localstorage.json"
with open(local_file, "w") as f:
    json.dump(storage_items, f, indent=2)

print("✔ LocalStorage cloned.")
print("Done.")
tab.stop()
