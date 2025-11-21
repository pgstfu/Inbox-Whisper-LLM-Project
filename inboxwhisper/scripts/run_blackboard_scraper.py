import os
import sys

# Add project root so imports work
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from nodes.blackboard_node import ingest_blackboard_full

print("Running Blackboard scraper (headless)...")

result = ingest_blackboard_full(
    headless=True,        # headless after login
    max_courses=8,        # adjust if needed
    download_files=True   # downloads PDFs, PPTs, DOCX, etc.
)

print("\n✔ Scraper completed.")
print("✔ Files saved in: ./files/")
print("✔ Summary saved at: ./files/blackboard_scrape_summary.json")

# Print trimmed JSON preview
import json
print("\n--- PREVIEW (first 1500 chars) ---\n")
print(json.dumps(result, indent=2)[:1500])
print("\n--- END PREVIEW ---")
