# nodes/blackboard_scraper.py
import os
import json
import time
import shutil
from pathlib import Path
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, TimeoutError as PlayTimeout
from utils.config import Config
from utils.playwright_auth import ensure_profile

DOWNLOAD_DIR = Path(Config.DOWNLOAD_DIR or "./files")
PROFILE_DIR = Path(".playwright_user_data")

DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

def _safe_filename(s: str):
    # minimal safe filename
    return "".join(c for c in s if c.isalnum() or c in " .-_()").strip()

def _resolve_url(base, href):
    return urljoin(base, href)

def download_file(page, link_url, dest_path: Path, timeout=30000):
    """
    Uses Playwright to navigate to file URL and download via response.
    Returns saved path.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with page.expect_download(timeout=timeout) as download_info:
        page.goto(link_url)
    download = download_info.value
    suggested = download.suggested_filename or dest_path.name
    final = dest_path.parent / _safe_filename(suggested)
    download.save_as(str(final))
    return final

def scrape_stream_and_courses(headless=True, max_courses=8, download_files=True):
    """
    Full scraper flow:
    - uses persistent profile (PROFILE_DIR). If not logged-in, user must login once via utils.playwright_auth.interactive_sso_login()
    - navigates to /ultra/stream and extracts announcements
    - fetches course links on stream or course list and downloads files from content
    """
    profile = ensure_profile()
    results = {"courses": [], "announcements": []}

    with sync_playwright() as p:
        browser = p.chromium.launch_persistent_context(user_data_dir=str(profile), headless=headless)
        page = browser.new_page()
        # navigate to stream
        stream_url = Config.BLACKBOARD_BASE_URL.rstrip("/") + "/ultra/stream"
        page.goto(stream_url, wait_until="networkidle", timeout=60000)
        time.sleep(1)

        # parse stream page
        html = page.content()
        # DEBUG: save HTML snapshot so we can inspect what Blackboard actually shows
        with open("debug_stream.html", "w", encoding="utf-8") as f:
            f.write(html)
        print("Saved debug_stream.html")
        soup = BeautifulSoup(html, "html.parser")

        # Announcements extraction heuristics: locate announcement-like elements
        # This may vary by institution theme
        # We'll look for elements containing 'Announcement' or classes containing 'announcement' or '.stream'
        for ann in soup.select(".announcement, .streamItem, .stream-card, .announcementCard"):
            text = ann.get_text(separator="\n", strip=True)
            if text:
                results["announcements"].append({"text": text})
        # fallback: find headlines in stream
        if not results["announcements"]:
            for h in soup.select("h2, h3, .stream, .item"):
                txt = h.get_text(strip=True)
                if txt:
                    results["announcements"].append({"text": txt})

        # Find course links (look for anchors that contain '/course' or '/ultra/course/')
        course_links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "/ultra/courses/" in href or "/course/" in href or "/ultra/course" in href:
                parsed = urlparse(href)
                if parsed.scheme:
                    url = href
                else:
                    url = urljoin(Config.BLACKBOARD_BASE_URL, href)
                if url not in course_links:
                    course_links.append(url)

        # Limit number of courses to scrape to avoid long runs
        if max_courses and len(course_links) > max_courses:
            course_links = course_links[:max_courses]

        # For each course, visit and inspect content and files
        for c_url in course_links:
            page.goto(c_url, wait_until="networkidle", timeout=60000)
            time.sleep(1)
            course_html = page.content()
            course_soup = BeautifulSoup(course_html, "html.parser")

            # Course title
            course_title = course_soup.find(class_="courseTitle") or course_soup.find("h1")
            course_name = course_title.get_text(strip=True) if course_title else c_url.split("/")[-1]

            course_obj = {"course_name": course_name, "url": c_url, "files": [], "announcements": []}

            # Find announcement blocks within course page
            for ann in course_soup.select(".announcement, .announcementItem, .courseAnnouncement, .announcement-list"):
                text = ann.get_text(separator="\n", strip=True)
                if text:
                    course_obj["announcements"].append({"text": text})

            # Find links that point to files (common file extensions)
            for a in course_soup.find_all("a", href=True):
                href = a["href"]
                if any(ext in href.lower() for ext in [".pdf", ".ppt", ".pptx", ".doc", ".docx", ".zip", ".xls", ".xlsx"]):
                    file_url = href if urlparse(href).scheme else urljoin(Config.BLACKBOARD_BASE_URL, href)
                    file_name = a.get_text(strip=True) or file_url.split("/")[-1]
                    file_name = _safe_filename(file_name)
                    dest_dir = DOWNLOAD_DIR / _safe_filename(course_name)
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    dest_path = dest_dir / file_name
                    try:
                        if download_files:
                            # robust download: use download API
                            saved = download_file(page, file_url, dest_path)
                            course_obj["files"].append({"name": saved.name, "path": str(saved), "url": file_url})
                        else:
                            course_obj["files"].append({"name": file_name, "path": None, "url": file_url})
                    except PlayTimeout as e:
                        course_obj["files"].append({"name": file_name, "path": None, "url": file_url, "error": "timeout"})
                    except Exception as e:
                        course_obj["files"].append({"name": file_name, "path": None, "url": file_url, "error": str(e)})

            results["courses"].append(course_obj)

        browser.close()

    # Save results summary
    summary_path = DOWNLOAD_DIR / "blackboard_scrape_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    return results
