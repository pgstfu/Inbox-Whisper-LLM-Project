# nodes/blackboard_node.py
from nodes.blackboard_scraper import scrape_stream_and_courses
from utils.config import Config

def ingest_blackboard_full(headless=True, max_courses=6, download_files=True):
    """
    LangGraph node wrapper that runs the full Playwright scraper.
    """
    # headless False only when you need to login interactively (first time)
    res = scrape_stream_and_courses(headless=headless, max_courses=max_courses, download_files=download_files)
    return res
