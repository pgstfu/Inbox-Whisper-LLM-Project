# utils/graph_pagination.py
import requests
from utils.config import Config

def graph_get_with_paging(url, access_token, page_limit=10):
    """
    Fetch up to page_limit pages from a Graph endpoint that returns @odata.nextLink.
    Returns combined list of message dicts (value).
    """
    headers = {"Authorization": f"Bearer {access_token}"}
    results = []
    cur = url
    pages = 0
    while cur and pages < page_limit:
        resp = requests.get(cur, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        vals = data.get("value", [])
        results.extend(vals)
        cur = data.get("@odata.nextLink")
        pages += 1
    return results
