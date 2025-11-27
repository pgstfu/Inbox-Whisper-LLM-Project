# nodes/email_ingest_batch.py
from utils.graph_pagination import graph_get_with_paging
from utils.config import Config
from utils.token_manager import get_token_silent
from utils.attachments import fetch_attachments_for_message
import datetime

# default: fetch 100 latest messages
def ingest_emails_batch(limit=100, domain_filter="@snu.edu.in"):
    access_token = get_token_silent()
    # request includes body preview
    url = "https://graph.microsoft.com/v1.0/me/messages?$top=50&$orderby=receivedDateTime DESC&$select=subject,from,bodyPreview,body,receivedDateTime"
    messages = graph_get_with_paging(url, access_token, page_limit=10)  # ~500 messages max with defaults
    # normalize and filter
    out = []
    for m in messages:
        from_addr = m.get("from", {}).get("emailAddress", {}).get("address", "")
        if domain_filter and not from_addr.endswith(domain_filter):
            continue
        out.append({
            "id": m.get("id"),
            "from": from_addr,
            "subject": m.get("subject"),
            "body": m.get("body", {}).get("content") or m.get("bodyPreview", ""),
            "received": m.get("receivedDateTime")
        })
        attachments = fetch_attachments_for_message(m.get("id"))
        out[-1]["attachments"] = attachments
        if len(out) >= limit:
            break
    return out
