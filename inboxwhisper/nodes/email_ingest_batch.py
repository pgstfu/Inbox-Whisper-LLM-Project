# nodes/email_ingest_batch.py

from utils.graph_pagination import graph_get_with_paging
from utils.token_manager import get_token_silent
from utils.attachments import fetch_attachments_for_message
from utils.html_cleaner import clean_html_to_text

def ingest_emails_batch(limit=200, domain_filter="@snu.edu.in"):
    token = get_token_silent()

    # Fetch base email metadata
    url = (
        "https://graph.microsoft.com/v1.0/me/messages?"
        "$top=50&"
        "$orderby=receivedDateTime DESC&"
        "$select=id,subject,from,body,bodyPreview,hasAttachments,receivedDateTime"
    )

    messages = graph_get_with_paging(url, token, page_limit=10)

    out = []
    for m in messages:

        # Filter non SNU emails
        sender = m.get("from",{}).get("emailAddress",{}).get("address","")
        if domain_filter and not sender.endswith(domain_filter):
            continue

        # Clean HTML body
        body_raw = m.get("body",{}).get("content","") or m.get("bodyPreview","")
        body_clean = clean_html_to_text(body_raw)

        # Attachments
        attachments = []
        if m.get("hasAttachments"):
            attachments = fetch_attachments_for_message(m.get("id"))

        out.append({
            "id": m.get("id"),
            "from": sender,
            "subject": m.get("subject"),
            "body": body_clean,
            "attachments": attachments,
            "received": m.get("receivedDateTime")
        })

        if len(out) >= limit:
            break

    return out
