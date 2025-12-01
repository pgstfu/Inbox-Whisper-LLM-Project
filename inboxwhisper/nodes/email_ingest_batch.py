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

    # Helper function to extract sender email with fallbacks
    def extract_sender_email(msg):
        """Extract sender email from message with multiple fallback methods."""
        sender_email = ""
        # Method 1: from.emailAddress.address
        from_field = msg.get("from")
        if from_field and isinstance(from_field, dict):
            email_address = from_field.get("emailAddress")
            if email_address and isinstance(email_address, dict):
                sender_email = email_address.get("address", "") or ""
        
        # Method 2: sender.emailAddress.address (fallback)
        if not sender_email:
            sender_field = msg.get("sender")
            if sender_field and isinstance(sender_field, dict):
                email_address = sender_field.get("emailAddress")
                if email_address and isinstance(email_address, dict):
                    sender_email = email_address.get("address", "") or ""
        
        return sender_email

    out = []
    for m in messages:
        # Extract sender email using improved method
        sender = extract_sender_email(m)
        
        # Filter non SNU emails (case-insensitive)
        if domain_filter:
            if not sender:
                continue  # Skip emails with no sender
            sender_lower = sender.lower()
            domain_lower = domain_filter.lower()
            if not sender_lower.endswith(domain_lower):
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
