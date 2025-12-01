# nodes/email_ingest_azure.py

from utils.graph_api import graph_get
from utils.html_cleaner import clean_html_to_text
from utils.attachments import fetch_attachments_for_message

def ingest_email_azure(access_token, domain_only="@snu.edu.in", max_check=10):
    """
    Fetch the latest email via MS Graph API & clean the body.
    
    Args:
        access_token: OAuth token for Microsoft Graph API
        domain_only: Domain filter (e.g., "@snu.edu.in"). Set to None to disable filtering.
        max_check: Maximum number of emails to check before giving up (default 10)
    
    Returns:
        dict: Email object with id, from, subject, body, attachments, received
              or error dict if no matching email found
    """

    # Fetch more emails to check if domain filter is enabled
    top_count = max_check if domain_only else 1
    url = (
        f"https://graph.microsoft.com/v1.0/me/messages?"
        f"$top={top_count}&"
        "$orderby=receivedDateTime DESC&"
        "$select=id,subject,from,body,bodyPreview,receivedDateTime,hasAttachments"
    )

    data = graph_get(url, access_token)

    if "value" not in data or not data["value"]:
        return {"error": "No emails found."}

    messages = data["value"]
    
    # Helper function to extract sender email from a message
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
    
    # Find first email that matches domain filter (if enabled)
    message = None
    for msg in messages:
        sender_email = extract_sender_email(msg)
        
        # If no domain filter, use first email
        if not domain_only:
            message = msg
            break
        
        # Check if sender matches domain (case-insensitive)
        if sender_email:
            sender_lower = sender_email.lower()
            domain_lower = domain_only.lower()
            if sender_lower.endswith(domain_lower):
                message = msg
                break
    
    # If no matching email found and filter is enabled
    if not message and domain_only:
        # Return error with debug info from first email
        first_msg = messages[0]
        first_sender = extract_sender_email(first_msg)
        return {
            "error": f"No {domain_only} emails found in the latest {len(messages)} emails.",
            "debug": {
                "checked_emails": len(messages),
                "first_email_sender": first_sender,
                "first_email_subject": first_msg.get("subject", "No subject"),
                "first_email_id": first_msg.get("id")
            }
        }
    
    if not message:
        return {"error": "No emails found."}
    
    sender_email = extract_sender_email(message)

    # CLEAN HTML BODY
    body_raw = (
        message.get("body", {}).get("content", "") 
        or message.get("bodyPreview", "")
    )
    body_clean = clean_html_to_text(body_raw)

    # FETCH ATTACHMENTS
    attachments = []
    if message.get("id") and message.get("hasAttachments"):
        attachments = fetch_attachments_for_message(message.get("id"))

    # BUILD RESULT
    email_dict = {
        "id": message.get("id"),
        "from": sender_email,
        "subject": message.get("subject"),
        "body": body_clean,
        "attachments": attachments,
        "received": message.get("receivedDateTime")
    }

    return email_dict
