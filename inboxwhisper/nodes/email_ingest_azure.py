# nodes/email_ingest_azure.py

from utils.graph_api import graph_get
from utils.html_cleaner import clean_html_to_text
from utils.attachments import fetch_attachments_for_message

def ingest_email_azure(access_token, domain_only="@snu.edu.in"):
    """
    Fetch the latest email via MS Graph API & clean the body.
    """

    url = (
        "https://graph.microsoft.com/v1.0/me/messages?"
        "$top=1&"
        "$orderby=receivedDateTime DESC&"
        "$select=id,subject,from,body,bodyPreview,receivedDateTime,hasAttachments"
    )

    data = graph_get(url, access_token)

    if "value" not in data or not data["value"]:
        return {"error": "No emails found."}

    message = data["value"][0]

    sender_email = (
        message.get("from", {})
        .get("emailAddress", {})
        .get("address", "")
    )

    # Optional domain filter
    if domain_only and not sender_email.endswith(domain_only):
        return {"error": f"No {domain_only} emails found."}

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
