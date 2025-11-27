from utils.graph_api import graph_get
from utils.html_cleaner import clean_html_to_text

def ingest_email_azure(access_token):
    """
    Fetch the latest email sent from @snu.edu.in using Microsoft Graph.
    """
    url = (
        "https://graph.microsoft.com/v1.0/me/messages?"
        "$search=\"from:@snu.edu.in\"&"
        "$top=1"
    )

    data = graph_get(url, access_token, consistency=True)

    if "value" not in data or len(data["value"]) == 0:
        return {"error": "No @snu.edu.in emails found"}

    message = data["value"][0]

    body_raw = mail.get("body", {}).get("content", "")
    body_clean = clean_html_to_text(body_raw)

    email_dict = {
        "from": sender_email,
        "subject": subject,
        "body": body_clean,
        "attachments": attachments,
        "received": mail["receivedDateTime"]
    }

    return email_dict
