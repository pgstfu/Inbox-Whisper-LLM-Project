from utils.graph_api import graph_get

def ingest_email_azure(access_token):
    """
    Fetch the most recent email from the user's Outlook inbox using Microsoft Graph.
    """

    # Get top 1 message, sorted by newest
    url = "https://graph.microsoft.com/v1.0/me/messages?$top=1&$orderby=receivedDateTime DESC"
    data = graph_get(url, access_token)

    message = data["value"][0]

    # Extract fields
    email_dict = {
        "from": message["from"]["emailAddress"]["address"],
        "subject": message.get("subject", ""),
        "body": message.get("body", {}).get("content", ""),
        "received": message.get("receivedDateTime", "")
    }

    return email_dict
