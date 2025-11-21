from utils.graph_api import graph_get

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

    email_dict = {
        "from": message["from"]["emailAddress"]["address"],
        "subject": message.get("subject", ""),
        "body": message.get("body", {}).get("content", ""),
        "received": message.get("receivedDateTime", "")
    }

    return email_dict
