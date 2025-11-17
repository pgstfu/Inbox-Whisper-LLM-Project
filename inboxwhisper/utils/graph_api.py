import requests

def graph_get(url, access_token):
    """
    Helper to call Microsoft Graph GET endpoints.
    """
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return resp.json()
