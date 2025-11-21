import requests

def graph_get(url, access_token, consistency=False):
    """
    Helper to call Microsoft Graph GET endpoints.
    Set consistency=True when using $search queries.
    """
    headers = {
        "Authorization": f"Bearer {access_token}"
    }

    if consistency:
        headers["ConsistencyLevel"] = "eventual"

    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return resp.json()
