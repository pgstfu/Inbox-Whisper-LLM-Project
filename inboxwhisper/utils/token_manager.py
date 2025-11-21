import time
import msal
from utils.config import Config

# Memory cache for tokens
TOKEN_STATE = {
    "access_token": None,
    "refresh_token": None,
    "expires_at": 0
}

def get_token_silent():
    """
    Uses MSAL to refresh the token if expired.
    Keeps access & refresh tokens in memory.
    """

    now = time.time()

    # If we have a valid token, use it
    if TOKEN_STATE["access_token"] and now < TOKEN_STATE["expires_at"]:
        return TOKEN_STATE["access_token"]

    # Otherwise, refresh using MSAL
    app = msal.ConfidentialClientApplication(
        Config.MS_CLIENT_ID,
        client_credential=Config.MS_CLIENT_SECRET,
        authority=f"https://login.microsoftonline.com/{Config.MS_TENANT_ID}"
    )

    refreshed = app.acquire_token_by_refresh_token(
        TOKEN_STATE["refresh_token"],
        scopes=Config.MS_SCOPES.split()
    )

    TOKEN_STATE["access_token"] = refreshed["access_token"]
    TOKEN_STATE["refresh_token"] = refreshed.get("refresh_token", TOKEN_STATE["refresh_token"])
    TOKEN_STATE["expires_at"] = now + int(refreshed["expires_in"]) - 60

    return TOKEN_STATE["access_token"]

def store_initial_token(token_dict):
    """
    Call this ONCE after OAuth login to set initial refresh+access tokens.
    """
    TOKEN_STATE["access_token"] = token_dict["access_token"]
    TOKEN_STATE["refresh_token"] = token_dict["refresh_token"]
    TOKEN_STATE["expires_at"] = time.time() + int(token_dict["expires_in"]) - 60
