# utils/token_manager.py
import time
import json
import os
import msal
from utils.config import Config

# In-memory state (also persist optionally)
TOKEN_STATE = {
    "access_token": None,
    "refresh_token": None,
    "expires_at": 0,
    "account": None  # MSAL account info (optional)
}

# Optionally persist tokens to file so UI restarts keep them
PERSIST_FILE = os.path.join(os.path.dirname(__file__), ".token_state.json")

def _save_state():
    try:
        with open(PERSIST_FILE, "w") as f:
            json.dump(TOKEN_STATE, f)
    except Exception:
        pass

def _load_state():
    if os.path.exists(PERSIST_FILE):
        try:
            with open(PERSIST_FILE, "r") as f:
                data = json.load(f)
                TOKEN_STATE.update(data)
        except Exception:
            pass

_load_state()

def store_initial_token(token_dict, account=None):
    """
    Call this ONCE after OAuth login to set initial refresh+access tokens.
    token_dict should be the dictionary returned by your OAuth callback server.
    """
    now = time.time()
    TOKEN_STATE["access_token"] = token_dict.get("access_token")
    TOKEN_STATE["refresh_token"] = token_dict.get("refresh_token")
    expires_in = int(token_dict.get("expires_in", 3600))
    TOKEN_STATE["expires_at"] = now + expires_in - 60
    TOKEN_STATE["account"] = account
    _save_state()
    return TOKEN_STATE["access_token"]

def get_msal_app():
    authority = f"https://login.microsoftonline.com/{Config.MS_TENANT_ID or 'common'}"
    app = msal.ConfidentialClientApplication(
        client_id=Config.MS_CLIENT_ID,
        client_credential=Config.MS_CLIENT_SECRET,
        authority=authority
    )
    return app

def get_token_silent(scopes=None):
    """
    Returns a valid access token, refreshing it if needed.
    Raises RuntimeError with clear message if no refresh token is available.
    """
    now = time.time()
    if scopes is None:
        scopes = Config.MS_SCOPES.split() if Config.MS_SCOPES else ["User.Read", "Mail.Read"]

    # 1) If cached and not expired, return
    if TOKEN_STATE.get("access_token") and now < TOKEN_STATE.get("expires_at", 0):
        return TOKEN_STATE["access_token"]

    app = get_msal_app()

    # 2) If we have account info, try acquire_token_silent
    if TOKEN_STATE.get("account"):
        account = TOKEN_STATE["account"]
        result = app.acquire_token_silent(scopes=scopes, account=account)
        if result and "access_token" in result:
            TOKEN_STATE["access_token"] = result["access_token"]
            TOKEN_STATE["expires_at"] = now + int(result.get("expires_in", 3600)) - 60
            _save_state()
            return TOKEN_STATE["access_token"]

    # 3) If refresh token exists, try refresh via MSAL
    rt = TOKEN_STATE.get("refresh_token")
    if rt:
        try:
            refreshed = app.acquire_token_by_refresh_token(rt, scopes=scopes)
            if refreshed and "access_token" in refreshed:
                TOKEN_STATE["access_token"] = refreshed["access_token"]
                TOKEN_STATE["refresh_token"] = refreshed.get("refresh_token", rt)
                TOKEN_STATE["expires_at"] = now + int(refreshed.get("expires_in", 3600)) - 60
                _save_state()
                return TOKEN_STATE["access_token"]
            else:
                # fallback to error with reason
                raise RuntimeError(f"Refresh failed: {refreshed}")
        except Exception as e:
            # Clear invalid refresh_token to allow re-auth flow
            TOKEN_STATE["refresh_token"] = None
            _save_state()
            raise RuntimeError(
                "Refresh token usage failed: " + str(e) +
                ". Please re-run OAuth flow to obtain a new token (run the callback server)."
            )

    # 4) No refresh token available — ask user to re-authorize
    raise RuntimeError(
        "No valid refresh_token available. Please perform the OAuth login flow again "
        "using your notebook/script that starts the OAuth callback server, then call store_initial_token(...) "
        "with the returned token dictionary."
    )
