import os
import msal
from utils.config import Config

def get_msal_app():
    """
    Creates the MSAL ConfidentialClientApplication for OAuth.
    """
    client_id = Config.MS_CLIENT_ID
    tenant = Config.MS_TENANT_ID or "common"
    client_secret = Config.MS_CLIENT_SECRET
    authority = f"https://login.microsoftonline.com/{tenant}"

    app = msal.ConfidentialClientApplication(
        client_id=client_id,
        client_credential=client_secret,
        authority=authority
    )
    return app


def get_auth_url():
    """
    Generates the Microsoft login URL for user authentication.
    """
    app = get_msal_app()
    scopes = Config.MS_SCOPES.split()

    auth_url = app.get_authorization_request_url(
        scopes=scopes,
        redirect_uri=Config.MS_REDIRECT_URI
    )
    return auth_url


def exchange_code_for_token(auth_code):
    """
    Exchanges the returned authorization code for an access token.
    """
    app = get_msal_app()
    scopes = Config.MS_SCOPES.split()

    token = app.acquire_token_by_authorization_code(
        code=auth_code,
        scopes=scopes,
        redirect_uri=Config.MS_REDIRECT_URI
    )
    return token
