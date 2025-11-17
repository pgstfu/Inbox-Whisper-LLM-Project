import http.server
import socketserver
import urllib.parse
from utils.azure_auth import exchange_code_for_token

last_token = None

class OAuthHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        global last_token
        
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)

        if "code" in params:
            auth_code = params["code"][0]

            # Exchange code for token
            token = exchange_code_for_token(auth_code)
            last_token = token

            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"Login successful! You can close this tab.")

        else:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"No auth code found in request.")

def run_callback_server(port=8000):
    print(f"Starting OAuth callback server on http://localhost:{port}/auth/callback ...")
    with socketserver.TCPServer(("", port), OAuthHandler) as httpd:
        httpd.handle_request()   # handle one request then stop

    return last_token
