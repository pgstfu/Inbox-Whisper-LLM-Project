import imaplib
import email
from email.header import decode_header
from utils.config import Config

def fetch_latest_email():
    """
    Connect to Outlook IMAP and read the latest email.
    Returns a dict with sender, subject, and body.
    """

    try:
        mail = imaplib.IMAP4_SSL(Config.IMAP_SERVER, int(Config.IMAP_PORT))
        mail.login(Config.OUTLOOK_EMAIL, Config.OUTLOOK_PASSWORD)

        mail.select("inbox")

        status, data = mail.search(None, "ALL")
        if status != "OK":
            raise RuntimeError("Failed to fetch email IDs")

        mail_ids = data[0].split()
        latest_id = mail_ids[-1]

        status, msg_data = mail.fetch(latest_id, "(RFC822)")
        if status != "OK":
            raise RuntimeError("Failed to fetch the email content")

        raw_msg = msg_data[0][1]
        msg = email.message_from_bytes(raw_msg)

        # ----------- Parse: From, Subject, Body -----------
        subject, encoding = decode_header(msg["Subject"])[0]
        if isinstance(subject, bytes):
            subject = subject.decode(encoding or "utf-8", errors="ignore")

        sender = msg.get("From")

        # Extract body
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    charset = part.get_content_charset() or "utf-8"
                    body += part.get_payload(decode=True).decode(charset, errors="ignore")
        else:
            body = msg.get_payload(decode=True).decode("utf-8", errors="ignore")

        return {
            "from": sender,
            "subject": subject,
            "body": body,
        }

    except imaplib.IMAP4.error as e:
        raise RuntimeError(f"IMAP Authentication failed: {e}")

    except Exception as e:
        raise RuntimeError(f"IMAP error: {e}")
