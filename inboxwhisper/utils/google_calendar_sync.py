# utils/google_calendar_sync.py
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ['https://www.googleapis.com/auth/calendar']

def create_service():
    flow = InstalledAppFlow.from_client_secrets_file('client_secret.json', SCOPES)
    creds = flow.run_local_server(port=0)
    service = build('calendar', 'v3', credentials=creds)
    return service

def add_event_from_task(service, task, calendarId='primary'):
    event = {
        'summary': task.get("title"),
        'description': task.get("summary"),
    }
    if task.get("due_date"):
        event['start'] = {'dateTime': task.get("due_date")}
        event['end'] = {'dateTime': task.get("due_date")}
    service.events().insert(calendarId=calendarId, body=event).execute()
