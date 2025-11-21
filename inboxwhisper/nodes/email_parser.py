from openai import OpenAI
from utils.config import Config
import json
import re

client = OpenAI(api_key=Config.OPENAI_API_KEY)

def parse_email(email_dict):
    """
    Parses email content and extracts structured academic info.
    Ensures the output is pure JSON without code fences or markdown.
    """

    prompt = f"""
    You are InboxWhisper+, an academic assistant.

    Extract structured JSON ONLY — no code fences, no backticks, no markdown.

    JSON schema:
    {{
      "type": "assignment" | "exam" | "announcement" | "task" | "other",
      "course": string | null,
      "date": string | null,
      "time": string | null,
      "deadline": string | null,
      "summary": string,
      "action_item": string,
      "raw_subject": string,
      "raw_body": string
    }}

    Email:
    SUBJECT: {email_dict['subject']}
    BODY: {email_dict['body']}

    Return strictly valid JSON only.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Always return ONLY valid JSON. Never use ``` or any markdown."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    content = response.choices[0].message.content

    # Remove accidental ```json ... ``` wrappers
    content = re.sub(r"```json", "", content, flags=re.IGNORECASE)
    content = re.sub(r"```", "", content)
    content = content.strip()

    # Attempt to parse JSON
    try:
        return json.loads(content)
    except Exception as e:
        return {
            "error": "Failed to parse JSON output",
            "exception": str(e),
            "raw_output": content
        }
