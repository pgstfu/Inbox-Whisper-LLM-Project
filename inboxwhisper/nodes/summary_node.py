from openai import OpenAI
from utils.config import Config

client = OpenAI(api_key=Config.OPENAI_API_KEY)

def summarize_parsed(parsed):
    """
    Generate a human-readable summary from parsed academic fields.
    """

    prompt = f"""
    Create a short human-readable summary of this academic task:

    Parsed task JSON: {parsed}

    Include:
    - Task type, course (if any), due date, and priority score.
    - The key action item sentence.
    - If student_slot_details is present, explicitly mention the slot/timing.

    Summary should be 2 concise sentences max.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content.strip()
