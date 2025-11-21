from openai import OpenAI
from utils.config import Config

client = OpenAI(api_key=Config.OPENAI_API_KEY)

def summarize_parsed(parsed):
    """
    Generate a human-readable summary from parsed academic fields.
    """

    prompt = f"""
    Create a short human-readable summary of this academic task:

    {parsed}

    Summary should be:
    - 1-2 sentences
    - Student-friendly
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content.strip()
