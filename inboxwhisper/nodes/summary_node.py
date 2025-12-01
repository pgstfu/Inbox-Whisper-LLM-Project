from openai import OpenAI
from utils.config import Config

client = OpenAI(api_key=Config.OPENAI_API_KEY)

def summarize_parsed(parsed):
    """
    Generates a concise, human-readable summary from a parsed task dictionary.
    
    This function takes the structured task data (from email_parser) and converts it
    into a natural language summary suitable for display in the UI. The prompt
    explicitly instructs the model to include key details like course, due date,
    location, and student-specific slot information.
    
    Args:
        parsed: Dict containing parsed task fields (type, course, due_date, location,
                action_item, student_slot_details, priority, etc.)
    
    Returns:
        str: A 2-sentence summary of the task, formatted for human reading.
    """

    prompt = f"""
    Create a concise, human-readable summary of this academic task for a student.

    Task details:
    {parsed}

    Summary requirements:
    - First sentence: State the task type (assignment/exam/viva/announcement), course code if available, and due date/time.
    - Second sentence: Mention the key action required (what the student needs to do).
    - If "location" is present, include the venue/room in the first sentence.
    - If "student_slot_details" exists, explicitly state the student's specific slot/timing in the second sentence.
    - If "priority" is high (>7), mention urgency.

    Keep it to exactly 2 sentences. Be specific and actionable.
    """

    # Use temperature=0.3 for slight variation while maintaining consistency
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content.strip()
