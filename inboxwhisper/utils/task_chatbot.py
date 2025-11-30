from openai import OpenAI
from utils.config import Config
from utils.task_presenter import concise_task_lines

client = OpenAI(api_key=Config.OPENAI_API_KEY)


def _task_score(task, question_lower):
    haystack = " ".join(
        filter(
            None,
            [
                task.get("title"),
                task.get("summary"),
                task.get("action_item"),
                task.get("course"),
                task.get("due_date"),
            ],
        )
    ).lower()
    score = 0
    for token in question_lower.split():
        if token in haystack:
            score += 1
    return score


def select_relevant_tasks(question, tasks, limit=5):
    question_lower = question.lower()
    scored = []
    for t in tasks:
        score = _task_score(t, question_lower)
        if score > 0:
            scored.append((score, t))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [t for _, t in scored[:limit]] or tasks[:1]


def answer_task_question(question, tasks):
    relevant = select_relevant_tasks(question, tasks)
    context_blocks = []
    for idx, task in enumerate(relevant, start=1):
        lines = concise_task_lines(task)
        context_blocks.append(f"Task {idx}:\n" + "\n".join(lines))

    context_text = "\n\n".join(context_blocks) if context_blocks else "No tasks available."

    prompt = f"""
You are InboxWhisper+, a personal academic assistant. You have access to the following tasks:

{context_text}

Answer this question using ONLY the provided tasks. If the answer is not in the tasks, say you cannot find it. Keep responses short, actionable, and cite the task number when helpful.

Question: {question}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that answers only from provided task data.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()

