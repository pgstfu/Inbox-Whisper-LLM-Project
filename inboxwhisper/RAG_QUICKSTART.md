# RAG Quick Start Guide

## Current Status

Your project currently uses **keyword-based RAG** in `utils/task_chatbot.py`. This works but has limitations:
- Misses synonyms (e.g., "exam" vs "test")
- Doesn't understand semantic similarity
- Poor handling of paraphrased questions

## Quick Upgrade (Recommended)

### Step 1: Install Dependencies

```bash
pip install sentence-transformers
```

### Step 2: Update Your Chatbot

Replace the import in `app.py`:

```python
# OLD:
from utils.task_chatbot import answer_task_question

# NEW:
from utils.task_chatbot_advanced import answer_task_question_advanced
```

### Step 3: Update the Chatbot Tab

In `app.py`, find the chatbot section and update:

```python
# OLD:
answer = answer_task_question(question, tasks_snapshot)

# NEW (with strategy selection):
strategy = st.selectbox(
    "Retrieval Strategy",
    ["hybrid", "semantic", "rerank", "temporal", "keyword"],
    index=0,
    help="hybrid = best balance, semantic = fast, rerank = most accurate"
)
answer = answer_task_question_advanced(question, tasks_snapshot, strategy=strategy)
```

That's it! Your chatbot now uses semantic search.

## Strategy Comparison

| Strategy | Speed | Accuracy | Best For |
|----------|-------|----------|----------|
| **keyword** | ⚡⚡⚡ Fastest | ⭐⭐ Basic | Small datasets, exact matches |
| **semantic** | ⚡⚡ Fast | ⭐⭐⭐ Good | General queries, synonyms |
| **hybrid** | ⚡⚡ Fast | ⭐⭐⭐⭐ Very Good | **Recommended default** |
| **rerank** | ⚡ Slower | ⭐⭐⭐⭐⭐ Best | When accuracy is critical |
| **temporal** | ⚡⚡ Fast | ⭐⭐⭐ Good | Recent tasks, time-sensitive queries |

## Testing

Try these queries to see the difference:

1. **"When is my operating systems exam?"**
   - Keyword: Might miss if email says "OS viva" instead of "exam"
   - Semantic: Finds it even with different wording

2. **"What do I need to submit for CSO203?"**
   - Keyword: Needs exact course code match
   - Semantic: Understands "CSO203" = "Operating Systems"

3. **"Where is my viva?"**
   - Keyword: Searches for "viva" keyword
   - Semantic: Also finds "oral exam", "presentation"

## Advanced: Custom Strategy

You can also create your own strategy by combining methods:

```python
# In app.py or a custom function
def custom_retrieval(question, tasks):
    # First filter by query type
    if "when" in question.lower() or "due" in question.lower():
        # For deadline questions, use temporal + semantic
        return select_relevant_tasks_temporal(question, tasks, limit=5)
    else:
        # For general questions, use hybrid
        return select_relevant_tasks_hybrid(question, tasks, limit=5)
```

## Performance Tips

1. **Cache Embeddings**: For production, pre-compute and store task embeddings in SQLite
2. **Batch Encoding**: The code already batches encoding (faster than one-by-one)
3. **Model Choice**: 
   - `all-MiniLM-L6-v2` (default): Fast, good for <1000 tasks
   - `all-mpnet-base-v2`: Slower but more accurate for larger datasets

## Troubleshooting

**"ModuleNotFoundError: No module named 'sentence_transformers'"**
- Run: `pip install sentence-transformers`

**Slow performance**
- Use "semantic" instead of "rerank" strategy
- Reduce `limit` parameter (fewer tasks retrieved)

**Poor results**
- Try "hybrid" strategy (combines keyword + semantic)
- Increase `limit` to retrieve more candidates
- Check if tasks have enough text in title/summary/action_item fields

## Next Steps

See `RAG_IMPLEMENTATIONS.md` for:
- Detailed explanations of each strategy
- Multi-vector RAG
- Query expansion
- Contextual RAG (conversation history)
- Task-type specific RAG


