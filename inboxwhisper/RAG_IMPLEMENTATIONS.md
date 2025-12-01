# RAG Implementation Strategies for InboxWhisper+

This document outlines various RAG (Retrieval Augmented Generation) approaches you can implement in your project, from simple improvements to advanced techniques.

## Current Implementation (Baseline)

**Location**: `utils/task_chatbot.py`

**Method**: Keyword-based retrieval
- Scores tasks by token overlap between question and task fields
- Returns top 5 tasks
- Passes context to LLM for generation

**Limitations**:
- No semantic understanding (misses synonyms, related concepts)
- Doesn't handle paraphrasing well
- No re-ranking or query expansion

---

## 1. Semantic Search RAG (Recommended First Upgrade)

**Concept**: Use embeddings to find semantically similar tasks, not just keyword matches.

### Implementation Options:

#### Option A: OpenAI Embeddings (Easiest)
```python
# utils/task_chatbot_semantic.py
from openai import OpenAI
import numpy as np
from typing import List, Dict

client = OpenAI(api_key=Config.OPENAI_API_KEY)

def get_embedding(text: str) -> List[float]:
    """Get embedding vector for text using OpenAI."""
    response = client.embeddings.create(
        model="text-embedding-3-small",  # or text-embedding-ada-002
        input=text
    )
    return response.data[0].embedding

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def select_relevant_tasks_semantic(question: str, tasks: List[Dict], limit: int = 5) -> List[Dict]:
    """Select tasks using semantic similarity."""
    # Get question embedding
    question_embedding = get_embedding(question)
    
    # Score each task
    scored = []
    for task in tasks:
        # Create searchable text from task fields
        task_text = " ".join([
            task.get("title", ""),
            task.get("summary", ""),
            task.get("action_item", ""),
            task.get("course", ""),
            task.get("location", "")
        ])
        
        # Get task embedding
        task_embedding = get_embedding(task_text)
        
        # Compute similarity
        similarity = cosine_similarity(question_embedding, task_embedding)
        scored.append((similarity, task))
    
    # Sort by similarity and return top N
    scored.sort(key=lambda x: x[0], reverse=True)
    return [task for _, task in scored[:limit]]
```

**Pros**: Better semantic understanding, handles synonyms
**Cons**: Requires API calls for each task (can be slow), costs money

#### Option B: Local Embeddings (Faster, Free)
```python
# Requires: pip install sentence-transformers
from sentence_transformers import SentenceTransformer
import numpy as np

# Load model once (cache it)
model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight, fast

def select_relevant_tasks_semantic_local(question: str, tasks: List[Dict], limit: int = 5) -> List[Dict]:
    """Select tasks using local semantic similarity."""
    # Get question embedding
    question_embedding = model.encode(question)
    
    # Batch encode all tasks (much faster)
    task_texts = []
    for task in tasks:
        task_text = " ".join([
            task.get("title", ""),
            task.get("summary", ""),
            task.get("action_item", ""),
            task.get("course", "")
        ])
        task_texts.append(task_text)
    
    task_embeddings = model.encode(task_texts)  # Batch encoding
    
    # Compute similarities
    similarities = np.dot(task_embeddings, question_embedding) / (
        np.linalg.norm(task_embeddings, axis=1) * np.linalg.norm(question_embedding)
    )
    
    # Get top N
    top_indices = np.argsort(similarities)[::-1][:limit]
    return [tasks[i] for i in top_indices]
```

**Pros**: Fast, free, works offline
**Cons**: Slightly less accurate than OpenAI embeddings

---

## 2. Hybrid Search RAG (Best of Both Worlds)

**Concept**: Combine keyword matching (BM25) with semantic search for better results.

```python
# utils/task_chatbot_hybrid.py
from sentence_transformers import SentenceTransformer
import numpy as np
from collections import Counter
import math

model = SentenceTransformer('all-MiniLM-L6-v2')

def bm25_score(query_tokens: List[str], task_text: str, k1: float = 1.5, b: float = 0.75) -> float:
    """BM25 keyword scoring."""
    task_tokens = task_text.lower().split()
    task_freq = Counter(task_tokens)
    doc_length = len(task_tokens)
    avg_doc_length = doc_length  # Simplified
    
    score = 0
    for token in query_tokens:
        if token in task_freq:
            tf = task_freq[token]
            idf = math.log((1 + 1) / (1 + 1))  # Simplified IDF
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_length / avg_doc_length)))
    return score

def select_relevant_tasks_hybrid(question: str, tasks: List[Dict], limit: int = 5, alpha: float = 0.5) -> List[Dict]:
    """
    Hybrid retrieval: combines BM25 (keyword) and semantic search.
    
    Args:
        alpha: Weight for semantic (0-1). 0.5 = equal weight.
    """
    query_tokens = question.lower().split()
    
    # Prepare task texts
    task_texts = []
    for task in tasks:
        task_text = " ".join([
            task.get("title", ""),
            task.get("summary", ""),
            task.get("action_item", ""),
            task.get("course", "")
        ])
        task_texts.append(task_text)
    
    # Semantic scores
    question_embedding = model.encode(question)
    task_embeddings = model.encode(task_texts)
    semantic_scores = np.dot(task_embeddings, question_embedding) / (
        np.linalg.norm(task_embeddings, axis=1) * np.linalg.norm(question_embedding)
    )
    
    # Normalize scores to 0-1
    semantic_scores = (semantic_scores - semantic_scores.min()) / (semantic_scores.max() - semantic_scores.min() + 1e-8)
    
    # BM25 scores
    bm25_scores = [bm25_score(query_tokens, text) for text in task_texts]
    bm25_scores = np.array(bm25_scores)
    if bm25_scores.max() > 0:
        bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)
    
    # Combine scores
    combined_scores = alpha * semantic_scores + (1 - alpha) * bm25_scores
    
    # Get top N
    top_indices = np.argsort(combined_scores)[::-1][:limit]
    return [tasks[i] for i in top_indices]
```

**Pros**: Combines precision (keywords) with recall (semantics)
**Cons**: More complex, requires tuning alpha parameter

---

## 3. Multi-Vector RAG

**Concept**: Store multiple representations of each task (summary, action_item, full text) and search across all of them.

```python
# utils/task_chatbot_multivector.py
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

def create_task_chunks(task: Dict) -> List[Dict]:
    """Split task into multiple searchable chunks."""
    chunks = []
    
    # Chunk 1: Title + Course
    if task.get("title") or task.get("course"):
        chunks.append({
            "type": "metadata",
            "text": f"{task.get('title', '')} {task.get('course', '')}".strip(),
            "task_id": task.get("id")
        })
    
    # Chunk 2: Summary
    if task.get("summary"):
        chunks.append({
            "type": "summary",
            "text": task.get("summary"),
            "task_id": task.get("id")
        })
    
    # Chunk 3: Action item
    if task.get("action_item"):
        chunks.append({
            "type": "action",
            "text": task.get("action_item"),
            "task_id": task.get("id")
        })
    
    # Chunk 4: Location + Due date
    location_date = f"{task.get('location', '')} {task.get('due_date', '')}".strip()
    if location_date:
        chunks.append({
            "type": "details",
            "text": location_date,
            "task_id": task.get("id")
        })
    
    return chunks

def select_relevant_tasks_multivector(question: str, tasks: List[Dict], limit: int = 5) -> List[Dict]:
    """Multi-vector retrieval: search across different task representations."""
    # Create chunks for all tasks
    all_chunks = []
    task_map = {}  # Map task_id -> full task
    for task in tasks:
        task_map[task.get("id")] = task
        chunks = create_task_chunks(task)
        all_chunks.extend(chunks)
    
    if not all_chunks:
        return []
    
    # Encode question and all chunks
    question_embedding = model.encode(question)
    chunk_texts = [chunk["text"] for chunk in all_chunks]
    chunk_embeddings = model.encode(chunk_texts)
    
    # Compute similarities
    similarities = np.dot(chunk_embeddings, question_embedding) / (
        np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(question_embedding)
    )
    
    # Get top chunks
    top_chunk_indices = np.argsort(similarities)[::-1][:limit * 2]  # Get more chunks
    
    # Deduplicate by task_id, keeping best chunk per task
    seen_tasks = {}
    for idx in top_chunk_indices:
        chunk = all_chunks[idx]
        task_id = chunk["task_id"]
        if task_id not in seen_tasks or similarities[idx] > seen_tasks[task_id][1]:
            seen_tasks[task_id] = (task_map[task_id], similarities[idx])
    
    # Return top tasks
    result = sorted(seen_tasks.values(), key=lambda x: x[1], reverse=True)[:limit]
    return [task for task, _ in result]
```

**Pros**: More granular search, can find tasks by any field
**Cons**: More complex, requires chunking logic

---

## 4. Query Expansion RAG

**Concept**: Expand the user's question with related terms before retrieval.

```python
# utils/task_chatbot_expanded.py
from openai import OpenAI

client = OpenAI(api_key=Config.OPENAI_API_KEY)

def expand_query(question: str) -> str:
    """Use LLM to expand query with synonyms and related terms."""
    prompt = f"""
    Expand this question with synonyms and related academic terms.
    Return the original question plus 2-3 related terms, separated by spaces.
    
    Question: {question}
    
    Expanded query (just the terms, no explanation):
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    
    expanded = response.choices[0].message.content.strip()
    # Combine original and expanded
    return f"{question} {expanded}"

def select_relevant_tasks_expanded(question: str, tasks: List[Dict], limit: int = 5) -> List[Dict]:
    """Retrieval with query expansion."""
    # Expand query
    expanded_query = expand_query(question)
    
    # Use existing semantic or keyword search with expanded query
    # (You can combine with any of the previous methods)
    return select_relevant_tasks_semantic_local(expanded_query, tasks, limit)
```

**Pros**: Better recall for related concepts
**Cons**: Extra LLM call, may introduce noise

---

## 5. Re-ranking RAG

**Concept**: Use a cross-encoder to re-rank initial retrieval results for better precision.

```python
# utils/task_chatbot_rerank.py
from sentence_transformers import CrossEncoder

# Load cross-encoder model (more accurate but slower than bi-encoder)
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def select_relevant_tasks_rerank(question: str, tasks: List[Dict], limit: int = 5, initial_k: int = 20) -> List[Dict]:
    """
    Two-stage retrieval: fast initial retrieval + accurate re-ranking.
    
    Args:
        initial_k: Number of candidates to retrieve before re-ranking
    """
    # Stage 1: Fast initial retrieval (use semantic or hybrid)
    from utils.task_chatbot_semantic import select_relevant_tasks_semantic_local
    candidates = select_relevant_tasks_semantic_local(question, tasks, limit=initial_k)
    
    if not candidates:
        return []
    
    # Stage 2: Re-rank with cross-encoder
    pairs = []
    for task in candidates:
        task_text = " ".join([
            task.get("title", ""),
            task.get("summary", ""),
            task.get("action_item", "")
        ])
        pairs.append([question, task_text])
    
    # Get re-ranking scores
    scores = reranker.predict(pairs)
    
    # Sort by re-ranking scores
    scored_candidates = list(zip(candidates, scores))
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    
    return [task for task, _ in scored_candidates[:limit]]
```

**Pros**: Best accuracy, two-stage approach balances speed and precision
**Cons**: Slower than single-stage, requires more computation

---

## 6. Contextual RAG (Conversation History)

**Concept**: Use previous conversation turns to improve retrieval.

```python
# utils/task_chatbot_contextual.py

def select_relevant_tasks_contextual(
    question: str, 
    tasks: List[Dict], 
    conversation_history: List[Dict],
    limit: int = 5
) -> List[Dict]:
    """
    Retrieval that considers conversation context.
    
    Args:
        conversation_history: List of {"role": "user/assistant", "content": "..."}
    """
    # Build context-aware query
    # Option 1: Concatenate recent questions
    recent_questions = [
        msg["content"] for msg in conversation_history[-3:] 
        if msg["role"] == "user"
    ]
    context_query = " ".join(recent_questions + [question])
    
    # Option 2: Use LLM to rewrite query with context
    # (More sophisticated but requires extra API call)
    
    # Use semantic search with context-aware query
    return select_relevant_tasks_semantic_local(context_query, tasks, limit)
```

**Pros**: Better for follow-up questions, maintains context
**Cons**: Requires storing conversation history

---

## 7. Temporal RAG (Recency Weighting)

**Concept**: Boost recently created/updated tasks in retrieval.

```python
# utils/task_chatbot_temporal.py
from datetime import datetime
import dateutil.parser

def select_relevant_tasks_temporal(question: str, tasks: List[Dict], limit: int = 5, recency_weight: float = 0.2) -> List[Dict]:
    """
    Retrieval with recency boost for recent tasks.
    
    Args:
        recency_weight: How much to weight recency (0-1)
    """
    # Get semantic scores
    semantic_scores = ...  # Use your semantic search method
    
    # Compute recency scores
    now = datetime.now()
    recency_scores = []
    for task in tasks:
        created = task.get("created_at")
        if created:
            try:
                created_dt = dateutil.parser.parse(created)
                days_ago = (now - created_dt.replace(tzinfo=None)).days
                # Exponential decay: newer = higher score
                recency = math.exp(-days_ago / 30)  # 30-day half-life
            except:
                recency = 0.5  # Default for unparseable dates
        else:
            recency = 0.5
        recency_scores.append(recency)
    
    # Normalize recency scores
    recency_scores = np.array(recency_scores)
    if recency_scores.max() > 0:
        recency_scores = (recency_scores - recency_scores.min()) / (recency_scores.max() - recency_scores.min() + 1e-8)
    
    # Combine semantic and recency
    combined = (1 - recency_weight) * semantic_scores + recency_weight * recency_scores
    
    # Return top N
    top_indices = np.argsort(combined)[::-1][:limit]
    return [tasks[i] for i in top_indices]
```

**Pros**: Prioritizes recent tasks, useful for time-sensitive queries
**Cons**: May miss older relevant tasks

---

## 8. Task-Type Specific RAG

**Concept**: Use different retrieval strategies based on query type (deadline questions vs. course questions).

```python
# utils/task_chatbot_typed.py

def classify_query_type(question: str) -> str:
    """Classify query into: deadline, course, location, general."""
    question_lower = question.lower()
    
    if any(word in question_lower for word in ["when", "due", "deadline", "date", "time"]):
        return "deadline"
    elif any(word in question_lower for word in ["course", "class", "subject"]):
        return "course"
    elif any(word in question_lower for word in ["where", "location", "room", "venue"]):
        return "location"
    else:
        return "general"

def select_relevant_tasks_typed(question: str, tasks: List[Dict], limit: int = 5) -> List[Dict]:
    """Retrieval with query-type-specific strategies."""
    query_type = classify_query_type(question)
    
    if query_type == "deadline":
        # Filter to tasks with due dates, then rank by deadline proximity
        dated_tasks = [t for t in tasks if t.get("due_date")]
        # Sort by due date (closest first)
        dated_tasks.sort(key=lambda t: t.get("due_date") or "9999-12-31")
        return dated_tasks[:limit]
    
    elif query_type == "course":
        # Extract course code from question, filter by course
        # Then use semantic search within filtered set
        course_keywords = [t.get("course") for t in tasks if t.get("course")]
        # Simple keyword match for course
        matching_tasks = [
            t for t in tasks 
            if any(course in question.lower() for course in [t.get("course", "").lower()])
        ]
        if matching_tasks:
            return select_relevant_tasks_semantic_local(question, matching_tasks, limit)
    
    elif query_type == "location":
        # Filter to tasks with locations, then semantic search
        located_tasks = [t for t in tasks if t.get("location")]
        return select_relevant_tasks_semantic_local(question, located_tasks, limit)
    
    else:
        # General semantic search
        return select_relevant_tasks_semantic_local(question, tasks, limit)
```

**Pros**: More precise for specific query types
**Cons**: Requires query classification logic

---

## Implementation Recommendations

### Quick Win (1-2 hours):
1. **Semantic Search with Local Embeddings** (#1, Option B)
   - Install: `pip install sentence-transformers`
   - Replace `select_relevant_tasks()` in `task_chatbot.py`
   - Immediate improvement in retrieval quality

### Medium Effort (4-6 hours):
2. **Hybrid Search** (#2)
   - Combines your current keyword approach with semantic
   - Best balance of speed and accuracy
   - Tune `alpha` parameter (0.5 is good starting point)

### Advanced (1-2 days):
3. **Re-ranking RAG** (#5)
   - Two-stage: fast initial retrieval + accurate re-ranking
   - Best accuracy, still reasonably fast
   - Install: `pip install sentence-transformers` (includes CrossEncoder)

### For Production:
4. **Multi-Vector + Hybrid + Re-ranking**
   - Most comprehensive approach
   - Best accuracy and recall
   - Requires caching embeddings for performance

---

## Performance Considerations

- **Embedding Caching**: Pre-compute and store task embeddings in SQLite to avoid recomputing
- **Batch Encoding**: Always encode multiple texts at once (sentence-transformers supports this)
- **Model Choice**: 
  - `all-MiniLM-L6-v2`: Fast, 384 dims, good for <1000 tasks
  - `all-mpnet-base-v2`: Slower, 768 dims, better accuracy
  - OpenAI `text-embedding-3-small`: Best accuracy, requires API calls

---

## Example: Complete Implementation

See `utils/task_chatbot_advanced.py` for a production-ready implementation combining:
- Semantic search (local embeddings)
- Hybrid keyword + semantic
- Re-ranking
- Temporal weighting
- Query expansion (optional)

