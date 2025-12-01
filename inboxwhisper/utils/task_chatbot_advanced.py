"""
Advanced RAG implementation for InboxWhisper+ chatbot.

This module provides multiple RAG strategies:
1. Semantic search using local embeddings (sentence-transformers)
2. Hybrid search (keyword + semantic)
3. Re-ranking with cross-encoder
4. Temporal weighting (boost recent tasks)

Usage:
    from utils.task_chatbot_advanced import answer_task_question_advanced
    
    answer = answer_task_question_advanced(
        question="When is my OS viva?",
        tasks=all_tasks,
        strategy="hybrid"  # or "semantic", "rerank", "hybrid_rerank"
    )
"""

from openai import OpenAI
from utils.config import Config
from utils.task_presenter import concise_task_lines
from typing import List, Dict, Optional
import numpy as np
from collections import Counter
import math
from datetime import datetime
import dateutil.parser

client = OpenAI(api_key=Config.OPENAI_API_KEY)

# Try to import sentence-transformers (optional dependency)
try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("Warning: sentence-transformers not installed. Install with: pip install sentence-transformers")
    print("Falling back to keyword-based retrieval.")


# Global model instances (loaded once, reused)
_embedding_model = None
_reranker_model = None


def get_embedding_model():
    """Lazy-load embedding model."""
    global _embedding_model
    if _embedding_model is None and HAS_SENTENCE_TRANSFORMERS:
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedding_model


def get_reranker_model():
    """Lazy-load reranker model."""
    global _reranker_model
    if _reranker_model is None and HAS_SENTENCE_TRANSFORMERS:
        _reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return _reranker_model


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def bm25_score(query_tokens: List[str], task_text: str, k1: float = 1.5, b: float = 0.75) -> float:
    """
    BM25 keyword scoring algorithm.
    
    Args:
        query_tokens: List of query terms
        task_text: Text to score against
        k1: Term frequency saturation parameter
        b: Length normalization parameter
    """
    task_tokens = task_text.lower().split()
    if not task_tokens:
        return 0.0
    
    task_freq = Counter(task_tokens)
    doc_length = len(task_tokens)
    avg_doc_length = doc_length  # Simplified (would normally be average across all docs)
    
    score = 0.0
    for token in query_tokens:
        if token in task_freq:
            tf = task_freq[token]
            # Simplified IDF (in production, compute from full corpus)
            idf = math.log(2.0)  # log((N + 1) / df) simplified
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_length / max(avg_doc_length, 1))))
    return score


def select_relevant_tasks_semantic(question: str, tasks: List[Dict], limit: int = 5) -> List[Dict]:
    """
    Semantic search using sentence-transformers embeddings.
    
    This is the recommended upgrade from keyword search.
    """
    if not HAS_SENTENCE_TRANSFORMERS or not tasks:
        # Fallback to keyword search
        return select_relevant_tasks_keyword(question, tasks, limit)
    
    model = get_embedding_model()
    
    # Encode question
    question_embedding = model.encode(question, convert_to_numpy=True)
    
    # Prepare task texts and encode in batch (much faster)
    task_texts = []
    for task in tasks:
        task_text = " ".join(filter(None, [
            task.get("title", ""),
            task.get("summary", ""),
            task.get("action_item", ""),
            task.get("course", ""),
            task.get("location", "")
        ]))
        task_texts.append(task_text)
    
    if not task_texts:
        return []
    
    # Batch encode all tasks
    task_embeddings = model.encode(task_texts, convert_to_numpy=True)
    
    # Compute cosine similarities
    similarities = np.dot(task_embeddings, question_embedding) / (
        np.linalg.norm(task_embeddings, axis=1) * np.linalg.norm(question_embedding) + 1e-8
    )
    
    # Get top N indices
    top_indices = np.argsort(similarities)[::-1][:limit]
    return [tasks[i] for i in top_indices]


def select_relevant_tasks_keyword(question: str, tasks: List[Dict], limit: int = 5) -> List[Dict]:
    """
    Original keyword-based retrieval (fallback).
    """
    question_lower = question.lower()
    query_tokens = question_lower.split()
    
    scored = []
    for task in tasks:
        haystack = " ".join(filter(None, [
            task.get("title"),
            task.get("summary"),
            task.get("action_item"),
            task.get("course"),
            task.get("due_date"),
        ])).lower()
        
        score = sum(1 for token in query_tokens if token in haystack)
        if score > 0:
            scored.append((score, task))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    return [task for _, task in scored[:limit]] if scored else tasks[:limit]


def select_relevant_tasks_hybrid(
    question: str, 
    tasks: List[Dict], 
    limit: int = 5, 
    alpha: float = 0.6
) -> List[Dict]:
    """
    Hybrid search: combines BM25 (keyword) and semantic search.
    
    Args:
        alpha: Weight for semantic search (0-1). 0.6 means 60% semantic, 40% keyword.
    """
    if not HAS_SENTENCE_TRANSFORMERS or not tasks:
        return select_relevant_tasks_keyword(question, tasks, limit)
    
    model = get_embedding_model()
    query_tokens = question.lower().split()
    
    # Prepare task texts
    task_texts = []
    for task in tasks:
        task_text = " ".join(filter(None, [
            task.get("title", ""),
            task.get("summary", ""),
            task.get("action_item", ""),
            task.get("course", "")
        ]))
        task_texts.append(task_text)
    
    if not task_texts:
        return []
    
    # Semantic scores
    question_embedding = model.encode(question, convert_to_numpy=True)
    task_embeddings = model.encode(task_texts, convert_to_numpy=True)
    semantic_scores = np.dot(task_embeddings, question_embedding) / (
        np.linalg.norm(task_embeddings, axis=1) * np.linalg.norm(question_embedding) + 1e-8
    )
    
    # Normalize semantic scores to 0-1
    if semantic_scores.max() > semantic_scores.min():
        semantic_scores = (semantic_scores - semantic_scores.min()) / (
            semantic_scores.max() - semantic_scores.min() + 1e-8
        )
    
    # BM25 scores
    bm25_scores = np.array([bm25_score(query_tokens, text) for text in task_texts])
    if bm25_scores.max() > bm25_scores.min():
        bm25_scores = (bm25_scores - bm25_scores.min()) / (
            bm25_scores.max() - bm25_scores.min() + 1e-8
        )
    
    # Combine scores
    combined_scores = alpha * semantic_scores + (1 - alpha) * bm25_scores
    
    # Get top N
    top_indices = np.argsort(combined_scores)[::-1][:limit]
    return [tasks[i] for i in top_indices]


def select_relevant_tasks_rerank(
    question: str, 
    tasks: List[Dict], 
    limit: int = 5, 
    initial_k: int = 20
) -> List[Dict]:
    """
    Two-stage retrieval: fast initial retrieval + accurate re-ranking.
    
    Args:
        initial_k: Number of candidates to retrieve before re-ranking
    """
    if not HAS_SENTENCE_TRANSFORMERS or not tasks:
        return select_relevant_tasks_keyword(question, tasks, limit)
    
    # Stage 1: Fast initial retrieval (semantic or hybrid)
    candidates = select_relevant_tasks_hybrid(question, tasks, limit=initial_k)
    
    if not candidates:
        return []
    
    # Stage 2: Re-rank with cross-encoder
    reranker = get_reranker_model()
    
    pairs = []
    for task in candidates:
        task_text = " ".join(filter(None, [
            task.get("title", ""),
            task.get("summary", ""),
            task.get("action_item", "")
        ]))
        pairs.append([question, task_text])
    
    # Get re-ranking scores
    rerank_scores = reranker.predict(pairs)
    
    # Sort by re-ranking scores
    scored_candidates = list(zip(candidates, rerank_scores))
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    
    return [task for task, _ in scored_candidates[:limit]]


def select_relevant_tasks_temporal(
    question: str, 
    tasks: List[Dict], 
    limit: int = 5, 
    recency_weight: float = 0.15
) -> List[Dict]:
    """
    Semantic search with recency boost for recent tasks.
    
    Args:
        recency_weight: How much to weight recency (0-1). 0.15 = 15% recency, 85% semantic.
    """
    if not HAS_SENTENCE_TRANSFORMERS or not tasks:
        return select_relevant_tasks_keyword(question, tasks, limit)
    
    model = get_embedding_model()
    
    # Get semantic scores
    question_embedding = model.encode(question, convert_to_numpy=True)
    task_texts = []
    for task in tasks:
        task_text = " ".join(filter(None, [
            task.get("title", ""),
            task.get("summary", ""),
            task.get("action_item", ""),
            task.get("course", "")
        ]))
        task_texts.append(task_text)
    
    task_embeddings = model.encode(task_texts, convert_to_numpy=True)
    semantic_scores = np.dot(task_embeddings, question_embedding) / (
        np.linalg.norm(task_embeddings, axis=1) * np.linalg.norm(question_embedding) + 1e-8
    )
    
    # Normalize semantic scores
    if semantic_scores.max() > semantic_scores.min():
        semantic_scores = (semantic_scores - semantic_scores.min()) / (
            semantic_scores.max() - semantic_scores.min() + 1e-8
        )
    
    # Compute recency scores
    now = datetime.now()
    recency_scores = []
    for task in tasks:
        created = task.get("created_at") or task.get("updated_at")
        if created:
            try:
                created_dt = dateutil.parser.parse(created)
                if created_dt.tzinfo:
                    created_dt = created_dt.replace(tzinfo=None)
                days_ago = (now - created_dt).days
                # Exponential decay: newer = higher score
                recency = math.exp(-days_ago / 30)  # 30-day half-life
            except:
                recency = 0.5
        else:
            recency = 0.5
        recency_scores.append(recency)
    
    recency_scores = np.array(recency_scores)
    if recency_scores.max() > recency_scores.min():
        recency_scores = (recency_scores - recency_scores.min()) / (
            recency_scores.max() - recency_scores.min() + 1e-8
        )
    
    # Combine semantic and recency
    combined = (1 - recency_weight) * semantic_scores + recency_weight * recency_scores
    
    # Get top N
    top_indices = np.argsort(combined)[::-1][:limit]
    return [tasks[i] for i in top_indices]


def answer_task_question_advanced(
    question: str,
    tasks: List[Dict],
    strategy: str = "hybrid",
    limit: int = 5
) -> str:
    """
    Advanced RAG-based question answering with multiple retrieval strategies.
    
    Args:
        question: User's question about tasks
        tasks: List of task dictionaries from database
        strategy: One of "semantic", "hybrid", "rerank", "temporal", "keyword"
        limit: Number of tasks to retrieve
    
    Returns:
        str: Answer generated by LLM based on retrieved tasks
    """
    # Select retrieval strategy
    if strategy == "semantic":
        relevant = select_relevant_tasks_semantic(question, tasks, limit)
    elif strategy == "hybrid":
        relevant = select_relevant_tasks_hybrid(question, tasks, limit)
    elif strategy == "rerank":
        relevant = select_relevant_tasks_rerank(question, tasks, limit)
    elif strategy == "temporal":
        relevant = select_relevant_tasks_temporal(question, tasks, limit)
    else:  # "keyword" or fallback
        relevant = select_relevant_tasks_keyword(question, tasks, limit)
    
    # Format context
    context_blocks = []
    for idx, task in enumerate(relevant, start=1):
        lines = concise_task_lines(task)
        context_blocks.append(f"Task {idx}:\n" + "\n".join(lines))
    
    context_text = "\n\n".join(context_blocks) if context_blocks else "No tasks available."
    
    # Generate answer with LLM
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


# Backward compatibility: keep original function name
def answer_task_question(question: str, tasks: List[Dict]) -> str:
    """Original function signature for backward compatibility."""
    return answer_task_question_advanced(question, tasks, strategy="hybrid" if HAS_SENTENCE_TRANSFORMERS else "keyword")


