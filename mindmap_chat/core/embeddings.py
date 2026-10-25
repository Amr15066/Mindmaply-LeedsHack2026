"""
Embedding utilities for similarity matching.
Uses the configured LLM client for embeddings.
"""

from typing import List, Sequence

import numpy as np

from mindmap_chat.llm.base import LLMClient


def compute_similarity(embedding1: Sequence[float], embedding2: Sequence[float]) -> float:
    """
    Compute cosine similarity between two embeddings.

    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector

    Returns:
        Similarity score in [-1, 1]; 0.0 for empty, zero or mismatched vectors
    """
    if len(embedding1) == 0 or len(embedding2) == 0 or len(embedding1) != len(embedding2):
        return 0.0
    a = np.asarray(embedding1, dtype=np.float64)
    b = np.asarray(embedding2, dtype=np.float64)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(a @ b / norm)


def cosine_batch(query: Sequence[float], matrix: np.ndarray) -> np.ndarray:
    """
    Cosine similarity of one query vector against every row of a matrix.

    Args:
        query: (D,) vector
        matrix: (N, D) array

    Returns:
        (N,) similarities; rows (or a query) with zero norm score 0.0
    """
    q = np.asarray(query, dtype=np.float64)
    m = np.asarray(matrix, dtype=np.float64)
    if m.size == 0:
        return np.zeros(len(m))
    norms = np.linalg.norm(m, axis=1) * np.linalg.norm(q)
    with np.errstate(divide="ignore", invalid="ignore"):
        sims = (m @ q) / norms
    return np.where(norms > 0, sims, 0.0)


def embed_text(llm_client: LLMClient, text: str) -> List[float]:
    """
    Generate embedding for text.

    Args:
        llm_client: LLM client instance
        text: Text to embed

    Returns:
        Embedding vector
    """
    return llm_client.embed(text)
