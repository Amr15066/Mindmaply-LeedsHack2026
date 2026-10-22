"""
Embedding utilities for similarity matching.
Uses the configured LLM client for embeddings.
"""

from typing import List
from mindmap_chat.llm.base import LLMClient


def compute_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """
    Compute cosine similarity between two embeddings.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        
    Returns:
        Similarity score (0-1)
    """
    if not embedding1 or not embedding2:
        return 0.0
    
    dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
    magnitude1 = sum(a * a for a in embedding1) ** 0.5
    magnitude2 = sum(b * b for b in embedding2) ** 0.5
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    return dot_product / (magnitude1 * magnitude2)


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
